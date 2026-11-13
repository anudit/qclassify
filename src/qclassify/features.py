"""Portable hashing, shape features and fixed intent hints. Keep identical in every runtime."""
import unicodedata

import numpy as np

HASH_BINS = 1024
SHAPE_FEATURES = 20
INPUT_DIM = HASH_BINS + SHAPE_FEATURES
FEATURE_VERSION = "nfkc-lower-fnv1a-char2to4-shape20-v3"
MAX_CHARS = 256
REQUEST_WORDS = frozenset("how what where when why who find show latest recent best compare search".split())
ARTIFACT_WORDS = frozenset("biography biographies papers publications tutorials courses interviews interview blog projects announcements github books news talks lecture lectures research tools libraries library frameworks software datasets opportunities jobs".split())
TECH_WORDS = frozenset("ai ml llm llms api sdk sql gpu cpu webgl webgpu onnx pytorch tensorflow javascript typescript python rust open source".split())


def canonicalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def fnv1a(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def featurize(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a nonempty string.")
    # Everything below sees lowercase text only: "Solar Energy" and "solar energy" are one input,
    # so the uppercase shape features are constant zero (slots kept to preserve the layout).
    raw = unicodedata.normalize("NFKC", text).strip()[:MAX_CHARS].lower()
    normalized = canonicalize(raw)
    vector = np.zeros(INPUT_DIM, dtype=np.float32)
    padded = "^" + normalized + "$"
    for size in (2, 3, 4):
        for start in range(len(padded) - size + 1):
            hashed = fnv1a(padded[start:start + size])
            # Use a separate bit for the sign to reduce collision bias.
            vector[hashed % HASH_BINS] += 1.0 if hashed & 0x80000000 else -1.0
    norm = np.linalg.norm(vector[:HASH_BINS])
    if norm:
        vector[:HASH_BINS] /= norm
    words = raw.split()
    tokens = normalized.split()
    length = len(raw)
    vector[HASH_BINS:] = [
        min(length / 80, 1), min(len(words) / 10, 1),
        sum(c.isdecimal() for c in raw) / length,
        sum(c.isupper() for c in raw) / length,
        sum(c.isalpha() for c in raw) / length,
        sum(c.isspace() for c in raw) / length,
        sum(not (c.isalnum() or c.isspace()) for c in raw) / length,
        sum(w[0].isupper() for w in words) / len(words),
        float("?" in raw), float("@" in raw), float("/" in raw),
        float("." in raw), float("-" in raw), float("'" in raw or "’" in raw),
        float(any(c.isalpha() and not c.isascii() for c in raw)),
        min(max(map(len, words)) / 24, 1),
        float(tokens[0] in REQUEST_WORDS),
        sum(t in ARTIFACT_WORDS for t in tokens) / len(tokens),
        sum(t in TECH_WORDS for t in tokens) / len(tokens),
        float("http://" in normalized or "https://" in normalized or "www." in normalized),
    ]
    return vector


def transform(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, INPUT_DIM), dtype=np.float32)
    return np.stack([featurize(text) for text in texts])
