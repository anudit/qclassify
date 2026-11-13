"""Feature contract, mirrored exactly by web/src/features.ts.

Every input is NFKC-normalized and lowercased first. The model sees two things:
  * one WORD vector per word (up to MAX_WORDS): hashed signed character 2-4-grams of "^word$" (L2-normalized),
    ten word-shape flags and the word's lexicon value (is it usually a name or an ordinary word?);
  * one GLOBAL vector for the whole string: hashed 2-4-grams of "^text$" (L2-normalized) and 20 shape features.
"""
from functools import lru_cache
from pathlib import Path
import unicodedata

import numpy as np

WORD_BINS = 2028
WORD_EXTRA = 12                       # 10 shape flags + lexicon value + known flag
WORD_DIM = 2048                       # WORD_BINS + WORD_EXTRA, zero-padded to the texture width WebGL 2 guarantees
GLOBAL_BINS = 1024
SHAPE_FEATURES = 20
GLOBAL_DIM = GLOBAL_BINS + SHAPE_FEATURES
LEX_BINS = 4096
MAX_WORDS = 16
MAX_CHARS = 256
FEATURE_VERSION = "nfkc-lower-words2to4-lex4096-global1024-v5"
REQUEST_WORDS = frozenset("how what where when why who find show latest recent best compare search".split())
ARTIFACT_WORDS = frozenset("biography biographies papers publications tutorials courses interviews interview blog projects announcements github books news talks lecture lectures research tools libraries library frameworks software datasets opportunities jobs".split())
TECH_WORDS = frozenset("ai ml llm llms api sdk sql gpu cpu webgl webgpu onnx pytorch tensorflow javascript typescript python rust open source".split())
LEXICON_PATH = Path(__file__).with_name("lexicon.bin")


def canonicalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def fnv1a(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


def decode_lexicon(packed: bytes) -> np.ndarray:
    if len(packed) != LEX_BINS // 2:
        raise ValueError("Invalid lexicon length")
    b = np.frombuffer(packed, np.uint8)
    codes = np.empty(LEX_BINS, np.int64)
    codes[0::2], codes[1::2] = b & 15, b >> 4
    if codes.max() > 14:
        raise ValueError("Invalid lexicon code")
    return ((codes - 7) / 7).astype(np.float32)


@lru_cache(maxsize=1)
def lexicon() -> np.ndarray:
    return decode_lexicon(LEXICON_PATH.read_bytes())


def lex_value(word: str, table: np.ndarray) -> float:
    return float(table[fnv1a("#" + word) % LEX_BINS])


def _grams(padded: str, vector: np.ndarray, bins: int) -> None:
    for size in (2, 3, 4):
        for start in range(len(padded) - size + 1):
            hashed = fnv1a(padded[start:start + size])
            # A separate hash bit picks the sign, to reduce collision bias.
            vector[hashed % bins] += 1.0 if hashed & 0x80000000 else -1.0
    norm = np.linalg.norm(vector[:bins])
    if norm:
        vector[:bins] /= norm


def _prepare(text: str) -> tuple[str, str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a nonempty string.")
    raw = unicodedata.normalize("NFKC", text).strip()[:MAX_CHARS].lower()
    return raw, canonicalize(raw)


def _shape(raw: str, normalized: str) -> list[float]:
    # Lowercase input: the two uppercase slots are constant zero (kept to preserve the layout).
    words, tokens, length = raw.split(), normalized.split(), len(raw)
    return [
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


def featurize(text: str, table: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """-> (words [n, WORD_DIM], global [GLOBAL_DIM]) with 1 <= n <= MAX_WORDS."""
    table = lexicon() if table is None else table
    raw, normalized = _prepare(text)
    tokens = normalized.split(" ")[:MAX_WORDS]
    n = len(tokens)
    words = np.zeros((n, WORD_DIM), np.float32)
    for i, tok in enumerate(tokens):
        _grams("^" + tok + "$", words[i], WORD_BINS)
        v = lex_value(tok, table)
        words[i, WORD_BINS:WORD_BINS + WORD_EXTRA] = [
            min(len(tok) / 16, 1), float(any(c.isdecimal() for c in tok)),
            float(any(c.isalpha() and not c.isascii() for c in tok)),
            float(tok in REQUEST_WORDS), float(tok in ARTIFACT_WORDS), float(tok in TECH_WORDS),
            float(n == 1), float(i == 0 and n > 1), float(i == n - 1 and n > 1), float(not tok.isalpha()),
            v, float(v != 0)]
    glob = np.zeros(GLOBAL_DIM, np.float32)
    _grams("^" + normalized + "$", glob, GLOBAL_BINS)
    glob[GLOBAL_BINS:] = _shape(raw, normalized)
    return words, glob


def transform(texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padded batch: words [N, MAX_WORDS, WORD_DIM], mask [N, MAX_WORDS], global [N, GLOBAL_DIM]."""
    words = np.zeros((len(texts), MAX_WORDS, WORD_DIM), np.float32)
    mask = np.zeros((len(texts), MAX_WORDS), np.float32)
    glob = np.zeros((len(texts), GLOBAL_DIM), np.float32)
    for r, text in enumerate(texts):
        w, g = featurize(text)
        words[r, :len(w)], mask[r, :len(w)], glob[r] = w, 1, g
    return words, mask, glob
