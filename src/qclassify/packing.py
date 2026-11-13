"""Browser weight blob (qclassify-webgl-v3), with the bit width chosen on validation data only.

Layout, little-endian:
  lexicon   LEX_BINS 4-bit codes (value = (code - 7) / 7), low nibble = even slot
  W1        word layer, stored hidden-major (W1.T, so each unit's 2048 weights are contiguous):
            f16 scale per group of 32, then `bits`-bit codes packed LSB-first (value = (code - levels) * scale)
  W3        global layer, stored the same way (W3.T)
  tail      f32: b1[H], w2[H], b2, mix[2], b3[G], w4[G], vs[20], c
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from qclassify.features import (FEATURE_VERSION, GLOBAL_BINS, GLOBAL_DIM, LEX_BINS, LEXICON_PATH, MAX_CHARS,
                                MAX_WORDS, SHAPE_FEATURES, WORD_BINS, WORD_DIM, transform)
from qclassify.model import logits, sigmoid

GROUP_SIZE = 32
FORMAT = "qclassify-webgl-v3"
LAYOUT = ["lexicon-u4", "w1-grouped", "w3-grouped", "tail-f32"]
TAIL = ["b1", "w2", "b2", "mix", "b3", "w4", "vs", "c"]


def matrix_bytes(count: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    return -(-count // group_size) * 2 + -(-count * bits // 8)


def blob_bytes(hidden: int, global_hidden: int, bits: int) -> int:
    tail = 2 * hidden + 1 + 2 + 2 * global_hidden + SHAPE_FEATURES + 1
    return LEX_BINS // 2 + matrix_bytes(WORD_DIM * hidden, bits) + matrix_bytes(GLOBAL_DIM * global_hidden, bits) + tail * 4


def pack_matrix(flat: np.ndarray, bits: int, group_size: int = GROUP_SIZE) -> bytes:
    levels = (1 << (bits - 1)) - 1
    scales, codes = [], []
    for start in range(0, len(flat), group_size):
        group = flat[start:start + group_size]
        scale = np.float16(max(float(np.max(np.abs(group))) / levels, 2 ** -24))
        scales.append(scale)
        codes.extend((np.clip(np.rint(group / float(scale)), -levels, levels).astype(int) + levels).tolist())
    packed = bytearray(-(-len(codes) * bits // 8))
    for i, code in enumerate(codes):
        byte, shift = divmod(i * bits, 8)
        value = code << shift
        packed[byte] |= value & 255
        if shift + bits > 8:
            packed[byte + 1] |= value >> 8
    return np.array(scales, dtype="<f2").tobytes() + bytes(packed)


def unpack_matrix(blob: bytes, offset: int, count: int, bits: int, group_size: int = GROUP_SIZE) -> np.ndarray:
    scale_count = -(-count // group_size)
    scales = np.frombuffer(blob, dtype="<f2", count=scale_count, offset=offset).astype(np.float32)
    codes = blob[offset + scale_count * 2: offset + matrix_bytes(count, bits, group_size)]
    out = np.empty(count, np.float32)
    levels = (1 << (bits - 1)) - 1
    for i in range(count):
        byte, shift = divmod(i * bits, 8)
        value = codes[byte] | ((codes[byte + 1] if byte + 1 < len(codes) else 0) << 8)
        out[i] = (((value >> shift) & ((1 << bits) - 1)) - levels) * scales[i // group_size]
    return out


def pack(params: dict, bits: int, lexicon: bytes) -> bytes:
    tail = np.concatenate([np.asarray(params[k], np.float32).reshape(-1) for k in TAIL]).astype("<f4")
    return (lexicon + pack_matrix(params["w1"].T.reshape(-1), bits) +
            pack_matrix(params["w3"].T.reshape(-1), bits) + tail.tobytes())


def unpack(blob: bytes, hidden: int, global_hidden: int, bits: int) -> tuple[dict, bytes]:
    if len(blob) != blob_bytes(hidden, global_hidden, bits):
        raise ValueError("Invalid packed weight length")
    offset = LEX_BINS // 2
    lexicon = blob[:offset]
    w1 = unpack_matrix(blob, offset, WORD_DIM * hidden, bits).reshape(hidden, WORD_DIM).T.copy()
    offset += matrix_bytes(WORD_DIM * hidden, bits)
    w3 = unpack_matrix(blob, offset, GLOBAL_DIM * global_hidden, bits).reshape(global_hidden, GLOBAL_DIM).T.copy()
    offset += matrix_bytes(GLOBAL_DIM * global_hidden, bits)
    tail = np.frombuffer(blob, dtype="<f4", offset=offset).astype(np.float32)
    sizes = {"b1": hidden, "w2": hidden, "b2": 1, "mix": 2, "b3": global_hidden, "w4": global_hidden,
             "vs": SHAPE_FEATURES, "c": 1}
    params, i = {"w1": w1, "w3": w3}, 0
    for k in TAIL:
        params[k] = tail[i:i + sizes[k]].copy()
        i += sizes[k]
    return params, lexicon


def probabilities(params: dict, texts: list[str]) -> np.ndarray:
    out = []
    for start in range(0, len(texts), 512):
        words, mask, glob = transform(texts[start:start + 512])
        out.append(sigmoid(logits(params, words, mask, glob)))
    return np.concatenate(out)


def measure(truth: np.ndarray, scores: np.ndarray) -> dict:
    clipped = np.clip(scores.astype(np.float64), 1e-12, 1 - 1e-12)
    return {"accuracy": float(np.mean((scores >= 0.5) == truth)),
            "loss": float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)))}


def export_packed(model_path: Path, out: Path, *, data: Path, splits: Path, embed: Path | None = None,
                  accuracy_tolerance=0.01, loss_tolerance=0.01, max_packed_bytes: int | None = 25_000) -> dict:
    import onnx
    from qclassify.export import export_onnx, load_params
    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model)
    props = {p.key: p.value for p in model.metadata_props}
    lexicon = LEXICON_PATH.read_bytes()
    if props.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Incompatible preprocessing version")
    if props.get("lexicon_sha256") != hashlib.sha256(lexicon).hexdigest():
        raise ValueError("Model was trained with a different lexicon")
    params = load_params(model_path)
    hidden, global_hidden = len(params["b1"]), len(params["b3"])
    lookup = {row["input"]: row["output"] == "person_name" for row in json.loads(data.read_text())}
    partitions = json.loads(splits.read_text())
    val_texts = partitions["validation"]
    val_y = np.array([lookup[t] for t in val_texts], dtype=int)
    baseline = measure(val_y, probabilities(params, val_texts))
    candidates, selected = [], None
    # Select on validation; test is evaluated only after freezing this choice.
    for bits in (3, 4, 5, 6, 8):
        blob = pack(params, bits, lexicon)
        decoded, _ = unpack(blob, hidden, global_hidden, bits)
        metric = measure(val_y, probabilities(decoded, val_texts))
        accepted = (metric["accuracy"] >= baseline["accuracy"] - accuracy_tolerance - 1e-10
                    and metric["loss"] <= baseline["loss"] + loss_tolerance)
        within = max_packed_bytes is None or len(blob) <= max_packed_bytes
        candidates.append({"bits": bits, "bytes": len(blob), "validation": metric, "accepted": accepted and within})
        if accepted and within and (selected is None or max_packed_bytes is not None):
            selected = (bits, blob, decoded, metric)  # with a budget: the widest width that fits
    if selected is None:
        raise ValueError("No packed model meets validation tolerances")
    bits, blob, decoded, validation = selected
    test_texts = partitions["test"]
    test_y = np.array([lookup[t] for t in test_texts], dtype=int)
    report = {"selection": f"Widest bit width meeting validation tolerances within {max_packed_bytes} packed bytes"
                           "; test unused for selection", "max_packed_bytes": max_packed_bytes,
              "accuracy_tolerance": accuracy_tolerance, "loss_tolerance": loss_tolerance,
              "baseline_validation": baseline, "candidates": candidates,
              "selected_bits": bits, "packed_bytes": len(blob), "validation": validation,
              "test": measure(test_y, probabilities(decoded, test_texts)),
              "baseline_test": measure(test_y, probabilities(params, test_texts)),
              "source_onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    manifest = {"format": FORMAT, "featureVersion": FEATURE_VERSION, "wordDim": WORD_DIM, "wordBins": WORD_BINS,
                "globalDim": GLOBAL_DIM, "globalBins": GLOBAL_BINS, "lexBins": LEX_BINS, "maxWords": MAX_WORDS,
                "hiddenDim": hidden, "globalHiddenDim": global_hidden, "maxChars": MAX_CHARS,
                "weightBytes": len(blob), "bits": bits, "groupSize": GROUP_SIZE,
                "sha256": hashlib.sha256(blob).hexdigest(), "layout": LAYOUT,
                "labels": ["search_query", "person_name"], "threshold": 0.5}
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(blob)
    (out / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "packing.json").write_text(json.dumps(report, indent=2) + "\n")
    export_onnx(decoded, out / "classifier.onnx")  # dequantized twin, for parity fixtures and downstream use
    if embed:
        embed.parent.mkdir(parents=True, exist_ok=True)
        embed.write_text('// Generated by qclassify-export-web; do not edit.\n' +
                         'export const manifest = ' + json.dumps(manifest, separators=(",", ":")) + ';\n' +
                         'const hex = "' + blob.hex() + '";\n' +
                         'export function decodeWeights(): ArrayBuffer {\n' +
                         '  const bytes = new Uint8Array(hex.length / 2);\n' +
                         '  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);\n' +
                         '  return bytes.buffer;\n}\n')
    return report
