"""Groupwise symmetric low-bit packing, chosen using validation data only."""
import hashlib
import json
from pathlib import Path

import numpy as np

from qclassify.features import transform

GROUP_SIZE = 32
LAYOUT = ["scales-f16", "codes-lsb", "b1-f32", "w2-f16", "b2-f32"]


def pack(params: dict, bits: int, group_size: int = GROUP_SIZE) -> bytes:
    flat = params["w1"].T.reshape(-1)
    levels = (1 << (bits - 1)) - 1
    scales = []
    codes = []
    for start in range(0, len(flat), group_size):
        group = flat[start:start + group_size]
        scale = np.float16(max(float(np.max(np.abs(group))) / levels, 2 ** -24))
        scales.append(scale)
        codes.extend((np.clip(np.rint(group / float(scale)), -levels, levels).astype(int) + levels).tolist())
    packed = bytearray((len(codes) * bits + 7) // 8)
    for i, code in enumerate(codes):
        byte, shift = divmod(i * bits, 8)
        value = code << shift
        packed[byte] |= value & 255
        if shift + bits > 8:
            packed[byte + 1] |= value >> 8
    return (np.array(scales, dtype="<f2").tobytes() + bytes(packed) +
            np.asarray(params["b1"], dtype="<f4").tobytes() +
            np.asarray(params["w2"].reshape(-1), dtype="<f2").tobytes() +
            np.asarray(params["b2"], dtype="<f4").tobytes())


def unpack(blob: bytes, inputs: int, hidden: int, bits: int, group_size: int = GROUP_SIZE) -> dict:
    count = inputs * hidden
    scale_count = (count + group_size - 1) // group_size
    code_bytes = (count * bits + 7) // 8
    expected = scale_count * 2 + code_bytes + hidden * 6 + 4
    if len(blob) != expected:
        raise ValueError("Invalid packed weight length")
    scales = np.frombuffer(blob, dtype="<f2", count=scale_count).astype(np.float32)
    start = scale_count * 2
    codes = blob[start:start + code_bytes]
    weights = np.empty(count, dtype=np.float32)
    for i in range(count):
        byte, shift = divmod(i * bits, 8)
        value = codes[byte] | ((codes[byte + 1] if byte + 1 < len(codes) else 0) << 8)
        code = (value >> shift) & ((1 << bits) - 1)
        weights[i] = (code - ((1 << (bits - 1)) - 1)) * scales[i // group_size]
    offset = start + code_bytes
    b1 = np.frombuffer(blob, dtype="<f4", count=hidden, offset=offset).copy()
    offset += hidden * 4
    w2 = np.frombuffer(blob, dtype="<f2", count=hidden, offset=offset).astype(np.float32)
    b2 = np.frombuffer(blob, dtype="<f4", count=1, offset=offset + hidden * 2).copy()
    return {"w1": weights.reshape(hidden, inputs).T.copy(), "b1": b1,
            "w2": w2[:, None], "b2": b2}


def probabilities(params: dict, features: np.ndarray) -> np.ndarray:
    logits = np.maximum(features @ params["w1"] + params["b1"], 0) @ params["w2"] + params["b2"]
    return np.exp(-np.logaddexp(0, -logits)).reshape(-1)


def measure(truth: np.ndarray, scores: np.ndarray) -> dict:
    clipped = np.clip(scores.astype(np.float64), 1e-12, 1 - 1e-12)
    return {"accuracy": float(np.mean((scores >= 0.5) == truth)),
            "loss": float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped)))}


def export_packed(model_path: Path, out: Path, *, data: Path, splits: Path,
                  embed: Path | None = None, accuracy_tolerance=0.002, loss_tolerance=0.01,
                  max_packed_bytes: int | None = None) -> dict:
    import onnx
    from onnx import numpy_helper
    from qclassify.features import FEATURE_VERSION, HASH_BINS, INPUT_DIM, MAX_CHARS
    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model)
    if {p.key: p.value for p in model.metadata_props}.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Incompatible preprocessing version")
    params = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
    hidden = len(params["b1"])
    if params["w1"].shape != (INPUT_DIM, hidden) or params["w2"].shape != (hidden, 1):
        raise ValueError("Unexpected model shape")
    rows = json.loads(data.read_text())
    partitions = json.loads(splits.read_text())
    lookup = {row["input"]: row["output"] == "person_name" for row in rows}
    val_texts = partitions["validation"]
    val_x = transform(val_texts)
    val_y = np.array([lookup[t] for t in val_texts], dtype=int)
    baseline = measure(val_y, probabilities(params, val_x))
    candidates = []
    selected = None
    # Select on validation; test is evaluated only after freezing this choice.
    for bits in (2, 3, 4, 5, 6, 8):
        blob = pack(params, bits)
        decoded = unpack(blob, INPUT_DIM, hidden, bits)
        metric = measure(val_y, probabilities(decoded, val_x))
        accepted = (metric["accuracy"] >= baseline["accuracy"] - accuracy_tolerance - 1e-10
                    and metric["loss"] <= baseline["loss"] + loss_tolerance)
        within_budget = max_packed_bytes is None or len(blob) <= max_packed_bytes
        candidates.append({"bits": bits, "bytes": len(blob), "validation": metric, "accepted": accepted and within_budget})
        if accepted and within_budget:
            # No budget: the smallest acceptable width. With a budget: spend it, the widest width that fits.
            if selected is None or max_packed_bytes is not None:
                selected = (bits, blob, decoded, metric)
    if selected is None:
        raise ValueError("No packed model meets validation tolerances")
    bits, blob, decoded, validation = selected
    test_texts = partitions["test"]
    test_x = transform(test_texts)
    test_y = np.array([lookup[t] for t in test_texts], dtype=int)
    report = {"selection": ("Smallest bit width meeting validation tolerances" if max_packed_bytes is None else
                            f"Widest bit width meeting validation tolerances within {max_packed_bytes} packed bytes")
                           + "; test unused for selection", "max_packed_bytes": max_packed_bytes,
              "accuracy_tolerance": accuracy_tolerance, "loss_tolerance": loss_tolerance,
              "baseline_validation": baseline, "candidates": candidates,
              "selected_bits": bits, "packed_bytes": len(blob), "validation": validation,
              "test": measure(test_y, probabilities(decoded, test_x)),
              "baseline_test": measure(test_y, probabilities(params, test_x)),
              "source_onnx_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    manifest = {"format": "qclassify-webgl-v2", "featureVersion": FEATURE_VERSION,
                "inputDim": INPUT_DIM, "hashBins": HASH_BINS, "hiddenDim": hidden,
                "maxChars": MAX_CHARS, "weightBytes": len(blob), "bits": bits,
                "groupSize": GROUP_SIZE, "sha256": hashlib.sha256(blob).hexdigest(),
                "layout": LAYOUT, "labels": ["search_query", "person_name"], "threshold": 0.5}
    out.mkdir(parents=True, exist_ok=True)
    (out / "weights.bin").write_bytes(blob)
    (out / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "packing.json").write_text(json.dumps(report, indent=2) + "\n")
    # Export an equivalent dequantized ONNX for browser parity and downstream use.
    from qclassify.export import export_onnx
    export_onnx(decoded, out / "classifier.onnx")
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
