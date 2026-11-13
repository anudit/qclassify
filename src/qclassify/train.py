import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from threadpoolctl import threadpool_limits

from qclassify.export import export_onnx
from qclassify.features import FEATURE_VERSION, MAX_CHARS, canonicalize, transform
from qclassify.model import fit, forward, sigmoid

LABELS = {"search_query": 0, "person_name": 1}


def load_dataset(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON array.")
    seen = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"input", "output"}:
            raise ValueError(f"Row {i}: expected input and output fields.")
        text, label = row["input"], row["output"]
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
            raise ValueError(f"Row {i}: expected 1..{MAX_CHARS} characters.")
        if label not in LABELS:
            raise ValueError(f"Row {i}: invalid output {label!r}.")
        key = canonicalize(text)
        if key in seen:
            raise ValueError(f"Duplicate normalized input: {text!r}")
        seen.add(key)
    if len({row["output"] for row in rows}) != 2:
        raise ValueError("Both classes must be present.")
    return rows


def split_indices(rows: list[dict], seed: int):
    # Sort first so merely reordering the JSON cannot change the split.
    rng = np.random.default_rng(seed)
    splits = [[], [], []]
    for label in LABELS:
        indices = sorted((i for i, row in enumerate(rows) if row["output"] == label),
                         key=lambda i: canonicalize(rows[i]["input"]))
        if len(indices) < 10:
            raise ValueError("Need at least 10 examples per class for splitting.")
        shuffled = rng.permutation(indices)
        a, b = int(len(indices) * 0.8), int(len(indices) * 0.9)
        for split, part in zip(splits, np.split(shuffled, [a, b])):
            split.extend(part.tolist())
    return [np.array(sorted(split)) for split in splits]


def evaluate(y: np.ndarray, probabilities: np.ndarray, rows: list[dict]) -> dict:
    truth = y.reshape(-1).astype(int)
    scores = probabilities.reshape(-1)
    predictions = (scores >= 0.5).astype(int)
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (truth, predictions), 1)
    classes = {}
    for label, i in LABELS.items():
        tp = int(confusion[i, i])
        precision = tp / max(1, int(confusion[:, i].sum()))
        recall = tp / max(1, int(confusion[i].sum()))
        classes[label] = {"precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / max(1e-12, precision + recall)}
    errors = [{**rows[i], "person_probability": float(scores[i])}
              for i in np.flatnonzero(predictions != truth)]
    return {"count": len(truth), "accuracy": float(np.mean(predictions == truth)),
            "confusion_matrix": confusion.tolist(),
            "confusion_order": list(LABELS), "classes": classes, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate and export a <100 KB neural classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/examples.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.hidden < 1 or args.epochs < 1:
        parser.error("--hidden and --epochs must be positive")
    rows = load_dataset(args.data)
    indices = split_indices(rows, args.seed)
    x = transform([r["input"] for r in rows])
    y = np.array([LABELS[r["output"]] for r in rows], dtype=np.float32)[:, None]
    train, validation, test = indices
    # Features lowercase every input, so casing variants would be duplicates: no augmentation.
    with threadpool_limits(limits=1):
        params, training = fit(x[train], y[train], x[validation], y[validation],
                               seed=args.seed, hidden=args.hidden, epochs=args.epochs, device=args.device)
    training["augmentation"] = "none: the feature extractor lowercases all input"
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "classifier.onnx"
    size = export_onnx(params, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    # Verify the whole corpus, not just a single example, with dynamic batch sizes.
    onnx_probabilities = session.run(None, {"features": x})[0]
    numpy_probabilities = sigmoid(forward(params, x)[1])
    np.testing.assert_allclose(onnx_probabilities, numpy_probabilities, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(session.run(None, {"features": x[:1]})[0],
                               numpy_probabilities[:1], atol=1e-6, rtol=1e-5)
    metrics = {name: evaluate(y[part], onnx_probabilities[part], [rows[i] for i in part])
               for name, part in zip(("train", "validation", "test"), indices)}
    report = {"dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "seed": args.seed, "feature_version": FEATURE_VERSION,
              "hidden_units": args.hidden, "class_counts": dict(Counter(r["output"] for r in rows)),
              **size, "max_onnx_error": float(np.max(np.abs(onnx_probabilities - numpy_probabilities))),
              "training": training, "metrics": metrics,
              "limitations": "Authored synthetic data; random holdout measures this corpus only. Validate on real traffic."}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {name: [rows[i]["input"] for i in part]
                for name, part in zip(("train", "validation", "test"), indices)}
    (args.data.parent / "splits.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model": str(path), **size,
                      "validation_accuracy": metrics["validation"]["accuracy"],
                      "test_accuracy": metrics["test"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
