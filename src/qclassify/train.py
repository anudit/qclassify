import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from threadpoolctl import threadpool_limits

from qclassify.export import export_onnx
from qclassify.features import FEATURE_VERSION, GLOBAL_BINS, MAX_CHARS, canonicalize, featurize, transform
from qclassify.model import fit, logits, sigmoid, sparse

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


def rows_for(texts: list[str]) -> list[dict]:
    out = []
    for text in texts:
        words, glob = featurize(text)
        out.append({"words": sparse(words), "global": sparse([glob])[0], "shape": glob[GLOBAL_BINS:].copy()})
    return out


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate and export the per-word name/query classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/examples.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42, help="fixes the train/validation/test split")
    parser.add_argument("--init-seeds", default="42,7,123,2024,99",
                        help="weight-init seeds to train; the lowest clean validation loss is kept (test is never used)")
    parser.add_argument("--hidden", type=int, default=13)
    parser.add_argument("--global-hidden", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--clean-weight", type=float, default=3.0,
                        help="loss weight of hand-authored rows vs mined rows (needs data/mined; ignored without it)")
    parser.add_argument("--l2", type=float, default=3e-4)
    args = parser.parse_args()
    if args.hidden < 1 or args.global_hidden < 1 or args.epochs < 1:
        parser.error("--hidden, --global-hidden and --epochs must be positive")
    rows = load_dataset(args.data)
    indices = split_indices(rows, args.seed)
    texts = [r["input"] for r in rows]
    y = np.array([LABELS[r["output"]] for r in rows], dtype=np.float32)
    train, validation, test = indices
    weights = np.ones(len(rows), dtype=np.float32)
    mined_dir = args.data.parent / "mined"
    if args.clean_weight != 1.0 and mined_dir.exists():
        mined = set()
        for name in ("names_multi", "names_single", "queries", "names_db"):
            if (mined_dir / f"{name}.json").exists():
                mined |= {canonicalize(t) for t in json.loads((mined_dir / f"{name}.json").read_text())}
        weights = np.array([1.0 if canonicalize(t) in mined else args.clean_weight for t in texts], dtype=np.float32)
        print(f"clean-weight {args.clean_weight}: {int((weights > 1).sum())} authored rows, {int((weights == 1).sum())} mined rows")
    # Early stopping and seed choice use the clean (authored) validation rows only: mined labels are noisy.
    clean_val = validation[weights[validation] > 1] if (weights > 1).any() else validation
    feats = rows_for(texts)
    runs = []
    with threadpool_limits(limits=4):
        for init_seed in [int(s) for s in args.init_seeds.split(",")]:
            params, info = fit([feats[i] for i in train], y[train], [feats[i] for i in clean_val], y[clean_val],
                               seed=init_seed, hidden=args.hidden, global_hidden=args.global_hidden,
                               epochs=args.epochs, l2=args.l2, weights=weights[train])
            runs.append((info["best_validation_loss"], init_seed, params, info))
            print(f"init seed {init_seed}: clean validation loss {info['best_validation_loss']:.5f}")
    _, init_seed, params, training = min(runs, key=lambda r: r[0])
    training.update({"init_seed": init_seed, "clean_weight": args.clean_weight, "l2": args.l2,
                     "seed_selection": {str(s): round(l, 6) for l, s, _, _ in runs},
                     "augmentation": "none: the feature extractor lowercases all input"})
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "classifier.onnx"
    size = export_onnx(params, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    probabilities = np.zeros(len(rows), np.float32)
    max_error = 0.0
    for start in range(0, len(rows), 512):  # verify the whole corpus against the NumPy reference
        words, mask, glob = transform(texts[start:start + 512])
        onnx_p = session.run(None, {"words": words, "word_mask": mask, "global": glob})[0].reshape(-1)
        ref = sigmoid(logits(params, words, mask, glob))
        np.testing.assert_allclose(onnx_p, ref, atol=1e-5, rtol=1e-4)
        max_error = max(max_error, float(np.max(np.abs(onnx_p - ref))))
        probabilities[start:start + 512] = onnx_p
    metrics = {name: evaluate(y[part], probabilities[part], [rows[i] for i in part])
               for name, part in zip(("train", "validation", "test"), indices)}
    report = {"dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "seed": args.seed, "feature_version": FEATURE_VERSION,
              "hidden_units": args.hidden, "global_hidden_units": args.global_hidden,
              "class_counts": dict(Counter(r["output"] for r in rows)),
              **size, "max_onnx_error": max_error, "training": training, "metrics": metrics,
              "limitations": "Authored synthetic data plus mined profile data; random holdout measures this corpus only. Validate on real traffic."}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {name: [rows[i]["input"] for i in part]
                for name, part in zip(("train", "validation", "test"), indices)}
    (args.data.parent / "splits.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"model": str(path), **size, "init_seed": init_seed,
                      "validation_accuracy": metrics["validation"]["accuracy"],
                      "test_accuracy": metrics["test"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
