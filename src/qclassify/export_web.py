"""Create packed browser weights, a parity ONNX and an optional embedded module."""
import argparse
import json
from pathlib import Path

from qclassify.packing import export_packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/classifier.onnx"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/web"))
    parser.add_argument("--data", type=Path, default=Path("data/examples.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    parser.add_argument("--embed", type=Path, default=Path("web/src/model.ts"))
    parser.add_argument("--max-packed-bytes", type=int, default=23_168,
                        help="packed-weight budget: 2x the 11,584 bytes of 0.2.1; pass 0 for smallest-within-tolerance")
    parser.add_argument("--accuracy-tolerance", type=float, default=0.01,
                        help="max validation accuracy drop from quantizing (0.01 = one point)")
    args = parser.parse_args()
    print(json.dumps(export_packed(args.model, args.out, data=args.data, splits=args.splits,
                                   embed=args.embed, accuracy_tolerance=args.accuracy_tolerance,
                                   max_packed_bytes=args.max_packed_bytes or None), indent=2))


if __name__ == "__main__":
    main()
