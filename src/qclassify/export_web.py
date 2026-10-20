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
    args = parser.parse_args()
    print(json.dumps(export_packed(args.model, args.out, data=args.data, splits=args.splits,
                                   embed=args.embed), indent=2))


if __name__ == "__main__":
    main()
