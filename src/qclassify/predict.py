import argparse
import json
from pathlib import Path

import onnxruntime as ort

from qclassify.features import FEATURE_VERSION, transform


class Classifier:
    def __init__(self, model: str | Path = "artifacts/classifier.onnx"):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        if self.session.get_modelmeta().custom_metadata_map.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Model and preprocessor versions do not match.")

    def predict(self, texts: list[str]) -> list[dict]:
        if not texts:
            return []
        probabilities = self.session.run(None, {"features": transform(texts)})[0].reshape(-1)
        return [{"input": text, "label": "person_name" if p >= 0.5 else "search_query",
                 "person_probability": float(p), "search_probability": float(1 - p)}
                for text, p in zip(texts, probabilities)]


def main():
    parser = argparse.ArgumentParser(description="Classify strings using ONNX Runtime.")
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--model", type=Path, default=Path("artifacts/classifier.onnx"))
    args = parser.parse_args()
    print(json.dumps(Classifier(args.model).predict(args.texts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
