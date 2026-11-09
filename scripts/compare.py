#!/usr/bin/env python3
"""Accuracy of one or more classifier.onnx files on held-out probes (never trained on).
  uv run python scripts/compare.py artifacts/classifier.onnx /tmp/old.onnx"""
import json, sys
from pathlib import Path
from qclassify.predict import Classifier

D = Path(__file__).resolve().parents[1] / "data"
old = json.loads((D / "probes.json").read_text())
mined = json.loads((D / "mined/probes_mined.json").read_text())
HAND = {"person_name": ["Vitalik", "Anatoly", "Aniket", "Devang", "Sarah", "Ilya", "Satoshi", "Gavin", "Hayden", "Subh", "tanmay", "Ryan Dahl"],
        "search_query": ["hackathon", "ethereum", "Hugging Face", "react native", "zero knowledge", "solidity", "web3", "devfolio", "bangalore", "machine learning", "who won ethglobal", "rust"]}
sets = {"authored probes (multi-word)": old, "mined probes": mined, "hand: one-word + tech": HAND}
for path in sys.argv[1:]:
    c = Classifier(path)
    print(path)
    for name, s in sets.items():
        res = {}
        for label, items in s.items():
            p = [r["person_probability"] for r in c.predict(items)]
            res[label] = sum((x >= .5) == (label == "person_name") for x in p) / len(p)
        one = [(x, label) for label, items in s.items() for x in items if len(x.split()) == 1]
        if one:
            pr = c.predict([x for x, _ in one])
            res["one-word"] = sum((r["person_probability"] >= .5) == (l == "person_name") for r, (_, l) in zip(pr, one)) / len(one)
        print(f"  {name:30}", {k: round(v, 3) for k, v in res.items()})
