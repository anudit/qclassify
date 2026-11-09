"""Freeze Python/ONNX references for Node and real-browser compatibility tests."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from qclassify.features import canonicalize, featurize, fnv1a

# Contract probes, not training examples. No accuracy labels are inferred here.
probes = [
    "Jane Doe", "weather today", "Ada Lovelace biography", "Sam Altman", "Andrej Karpathy",
    "open source inference engines", "J. R. Smith", "Sean O'Brien", "Anne-Marie Wilson",
    "José García", "İpek Yılmaz", "Straße", "ΟΣ", "ΟΣΑ", "οσ", "ὈΔΥΣΣΕΎΣ",
    "张伟", "佐藤太郎", "김민준", "محمد علي", "अर्जुन शर्मा", "ค้นหาโครงการ",
    "latest WebGPU projects", "C++", "AI", "RAG", "Hugging Face", "vLLM", "ONNX Runtime",
    "㍿ 東京", "Ａａｒａｖ　Ｓｈａｒｍａ", "e\u0301lodie", "Zoë 😀", "𝓙𝓸𝓱𝓷 𝓢𝓶𝓲𝓽𝓱",
    "\tMary\nAnn\rWilson\t", "\u001cAlice\u0085Smith\u001f", "\ufeffJane Doe\ufeff",
    "\u2003Bob\u202fLee\u3000", "\u200bword\u200b", "१२३", "²Ⅷ①", "name@domain.com",
    "@someone", "alice123", "a", "?", "0", "𐐀𐐁 𐐨𐐩", "Müller-Straße",
    "A" * 300, "long " * 100, "😀" * 300, "Ａ" * 300, "x" * 255 + " y",
]
rows = json.loads(Path("data/examples.json").read_text())
# Include varied rows from every authored section; do not use for model selection.
AUTHORED = 5909  # the corpus starts with the authored rows; mined profile names (appended after) are not published
probes.extend(row["input"] for row in rows[:AUTHORED][::31])
session = ort.InferenceSession("artifacts/web/classifier.onnx", providers=["CPUExecutionProvider"])
fixtures = []
for text in probes:
    vector = featurize(text)
    score = float(session.run(None, {"features": vector[None]})[0][0, 0])
    fixtures.append({"input": text, "canonical": canonicalize(text), "hash": fnv1a(text),
                     "features": [[int(i), float(vector[i])] for i in np.flatnonzero(vector)],
                     "personProbability": score})
Path("data/parity.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n")
print(f"Wrote {len(fixtures)} cross-runtime contract probes")
