#!/usr/bin/env python3
"""Fold the committed hand-written lists in data/authored/ into examples.json, names.txt and queries.txt:
topics.txt (search_query) and famous.txt (person_name).

Skips anything already in the corpus and any held-out probe. Idempotent. Works on the sanitized committed
corpus and on the local one that also holds data/mined.
Remember to update tests/test_pipeline.py counts afterwards.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
canon = lambda s: " ".join(s.lower().split())

LISTS = (("authored/topics.txt", "search_query"), ("authored/famous.txt", "person_name"))
PROBES = ("probes.json", "probes_topics.json", "probes_famous.json", "mined/probes_mined.json")

rows = json.loads((D / "examples.json").read_text())
have = {canon(r["input"]) for r in rows}
held = {canon(x) for f in PROBES if (D / f).exists()
        for v in json.loads((D / f).read_text()).values() for x in v}

added = {"search_query": [], "person_name": []}
for path, label in LISTS:
    for text in (l.strip() for l in (D / path).read_text().splitlines() if l.strip()):
        key = canon(text)
        if key in have or key in held:
            continue
        have.add(key)
        rows.append({"input": text, "output": label})
        added[label].append(text)

(D / "examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
for label, f in (("person_name", "names.txt"), ("search_query", "queries.txt")):
    (D / f).write_text("\n".join(r["input"] for r in rows if r["output"] == label) + "\n")
print({k: len(v) for k, v in added.items()}, "total", len(rows))
