#!/usr/bin/env python3
"""Fold data/authored/topics.txt (committed, hand-written topic queries) into examples.json + queries.txt.

Skips anything already in the corpus, any held-out probe, and any string already labelled a name.
Idempotent. Works on the sanitized committed corpus and on the local one that also holds data/mined.
Remember to update tests/test_pipeline.py counts afterwards.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
canon = lambda s: " ".join(s.lower().split())

rows = json.loads((D / "examples.json").read_text())
have = {canon(r["input"]) for r in rows}
held = {canon(x) for f in ("probes.json", "probes_topics.json", "mined/probes_mined.json") if (D / f).exists()
        for v in json.loads((D / f).read_text()).values() for x in v}
topics = [l.strip() for l in (D / "authored/topics.txt").read_text().splitlines() if l.strip()]

added = []
for s in topics:
    k = canon(s)
    if k in have or k in held:
        continue
    have.add(k)
    rows.append({"input": s, "output": "search_query"})
    added.append(s)

(D / "examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
(D / "queries.txt").write_text("\n".join(r["input"] for r in rows if r["output"] == "search_query") + "\n")
print({"search_query": len(added)}, "total", len(rows))
