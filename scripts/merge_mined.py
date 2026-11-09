#!/usr/bin/env python3
"""Fold data/mined/*.json (scripts/mine_people.py) into examples.json + names.txt + queries.txt.

Skips anything already in the corpus, in probes.json / probes_mined.json (held out), or labelled both ways.
Idempotent: a second run adds nothing. Remember to update tests/test_pipeline.py counts afterwards.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
canon = lambda s: " ".join(s.lower().split())

rows = json.loads((D / "examples.json").read_text())
have = {canon(r["input"]) for r in rows}
held = {canon(x) for v in json.loads((D / "probes.json").read_text()).values() for x in v}
held |= {canon(x) for v in json.loads((D / "mined/probes_mined.json").read_text()).values() for x in v}
M = D / "mined"
names = json.loads((M / "names_multi.json").read_text()) + json.loads((M / "names_single.json").read_text())
queries = json.loads((M / "queries.json").read_text())
clash = {canon(x) for x in names} & {canon(x) for x in queries}  # same string, both labels: drop

added = {"person_name": [], "search_query": []}
for label, items in (("person_name", names), ("search_query", queries)):
    for s in items:
        k = canon(s)
        if k in have or k in held or k in clash:
            continue
        have.add(k)
        rows.append({"input": s, "output": label})
        added[label].append(s)

(D / "examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
for label, f in (("person_name", "names.txt"), ("search_query", "queries.txt")):
    (D / f).write_text("\n".join(r["input"] for r in rows if r["output"] == label) + "\n")
print({k: len(v) for k, v in added.items()}, "total", len(rows))
