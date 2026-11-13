#!/usr/bin/env python3
"""Add 3,000 real profile names from the dumpyard hackathon_people table to the LOCAL corpus.

Input: data/mined/db_names_sample.json, a random sample of distinct 2-4 word names pulled from the database
(gitignored, never committed). Skips corpus rows and held-out probes; the pick is seeded, so a rerun adds nothing.
Writes data/mined/names_db.json (so training treats these as mined, weight 1) and appends them to examples.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
COUNT = 3000
canon = lambda s: " ".join(s.lower().split())

rows = json.loads((D / "examples.json").read_text())
out = D / "mined/names_db.json"
if out.exists():
    raise SystemExit(f"{out} exists: already merged")
have = {canon(r["input"]) for r in rows}
held = {canon(x) for f in ("probes.json", "probes_topics.json", "probes_famous.json", "mined/probes_mined.json")
        for v in json.loads((D / f).read_text()).values() for x in v}
pool = sorted({n for n in json.loads((D / "mined/db_names_sample.json").read_text())
               if canon(n) not in have and canon(n) not in held})
pool = list(dict((canon(n), n) for n in pool).values())  # one spelling per normalized name
add = [pool[i] for i in sorted(np.random.default_rng(0).choice(len(pool), COUNT, replace=False))]
out.write_text(json.dumps(add, ensure_ascii=False) + "\n")
rows += [{"input": n, "output": "person_name"} for n in add]
(D / "examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
for label, f in (("person_name", "names.txt"), ("search_query", "queries.txt")):
    (D / f).write_text("\n".join(r["input"] for r in rows if r["output"] == label) + "\n")
print("added", len(add), "total", len(rows))
