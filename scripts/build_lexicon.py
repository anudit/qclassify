#!/usr/bin/env python3
"""Build src/qclassify/lexicon.bin: 4096 hashed slots, each a signed 4-bit "name vs ordinary word" value.

value(word) = clip(name_zipf - word_zipf, ±3.5) / 3.5, rounded to sevenths; +1 = mostly a name, -1 = mostly a word.
Slot = fnv1a("#" + word) % 4096; on a collision the more frequent word keeps the slot; empty slots read 0.

Inputs (downloaded once into a directory, not committed):
  census/Names_2010Census.csv  US Census 2010 surnames (public domain)
  hadley.csv                   US SSA baby names 1880-2008, github.com/hadley/data-baby-names (public domain)
  wd_given.csv, wd_family.csv  Wikidata labels of given-name (Q202444) / family-name (Q101352) items (CC0)
  word frequencies             the `wordfreq` package, English (CC-BY-SA 4.0: see README attribution)

  uv run --with wordfreq python scripts/build_lexicon.py /path/to/downloads
"""
import csv, math, re, sys, collections
from pathlib import Path

import numpy as np
from wordfreq import top_n_list, zipf_frequency

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qclassify.features import LEX_BINS, fnv1a  # noqa: E402

L = Path(sys.argv[1])
ok = re.compile(r"^[a-z][a-z'\-]{1,20}$")
name = collections.defaultdict(float)
for r in csv.DictReader(open(L / "census/Names_2010Census.csv")):
    w = r["name"].lower()
    if ok.match(w) and r["count"].isdigit():
        name[w] = max(name[w], math.log10(int(r["count"]) / 308.7e6 * 1e9))
for r in csv.DictReader(open(L / "hadley.csv")):
    w = r["name"].lower()
    if ok.match(w):
        name[w] = max(name[w], math.log10(float(r["percent"]) * 1e9))
for f in ("wd_given.csv", "wd_family.csv"):
    for line in (L / f).read_text().splitlines()[1:]:
        w = line.strip().strip('"').lower()
        if ok.match(w):
            name[w] = max(name[w], 2.5)
table = np.zeros(LEX_BINS, np.int64)  # level in -7..7
prio = np.full(LEX_BINS, -1.0)
for w in sorted(set(top_n_list("en", 40000)) | set(name)):
    if not ok.match(w):
        continue
    nz, wz = round(name.get(w, 0.0), 2), round(zipf_frequency(w, "en"), 2)
    level = round(max(-3.5, min(3.5, nz - wz)) / 3.5 * 7)
    b = fnv1a("#" + w) % LEX_BINS
    if max(nz, wz) > prio[b]:
        prio[b], table[b] = max(nz, wz), level
codes = (table + 7).astype(np.uint8)
packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)  # low nibble = even slot
out = Path(__file__).resolve().parents[1] / "src/qclassify/lexicon.bin"
out.write_bytes(packed.tobytes())
print(f"{len(name)} name words; wrote {out} ({len(packed)} bytes, {int((table != 0).sum())} nonzero slots)")
