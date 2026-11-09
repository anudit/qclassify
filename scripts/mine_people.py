#!/usr/bin/env python3
"""Mine extra training data from the Hackathon_People / Hackathon_Projects data behind Dumpyard.

Names  : real profile names (name / gh_name / tw_name) of ~950k hackathon people.
Single : first names that recur across many profiles ("Vitalik", "Subh") -> one-word person lookups.
Queries: what those same profiles say about *topics*: skills, interests, job titles, companies, cities and
         project titles -> one-word and short topic searches ("react", "machine learning", "Bangalore").

Names that are also topic words (Rust, Swift, Jordan...) are dropped from both sides, so a one-word label
is only kept when the data agrees on it. A held-out slice is written to probes_mined.json, never trained on.

  uv run python scripts/mine_people.py            # writes data/mined/*.json
  uv run python scripts/merge_mined.py            # folds them into examples.json / names.txt / queries.txt
"""
import collections, json, random, re, sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SDB = Path("/Users/anudit/Documents/GitHub/projectssdb")
PEOPLE = SDB / "Hackathon_People/hackathon_people_github.parquet"
PROJECTS = SDB / "Hackathon_Projects/hackathon_projects.parquet"
MISSING = SDB / "Hackathon_Projects/devpost/out/missing_projects.jsonl"
OUT = ROOT / "data/mined"
rng = random.Random(7)

NOT_PERSON = set("""team inc llc ltd labs lab studio studios dev devs tech technologies technology solutions group
hack hacks hackathon club society official corp company co org foundation community university college school
institute academy project projects app apps software systems digital media network works ventures capital ai bot
the and of for of""".split())
NAME_TOKEN = re.compile(r"^[^\W\d_]+(?:[-'’.][^\W\d_]+)*\.?$", re.U)


def good_name(s: str):
    if not isinstance(s, str):
        return None
    s = " ".join(s.split())
    toks = s.split()
    if not 2 <= len(toks) <= 4 or len(s) > 40:
        return None
    for t in toks:
        if not NAME_TOKEN.match(t) or len(t.strip(".")) < 2:
            return None
        if t.lower().strip(".") in NOT_PERSON:
            return None
        if re.search(r"[a-z][A-Z]", t) and not re.match(r"^(Mc|Mac|De|Di|Le|La|Van|Von|O')", t):
            return None
    if sum(t.isupper() and len(t) > 3 for t in toks):  # SHOUTED handles / org names
        return None
    return s


def key(s: str) -> str:
    return " ".join(s.lower().split())


def title_case_ok(s: str) -> bool:
    # plain words only: no quotes, slashes, underscores, emoji, urls or sentences
    return bool(s) and 2 <= len(s) <= 40 and bool(re.fullmatch(r"[^\W_]+(?:[ .+#&'-][^\W_]+)*[+#]*", s, re.U))


print("reading people…", file=sys.stderr)
df = pq.read_table(PEOPLE, columns=["source", "name", "gh_name", "tw_name", "skills", "interests", "job_title",
                                    "gh_company", "location", "gh_location"]).to_pandas()

# ---- person names -------------------------------------------------------------------------------------
names: dict[str, str] = {}
for col in ("name", "gh_name", "tw_name"):
    for v in df[col].dropna().unique():
        n = good_name(v)
        if n and key(n) not in names:
            names[key(n)] = n
print(f"{len(names)} distinct multi-word names", file=sys.stderr)

# ---- topic vocabulary ---------------------------------------------------------------------------------
def counted(series, minimum, many=True):
    c = collections.Counter()
    for v in series.dropna():
        items = list(v) if many else [v]
        for it in items:
            if isinstance(it, str):
                it = " ".join(it.split())
                if title_case_ok(it):
                    c[it.lower()] += 1
    return {k: n for k, n in c.items() if n >= minimum}

skills = counted(df["skills"], 15)
interests = counted(df["interests"], 15)
jobs = counted(df["job_title"], 25, many=False)
companies = counted(df["gh_company"].str.lstrip("@"), 12, many=False)
cities = {}
for col in ("location", "gh_location"):
    for k, n in counted(df[col].map(lambda s: s.split(",")[0].strip() if isinstance(s, str) else None), 40, many=False).items():
        cities[k] = cities.get(k, 0) + n
topics = {**skills, **interests}
print(f"skills {len(skills)} interests {len(interests)} jobs {len(jobs)} companies {len(companies)} cities {len(cities)}", file=sys.stderr)

# ---- one-word names: first names that recur ------------------------------------------------------------
first = collections.Counter(n.split()[0].lower().strip(".") for n in names.values())
last = collections.Counter(n.split()[-1].lower().strip(".") for n in names.values())
try:
    common_words = {w.strip() for w in open("/usr/share/dict/words") if w[:1].islower()}
except OSError:
    common_words = set()
topic_words = set(topics) | set(jobs) | set(companies) | set(cities)
topic_words |= {w for k in topic_words for w in k.split() if len(w) > 3}  # "rust" in "rust developer" stays a topic
existing_q = {key(x) for x in (ROOT / "data/queries.txt").read_text().splitlines()}
singles = {}
for tok, n in first.items():
    if n < 6 or not tok.isalpha() or not tok.isascii() and n < 20 or not 3 <= len(tok) <= 14:
        continue
    if last[tok] > n or tok in common_words:  # mostly a surname, or an ordinary word
        continue
    if tok in topic_words or tok in existing_q or tok in NOT_PERSON:
        continue
    singles[tok] = n
print(f"{len(singles)} one-word first names (>=6 profiles)", file=sys.stderr)

# ---- project titles ------------------------------------------------------------------------------------
titles = set()
try:
    t = pq.read_table(PROJECTS, columns=["title"]).to_pandas()["title"]
    titles.update(x for x in t.dropna().unique() if isinstance(x, str))
except Exception as e:
    print("projects parquet:", e, file=sys.stderr)
if MISSING.exists():
    for line in MISSING.open():
        try:
            titles.add(json.loads(line)["title"])
        except Exception:
            pass
proj = []
name_toks = set(singles)
for ttl in titles:
    ttl = " ".join(ttl.split())
    w = ttl.split()
    if 2 <= len(w) <= 4 and first.get(w[0].lower().strip("."), 0) < 3 and title_case_ok(ttl) and not any(re.search(r"\d", x) for x in w) and ttl.lower() not in names:
        if len(w) >= 2 and all(x.lower().strip(".") in name_toks for x in w):
            continue  # a title that reads like a person ("Sarah Connor") is not a clean negative
        proj.append(ttl)
print(f"{len(proj)} usable project titles", file=sys.stderr)

# ---- assemble ------------------------------------------------------------------------------------------
def sample(items, n):
    items = sorted(set(items))
    rng.shuffle(items)
    return items[:n]

def cap(w):  # "react native" -> as written (lowercase) ; variants come from training augmentation
    return w

def person_like(n):
    toks = [t.lower().strip(".") for t in n.split()]
    if first[toks[0]] < 3 or toks[0] in topic_words or toks[0] in common_words and last[toks[0]] > first[toks[0]]:
        return False
    return not any(t in topic_words or (t in common_words and last[t] < 5) for t in toks[1:])

person_multi = sample([n for n in names.values() if person_like(n)], 6000)
person_single = sorted(singles, key=lambda t: -singles[t])
# keep the head (very common) and a random slice of the tail so rare names are represented too
person_single = person_single[:1200] + sample(person_single[1200:], 1800)
person_single = [t.capitalize() if rng.random() < .5 else t for t in person_single]

q_single = sample([k for k in topics if " " not in k], 2600)
q_multi = sample([k for k in topics if " " in k], 900)
q_jobs = sample(jobs, 400)
q_cos = sample([c for c in companies if key(c) not in names], 900)
q_cities = sample(cities, 500)
q_proj = sample(proj, 2400)
queries = q_single + q_multi + q_jobs + q_cos + q_cities + q_proj
queries = [q.title() if (" " in q and rng.random() < .35) else q for q in queries]

# one-word *people* seen as profile names only (no surname) and clearly not topics are already in `singles`.
# lexicon sources for the feature experiment: recurring first names, rarer first names (never a lexicon hit,
# used only to test generalisation), and the topic vocabulary
rare = [t for t, n in first.items() if 3 <= n <= 5 and t.isalpha() and t.isascii() and 4 <= len(t) <= 12
        and t not in common_words and t not in topic_words and last[t] <= n]
(OUT / "first_names.json").write_text(json.dumps(singles, ensure_ascii=False))
(OUT / "rare_first_names.json").write_text(json.dumps(sorted(rare)[::1], ensure_ascii=False))
(OUT / "topic_words.json").write_text(json.dumps(sorted(w for w in topic_words if " " not in w), ensure_ascii=False))

# held-out slice for honest evaluation
def split_probe(items, frac=0.08):
    items = list(dict.fromkeys(items))
    rng.shuffle(items)
    k = int(len(items) * frac)
    return items[k:], items[:k]

pm, pm_probe = split_probe(person_multi)
ps, ps_probe = split_probe(person_single)
qq, q_probe = split_probe(queries)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "names_multi.json").write_text(json.dumps(pm, ensure_ascii=False, indent=1))
(OUT / "names_single.json").write_text(json.dumps(ps, ensure_ascii=False, indent=1))
(OUT / "queries.json").write_text(json.dumps(qq, ensure_ascii=False, indent=1))
(OUT / "probes_mined.json").write_text(json.dumps({"person_name": pm_probe + ps_probe, "search_query": q_probe},
                                                  ensure_ascii=False, indent=1))
print(json.dumps({"names_multi": len(pm), "names_single": len(ps), "queries": len(qq),
                  "probes": {"person": len(pm_probe) + len(ps_probe), "query": len(q_probe)}}))
