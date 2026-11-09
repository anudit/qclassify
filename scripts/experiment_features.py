"""Feature experiment: do whole-word hashes / a first-name lexicon beat the v2 char-n-gram features?
Evaluates on probes the lexicon never saw. uv run python scripts/experiment_features.py"""
import json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort  # noqa
from qclassify import features as F
from qclassify.model import fit, forward, sigmoid
from qclassify.train import load_dataset, split_indices, LABELS

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
WORD_BINS = 1024

mined_probes = json.loads((D / "mined/probes_mined.json").read_text())
old_probes = json.loads((D / "probes.json").read_text())
HAND = {"person_name": ["Vitalik", "Anatoly", "Aniket", "Devang", "Sarah", "Ilya", "Satoshi", "Gavin", "Hayden", "Subh", "tanmay", "Ryan Dahl"],
        "search_query": ["hackathon", "ethereum", "Hugging Face", "react native", "zero knowledge", "solidity", "web3", "devfolio", "bangalore", "machine learning", "who won ethglobal", "rust"]}
held = {F.canonicalize(x) for s in (mined_probes, old_probes, HAND) for v in s.values() for x in v}

rows = load_dataset(D / "examples.json")
# lexicon: every single-token *name* and single-token *query* in the corpus (+ the unused tail of first names),
# minus anything held out. Words only; the label of a multi-word example is not used.
names_tail = json.loads((D / "mined/names_single.json").read_text())
person_words = {F.canonicalize(r["input"]) for r in rows if r["output"] == "person_name" and len(r["input"].split()) == 1}
person_words |= {F.canonicalize(r["input"]).split()[0] for r in rows if r["output"] == "person_name" and len(r["input"].split()) > 1}
topic_words = {F.canonicalize(r["input"]) for r in rows if r["output"] == "search_query" and len(r["input"].split()) == 1}
person_words -= held; topic_words -= held; person_words -= topic_words

LEX = set(json.loads((D / "mined/first_names.json").read_text()))
TOPICS = set(json.loads((D / "mined/topic_words.json").read_text()))
RARE = json.loads((D / "mined/rare_first_names.json").read_text())[:400]
mined_probes = {**mined_probes}
rng = np.random.default_rng(3)

def feats(text, mode, drop=0.0):
    base = F.featurize(text)
    if mode == "v2":
        return base
    if mode == "lex2":  # 2 extra bits, from word lists that were not built from labels; dropout on the name bit in training
        toks = F.canonicalize(text).split()
        hit = float(toks[0] in LEX and rng.random() >= drop)
        top = float(any(t in TOPICS for t in toks) and rng.random() >= drop)
        return np.concatenate([base, np.array([hit, top], dtype=np.float32)])
    toks = F.canonicalize(text).split()
    extra = np.zeros(WORD_BINS + (2 if mode == "lex" else 0), dtype=np.float32)
    for t in toks:
        h = F.fnv1a("w:" + t)
        extra[h % WORD_BINS] += 1.0 if h & 0x80000000 else -1.0
    extra[:WORD_BINS] /= max(1, len(toks)) ** 0.5
    if mode == "lex":
        extra[WORD_BINS] = float(toks[0] in person_words)           # first token is a known first name
        extra[WORD_BINS + 1] = float(any(t in topic_words for t in toks))  # some token is a known topic word
    return np.concatenate([base, extra])

def transform(texts, mode, drop=0.0):
    return np.stack([feats(t, mode, drop) for t in texts]).astype(np.float32)

def run(mode, hidden=16, bits=4):
    train, val, test = split_indices(rows, 42)
    y = np.array([LABELS[r["output"]] for r in rows], dtype=np.float32)[:, None]
    texts = [r["input"] for r in rows]
    tt = [texts[i] for i in train]
    aug = tt + [t.lower() for t in tt] + [t.upper() for t in tt] + [t.title() for t in tt]
    ax = transform(aug, mode, float(__import__("os").environ.get("DROP", "0.3"))); ay = np.tile(y[train], (4, 1))
    vx = transform([texts[i] for i in val], mode); tx = transform([texts[i] for i in test], mode)
    params, info = fit(ax, ay, vx, y[val], seed=42, hidden=hidden, epochs=400, device="mps", augmentation_blocks=4)
    prob = lambda x: sigmoid(forward(params, x)[1]).reshape(-1)
    res = {"mode": mode, "params": int(sum(v.size for v in params.values())), "epoch": info["best_epoch"],
           "test_acc": float(np.mean((prob(tx) >= .5) == (y[test].reshape(-1) == 1)))}
    rare = {"person_name": RARE}
    for name, s in (("authored", old_probes), ("mined", mined_probes), ("hand", HAND), ("rare-names", rare)):
        pe, one = {}, []
        for label, items in s.items():
            p = prob(transform(items, mode))
            ok = (p >= .5) == (label == "person_name")
            pe[label[:3]] = round(float(ok.mean()), 3)
            one += [bool(o) for o, x in zip(ok, items) if len(x.split()) == 1]
        pe["1w"] = round(float(np.mean(one)), 3) if one else None
        res[name] = pe
    return res

for mode in sys.argv[1:] or ["v2", "word", "lex"]:
    print(json.dumps(run(mode)), flush=True)
