"""Per-word network + whole-string branch. Trained on sparse inputs (EmbeddingBag == Linear on a sparse vector).

  word score   s_w   = w2 . relu(W1 x_w + b1) + b2                  (shared by every word)
  pooled             = mix0 * softmin_w(s_w) + mix1 * mean_w(s_w)    (a person only if every word looks like a name)
  global             = w4 . relu(W3 g + b3) + vs . shape(g)          (bigrams across words, '?', URLs, length...)
  P(person)          = sigmoid(pooled + global + c)
"""
import math

import numpy as np
import torch
from torch import nn

from qclassify.features import GLOBAL_BINS, GLOBAL_DIM, WORD_DIM

PARAM_ORDER = ["w1", "b1", "w2", "b2", "mix", "w3", "b3", "w4", "vs", "c"]


def sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -value))


def logits(params: dict, words: np.ndarray, mask: np.ndarray, glob: np.ndarray) -> np.ndarray:
    """NumPy reference. words [N, W, WORD_DIM], mask [N, W], glob [N, GLOBAL_DIM] -> [N]."""
    hidden = np.maximum(words @ params["w1"] + params["b1"], 0)
    s = np.clip(hidden @ params["w2"] + params["b2"][0], -30, 30)
    softmin = -np.log((np.exp(-s) * mask).sum(1))
    mean = (s * mask).sum(1) / mask.sum(1)
    g = np.maximum(glob @ params["w3"] + params["b3"], 0) @ params["w4"]
    return (params["mix"][0] * softmin + params["mix"][1] * mean + g +
            glob[:, GLOBAL_BINS:] @ params["vs"] + params["c"][0])


def sparse(vectors) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.flatnonzero(v), v[np.flatnonzero(v)]) for v in vectors]


class Net(nn.Module):
    def __init__(self, hidden: int, global_hidden: int):
        super().__init__()
        k1, k3 = 1 / math.sqrt(WORD_DIM), 1 / math.sqrt(GLOBAL_DIM)
        self.w1 = nn.EmbeddingBag(WORD_DIM, hidden, mode="sum")
        self.b1 = nn.Parameter(torch.empty(hidden).uniform_(-k1, k1))
        nn.init.uniform_(self.w1.weight, -k1, k1)  # = nn.Linear's default init
        self.out = nn.Linear(hidden, 1)
        self.mix = nn.Parameter(torch.tensor([1.0, 0.3]))
        self.w3 = nn.EmbeddingBag(GLOBAL_DIM, global_hidden, mode="sum")
        self.b3 = nn.Parameter(torch.empty(global_hidden).uniform_(-k3, k3))
        nn.init.uniform_(self.w3.weight, -k3, k3)
        self.w4 = nn.Linear(global_hidden, 1)
        self.vs = nn.Linear(GLOBAL_DIM - GLOBAL_BINS, 1)

    def forward(self, batch):
        (wi, wv, wo), seg, n, (gi, gv, go), shape = batch
        s = self.out(torch.relu(self.w1(wi, wo, per_sample_weights=wv) + self.b1)).squeeze(-1).clamp(-30, 30)
        softmin = -torch.zeros(n).index_add_(0, seg, torch.exp(-s)).log()
        mean = torch.zeros(n).index_add_(0, seg, s) / torch.zeros(n).index_add_(0, seg, torch.ones_like(s))
        g = self.w4(torch.relu(self.w3(gi, go, per_sample_weights=gv) + self.b3)).squeeze(-1)
        return self.mix[0] * softmin + self.mix[1] * mean + g + self.vs(shape).squeeze(-1)

    def params(self) -> dict:
        f = lambda t: t.detach().cpu().numpy().astype(np.float32).copy()
        return {"w1": f(self.w1.weight), "b1": f(self.b1), "w2": f(self.out.weight).reshape(-1),
                "b2": f(self.out.bias), "mix": f(self.mix), "w3": f(self.w3.weight), "b3": f(self.b3),
                "w4": f(self.w4.weight).reshape(-1), "vs": f(self.vs.weight).reshape(-1),
                "c": f(self.w4.bias + self.vs.bias)}


def _flat(items):
    idx, val, off, total = [], [], [], 0
    for i, v in items:
        off.append(total); idx.append(i); val.append(v); total += len(i)
    return (torch.from_numpy(np.concatenate(idx)).long(), torch.from_numpy(np.concatenate(val)).float(),
            torch.tensor(off, dtype=torch.long))


def collate(rows: list[dict]):
    words = [w for r in rows for w in r["words"]]
    seg = torch.tensor([j for j, r in enumerate(rows) for _ in r["words"]])
    return (_flat(words), seg, len(rows), _flat([r["global"] for r in rows]),
            torch.from_numpy(np.stack([r["shape"] for r in rows])))


def fit(train: list[dict], y, validation: list[dict], validation_y, *, seed=42, hidden=13, global_hidden=4,
        epochs=400, batch_size=128, learning_rate=0.003, l2=3e-4, patience=35, weights=None):
    """rows: {"words": [(idx, val)...], "global": (idx, val), "shape": float32[20]}. CPU: sparse ops."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(hidden, global_hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    w = torch.from_numpy(weights if weights is not None else np.ones(len(train), np.float32))
    ty, vy = torch.from_numpy(np.asarray(y, np.float32)), torch.from_numpy(np.asarray(validation_y, np.float32))
    vbatch = collate(validation)
    best, best_loss, best_epoch, history = None, float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch_size):
            b = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(collate([train[i] for i in b])), ty[b])
            ((loss * w[b]).sum() / w[b].sum()).backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(criterion(net(vbatch), vy).mean())
        history.append({"epoch": epoch, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best = {k: v.clone() for k, v in net.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"seed={seed} epoch={epoch} validation_loss={val_loss:.6f} best={best_epoch}", flush=True)
        if epoch - best_epoch >= patience:
            break
    net.load_state_dict(best)
    return net.params(), {"device": "cpu", "torch_version": torch.__version__, "best_epoch": best_epoch,
                          "epochs_run": epoch, "best_validation_loss": best_loss, "history": history}
