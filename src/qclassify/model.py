"""Train on MPS; convert the best checkpoint to portable NumPy weights."""
import numpy as np
import torch
from torch import nn


# original, lower, upper, title: search boxes are mostly typed lowercase
CASE_MIX = [0.55, 0.35, 0.05, 0.05]


def sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -value))


def forward(params: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(x @ params["w1"] + params["b1"], 0)
    return hidden, hidden @ params["w2"] + params["b2"]


def fit(x, y, validation_x, validation_y, *, seed=42, hidden=8,
        epochs=400, batch_size=128, learning_rate=0.003, l2=0.001,
        patience=35, device="mps", augmentation_blocks=1, weights=None, validation_weights=None):
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Use an MPS-capable Mac and PyTorch build, "
                           "or explicitly select --device cpu. No silent CPU fallback.")
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=l2)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    # per-example weights (optional): clean authored rows can count more than noisy mined ones
    wt = torch.from_numpy(weights if weights is not None else np.ones(len(x), np.float32)).to(device)
    wv = torch.from_numpy(validation_weights if validation_weights is not None else np.ones(len(validation_x), np.float32)).to(device)
    tx, ty, vx, vy = [torch.from_numpy(v).to(device) for v in (x, y, validation_x, validation_y)]
    best, best_loss, best_epoch = None, float("inf"), 0
    history = []
    epoch_rows = len(x) // augmentation_blocks
    for epoch in range(1, epochs + 1):
        network.train()
        for start in range(0, epoch_rows, batch_size):
            if start == 0:
                order = rng.permutation(epoch_rows)
                if augmentation_blocks == 4:
                    order += rng.choice(4, size=epoch_rows, p=CASE_MIX) * epoch_rows
            indices = torch.tensor(order[start:start + batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = (criterion(network(tx[indices]), ty[indices]).reshape(-1) * wt[indices]).sum() / wt[indices].sum()
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            val_loss = float(((criterion(network(vx), vy).reshape(-1) * wv).sum() / wv.sum()).item())
        history.append({"epoch": epoch, "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"device={device} epoch={epoch} validation_loss={val_loss:.6f} best={best_epoch}", flush=True)
        if epoch - best_epoch >= patience:
            break
    params = {"w1": best["0.weight"].numpy().T.copy(), "b1": best["0.bias"].numpy().copy(),
              "w2": best["2.weight"].numpy().T.copy(), "b2": best["2.bias"].numpy().copy()}
    return params, {"device": device, "torch_version": torch.__version__, "best_epoch": best_epoch,
                    "epochs_run": epoch, "history": history}
