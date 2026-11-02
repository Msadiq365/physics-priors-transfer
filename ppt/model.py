"""Small 1D CNN shared by every learned arm, and its training loop (frozen v2.1 settings)."""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("TRAIN_THREADS", "1")))   # fixed: thread count changes results
LR, WD, BATCH = 1e-3, 1e-4, 64
AMP_JIT, NOISE = (0.9, 1.1), 0.05
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))   # dev change 2 Nov 2026 (G1 recipe)


class Net(nn.Module):
    def __init__(self, in_ch, c1=16, n_cls=3, dropout=DROPOUT):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, c1, 9, padding=4), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(c1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(32, n_cls)

    def forward(self, x):
        return self.fc(self.drop(self.body(x).squeeze(-1)))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _as3d(X):
    X = np.asarray(X, np.float32)
    return X[:, None, :] if X.ndim == 2 else X


def train(X, y, seed, epochs, in_ch=None, c1=16):
    X = _as3d(X); y = np.asarray(y)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(X.shape[1] if in_ch is None else in_ch, c1)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    counts = np.bincount(y, minlength=3).astype(float)
    w = torch.tensor(np.where(counts > 0, counts.sum() / (3 * np.maximum(counts, 1)), 0.0), dtype=torch.float32)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // BATCH)):
            xb = Xt[b] * torch.empty(len(b), 1, 1).uniform_(*AMP_JIT) + NOISE * torch.randn_like(Xt[b])
            loss = F.cross_entropy(net(xb), yt[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def predict_proba(net, X):
    net.eval()
    return F.softmax(net(torch.tensor(_as3d(X))), -1).numpy()
