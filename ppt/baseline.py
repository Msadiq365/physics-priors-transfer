"""C0: train-free envelope baseline (methodology section 'Models, arms and controls').

Per window: P_O = sum over k = 1..3 of (peak within +-2% of k*BPFO) / (median of S within +-0.5 order of k*BPFO),
and P_I likewise for BPFI. Predict the larger family if its score exceeds a threshold, otherwise healthy.
The threshold is chosen on source bearings only.
"""
import numpy as np

from .signal import BASE_GRID

CLASSES = ("N", "IR", "OR")


def prominence(sn_order, fault_order, harmonics=(1, 2, 3), tol=0.02, half_band=0.5):
    sn = np.atleast_2d(sn_order)
    total = np.zeros(len(sn))
    for k in harmonics:
        c = k * fault_order
        pk = (BASE_GRID >= c * (1 - tol)) & (BASE_GRID <= c * (1 + tol))
        bg = (BASE_GRID >= c - half_band) & (BASE_GRID <= c + half_band) & ~pk
        total += sn[:, pk].max(axis=1) / (np.median(sn[:, bg], axis=1) + 1e-12)
    return total


def scores(sn_order, bpfo, bpfi):
    return prominence(sn_order, bpfo), prominence(sn_order, bpfi)


def predict(p_o, p_i, tau):
    pred = np.where(p_i >= p_o, 1, 2)            # 1 = IR, 2 = OR
    return np.where(np.maximum(p_o, p_i) > tau, pred, 0)


def macro_f1(y, p, n_cls=3):
    f = []
    for c in range(n_cls):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def fit_threshold(p_o, p_i, y, grid=None):
    """Threshold maximising source macro-F1 (source bearings only)."""
    grid = np.quantile(np.maximum(p_o, p_i), np.linspace(0.01, 0.99, 197)) if grid is None else grid
    best = max(grid, key=lambda t: macro_f1(y, predict(p_o, p_i, t)))
    return float(best)


# ---- probabilistic C0 (registered): softmax of log margins with one temperature fitted on source only ----
def log_margins(p_o, p_i, tau):
    """Class scores (N, IR, OR) = log(tau, P_I, P_O). argmax reproduces predict() exactly (ties go to N, then IR)."""
    return np.log(np.stack([np.full_like(p_o, tau), p_i, p_o], axis=1) + 1e-12)


def soft(p_o, p_i, tau, T):
    z = log_margins(p_o, p_i, tau) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


T_GRID = np.exp(np.linspace(np.log(0.02), np.log(20.0), 121))


def fit_temperature(p_o, p_i, y, bearing):
    """Leave-one-source-bearing-out: tau refitted without the held-out bearing, T chosen to minimise the
    mean unit Brier score of held-out bearings. Source data only."""
    bearing = np.asarray(bearing); held = {}
    for b in np.unique(bearing):
        tr = bearing != b
        held[b] = fit_threshold(p_o[tr], p_i[tr], y[tr])
    def loss(T):
        out = []
        for b, tau in held.items():
            te = bearing == b
            P = soft(p_o[te], p_i[te], tau, T)
            for c in np.unique(y[te]):
                f = P[y[te] == c].mean(0); out.append(np.sum((f - np.eye(3)[c]) ** 2))
        return float(np.mean(out))
    return float(min(T_GRID, key=loss))
