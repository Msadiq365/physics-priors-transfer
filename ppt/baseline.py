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
