"""Arms M1 to M7: input representations, the Gaussian weighting layer (M4, M5) and parameter matching.

Every arm uses the same CNN body; only the input (and for M4/M5 a front weighting layer) changes.
Widths of the first convolution are chosen so all learned arms stay within 5% of a common parameter count.
"""
import numpy as np
import torch
import torch.nn as nn

from . import signal as S
from .model import Net, n_params

M2_GRID = S.BASE_GRID[: int(round(25.0 / S.ORDER_STEP))]          # 0 to 25 orders, 1,000 bins
SIGMA_MIN, SIGMA_MAX = 0.03, 0.25


def centres(orders, harmonics=(1, 2, 3, 4)):
    return np.array([h * orders[k] for k in ("BPFO", "BPFI") for h in harmonics], np.float32)


class WeightLayer(nn.Module):
    """K Gaussian bumps on the order axis; output is the spectrum times each bump (K channels).
    learn_centre=False: centres are given (physics, M4); True: centres learned (M5)."""

    def __init__(self, c0, learn_centre):
        super().__init__()
        self.register_buffer("grid", torch.tensor(M2_GRID, dtype=torch.float32))
        c0 = torch.tensor(c0, dtype=torch.float32)
        self.learn_centre = learn_centre
        if learn_centre:
            self.c = nn.Parameter(c0.clone())
        else:
            self.register_buffer("c", c0.clone())
        self.log_sigma = nn.Parameter(torch.log(torch.clamp(0.005 * c0, min=0.05)))
        self.amp = nn.Parameter(torch.ones(len(c0))) if not learn_centre else None   # M4: amplitude+width; M5: centre+width

    def forward(self, x, override_centres=None):
        c = override_centres if override_centres is not None else self.c
        c = c.clamp(0.5, 24.0)
        s = self.log_sigma.exp().clamp(SIGMA_MIN, SIGMA_MAX)
        g = torch.exp(-0.5 * ((self.grid[None, :] - c[:, None]) / s[:, None]) ** 2)   # (K, L)
        if self.amp is not None:
            g = g * self.amp[:, None]
        return x * g[None, :, :]                                                      # x: (B, 1, L)


class WeightedNet(nn.Module):
    def __init__(self, c0, learn_centre, c1):
        super().__init__()
        self.layer = WeightLayer(c0, learn_centre)
        self.net = Net(len(c0), c1)
        self.test_centres = None             # M4: set to the target geometry's centres at test time

    def forward(self, x):
        oc = None if self.test_centres is None else torch.tensor(self.test_centres, dtype=torch.float32)
        return self.net(self.layer(x, oc))


def represent(arm, sn_order, sn_hz, orders):
    """Input array for an arm, using the geometry `orders` of the bearing the windows come from."""
    if arm == "M1":
        return S.rep_m1(sn_hz)
    if arm in ("M2", "M4", "M5"):
        return S.rep_m2(sn_order)
    if arm in ("M3", "M6"):
        return S.rep_m3(sn_order, orders["BPFO"], orders["BPFI"])
    if arm == "M7":
        return S.rep_m7(sn_order, orders["BPFO"], orders["BPFI"])
    raise ValueError(arm)


# ---------------- parameter matching ----------------
REFERENCE_C1 = 16


def build(arm, source_orders, c1=None, seed=0):
    torch.manual_seed(seed)
    if arm in ("M4", "M5"):
        c0 = centres(source_orders) if arm == "M4" else np.sort(np.random.default_rng(seed).uniform(1, 16, 8)).astype(np.float32)
        return WeightedNet(c0, learn_centre=(arm == "M5"), c1=c1 or matched_c1(arm))
    in_ch = 2 if arm in ("M3", "M6") else 1
    return Net(in_ch, c1 or matched_c1(arm))


def target_params():
    return n_params(WeightedNet(np.ones(8, np.float32), False, REFERENCE_C1))


def matched_c1(arm):
    """First-layer width giving a parameter count closest to the M4 reference (M4 uses REFERENCE_C1)."""
    if arm == "M4":
        return REFERENCE_C1
    goal = target_params()
    def count(c1):
        if arm == "M5":
            return n_params(WeightedNet(np.ones(8, np.float32), True, c1))
        return n_params(Net(2 if arm in ("M3", "M6") else 1, c1))
    return min(range(4, 64), key=lambda c1: abs(count(c1) - goal))
