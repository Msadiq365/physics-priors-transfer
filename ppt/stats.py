"""Registered statistics: per-physical-bearing Brier score, exact paired sign-flip test, Holm, intervals.

Unit forecast = window probabilities averaged over windows (and seeds); unit Brier = sum_k (p_k - y_k)^2 (0 to 2);
bearing score = mean over that physical bearing's units. Paired difference d_i per bearing; exact two-sided
sign-flip test over all 2^n sign patterns, statistic = mean(d).
"""
import itertools
import numpy as np

CLASSES = ("N", "IR", "OR")


def bearing_brier(meta, P):
    units = {}
    for m, p in zip(meta, P):
        units.setdefault((m["bearing_id"], m["state"], m["cls"]), []).append(p)
    per = {}
    for (b, _, c), ps in units.items():
        f = np.mean(ps, axis=0); y = np.eye(3)[CLASSES.index(c)]
        per.setdefault(b, []).append(float(np.sum((f - y) ** 2)))
    return {b: float(np.mean(v)) for b, v in sorted(per.items())}


def paired(ref, comp):
    """d_i = ref_i - comp_i (positive favours comp, since lower Brier is better)."""
    keys = sorted(ref)
    assert keys == sorted(comp), "arms scored on different bearings"
    return np.array([ref[k] - comp[k] for k in keys])


def _signs(n):
    return np.array(list(itertools.product((1, -1), repeat=n)), float)


def signflip_p(d):
    d = np.asarray(d, float); S = _signs(len(d))
    null = np.abs(S @ d) / len(d)
    return float(np.mean(null >= abs(d.mean()) - 1e-12))


def signflip_ci(d, alpha=0.05, grid=2001):
    """Interval for a constant shift: all delta not rejected by the sign-flip test of d - delta."""
    d = np.asarray(d, float); span = np.ptp(d) + 1e-9
    deltas = np.linspace(d.min() - span, d.max() + span, grid)
    keep = [x for x in deltas if signflip_p(d - x) > alpha]
    return (float(min(keep)), float(max(keep))) if keep else (float("nan"), float("nan"))


def bootstrap_ci(d, n=10000, seed=2026, alpha=0.05):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def holm(pvals, alpha=0.05):
    """dict name -> p. Returns name -> (adjusted p, rejected)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); k = len(items); out = {}; running = 0.0; stop = False
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (k - i) * p)); running = adj
        if p > alpha / (k - i):
            stop = True
        out[name] = (adj, not stop)
    return out


def sign_test_correctness(correct_a, correct_b):
    """Exact two-sided sign test on discordant bearings (sensitivity analysis)."""
    from scipy.stats import binomtest
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return (1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)), n10, n01
