"""Minimum detectable difference for the confirmatory pair (Paderborn -> UORED), by simulation.

Target: 10 physical bearings (5 IR, 5 OR), each giving a healthy and a faulty unit: 20 units (10 N, 5 IR, 5 OR).
Two arms A and B; each unit is correct with probability pA or pB (= pA + d), correlation rho between arms
(Gaussian copula); a wrong unit gets one of the other two classes at random. Test: paired permutation that swaps
the two arms' predictions per physical bearing (both its units together), statistic = macro-F1(B) - macro-F1(A),
two-sided. Inputs from the development dry run: learned-arm unit accuracy 0.3 to 0.6, between-arm correlation
-0.08 to 0.21. Reported: power at alpha 0.05 and at the strictest Holm level 0.05/3.
"""
import json, numpy as np
from scipy.stats import norm
rng = np.random.default_rng(2026)
Y = np.array([0]*10 + [1]*5 + [2]*5); BEAR = np.r_[np.arange(10), np.arange(5), np.arange(5, 10)]

def macro_f1(y, p):
    f = []
    for c in range(3):
        tp = ((y == c) & (p == c)).sum(-1); fp = ((y != c) & (p == c)).sum(-1); fn = ((y == c) & (p != c)).sum(-1)
        f.append(np.where(tp == 0, 0.0, 2*tp / np.maximum(2*tp + fp + fn, 1)))
    return np.mean(f, axis=0)

def preds(correct):
    wrong = (Y + rng.integers(1, 3, size=correct.shape)) % 3
    return np.where(correct, Y, wrong)

def sim(pA, d, rho, n_sim=400, n_perm=1000):
    pB = min(pA + d, 0.99)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=(n_sim, 20))
    A = preds(z[..., 0] < norm.ppf(pA)); B = preds(z[..., 1] < norm.ppf(pB))
    obs = macro_f1(Y, B) - macro_f1(Y, A)
    swap = rng.integers(0, 2, size=(n_perm, 10)).astype(bool)[:, BEAR]          # per bearing
    pv = np.empty(n_sim)
    for i in range(n_sim):
        PA = np.where(swap, B[i], A[i]); PB = np.where(swap, A[i], B[i])
        null = macro_f1(Y, PB) - macro_f1(Y, PA)
        pv[i] = (np.sum(np.abs(null) >= abs(obs[i])) + 1) / (n_perm + 1)
    return obs.mean(), np.mean(pv < 0.05), np.mean(pv < 0.05/3)

out = []
for pA in (0.4, 0.6):
    for rho in (0.0, 0.2, 0.4):
        for d in (0.0, 0.1, 0.2, 0.3, 0.4):
            dm, p05, pH = sim(pA, d, rho)
            out.append(dict(pA=pA, rho=rho, d_acc=d, mean_dF1=round(float(dm), 3), power_05=round(float(p05), 2), power_holm=round(float(pH), 2)))
            print(f"pA {pA} rho {rho} d_acc {d:.1f}: mean dF1 {dm:+.3f}  power(0.05) {p05:.2f}  power(Holm 0.0167) {pH:.2f}", flush=True)
json.dump(out, open('mdd_sim.json', 'w'), indent=1)
