"""Transfer runner: train on source bearings, predict target windows. No target label is read until scoring,
and no target window touches training, epoch choice or preprocessing (M6 uses unlabelled healthy-reference
windows from a pool that is never scored).

Epochs: nested grouped CV on source bearings (checkpoints CHK), except CWRU as source (fixed 30, see DEV_LOG).
M4 is trained with each source window's own centres and evaluated with each target window's own centres.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import arms as A
from . import baseline as B
from . import model as Mo
from . import stats as S
from .geometry import orders_for
from .isolation import assert_isolated, grouped_folds

CHK = [5, 10, 15, 20, 25, 30, 40, 50, 60]
CWRU_FIXED_EPOCHS = 30
SEEDS = range(5)

# Healthy-reference pools: never trained on, never scored, by any arm (registered)
POOL = {"PAD": lambda m: m["bearing_id"] in ("PAD_K005", "PAD_K006"),
        "UORED": lambda m: int(m["bearing_id"].split("_")[1]) >= 11}


def usable(m):
    """Windows that may be trained on or scored: not in a reference pool, not a 'developing' state."""
    pool = POOL.get(m["dataset"])
    return m["state"] != "developing" and not (pool and pool(m))


def geo(m):
    return orders_for(m.get("geometry", m["dataset"]))


def healthy_reference(meta, X, dataset):
    idx = [i for i, m in enumerate(meta) if m["dataset"] == dataset and POOL[dataset](m) and m["cls"] == "N"]
    if not idx:
        raise ValueError(f"no healthy-reference pool for {dataset}")
    return np.median(X[idx], axis=0)


def _inputs(arm, cache, idx, ds, **kw):
    meta, X, H = cache["meta"], cache["order"], cache["hz"]
    if arm == "WDCNN":
        return cache["raw"][idx]
    Xs = X[idx]
    if arm == "M6":
        Xs = Xs / (healthy_reference(meta, X, ds)[None, :] + 1e-12)
    return A.represent(arm, Xs, H[idx], [geo(meta[i]) for i in idx], **kw)


def _fit(arm, R, y, seed, epochs, checkpoints=(), R_val=None):
    if arm == "WDCNN":
        from .wdcnn import WDCNN
        torch.manual_seed(seed); net = WDCNN()
    else:
        net = A.build(arm, None, seed=seed) if arm != "M4" else A.build(arm, {"BPFO": 1.0, "BPFI": 1.0}, seed=seed)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=Mo.LR, weight_decay=Mo.WD)
    c = np.bincount(y, minlength=3)
    w = torch.tensor(np.where(c > 0, c.sum() / (3 * np.maximum(c, 1)), 0.0), dtype=torch.float32)
    Xt, yt = torch.tensor(Mo._as3d(R)), torch.tensor(y)
    L = Xt.shape[-1] - (A.N_CENTRES if arm == "M4" else 0)          # augmentation never touches M4's centres
    curve = {}
    for ep in range(1, epochs + 1):
        net.train()
        for b in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // Mo.BATCH)):
            xb = Xt[b].clone()
            xb[..., :L] = xb[..., :L] * torch.empty(len(b), 1, 1).uniform_(*Mo.AMP_JIT) + Mo.NOISE * torch.randn_like(xb[..., :L])
            loss = F.cross_entropy(net(xb), yt[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in checkpoints:
            curve[ep] = predict(net, R_val)
    return net, curve


@torch.no_grad()
def predict(net, R):
    net.eval()
    return F.softmax(net(torch.tensor(Mo._as3d(R))), -1).numpy()


def unit_scores(meta, P):
    units = {}
    for m, p in zip(meta, P):
        units.setdefault((m["bearing_id"], m["state"], m["cls"]), []).append(p)
    keys = sorted(units)
    yt = np.array([B.CLASSES.index(k[2]) for k in keys])
    prob = np.array([np.mean(units[k], axis=0) for k in keys])
    return keys, yt, prob


def true_class_prob_by_bearing(meta, P):
    """Registered inferential metric: mean probability of the true class, per physical bearing."""
    keys, yt, prob = unit_scores(meta, P)
    per = {}
    for k, c, p in zip(keys, yt, prob):
        per.setdefault(k[0], []).append(p[c])
    return {b: float(np.mean(v)) for b, v in per.items()}


def choose_epochs(arm, R, y, meta, src_ds):
    if src_ds == "CWRU":
        return CWRU_FIXED_EPOCHS
    folds = grouped_folds(meta, n_folds=3, seed=202)
    acc = {e: [] for e in CHK}
    for f in folds:
        te = np.array([m["bearing_id"] in f for m in meta]); tr = ~te
        if not te.any() or not tr.any():
            continue                      # an empty inner fold (few bearings) carries no information
        per_seed = [_fit(arm, R[tr], y[tr], s, max(CHK), CHK, R[te])[1] for s in SEEDS]
        for e in CHK:
            _, yt, prob = unit_scores([m for m, t in zip(meta, te) if t], np.mean([cv[e] for cv in per_seed], axis=0))
            acc[e].append((yt, prob.argmax(1)))
    if not acc[CHK[0]]:
        return CWRU_FIXED_EPOCHS          # no usable inner fold: same fixed rule as a CWRU source
    score = lambda e: B.macro_f1(np.concatenate([a[0] for a in acc[e]]), np.concatenate([a[1] for a in acc[e]]))
    return max(CHK, key=lambda e: (score(e), -e))


def select(cache, ds, filt=None):
    return np.array([i for i, m in enumerate(cache["meta"]) if m["dataset"] == ds and usable(m) and (filt is None or filt(m))])


def run(arm, cache, src_ds, tgt_ds, out_dir, src_filter=None, tgt_filter=None, seeds=SEEDS, tag=None, controls=False):
    meta = cache["meta"]
    si, ti = select(cache, src_ds, src_filter), select(cache, tgt_ds, tgt_filter)
    smeta, tmeta = [meta[i] for i in si], [meta[i] for i in ti]
    assert_isolated(smeta, tmeta)
    Rs, Rt = _inputs(arm, cache, si, src_ds), _inputs(arm, cache, ti, tgt_ds)
    ys = np.array([B.CLASSES.index(m["cls"]) for m in smeta])
    E = choose_epochs(arm, Rs, ys, smeta, src_ds)
    tag = tag or f"{src_ds}2{tgt_ds}"
    os.makedirs(out_dir, exist_ok=True)
    probs, nets = [], []
    for s in seeds:
        net, _ = _fit(arm, Rs, ys, s, E)
        probs.append(predict(net, Rt)); nets.append(net)
        torch.save(net.state_dict(), os.path.join(out_dir, f"{tag}_{arm}_seed{s}.pt"))
    P = np.mean(probs, axis=0)
    keys, yt, prob = unit_scores(tmeta, P); yp = prob.argmax(1)
    res = dict(arm=arm, source=src_ds, target=tgt_ds, tag=tag, epochs=E, n_source_windows=len(si), n_target_windows=len(ti),
               bearing_macro_f1=B.macro_f1(yt, yp),
               window_macro_f1=B.macro_f1(np.array([B.CLASSES.index(m["cls"]) for m in tmeta]), P.argmax(1)),
               true_class_prob=true_class_prob_by_bearing(tmeta, P),
               brier=S.bearing_brier(tmeta, P),
               n_classes_predicted=int(len(set(yp.tolist()))),
               units=[(k[0], k[1], k[2], B.CLASSES[p]) for k, p in zip(keys, yp)])
    if controls and arm == "M4":
        res["controls"] = m4_controls(nets, cache, ti, tgt_ds, tmeta)
    np.savez_compressed(os.path.join(out_dir, f"{tag}_{arm}_probs.npz"), P=P.astype(np.float32),
                        per_seed=np.array(probs, np.float32), target_meta=json.dumps(tmeta))
    json.dump(res, open(os.path.join(out_dir, f"{tag}_{arm}.json"), "w"), indent=1)
    return res


def m4_controls(nets, cache, ti, tgt_ds, tmeta, n_random=20, seed=7):
    """Forward passes only: target centres perturbed by -5, -2, +2, +5 %, and 20 random centre sets (1 to 16 orders)."""
    def score(R):
        P = np.mean([predict(n, R) for n in nets], axis=0)
        _, yt, prob = unit_scores(tmeta, P)
        return B.macro_f1(yt, prob.argmax(1))
    out = {f"pert{p:+d}": score(_inputs("M4", cache, ti, tgt_ds, centre_scale=1 + p / 100)) for p in (-5, -2, 2, 5)}
    rng = np.random.default_rng(seed)
    out["random"] = [score(_inputs("M4", cache, ti, tgt_ds, centre_override=np.sort(rng.uniform(1, 16, 8)))) for _ in range(n_random)]
    return out


def label_shuffle_control(cache, src_ds, tgt_ds, seed=11):
    """G4: M2 trained on source labels permuted across physical bearings; must score near chance."""
    meta = cache["meta"]; si, ti = select(cache, src_ds), select(cache, tgt_ds)
    smeta, tmeta = [meta[i] for i in si], [meta[i] for i in ti]
    bears = sorted({m["bearing_id"] for m in smeta}); lab = {b: next(m["cls"] for m in smeta if m["bearing_id"] == b) for b in bears}
    perm = dict(zip(bears, np.random.default_rng(seed).permutation([lab[b] for b in bears])))
    ys = np.array([B.CLASSES.index(perm[m["bearing_id"]]) for m in smeta])
    Rs, Rt = _inputs("M2", cache, si, src_ds), _inputs("M2", cache, ti, tgt_ds)
    E = CWRU_FIXED_EPOCHS
    P = np.mean([predict(_fit("M2", Rs, ys, s, E)[0], Rt) for s in SEEDS], axis=0)
    _, yt, prob = unit_scores(tmeta, P)
    return B.macro_f1(yt, prob.argmax(1))


def run_c0(cache, src_ds, tgt_ds, src_filter=None, tgt_filter=None):
    meta, X = cache["meta"], cache["order"]
    si, ti = select(cache, src_ds, src_filter), select(cache, tgt_ds, tgt_filter)
    ys = np.array([B.CLASSES.index(meta[i]["cls"]) for i in si])
    def sc(idx):
        po, pi = np.zeros(len(idx)), np.zeros(len(idx))
        for j, i in enumerate(idx):
            o = geo(meta[i]); a, b = B.scores(X[i:i + 1], o["BPFO"], o["BPFI"]); po[j], pi[j] = a[0], b[0]
        return po, pi
    po, pi = sc(si); tau = B.fit_threshold(po, pi, ys)
    T = B.fit_temperature(po, pi, ys, [meta[i]["bearing_id"] for i in si])
    qo, qi = sc(ti); pred = B.predict(qo, qi, tau)
    Psoft = B.soft(qo, qi, tau, T)
    assert np.array_equal(Psoft.argmax(1), pred), "soft C0 must keep every C0 decision"
    tmeta = [meta[i] for i in ti]
    keys, yt, prob = unit_scores(tmeta, np.eye(3)[pred]); yp = prob.argmax(1)
    return dict(arm="C0", source=src_ds, target=tgt_ds, tau=tau, temperature=T, bearing_macro_f1=B.macro_f1(yt, yp),
                brier=S.bearing_brier(tmeta, Psoft), brier_onehot=S.bearing_brier(tmeta, np.eye(3)[pred]),
                window_P=Psoft.tolist(),
                true_class_prob=true_class_prob_by_bearing(tmeta, np.eye(3)[pred]),
                n_classes_predicted=int(len(set(yp.tolist()))),
                units=[(k[0], k[1], k[2], B.CLASSES[p]) for k, p in zip(keys, yp)])
