"""Transfer runner: train on all source bearings, predict every target window. No target label is read
until scoring, and no target window touches training, epoch choice or preprocessing.

Epochs: nested grouped CV on source bearings (checkpoints CHK), except CWRU as source (fixed 30, see DEV_LOG).
M4 is trained with source-geometry centres and evaluated with target-geometry centres (recentring).
Outputs per run: window probabilities, bearing-state scores, chosen epochs, and saved weights.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import arms as A
from . import baseline as B
from . import model as Mo
from .geometry import orders_for
from .isolation import assert_isolated, grouped_folds

CHK = [5, 10, 15, 20, 25, 30, 40, 50, 60]
CWRU_FIXED_EPOCHS = 30
SEEDS = range(5)


def geometry_key(meta_row):
    return meta_row.get("geometry", meta_row["dataset"])


def _fit(arm, R, y, seed, epochs, src_orders, checkpoints=(), R_val=None):
    net = A.build(arm, src_orders, seed=seed)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=Mo.LR, weight_decay=Mo.WD)
    c = np.bincount(y, minlength=3)
    w = torch.tensor(np.where(c > 0, c.sum() / (3 * np.maximum(c, 1)), 0.0), dtype=torch.float32)
    Xt, yt = torch.tensor(Mo._as3d(R)), torch.tensor(y)
    curve = {}
    for ep in range(1, epochs + 1):
        net.train()
        for b in np.array_split(rng.permutation(len(Xt)), max(1, len(Xt) // Mo.BATCH)):
            xb = Xt[b] * torch.empty(len(b), 1, 1).uniform_(*Mo.AMP_JIT) + Mo.NOISE * torch.randn_like(Xt[b])
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
    """Average window probabilities per bearing state; return units, true and predicted class indices."""
    units = {}
    for m, p in zip(meta, P):
        units.setdefault((m["bearing_id"], m["state"], m["cls"]), []).append(p)
    keys = sorted(units)
    yt = np.array([B.CLASSES.index(k[2]) for k in keys])
    prob = np.array([np.mean(units[k], axis=0) for k in keys])
    return keys, yt, prob


def choose_epochs(arm, R, y, meta, src_ds, src_orders):
    if src_ds == "CWRU":
        return CWRU_FIXED_EPOCHS
    folds = grouped_folds(meta, n_folds=3, seed=202)
    acc = {e: [] for e in CHK}
    for f in folds:
        te = np.array([m["bearing_id"] in f for m in meta]); tr = ~te
        per_seed = []
        for s in SEEDS:
            _, cv = _fit(arm, R[tr], y[tr], s, max(CHK), src_orders, CHK, R[te])
            per_seed.append(cv)
        for e in CHK:
            P = np.mean([cv[e] for cv in per_seed], axis=0)
            _, yt, prob = unit_scores([m for m, t in zip(meta, te) if t], P)
            acc[e].append((yt, prob.argmax(1)))
    def score(e):
        yt = np.concatenate([a[0] for a in acc[e]]); yp = np.concatenate([a[1] for a in acc[e]])
        return B.macro_f1(yt, yp)
    return max(CHK, key=lambda e: (score(e), -e))


def run(arm, cache, src_ds, tgt_ds, out_dir, src_filter=None, tgt_filter=None, seeds=SEEDS):
    """Train `arm` on source dataset windows, predict target windows. Returns a result dict."""
    meta, X, H = cache["meta"], cache["order"], cache["hz"]
    si = np.array([i for i, m in enumerate(meta) if m["dataset"] == src_ds and (src_filter is None or src_filter(m))])
    ti = np.array([i for i, m in enumerate(meta) if m["dataset"] == tgt_ds and (tgt_filter is None or tgt_filter(m))])
    smeta, tmeta = [meta[i] for i in si], [meta[i] for i in ti]
    assert_isolated(smeta, tmeta)
    so, to = orders_for(src_ds), orders_for(tgt_ds)
    Rs = A.represent(arm, X[si], H[si], so)
    Rt = A.represent(arm, X[ti], H[ti], to)
    ys = np.array([B.CLASSES.index(m["cls"]) for m in smeta])
    E = choose_epochs(arm, Rs, ys, smeta, src_ds, so)
    os.makedirs(out_dir, exist_ok=True)
    probs = []
    for s in seeds:
        net, _ = _fit(arm, Rs, ys, s, E, so)
        if arm == "M4":
            net.test_centres = A.centres(to)          # recentring to target geometry
        probs.append(predict(net, Rt))
        torch.save(net.state_dict(), os.path.join(out_dir, f"{src_ds}2{tgt_ds}_{arm}_seed{s}.pt"))
    P = np.mean(probs, axis=0)
    keys, yt, prob = unit_scores(tmeta, P)
    yp = prob.argmax(1)
    res = dict(arm=arm, source=src_ds, target=tgt_ds, epochs=E, n_source_windows=len(si), n_target_windows=len(ti),
               bearing_macro_f1=B.macro_f1(yt, yp),
               window_macro_f1=B.macro_f1(np.array([B.CLASSES.index(m["cls"]) for m in tmeta]), P.argmax(1)),
               n_classes_predicted=int(len(set(yp.tolist()))),
               units=[(k[0], k[1], k[2], B.CLASSES[p]) for k, p in zip(keys, yp)])
    np.savez_compressed(os.path.join(out_dir, f"{src_ds}2{tgt_ds}_{arm}_probs.npz"), P=P.astype(np.float32),
                        per_seed=np.array(probs, np.float32), target_meta=json.dumps(tmeta))
    json.dump(res, open(os.path.join(out_dir, f"{src_ds}2{tgt_ds}_{arm}.json"), "w"), indent=1)
    return res


def run_c0(cache, src_ds, tgt_ds, src_filter=None, tgt_filter=None):
    meta, X = cache["meta"], cache["order"]
    si = [i for i, m in enumerate(meta) if m["dataset"] == src_ds and (src_filter is None or src_filter(m))]
    ti = [i for i, m in enumerate(meta) if m["dataset"] == tgt_ds and (tgt_filter is None or tgt_filter(m))]
    so, to = orders_for(src_ds), orders_for(tgt_ds)
    ys = np.array([B.CLASSES.index(meta[i]["cls"]) for i in si])
    po, pi = B.scores(X[si], so["BPFO"], so["BPFI"])
    tau = B.fit_threshold(po, pi, ys)
    qo, qi = B.scores(X[ti], to["BPFO"], to["BPFI"])
    pred = B.predict(qo, qi, tau)
    keys, yt, prob = unit_scores([meta[i] for i in ti], np.eye(3)[pred])
    yp = prob.argmax(1)
    return dict(arm="C0", source=src_ds, target=tgt_ds, tau=tau, bearing_macro_f1=B.macro_f1(yt, yp),
                n_classes_predicted=int(len(set(yp.tolist()))),
                units=[(k[0], k[1], k[2], B.CLASSES[p]) for k, p in zip(keys, yp)])
