"""Exploratory (added after the confirmatory analysis, prompted by the MSSP guidelines for machine-learning papers):
per-arm hyperparameter search on the SOURCE only, then transfer. Grid: learning rate {5e-4, 1e-3, 2e-3} x weight decay
{1e-4, 1e-3} x dropout {0.1, 0.3, 0.5} (18 configurations, includes the registered recipe). Each configuration is scored
by grouped 3-fold cross-validation over source bearings (3 seeds, checkpoints 5 to 60 epochs, bearing-level macro-F1);
the best configuration and epoch are then trained with 5 seeds on all source bearings and scored on the target.
Usage: python3 tune_run.py <worker k> <n workers>"""
import itertools, json, os, sys, time
import numpy as np
sys.path.insert(0, '.')
from ppt import transfer as T, baseline as B, model as Mo, stats as S
from ppt.data import load_cache
from ppt.isolation import assert_isolated, grouped_folds
GRID = list(itertools.product((5e-4, 1e-3, 2e-3), (1e-4, 1e-3), (0.1, 0.3, 0.5)))
PAIRS = [('PAD', 'UORED'), ('UORED', 'PAD')]
ARMS = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'WDCNN']
OUT = 'runs/tuned'; os.makedirs(OUT, exist_ok=True)

def cv_score(arm, R, y, meta, seeds=(0, 1, 2)):
    folds = grouped_folds(meta, n_folds=3, seed=202); acc = {e: [] for e in T.CHK}
    for f in folds:
        te = np.array([m['bearing_id'] in f for m in meta]); tr = ~te
        if not te.any() or not tr.any(): continue
        curves = [T._fit(arm, R[tr], y[tr], s, max(T.CHK), T.CHK, R[te])[1] for s in seeds]
        for e in T.CHK:
            _, yt, prob = T.unit_scores([m for m, t in zip(meta, te) if t], np.mean([c[e] for c in curves], 0))
            acc[e].append((yt, prob.argmax(1)))
    sc = {e: B.macro_f1(np.concatenate([a[0] for a in acc[e]]), np.concatenate([a[1] for a in acc[e]])) for e in T.CHK}
    e = max(T.CHK, key=lambda e: (sc[e], -e)); return sc[e], e

k, n = int(sys.argv[1]), int(sys.argv[2])
cache = load_cache(); cache['raw'] = np.load('data/raw_segments.npy'); meta = cache['meta']
jobs = [(s, t, a) for s, t in PAIRS for a in ARMS if not (a == 'M6' and False)]
for i, (src, tgt, arm) in enumerate(jobs):
    if i % n != k: continue
    path = f'{OUT}/{src}2{tgt}_{arm}.json'
    if os.path.exists(path): continue
    t0 = time.time()
    si, ti = T.select(cache, src), T.select(cache, tgt); smeta, tmeta = [meta[i] for i in si], [meta[i] for i in ti]
    assert_isolated(smeta, tmeta)
    Rs, Rt = T._inputs(arm, cache, si, src), T._inputs(arm, cache, ti, tgt); ys = np.array([B.CLASSES.index(m['cls']) for m in smeta])
    search = []
    for lr, wd, dp in GRID:
        Mo.LR, Mo.WD, T.DROPOUT_OVERRIDE = lr, wd, dp
        sc, ep = cv_score(arm, Rs, ys, smeta); search.append(dict(lr=lr, wd=wd, dropout=dp, cv_macro_f1=sc, epochs=ep))
    best = max(search, key=lambda r: (r['cv_macro_f1'], -r['epochs']))
    Mo.LR, Mo.WD, T.DROPOUT_OVERRIDE = best['lr'], best['wd'], best['dropout']
    P = np.mean([T.predict(T._fit(arm, Rs, ys, s, best['epochs'])[0], Rt) for s in T.SEEDS], 0)
    keys, yt, prob = T.unit_scores(tmeta, P)
    res = dict(arm=arm, source=src, target=tgt, tag=f'{src}2{tgt}', best=best, search=search,
               bearing_macro_f1=B.macro_f1(yt, prob.argmax(1)), brier=S.bearing_brier(tmeta, P),
               n_classes_predicted=int(len(set(prob.argmax(1).tolist()))))
    json.dump(res, open(path, 'w'), indent=1)
    Mo.LR, Mo.WD, T.DROPOUT_OVERRIDE = 1e-3, 1e-4, None
    print(f'[{k}] {src}2{tgt} {arm}: best {best} -> target macro-F1 {res["bearing_macro_f1"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
print(f'[{k}] all done', flush=True)
