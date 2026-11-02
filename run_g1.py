"""Gate G1: does the pipeline learn inside each source dataset? (development data only)

Leave-bearings-out cross-validation within CWRU and within Paderborn, arms M2 and M3, 5 seeds, fixed 50 epochs.
Scoring is bearing-state level: probabilities averaged over windows and seeds, then argmax.
Paderborn: 3 folds of whole bearings, stratified by class; metric macro-F1 over the 17 bearing states.
CWRU: one healthy bearing, so it always trains; metric is mean recall of held-out IR and OR bearings (as in v2.1).
Pass criterion (methodology gate G1): at least 0.70 for both M2 and M3 in both datasets.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, '.')
from ppt import signal as S, baseline as B
from ppt.geometry import orders_for
from ppt.isolation import assert_isolated, grouped_folds
from ppt.model import train, predict_proba

z = np.load('data/dev_cache.npz'); meta = json.loads(str(z['meta'])); X = z['order']
SEEDS, EPOCHS = range(5), 50
out = {}
t0 = time.time()
for ds in ('PAD', 'CWRU'):
    fo = orders_for(ds)
    idx = np.array([i for i, m in enumerate(meta) if m['dataset'] == ds])
    mt = [meta[i] for i in idx]
    y = np.array([B.CLASSES.index(m['cls']) for m in mt])
    reps = {'M2': S.rep_m2(X[idx]), 'M3': S.rep_m3(X[idx], fo['BPFO'], fo['BPFI'])}
    folds = grouped_folds(mt, n_folds=3, seed=101)
    for arm, R in reps.items():
        unit_prob = {}
        for f in folds:
            te = np.array([m['bearing_id'] in f for m in mt]); tr = ~te
            assert_isolated([m for m, t in zip(mt, tr) if t], [m for m, t in zip(mt, te) if t])
            P = np.mean([predict_proba(train(R[tr], y[tr], s, EPOCHS), R[te]) for s in SEEDS], axis=0)
            for m, p in zip([m for m, t in zip(mt, te) if t], P):
                unit_prob.setdefault((m['bearing_id'], m['state'], m['cls']), []).append(p)
        units = sorted(unit_prob)
        yt = np.array([B.CLASSES.index(u[2]) for u in units])
        yp = np.array([np.mean(unit_prob[u], axis=0).argmax() for u in units])
        if ds == 'CWRU':
            score = float(np.mean([np.mean(yp[yt == c] == c) for c in (1, 2)]))
            name = 'mean IR/OR recall (bearing level)'
        else:
            score = B.macro_f1(yt, yp); name = 'macro-F1 (bearing level)'
        out[f'{ds}_{arm}'] = {'metric': name, 'score': round(score, 3), 'pass': score >= 0.70,
                              'units': [(u[0], u[2], B.CLASSES[p]) for u, p in zip(units, yp)]}
        print(f'{ds} {arm}: {name} = {score:.3f}  {"PASS" if score >= 0.70 else "FAIL"}  ({time.time()-t0:.0f}s)', flush=True)
json.dump(out, open('results_g1.json', 'w'), indent=1)
