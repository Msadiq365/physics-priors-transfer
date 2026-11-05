"""Smoke and physics checks on the small raw sample (development data only)."""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')
from ppt import signal as S, baseline as B
from ppt.geometry import orders_for
from ppt.loaders import cwru_manifest, pad_manifest
from ppt.features import build_cache
from ppt.isolation import assert_isolated, grouped_folds

SCR = os.environ.get('PPT_SCRATCH', 'scratch'); os.makedirs(SCR, exist_ok=True)
SUB = os.environ.get('PPT_SUBSET', 'data/subset')
man = cwru_manifest(SUB + '/CWRU') + pad_manifest(SUB + '/Paderborn', origins=('healthy', 'real', 'artificial'))
print('recordings:', len(man), {d: sum(e['dataset'] == d for e in man) for d in ('CWRU', 'PAD')})
info = build_cache(man, os.path.join(SCR, 'cache_subset.npz'))
print('cache:', {k: v for k, v in info.items() if k != 'sha256'})
z = np.load(os.path.join(SCR, 'cache_subset.npz'))
meta = json.loads(str(z['meta'])); X = z['order']
print('order grid bins', X.shape, 'hz bins', z['hz'].shape, 'step', S.ORDER_STEP)

ok = True
for ds in ('CWRU', 'PAD'):
    fo = orders_for(ds)
    idx = np.array([i for i, m in enumerate(meta) if m['dataset'] == ds])
    y = np.array([B.CLASSES.index(meta[i]['cls']) for i in idx])
    po, pi = B.scores(X[idx], fo['BPFO'], fo['BPFI'])
    for c in (0, 1, 2):
        print(f"{ds} {B.CLASSES[c]:>2}: mean P_O={po[y==c].mean():6.2f}  P_I={pi[y==c].mean():6.2f}  windows={np.sum(y==c)}")
    # physics check: IR windows score higher on BPFI than OR windows do, and vice versa
    if not (pi[y == 1].mean() > pi[y == 2].mean() and po[y == 2].mean() > po[y == 1].mean()):
        print('  physics check FAILED for', ds); ok = False
    m3 = S.rep_m3(X[idx], fo['BPFO'], fo['BPFI'])
    m7 = S.rep_m7(X[idx], fo['BPFO'], fo['BPFI'])
    print(f"  shapes M2 {S.rep_m2(X[idx]).shape} M3 {m3.shape} M7 {m7.shape}")
    u = np.linspace(0, 6, 600, endpoint=False)
    near1 = (u > 0.97) & (u < 1.03)
    print(f"  M3 IR windows: mean BPFI-channel level at fault order 1 = {m3[y==1][:,1][:,near1].max(1).mean():.2f} vs channel median {np.median(m3[y==1][:,1]):.2f}")
folds = grouped_folds([m for m in meta if m['dataset'] == 'CWRU'])
print('CWRU folds:', [sorted(f) for f in folds])
tr = [m for m in meta if m['bearing_id'] not in folds[0] and m['dataset']=='CWRU']; te = [m for m in meta if m['bearing_id'] in folds[0]]
print('isolation assert:', assert_isolated(tr, te))
try:
    assert_isolated(meta, meta[:5]); print('isolation negative test FAILED'); ok = False
except AssertionError: print('isolation negative test: caught leak as expected')
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
