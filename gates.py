"""Gates G2 and G3 on the sealed data (definitions fixed in DEV_LOG before running)."""
import hashlib, json, sys, time, numpy as np, torch
sys.path.insert(0, '.')
from ppt import transfer as T, baseline as B, signal as S, model as Mo
from ppt.data import load_cache
from ppt.geometry import load_registry, orders_for
from ppt.isolation import assert_isolated, grouped_folds
from ppt.loaders import sealed_manifest, load_sealed
out = {}
# ---------- G3 ----------
sha = hashlib.sha256(open('data/sealed_export.npz', 'rb').read()).hexdigest()
g3 = dict(sealed_sha_ok=sha == '530894752c4d11dc676fd71e05ee7a5dd99cb26071c50b381710ed2bdc58932e')
frozen = dict(CWRU=(3.585, 5.415), PAD=(3.054, 4.946), UORED=(3.050, 4.950), KAIST=(3.577, 5.423), HUST_6204=(3.093, 4.907),
              HUST_6205=(3.588, 5.412), HUST_6206=(3.620, 5.380), HUST_6207=(3.575, 5.425), HUST_6208=(3.600, 5.400))
g3['registry'] = {k: [round(orders_for(k)['BPFO'], 3), round(orders_for(k)['BPFI'], 3), abs(orders_for(k)['BPFO'] - v[0]) < 0.006 and abs(orders_for(k)['BPFI'] - v[1]) < 0.006] for k, v in frozen.items()}
spd = []
for e in sealed_manifest('data/sealed_export.npz'):
    if e['dataset'] != 'UORED': continue
    x, fs, hz = load_sealed(e); import os; os.environ['UORED_SPEED']='stored'; _, _, hz_st = load_sealed(e); del os.environ['UORED_SPEED']
    X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))); f = np.fft.rfftfreq(len(x), 1 / fs)
    band = (f >= 0.9 * hz) & (f <= 1.1 * hz); pk = float(f[band][np.argmax(X[band])])
    spd.append(dict(rec=e['path'].split('::')[1], used_hz=round(hz, 3), stored_hz=round(hz_st, 3), stored_rel=round(abs(pk-hz_st)/hz_st, 4), peak_hz=round(pk, 3), rel=round(abs(pk - hz) / hz, 4)))
g3['uored_speed'] = dict(n=len(spd), n_within_2pct=sum(s['rel'] <= 0.02 for s in spd), stored_n_within_2pct=sum(s['stored_rel'] <= 0.02 for s in spd), worst=sorted(spd, key=lambda s: -s['rel'])[:5])
out['G3'] = g3; print('G3', json.dumps(g3, indent=0)[:1500], flush=True)
# ---------- G2 ----------
cache = load_cache(); meta = cache['meta']
def auroc(score, pos):
    from scipy.stats import rankdata
    r = rankdata(score); n1 = pos.sum(); n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def c0_auroc(ds):
    idx = T.select(cache, ds); units = {}
    for i in idx:
        m = meta[i]; o = T.geo(m); po, pi = B.scores(cache['order'][i:i + 1], o['BPFO'], o['BPFI'])
        units.setdefault((m['bearing_id'], m['state'], m['cls']), []).append((np.log(po[0]), np.log(pi[0])))
    k = sorted(units); v = np.array([np.mean(units[u], 0) for u in k]); c = np.array([u[2] for u in k])
    return dict(IR=auroc(v[:, 1], c == 'IR'), OR=auroc(v[:, 0], c == 'OR'))
def ceiling(ds):
    idx = T.select(cache, ds); mt = [meta[i] for i in idx]; y = np.array([B.CLASSES.index(m['cls']) for m in mt])
    R = T._inputs('M2', cache, idx, ds); nN = len({m['bearing_id'] for m in mt if m['cls'] == 'N'})
    fixed = nN < 3; probs = {}; chosen = []
    for f in grouped_folds(mt, 3, seed=101):
        te = np.array([m['bearing_id'] in f for m in mt]); tr = ~te
        if not te.any(): continue
        mtr = [m for m, t in zip(mt, tr) if t]; assert_isolated(mtr, [m for m, t in zip(mt, te) if t])
        E = 30 if fixed else T.choose_epochs('M2', R[tr], y[tr], mtr, ds)
        chosen.append(E)
        P = np.mean([T.predict(T._fit('M2', R[tr], y[tr], s, E)[0], R[te]) for s in T.SEEDS], 0)
        for m, p in zip([m for m, t in zip(mt, te) if t], P): probs.setdefault((m['bearing_id'], m['state'], m['cls']), []).append(p)
    k = sorted(probs); yt = np.array([B.CLASSES.index(u[2]) for u in k]); yp = np.array([np.mean(probs[u], 0).argmax() for u in k])
    sc = float(np.mean([np.mean(yp[yt == c] == c) for c in (1, 2) if np.any(yt == c)])) if fixed else B.macro_f1(yt, yp)
    return dict(score=round(sc, 3), metric='mean IR/OR recall' if fixed else 'macro-F1', epochs=chosen, units_scored=len(k))
g2 = {}; t0 = time.time()
for ds in ('UORED', 'KAIST', 'HUST', 'PAD', 'CWRU'):
    a = c0_auroc(ds); cl = ceiling(ds); au = (a['IR'] + a['OR']) / 2
    g2[ds] = dict(c0_auroc_IR=round(a['IR'], 3), c0_auroc_OR=round(a['OR'], 3), c0_auroc=round(au, 3), ceiling=cl,
                  pass_=bool(au >= 0.65 and cl['score'] >= 0.60))
    print(ds, g2[ds], f'({time.time() - t0:.0f}s)', flush=True)
out['G2'] = g2
json.dump(out, open('results/gates.json', 'w'), indent=1)
