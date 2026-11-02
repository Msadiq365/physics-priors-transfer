"""Minimum detectable difference for the registered endpoint (per-bearing Brier, exact sign-flip, n = 10 bearings).
Spread of paired per-bearing differences taken from the development dry run (all learned-arm pairs and each
learned arm versus probabilistic C0, both directions). Power by simulation: 10 differences drawn (a) from a normal
with that SD, (b) by resampling the centred empirical differences; shift delta added."""
import json, glob, os, sys, numpy as np
sys.path.insert(0, '.')
from ppt import stats as S, transfer as T
c = np.load('data/dev_cache.npz'); cache = dict(meta=json.loads(str(c['meta'])), order=c['order'], hz=c['hz'])
scores = {}
for f in sorted(glob.glob('runs/dry/*_probs.npz')):
    tag, arm = os.path.basename(f)[:-10].split('_')
    z = np.load(f); meta = json.loads(str(z['target_meta']))
    keep = [i for i, m in enumerate(meta) if T.usable(m)]
    scores.setdefault(tag, {})[arm] = S.bearing_brier([meta[i] for i in keep], z['P'][keep])
for tag, (s, t) in {'CWRU2PAD': ('CWRU', 'PAD'), 'PAD2CWRU': ('PAD', 'CWRU')}.items():
    r = T.run_c0(cache, s, t); scores[tag]['C0'] = r['brier']
    print(tag, 'C0 temperature', round(r['temperature'], 3), '| C0 Brier soft', round(np.mean(list(r['brier'].values())), 3),
          'one-hot', round(np.mean(list(r['brier_onehot'].values())), 3))
diffs, sds = [], {}
for tag, arms in scores.items():
    print(tag, {a: round(float(np.mean(list(v.values()))), 3) for a, v in sorted(arms.items())})
    names = sorted(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = S.paired(arms[a], arms[b]); diffs.append(d - d.mean()); sds[f'{tag}:{a}-{b}'] = float(d.std(ddof=1))
pool = np.concatenate(diffs); sd = float(np.median(list(sds.values())))
for k in ('CWRU2PAD','PAD2CWRU'):
    v=[x for n,x in sds.items() if n.startswith(k)]; print(k,'median SD',round(float(np.median(v)),3))
print(f'paired-difference SD: median {sd:.3f}, range {min(sds.values()):.3f} to {max(sds.values()):.3f}')
rng = np.random.default_rng(2026); Sg = S._signs(10)
def power(delta, gen, alpha, n_sim=2000):
    d = gen((n_sim, 10)) + delta
    null = np.abs(d @ Sg.T) / 10; obs = np.abs(d.mean(1))[:, None]
    return float(np.mean(np.mean(null >= obs - 1e-12, axis=1) <= alpha))
gens = {'normal': lambda sh: rng.normal(0, sd, sh), 'empirical': lambda sh: rng.choice(pool, size=sh)}
out = dict(sd_median=sd, sd_range=[min(sds.values()), max(sds.values())], rows=[])
for name, g in gens.items():
    for alpha in (0.05, 0.05 / 3):
        grid = np.round(np.arange(0.05, 2.01, 0.05), 2); pw = [power(x, g, alpha) for x in grid]
        mdd = next((float(x) for x, p in zip(grid, pw) if p >= 0.8), None)
        out['rows'].append(dict(generator=name, alpha=round(alpha, 4), mdd_80=mdd))
        print(f'{name:9s} alpha {alpha:.4f}: MDD (80% power) = {mdd}  Brier units')
json.dump(out, open('mdd_brier.json', 'w'), indent=1)
