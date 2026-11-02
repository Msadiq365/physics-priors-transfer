"""All registered runs: 12 directed pairs (confirmatory first) and the HUST leave-one-type-out case study.
Usage: python3 run_all.py <worker k> <n workers>. Skips jobs whose result file exists."""
import json, os, sys, time
sys.path.insert(0, '.')
from ppt import transfer as T
from ppt.data import load_cache
OUT = 'runs/main'; os.makedirs(OUT, exist_ok=True)
PAIRS = [('PAD', 'UORED'), ('UORED', 'PAD'), ('CWRU', 'KAIST'), ('KAIST', 'CWRU'),
         ('CWRU', 'PAD'), ('CWRU', 'UORED'), ('KAIST', 'PAD'), ('KAIST', 'UORED'),
         ('PAD', 'CWRU'), ('UORED', 'CWRU'), ('PAD', 'KAIST'), ('UORED', 'KAIST')]
POOLED = {'PAD', 'UORED'}
jobs = []
for s, t in PAIRS:
    tag = f'{s}2{t}'
    jobs.append((tag, 'C0', s, t, None, None))
    for arm in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']:
        if arm == 'M6' and not (s in POOLED and t in POOLED): continue
        jobs.append((tag, arm, s, t, None, None))
    if tag == 'PAD2UORED': jobs.append((tag, 'SHUFFLE', s, t, None, None))
for ty in ('6204', '6205', '6206', '6207', '6208'):
    tag = f'HUST_LOTO_{ty}'
    sf = lambda m, ty=ty: m.get('bearing_type', m['geometry'][-4:]) != ty
    tf = lambda m, ty=ty: m.get('bearing_type', m['geometry'][-4:]) == ty
    for arm in ['C0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M7']:
        jobs.append((tag, arm, 'HUST', 'HUST', sf, tf))
k, n = int(sys.argv[1]), int(sys.argv[2])
cache = load_cache(); t0 = time.time()
for i, (tag, arm, s, t, sf, tf) in enumerate(jobs):
    if i % n != k: continue
    path = f'{OUT}/{tag}_{arm}.json'
    if os.path.exists(path): continue
    t1 = time.time()
    if arm == 'C0':
        r = T.run_c0(cache, s, t, sf, tf); r.pop('window_P', None); r['tag'] = tag; json.dump(r, open(path, 'w'), indent=1)
    elif arm == 'SHUFFLE':
        json.dump(dict(tag=tag, arm='SHUFFLE_M2', bearing_macro_f1=T.label_shuffle_control(cache, s, t)), open(path, 'w'), indent=1)
    else:
        r = T.run(arm, cache, s, t, OUT, sf, tf, tag=tag, controls=(arm == 'M4'))
    print(f'[{k}] {tag} {arm} done in {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)', flush=True)
print(f'[{k}] all done', flush=True)
