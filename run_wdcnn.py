"""Exploratory WDCNN arm (added after the confirmatory analysis): same 17 settings, splits and recipe.
Usage: python3 run_wdcnn.py <worker k> <n workers>."""
import json, os, sys, time, numpy as np
sys.path.insert(0, '.')
from ppt import transfer as T
from ppt.data import load_cache

PAIRS = [('PAD', 'UORED'), ('UORED', 'PAD'), ('CWRU', 'KAIST'), ('KAIST', 'CWRU'), ('CWRU', 'PAD'), ('CWRU', 'UORED'),
         ('KAIST', 'PAD'), ('KAIST', 'UORED'), ('PAD', 'CWRU'), ('UORED', 'CWRU'), ('PAD', 'KAIST'), ('UORED', 'KAIST')]
jobs = [(f'{s}2{t}', s, t, None, None) for s, t in PAIRS]
for ty in ('6204', '6205', '6206', '6207', '6208'):
    jobs.append((f'HUST_LOTO_{ty}', 'HUST', 'HUST', lambda m, ty=ty: m['geometry'][-4:] != ty, lambda m, ty=ty: m['geometry'][-4:] == ty))
k, n = int(sys.argv[1]), int(sys.argv[2])
cache = load_cache(); cache['raw'] = np.load('data/raw_segments.npy'); OUT = 'runs/main'
for i, (tag, s, t, sf, tf) in enumerate(jobs):
    if i % n != k or os.path.exists(f'{OUT}/{tag}_WDCNN.json'): continue
    t1 = time.time(); T.run('WDCNN', cache, s, t, OUT, sf, tf, tag=tag)
    print(f'[{k}] {tag} WDCNN done in {time.time() - t1:.0f}s', flush=True)
print(f'[{k}] all done', flush=True)
