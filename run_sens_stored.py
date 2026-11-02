"""Sensitivity for Deviation 1: the confirmatory pair with the registered (stored) UORED speed."""
import json, os, sys
sys.path.insert(0, '.')
from ppt import transfer as T
from ppt.data import load_cache
OUT = 'runs/sens_stored'; os.makedirs(OUT, exist_ok=True)
cache = load_cache(('data/dev_cache.npz', 'data/sealed_cache_stored.npz'))
r = T.run_c0(cache, 'PAD', 'UORED'); r.pop('window_P', None); r['tag'] = 'PAD2UORED'; json.dump(r, open(f'{OUT}/PAD2UORED_C0.json', 'w'), indent=1)
for arm in ('M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'):
    if not os.path.exists(f'{OUT}/PAD2UORED_{arm}.json'):
        T.run(arm, cache, 'PAD', 'UORED', OUT, tag='PAD2UORED'); print(arm, 'done', flush=True)
