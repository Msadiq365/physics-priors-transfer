"""Development dry run: CWRU <-> Paderborn, every arm (except M6), 5 seeds. Development data only."""
import json, sys, time
import numpy as np
sys.path.insert(0, '.')
from ppt import transfer as T
z = np.load('data/dev_cache.npz')
cache = dict(meta=json.loads(str(z['meta'])), order=z['order'], hz=z['hz'])
out = []
t0 = time.time()
for src, tgt in [('CWRU', 'PAD'), ('PAD', 'CWRU')]:
    r = T.run_c0(cache, src, tgt); out.append(r)
    print(f"{src}->{tgt} C0: bearing macro-F1 {r['bearing_macro_f1']:.3f}, classes predicted {r['n_classes_predicted']}", flush=True)
    for arm in ['M1', 'M2', 'M3', 'M4', 'M5', 'M7']:
        t1 = time.time()
        r = T.run(arm, cache, src, tgt, 'runs/dry')
        out.append(r)
        print(f"{src}->{tgt} {arm}: bearing macro-F1 {r['bearing_macro_f1']:.3f}, window {r['window_macro_f1']:.3f}, "
              f"classes {r['n_classes_predicted']}, epochs {r['epochs']}, {time.time()-t1:.0f}s", flush=True)
json.dump(out, open('runs/dry/summary.json', 'w'), indent=1)
print(f'done in {time.time()-t0:.0f}s', flush=True)
