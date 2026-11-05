"""Tests on development data and on a synthetic file shaped like the sealed export (no sealed signal is used)."""
import json, sys, os, numpy as np
sys.path.insert(0, '.')
from ppt import transfer as T, arms as A
from ppt.loaders import sealed_manifest, load_sealed, refine_shaft_hz
SCR = os.environ.get('PPT_SCRATCH', 'scratch'); os.makedirs(SCR, exist_ok=True)
ok = True
# 1. synthetic sealed-format file built from CWRU development signals, to test loaders and speed handling
z = np.load('data/dev_raw.npz'); sig = z['CWRU__105__sig'].astype(np.float32)
man = [dict(dataset='UORED', file='H_3_0', letter='H', bearing=3, state='healthy'),
       dict(dataset='KAIST', file='0Nm_BPFI_03', load_nm=0, condition='BPFI_03'),
       dict(dataset='HUST', file='I502', letter='I', bearing_type='6205', load_code=2, shaft_hz=24.93)]
spd = np.zeros(len(sig), np.float32); spd[:4] = np.array([1797/3.5*0.3, 1797/3.5*0.4, 1797/3.5*0.3, 0])   # spread impulse, sum = 1797/3.5
np.savez(f'{SCR}/fake_sealed.npz', __manifest__=np.frombuffer(json.dumps(man).encode(), np.uint8),
         UORED__H_3_0__acc=sig, UORED__H_3_0__speed=spd, **{f'KAIST__0Nm_BPFI_03__{c}': sig for c in ('xA', 'yA', 'xB', 'yB')}, HUST__I502__acc=sig)
for e in sealed_manifest(f'{SCR}/fake_sealed.npz'):
    x, fs, hz = load_sealed(e)
    print(f"{e['dataset']:5s} {e['bearing_id']:14s} {e['cls']:2s} geometry {e['geometry']:9s} shaft {hz:.2f} Hz")
    if e['dataset'] == 'UORED' and abs(hz - 1797 / 60) > 0.05: ok = False; print('  UORED speed FAILED')
print('KAIST refine on CWRU 1797 rpm signal near 29.95 Hz nominal:', round(refine_shaft_hz(sig, 12000, 29.95), 2))
# 2. development cache: pools and exclusions, M6 normalisation, M4 per-window centres, controls
c = np.load('data/dev_cache.npz'); cache = dict(meta=json.loads(str(c['meta'])), order=c['order'], hz=c['hz'])
pad = T.select(cache, 'PAD'); ids = {cache['meta'][i]['bearing_id'] for i in pad}
print('PAD usable bearings:', len(ids), '| pool excluded:', not ({'PAD_K005', 'PAD_K006'} & ids))
if {'PAD_K005', 'PAD_K006'} & ids: ok = False
R6 = T._inputs('M6', cache, pad[:5], 'PAD'); R4 = T._inputs('M4', cache, pad[:5], 'PAD')
print('M6 input', R6.shape, '| M4 input', R4.shape, 'centres tail', np.round(R4[0, -8:], 2))
r = T.run('M4', cache, 'CWRU', 'PAD', f'{SCR}/test_runs', seeds=range(1), controls=True)
print('M4 CWRU->PAD (1 seed): macro-F1', round(r['bearing_macro_f1'], 3), '| controls', {k: (round(v, 3) if not isinstance(v, list) else round(float(np.mean(v)), 3)) for k, v in r['controls'].items()})
print('true-class prob per bearing (first 3):', dict(list(r['true_class_prob'].items())[:3]))
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
