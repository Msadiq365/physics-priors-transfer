"""Registered analysis: H1 to H3 on Paderborn -> UORED (per-bearing Brier, exact sign-flip, Holm), estimates,
exploratory arm-vs-C0 comparisons, sensitivity analyses, and a summary table of every pair."""
import glob, json, os, sys, numpy as np
sys.path.insert(0, '.')
from ppt import stats as S
D = sys.argv[1] if len(sys.argv) > 1 else 'runs/main'
R = {}
for f in glob.glob(f'{D}/*.json'):
    r = json.load(open(f)); R[(r.get('tag'), r['arm'])] = r
def brier(tag, arm): return R[(tag, arm)]['brier']
out = {}
tag = 'PAD2UORED'
H = {'H1': ('M2', 'M3'), 'H2': ('M5', 'M4'), 'H3': ('C0', 'M4')}       # d = ref - comp; positive favours comp
tests = {}
for h, (ref, comp) in H.items():
    d = S.paired(brier(tag, ref), brier(tag, comp))
    tests[h] = dict(ref=ref, comp=comp, mean_d=float(d.mean()), p=S.signflip_p(d), ci_signflip=S.signflip_ci(d),
                    ci_boot=S.bootstrap_ci(d), n=len(d), d=d.round(4).tolist(),
                    mean_brier={ref: float(np.mean(list(brier(tag, ref).values()))), comp: float(np.mean(list(brier(tag, comp).values())))})
adj = S.holm({h: t['p'] for h, t in tests.items()})
for h in tests: tests[h]['p_holm'], tests[h]['reject'] = adj[h]
out['confirmatory'] = tests
# sensitivity for H3
c0 = R[(tag, 'C0')]; d1 = S.paired(c0['brier_onehot'], brier(tag, 'M4'))
cor = lambda r: {b: all(u[2] == u[3] for u in r['units'] if u[0] == b) for b in {u[0] for u in r['units']}}
ca, cb = cor(c0), cor(R[(tag, 'M4')]); keys = sorted(ca)
out['H3_sensitivity'] = dict(c0_onehot=dict(mean_d=float(d1.mean()), p=S.signflip_p(d1)),
                             correctness_sign_test=dict(zip(('p', 'c0_only', 'm4_only'), S.sign_test_correctness([ca[k] for k in keys], [cb[k] for k in keys]))))
# exploratory: every learned arm and their mean vs C0
expl = {}
arms = [a for (t, a) in R if t == tag and a.startswith('M')]
for a in sorted(arms):
    d = S.paired(brier(tag, 'C0'), brier(tag, a)); expl[a] = dict(mean_d=float(d.mean()), p_unadjusted=S.signflip_p(d), ci=S.signflip_ci(d))
pooled = {b: float(np.mean([brier(tag, a)[b] for a in arms])) for b in brier(tag, 'C0')}
d = S.paired(brier(tag, 'C0'), pooled); expl['mean_of_learned'] = dict(mean_d=float(d.mean()), p_unadjusted=S.signflip_p(d), ci=S.signflip_ci(d))
out['exploratory_vs_C0'] = expl
# every pair, every arm: descriptive table
tab = {}
for (t, a), r in sorted(R.items()):
    if a == 'SHUFFLE_M2': tab.setdefault(t, {})[a] = dict(macro_f1=round(r['bearing_macro_f1'], 3)); continue
    tab.setdefault(t, {})[a] = dict(macro_f1=round(r['bearing_macro_f1'], 3), brier=round(float(np.mean(list(r['brier'].values()))), 3),
                                    classes=r['n_classes_predicted'], **({'controls': {k: (round(v, 3) if not isinstance(v, list) else [round(min(v), 3), round(float(np.mean(v)), 3), round(max(v), 3)]) for k, v in r['controls'].items()}} if 'controls' in r else {}))
out['table'] = tab
json.dump(out, open(f'results/analysis_{os.path.basename(D)}.json', 'w'), indent=1)
for h, t in tests.items():
    print(f"{h}: {t['comp']} vs {t['ref']}: mean Brier {t['mean_brier']} | d {t['mean_d']:+.3f} sign-flip CI {np.round(t['ci_signflip'],3)} p {t['p']:.4f} Holm {t['p_holm']:.4f} reject {t['reject']}")
