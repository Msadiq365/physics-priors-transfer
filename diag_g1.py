"""Development diagnosis for failed G1 (CWRU/PAD only): C0 under the same folds; CNN score versus epochs."""
import json, sys, numpy as np, torch
sys.path.insert(0,'.')
from ppt import signal as S, baseline as B
from ppt.geometry import orders_for
from ppt.isolation import grouped_folds
from ppt import model as Mo
z=np.load('data/dev_cache.npz'); meta=json.loads(str(z['meta'])); X=z['order']
CHK=[5,10,20,30,50,80]
def unit_score(ds, mt, te_units_probs):
    units=sorted(te_units_probs); yt=np.array([B.CLASSES.index(u[2]) for u in units])
    yp=np.array([np.mean(te_units_probs[u],0).argmax() for u in units])
    return float(np.mean([np.mean(yp[yt==c]==c) for c in (1,2)])) if ds=='CWRU' else B.macro_f1(yt,yp)
for ds in ('PAD','CWRU'):
    fo=orders_for(ds); idx=np.array([i for i,m in enumerate(meta) if m['dataset']==ds]); mt=[meta[i] for i in idx]
    y=np.array([B.CLASSES.index(m['cls']) for m in mt]); folds=grouped_folds(mt,3,seed=101)
    po,pi=B.scores(X[idx],fo['BPFO'],fo['BPFI'])
    # C0: threshold on training bearings, vote per unit
    up={}
    for f in folds:
        te=np.array([m['bearing_id'] in f for m in mt]); tr=~te
        tau=B.fit_threshold(po[tr],pi[tr],y[tr]); pr=B.predict(po[te],pi[te],tau)
        for m,p in zip([m for m,t in zip(mt,te) if t],pr): up.setdefault((m['bearing_id'],m['state'],m['cls']),[]).append(np.eye(3)[p])
    print(f'{ds} C0: {unit_score(ds,mt,up):.3f}', flush=True)
    for arm,R in {'M2':S.rep_m2(X[idx]),'M3':S.rep_m3(X[idx],fo['BPFO'],fo['BPFI'])}.items():
        R3=Mo._as3d(R); curve={e:{} for e in CHK}
        for f in folds:
            te=np.array([m['bearing_id'] in f for m in mt]); tr=~te
            for s in range(3):
                torch.manual_seed(s); rng=np.random.default_rng(s); net=Mo.Net(R3.shape[1])
                opt=torch.optim.Adam(net.parameters(),lr=Mo.LR,weight_decay=Mo.WD)
                c=np.bincount(y[tr],minlength=3); w=torch.tensor(c.sum()/(3*np.maximum(c,1)),dtype=torch.float32)
                Xt=torch.tensor(R3[tr]); yt=torch.tensor(y[tr])
                for ep in range(1,max(CHK)+1):
                    net.train()
                    for b in np.array_split(rng.permutation(len(Xt)),max(1,len(Xt)//64)):
                        xb=Xt[b]*torch.empty(len(b),1,1).uniform_(0.9,1.1)+0.05*torch.randn_like(Xt[b])
                        loss=torch.nn.functional.cross_entropy(net(xb),yt[b],weight=w); opt.zero_grad(); loss.backward(); opt.step()
                    if ep in CHK:
                        P=Mo.predict_proba(net,R3[te])
                        for m,p in zip([m for m,t in zip(mt,te) if t],P): curve[ep].setdefault((m['bearing_id'],m['state'],m['cls']),[]).append(p)
        print(f'{ds} {arm} by epoch: '+'  '.join(f'{e}:{unit_score(ds,mt,curve[e]):.2f}' for e in CHK), flush=True)
