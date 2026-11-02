"""Gate G1, version 2: epoch count chosen by inner grouped cross-validation on training bearings only
(methodology section 'Training and model selection'). Outer folds and scoring as in run_g1.py."""
import json, sys, time, numpy as np, torch
sys.path.insert(0,'.')
from ppt import signal as S, baseline as B
from ppt.geometry import orders_for
from ppt.isolation import assert_isolated, grouped_folds
from ppt import model as Mo
CHK=[5,10,15,20,25,30,40,50,60]; SEEDS=range(5)
CACHE=sys.argv[1] if len(sys.argv)>1 else 'data/dev_cache.npz'; OUT=sys.argv[2] if len(sys.argv)>2 else 'results_g1_v2.json'
CWRU_FIXED_EPOCHS=30
z=np.load(CACHE); meta=json.loads(str(z['meta'])); X=z['order']

def score(ds, probs):
    units=sorted(probs); yt=np.array([B.CLASSES.index(u[2]) for u in units]); yp=np.array([np.mean(probs[u],0).argmax() for u in units])
    return (float(np.mean([np.mean(yp[yt==c]==c) for c in (1,2) if np.any(yt==c)])) if ds=='CWRU' else B.macro_f1(yt,yp)), units, yp

def curve(R3, y, tr, te, mt, seed):
    """Train on tr, return {epoch: [(unit, prob), ...]} on te at checkpoints."""
    torch.manual_seed(seed); rng=np.random.default_rng(seed); net=Mo.Net(R3.shape[1])
    opt=torch.optim.Adam(net.parameters(),lr=Mo.LR,weight_decay=Mo.WD)
    c=np.bincount(y[tr],minlength=3); w=torch.tensor(np.where(c>0,c.sum()/(3*np.maximum(c,1)),0.),dtype=torch.float32)
    Xt=torch.tensor(R3[tr]); yt=torch.tensor(y[tr]); out={}
    tem=[m for m,t in zip(mt,te) if t]
    for ep in range(1,max(CHK)+1):
        net.train()
        for b in np.array_split(rng.permutation(len(Xt)),max(1,len(Xt)//64)):
            xb=Xt[b]*torch.empty(len(b),1,1).uniform_(*Mo.AMP_JIT)+Mo.NOISE*torch.randn_like(Xt[b])
            loss=torch.nn.functional.cross_entropy(net(xb),yt[b],weight=w); opt.zero_grad(); loss.backward(); opt.step()
        if ep in CHK and te.any():
            P=Mo.predict_proba(net,R3[te]); out[ep]=[((m['bearing_id'],m['state'],m['cls']),p) for m,p in zip(tem,P)]
    return net, out

res={}; t0=time.time()
for ds in ('PAD','CWRU'):
    fo=orders_for(ds); idx=np.array([i for i,m in enumerate(meta) if m['dataset']==ds]); mt=[meta[i] for i in idx]
    y=np.array([B.CLASSES.index(m['cls']) for m in mt]); outer=grouped_folds(mt,3,seed=101)
    for arm,R in {'M2':S.rep_m2(X[idx]),'M3':S.rep_m3(X[idx],fo['BPFO'],fo['BPFI'])}.items():
        R3=Mo._as3d(R); final={}; chosen=[]
        for f in outer:
            te=np.array([m['bearing_id'] in f for m in mt]); tr=~te
            assert_isolated([m for m,t in zip(mt,tr) if t],[m for m,t in zip(mt,te) if t])
            mtr=[m for m,t in zip(mt,tr) if t]; inner=grouped_folds(mtr,3,seed=202)
            inner_scores={e:{} for e in CHK}
            for g in (inner if ds!='CWRU' else []):
                ite=np.zeros(len(mt),bool); ite[np.where(tr)[0][[m['bearing_id'] in g for m in mtr]]]=True
                itr=tr & ~ite
                for s in SEEDS:
                    _,cv=curve(R3,y,itr,ite,mt,s)
                    for e,lst in cv.items():
                        for u,p in lst: inner_scores[e].setdefault(u,[]).append(p)
            E=CWRU_FIXED_EPOCHS if ds=='CWRU' else max(CHK,key=lambda e:(score(ds,inner_scores[e])[0],-e)); chosen.append(E)
            for s in SEEDS:
                net=Mo.train(R3[tr],y[tr],s,E)
                for m,p in zip([m for m,t in zip(mt,te) if t],Mo.predict_proba(net,R3[te])): final.setdefault((m['bearing_id'],m['state'],m['cls']),[]).append(p)
        sc,units,yp=score(ds,final)
        res[f'{ds}_{arm}']={'score':round(sc,3),'pass':sc>=0.70,'epochs_chosen':chosen,'units':[(u[0],u[2],B.CLASSES[p]) for u,p in zip(units,yp)]}
        print(f'{ds} {arm}: {sc:.3f} {"PASS" if sc>=0.70 else "FAIL"} epochs {chosen} ({time.time()-t0:.0f}s)',flush=True)
json.dump(res,open(OUT,'w'),indent=1)
print('done')
