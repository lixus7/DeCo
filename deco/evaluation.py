"""Validity-aware streaming metrics; unknown traffic states are not normal traffic."""
import csv
import numpy as np
from common import CHANNELS,INPUTS,valid_data
STATES=['all','onset','stays_congested','recovery','stays_normal']
POLICIES=['include_legal_zero','exclude_zero_targets']
STATUS={'val':'validation','test':'test'}

def safe(x):
    a=np.asarray(x)
    if a.ndim>1:return [safe(v) for v in a]
    return [float(v) if np.isfinite(v) else None for v in a]

def divide(a,n):return np.divide(a,n,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(n)),np.nan,dtype=np.float64),where=n>0)

def states(raw,ids,dataset,valid=None):
    if valid is None:valid=valid_data(dataset)
    cfg=INPUTS[dataset]['congestion'];ix=ids[:,None]+12+np.arange(12)
    truth=raw[ix];cur=raw[ids+11]
    current=((cur[...,2]<=cfg['speed_le'])&(cur[...,1]>=cfg['occupancy_ge']))[:,None]
    future=(truth[...,2]<=cfg['speed_le'])&(truth[...,1]>=cfg['occupancy_ge'])
    known=(valid[ids+11][...,[1,2]].all(-1))[:,None]&valid[ix][...,[1,2]].all(-1)
    return truth,np.stack([np.ones_like(future),known&~current&future,known&current&future,known&current&~future,known&~current&~future])

def score(raw,ids,prediction,channels=(0,1,2),dataset=None,cohort=None,metadata=None,valid=None):
    assert dataset in INPUTS
    if valid is None:valid=valid_data(dataset)
    channels=list(channels);ids=np.asarray(ids)
    assert prediction.shape==(len(ids),12,raw.shape[1],len(channels)) and len(ids)>0 and np.all(np.diff(ids)>0)
    shape=(2,5,12,len(channels));n=np.zeros(shape,np.int64);sae=np.zeros(shape);sse=np.zeros(shape);signed=np.zeros(shape)
    for at in range(0,len(ids),64):
        sid=ids[at:at+64];truth,masks=states(raw,sid,dataset,valid)
        truth=truth[...,channels].astype(np.float64);ok=valid[sid[:,None]+12+np.arange(12)][...,channels]
        err=np.asarray(prediction[at:at+64],np.float64)-truth;assert np.isfinite(err).all()
        for st,mask in enumerate(masks):
            for m in range(2):
                use=ok&mask[...,None]
                if m:use&=truth!=0
                n[m,st]+=use.sum((0,2));sae[m,st]+=np.where(use,np.abs(err),0).sum((0,2))
                sse[m,st]+=np.where(use,err*err,0).sum((0,2));signed[m,st]+=np.where(use,err,0).sum((0,2))
    assert (n[:,1:].sum(1)<=n[:,0]).all()
    def metric(m):
        count=n[m,0];a=sae[m,0];sq=sse[m,0]
        return dict(mae=safe(divide(a.sum(0),count.sum(0))),rmse=safe(np.sqrt(divide(sq.sum(0),count.sum(0)))),
            mae_by_horizon=safe(divide(a,count)),rmse_by_horizon=safe(np.sqrt(divide(sq,count))),n_by_horizon=count.tolist(),
            state_coverage=safe(divide(n[m,1:].sum((0,1)),count.sum(0))),
            state_coverage_by_horizon=safe(divide(n[m,1:].sum(0),count)),samples=len(ids),nodes=raw.shape[1],legal_zero_included=m==0)
    result=metric(0);result['nonzero']=metric(1);rows=[]
    for m,policy in enumerate(POLICIES):
        for st,state in enumerate(STATES):
            for lead in [None,*range(12)]:
                for c,ch in enumerate(channels):
                    ix=(m,st,slice(None) if lead is None else lead,c)
                    cnt=int(n[ix].sum());a=float(sae[ix].sum());sq=float(sse[ix].sum());b=float(signed[ix].sum())
                    allcnt=int(n[m,0,:,c].sum()) if lead is None else int(n[m,0,lead,c])
                    known=int(n[m,1:,:,c].sum()) if lead is None else int(n[m,1:,lead,c].sum())
                    rows.append(dict(metadata or {},dataset=dataset,cohort=cohort,evidence_status=STATUS.get(cohort,'internal OOF'),mask=policy,state=state,
                        lead='avg' if lead is None else lead+1,channel=CHANNELS[ch],n=cnt,sae=a,sse=sq,signed_sum=b,
                        mae=a/cnt if cnt else None,rmse=(sq/cnt)**.5 if cnt else None,bias=b/cnt if cnt else None,
                        state_known_n=known,all_valid_target_n=allcnt,state_coverage=known/allcnt if allcnt else None))
    return result,rows

def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
