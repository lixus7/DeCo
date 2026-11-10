"""Recent12 time purge and fold-local causal inputs; no network-wide source purge."""
import hashlib
import numpy as np

def label_mask(ids,length):
    mask=np.zeros(length,bool)
    if len(ids):mask[(ids[:,None]+12+np.arange(12)).ravel()]=True
    return mask

def time_purge(ids,forbidden):
    return ids[~forbidden[ids[:,None]+np.arange(24)].any(1)]

def time_partition(ids,cfg,fold):
    if fold is None:
        return dict(train=ids['train'],val=ids['val'],held=np.array([],np.int64),scaler_rows=np.arange(cfg['train_end']))
    assert 0<=fold<12
    held=np.array_split(ids['train'],12)[fold]
    held_mask=label_mask(held,cfg['length'])
    allowed=time_purge(ids['train'],held_mask)
    n=max(24,int(.1*len(allowed)));val=allowed[-n:]
    train_mask=held_mask|label_mask(val,cfg['length'])
    train=time_purge(allowed[:-n],train_mask)
    assert len(train)>0 and len(val)>0
    return dict(train=train,val=val,held=held)

def source_indices(valid,blocked):
    assert valid.ndim==3 and valid.shape[-1]==3 and valid.dtype==np.bool_
    assert blocked.shape==(len(valid),) and blocked.dtype==np.bool_
    times=np.arange(len(valid),dtype=np.int32)[:,None,None]
    source=np.where(valid&~blocked[:,None,None],times,-1)
    np.maximum.accumulate(source,axis=0,out=source)
    return source

def refill(raw,valid,blocked,with_sources=False):
    assert raw.shape==valid.shape
    source=source_indices(valid,blocked);values=np.empty(raw.shape,np.float32)
    for at in range(0,len(raw),128):
        s=source[at:at+128]
        take=np.take_along_axis(raw,np.maximum(s,0),axis=0)
        values[at:at+128]=np.where(s>=0,take,0)
        assert np.isfinite(values[at:at+128]).all()
        assert not blocked[np.maximum(s,0)][s>=0].any()
        assert (s<=np.arange(at,min(at+128,len(raw)))[:,None,None]).all()
    return (values,source) if with_sources else values

def allowed_scaler_rows(part,source,blocked):
    used=np.unique((part['train'][:,None]+np.arange(24)).ravel());extra=[]
    for at in range(0,len(part['train']),128):
        s=source[part['train'][at:at+128,None]+np.arange(12)]
        assert not blocked[np.maximum(s,0)][s>=0].any()
        extra.append(np.unique(s[s>=0]))
    rows=np.unique(np.concatenate([used,*extra]))
    assert not blocked[rows].any()
    return rows

class InputContexts:
    """TRAIN/inner-VAL use distinct refills. OOF/official VAL/TEST use global inputs."""
    def __init__(self,raw,valid,global_causal,part,mean,std,fold):
        global_norm=(np.asarray(global_causal,np.float32)-mean)/std
        self.norm={'forecast':global_norm}
        self.audit=dict(fold=fold,forecast='global causal input; only observations before each origin',leading_placeholder=0,extra_input_channels=False)
        if fold is None:
            self.norm.update(train=global_norm,selection=global_norm)
            self.audit.update(train='global',selection='official VAL, global')
        else:
            held=label_mask(part['held'],len(raw));training=held|label_mask(part['val'],len(raw))
            for role,blocked in [('train',training),('selection',held)]:
                filled=refill(raw,valid,blocked)
                self.norm[role]=(filled-mean)/std
                self.audit[role]=dict(blocked_label_times=int(blocked.sum()),blocked_sha256=hashlib.sha256(blocked.tobytes()).hexdigest())
            assert np.array_equal(time_purge(part['train'],training),part['train'])
            assert np.array_equal(time_purge(part['val'],held),part['val'])
    def batch(self,sids,role):
        assert role in ['train','selection','forecast']
        return self.norm[role][sids[:,None]+np.arange(12)]
