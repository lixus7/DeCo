"""Paths, seeding, data access, scalers and result records."""
import hashlib,json,os,random,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('DECO_DATA',REPO/'data'))
PREPARED=Path(os.environ.get('DECO_PREPARED',REPO/'prepared'))
CONFIG=Path(os.environ.get('DECO_DATASETS',REPO/'configs/datasets.json'))
INPUTS=json.loads(CONFIG.read_text())
CHANNELS=['flow','occupancy','speed']

def _cuda():
    import torch
    return torch.cuda.is_available()

DEVICE='cuda' if _cuda() else 'cpu'

def sync():
    import torch
    if _cuda():torch.cuda.synchronize()

def reset_peak():
    import torch
    if _cuda():torch.cuda.reset_peak_memory_stats()

def peak_gib():
    import torch
    return torch.cuda.max_memory_allocated()/1024**3 if _cuda() else 0.

def device_name():
    import torch
    return torch.cuda.get_device_name(0) if _cuda() else 'cpu'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def seed_all(seed):
    import numpy as np,torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if _cuda():torch.cuda.manual_seed_all(seed)

def array(ds,name):
    import numpy as np
    return np.load(DATA/ds/name,mmap_mode='r',allow_pickle=False)

def raw_data(ds):return array(ds,'labels.npy')

def valid_data(ds):return array(ds,'valid.npy')

def causal_data(ds):return array(ds,'causal.npy')

def split_ids(ds,history='recent12'):
    import numpy as np
    assert history=='recent12'
    cfg=INPUTS[ds]
    return {k:np.arange(a,b-23,dtype=np.int64) for k,(a,b) in {
        'train':(0,cfg['train_end']),'val':(cfg['train_end'],cfg['val_end']),'test':(cfg['val_end'],cfg['length'])}.items()}

def histories(ids,history='recent12'):
    import numpy as np
    assert history=='recent12'
    return ids[:,None]+np.arange(12)

def training_partition(ds,history='recent12',fold=None):
    import numpy as np
    assert history=='recent12'
    name='final' if fold is None else f'f{fold:02d}'
    with np.load(PREPARED/ds/f'part_{name}.npz',allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def scaler(raw,rows,valid):
    import numpy as np
    a=np.zeros(3,np.float64);sq=np.zeros(3,np.float64);n=np.zeros(3,np.int64)
    for at in range(0,len(rows),128):
        ix=rows[at:at+128];x=np.asarray(raw[ix],np.float64);m=valid[ix]
        a+=np.where(m,x,0).sum((0,1));sq+=np.where(m,x*x,0).sum((0,1));n+=m.sum((0,1))
    assert (n>0).all()
    mean=a/n;std=np.sqrt(np.maximum(sq/n-mean*mean,0))
    assert np.isfinite(mean).all() and (std>0).all()
    return mean.astype(np.float32),std.astype(np.float32)

def metrics(raw,ids,prediction,channels=(0,1,2),dataset=None,cohort=None):
    from evaluation import score
    assert dataset in INPUTS,'Dataset required to bind validity mask'
    return score(raw,ids,prediction,channels,dataset,cohort)[0]

def output_root():
    p=Path(os.environ.get('DECO_OUTPUT',REPO/'results')).resolve()
    p.mkdir(parents=True,exist_ok=True);return p

def finish(out,task,smoke,started,extra=None):
    from datetime import datetime,timezone
    data=dict(task=task,smoke=bool(smoke),complete=True,finished_at=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic()-started,config_sha256=sha(CONFIG),
        files={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name not in ['COMPLETE.json','RUN.log','FAILED.json','last.pt','RUNNING.lock','RUN_STATE.json','RUN.json','PAUSED.json'] and '.tmp.' not in p.name},
        **(extra or {}))
    dump(out/'COMPLETE.json',data);return data
