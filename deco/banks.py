"""On-disk float32 banks; bounded batches, no complete bank copied to CUDA."""
import json
import time
import numpy as np
from common import INPUTS, dump, finish, raw_data, scaler, sha, split_ids, valid_data


def assemble(task, out, results, smoke=False):
    start = time.monotonic(); ds = task['dataset']; ids = split_ids(ds)
    if smoke:
        ids = dict(train=np.concatenate([np.load(results/f'r12-{ds}-{task["seed"]}-f{k:02d}-c0'/'held_ids.npy') for k in range(12)]),
                   val=ids['val'][:4],test=ids['test'][:4])
    n = INPUTS[ds]['nodes']; sample_id = np.concatenate(list(ids.values()))
    role = np.concatenate([np.full(len(v), i, np.uint8) for i,v in enumerate(ids.values())])
    eval_ids = np.concatenate([ids['val'], ids['test']]); raw = raw_data(ds)
    mean, std = scaler(raw, np.arange(INPUTS[ds]['train_end']), valid_data(ds))
    for name, a in [('sample_id',sample_id), ('role',role), ('scaler_mean',mean), ('scaler_std',std)]:
        np.save(out/(name+'.npy'), a)
    teacher = np.lib.format.open_memmap(out/'teacher_prediction.npy', mode='w+', dtype=np.float32,
                                     shape=(len(sample_id),12,n,3))
    fold = np.lib.format.open_memmap(out/'fold_prediction.npy', mode='w+', dtype=np.float32,
                                  shape=(12,len(eval_ids),12,n,3))
    seen = np.zeros((len(ids['train']),3),np.int16)
    provenance = []
    for k in range(12):
        for c in range(3):
            key = f'r12-{ds}-{task["seed"]}-f{k:02d}-c{c}'
            d = results/key
            co = json.loads((d/'COMPLETE.json').read_text())
            assert co['smoke'] == smoke and co['task']['fold'] == k and co['task']['seed'] == task['seed']
            hids = np.load(d/'held_ids.npy'); ix = np.searchsorted(ids['train'],hids)
            assert np.array_equal(ids['train'][ix],hids)
            teacher[ix,...,c] = np.load(d/'held_prediction.npy',mmap_mode='r')[...,0]; seen[ix,c] += 1
            for label, offset in [('val',0),('test',len(ids['val']))]:
                assert np.array_equal(np.load(d/f'{label}_ids.npy'),ids[label])
                fold[k,offset:offset+len(ids[label]),...,c] = np.load(d/f'{label}_prediction.npy',mmap_mode='r')[...,0]
            provenance.append(dict(task=key, complete_sha256=sha(d/'COMPLETE.json')))
    assert (seen == 1).all(), 'Every OOF target must be predicted by exactly one held-fold model'
    for i in range(0, len(eval_ids), 64):
        teacher[len(ids['train'])+i:len(ids['train'])+i+64] = np.asarray(fold[:,i:i+64],np.float32).mean(0)
    final = np.lib.format.open_memmap(out/'final_prediction.npy', mode='w+', dtype=np.float32,
                                      shape=(len(eval_ids),12,n,3))
    for c in range(3):
        key = f'r12-{ds}-{task["seed"]}-final-c{c}'; d = results/key
        co = json.loads((d/'COMPLETE.json').read_text())
        assert co['smoke'] == smoke and co['task']['fold'] is None and co['task']['channel'] == c
        for label, offset in [('val',0),('test',len(ids['val']))]:
            assert np.array_equal(np.load(d/f'{label}_ids.npy'),ids[label])
            final[offset:offset+len(ids[label]),...,c] = np.load(d/f'{label}_prediction.npy',mmap_mode='r')[...,0]
        provenance.append(dict(task=key, complete_sha256=sha(d/'COMPLETE.json'), role='final_inference_only'))
    # Same raw-unit accumulation used by the dual-deployment identity predictor.
    for i in range(0, len(eval_ids), 64):
        teacher[len(ids['train'])+i:len(ids['train'])+i+64] = np.asarray(fold[:,i:i+64],np.float64).mean(0)
    teacher.flush(); fold.flush(); final.flush()
    dump(out/'PROVENANCE.json',provenance)
    return finish(out,task,smoke,start,dict(history='recent12',oof_exactly_once=True))


class Bank:
    def __init__(self, path, dataset, history):
        self.path = path; self.dataset = dataset; self.history = history
        for attr, file in [('ids','sample_id'),('role','role'),('teacher','teacher_prediction'),
                           ('folds','fold_prediction'),('mean','scaler_mean'),('std','scaler_std')]:
            setattr(self,attr,np.load(path/(file+'.npy'),mmap_mode='r'))
        self.mean=np.asarray(self.mean,np.float32);self.std=np.asarray(self.std,np.float32)
        self.final = np.load(path/'final_prediction.npy',mmap_mode='r') if (path/'final_prediction.npy').exists() else None
        first = 0
        self.rows = {'train': np.flatnonzero((self.role==0)&(self.ids>=first)),
                     'val':np.flatnonzero(self.role==1), 'test':np.flatnonzero(self.role==2)}
        self.eval_rows = np.flatnonzero(self.role>0); self.n_oof=int((self.role==0).sum())
        assert (self.role[:self.n_oof]==0).all()
        assert np.array_equal(self.eval_rows,np.arange(self.n_oof,len(self.ids)))
        assert np.all(np.diff(self.ids)>0) and (self.std>0).all()
        assert self.teacher.shape==(len(self.ids),12,INPUTS[dataset]['nodes'],3)
        assert self.folds.shape==(12,len(self.eval_rows),12,INPUTS[dataset]['nodes'],3)
        if self.final is not None:
            assert self.final.shape==(len(self.eval_rows),12,INPUTS[dataset]['nodes'],3)

    def check_ids(self):
        ids=split_ids(self.dataset,self.history)
        for label in ids:
            assert np.array_equal(self.ids[self.rows[label]],ids[label]),label
