"""OOF-only residual fitting; nonlinear per-fold inference before averaging."""
import numpy as np
import torch
from common import DEVICE, INPUTS, histories, raw_data, valid_data, causal_data
from banks import Bank


class Data:
    def __init__(self, task, bank_path):
        self.task=task; self.bank=Bank(bank_path,task['dataset'],task['history'])
        self.raw=raw_data(task['dataset']); self.valid=valid_data(task['dataset']); self.norm=(causal_data(task['dataset'])-self.bank.mean)/self.bank.std
        self.n=INPUTS[task['dataset']]['nodes']

    def points(self, cohort, smoke=False):
        rows=self.bank.rows[cohort]
        if smoke: rows=rows[:2]
        return (rows[:,None]*self.n+np.arange(self.n)).ravel()

    def batch(self, flat, fold=None, device=DEVICE):
        row,node=flat//self.n,flat%self.n; sid=self.bank.ids[row]
        h=self.norm[histories(sid,self.task['history']),node[:,None]]
        if self.task['history']=='recent12':
            # Keep original 48-input encoders and capacity; unavailable lag slots are zero.
            h=np.concatenate([h,np.zeros((len(h),36,3),np.float32)],1)
        if fold == 'FINAL':
            if self.bank.final is None or (row < self.bank.n_oof).any():
                raise ValueError('FINAL is inference-only and must not replace TRAIN OOF')
            base=self.bank.final[row-self.bank.n_oof,:,node]
        else:
            base=(self.bank.teacher[row,:,node] if fold is None else self.bank.folds[fold,row-self.bank.n_oof,:,node])
        base=(np.asarray(base,np.float32)-self.bank.mean)/self.bank.std
        yraw=self.raw[sid[:,None]+12+np.arange(12),node[:,None]]
        y=(yraw-self.bank.mean)/self.bank.std
        return [torch.as_tensor(x,device=device) for x in (h,base,node,y,self.valid[sid[:,None]+12+np.arange(12),node[:,None]] & (yraw!=0))]

