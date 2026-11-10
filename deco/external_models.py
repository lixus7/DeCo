"""Wrappers for STAEformer, HimNet and D2STGNN; see third_party/README.md for the upstream code."""
import datetime,sys
import numpy as np
import torch
from torch import nn
from common import REPO,PREPARED,INPUTS

def channels(variant):
    names={'flow':0,'occupancy':1,'speed':2}
    out=[0,1,2] if variant=='joint' else [names[variant.split('_')[1]]]
    return (out if variant.startswith('siso_') else [0,1,2]),out

class Baseline(nn.Module):
    def __init__(self,name,ds,variant):
        super().__init__();self.name=name;self.ds=ds;self.inc,self.outc=channels(variant)
        sys.path.insert(0,str(REPO/'third_party'))
        self.offset=datetime.date.fromisoformat(INPUTS[ds]['start_date']).weekday()
        N=INPUTS[ds]['nodes'];ci=len(self.inc);co=len(self.outc)
        self.register_buffer('batches_seen',torch.tensor(0,dtype=torch.long))
        if name=='staeformer':
            from staeformer_core import STAEformer
            self.core=STAEformer(num_nodes=N,in_steps=12,out_steps=12,input_dim=ci+2,output_dim=co,
                input_embedding_dim=24,tod_embedding_dim=24,dow_embedding_dim=24,spatial_embedding_dim=0,
                adaptive_embedding_dim=80,feed_forward_dim=256,num_heads=4,num_layers=3,dropout=.1)
        elif name=='himnet':
            from himnet_core import HimNet
            self.core=HimNet(N,input_dim=ci+2,output_dim=co,out_steps=12,hidden_dim=64,use_teacher_forcing=True)
        elif name=='d2stgnn':
            from d2stgnn import D2STGNN
            walk=torch.as_tensor(np.load(PREPARED/ds/'walk.npy'))
            self.core=D2STGNN(num_feat=ci,output_features=co,num_hidden=32,dropout=.1,seq_length=12,k_t=3,k_s=2,
                gap=3,num_nodes=N,adjs=[walk,walk],num_layers=5,num_modalities=2,node_hidden=12,time_emb_dim=12,time_in_day_size=288,day_in_week_size=7)
        else:raise ValueError(name)

    def calendar(self,sid,n,start=0,normalized=False):
        t=sid[:,None]+torch.arange(start,start+12,device=sid.device)
        cal=torch.stack([(t%288)/288.,((t//288+self.offset)%7)/(7. if normalized else 1.)],-1)
        return cal[:,:,None].expand(-1,-1,n,-1)

    def forward(self,history,sid,target=None,valid=None):
        x=history[...,self.inc];cal=self.calendar(sid,x.shape[2],normalized=self.name=='d2stgnn')
        packet=torch.cat([x,cal],-1)
        if self.name=='d2stgnn':return self.core(packet,None,int(self.batches_seen),0,self.training)
        if self.name=='staeformer':return self.core(packet)
        if self.training:
            assert target is not None and valid is not None
            target=torch.where(valid,target,torch.full_like(target,float('nan')))
        return self.core(packet,self.calendar(sid,x.shape[2],12),labels=target,batches_seen=int(self.batches_seen))
