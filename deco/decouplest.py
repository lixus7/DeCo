"""DecoupleST: a graph-propagated branch plus a local remainder branch, with a direct linear map.

SISO input packets carry one variable in the first of nine channels; the three unused history blocks stay zero.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(inside, hidden, outside):
    return nn.Sequential(nn.Linear(inside, hidden), nn.GELU(), nn.Linear(hidden, outside))


class FF(nn.Module):
    def __init__(self, d):
        super().__init__(); self.norm = nn.LayerNorm(d); self.net = mlp(d, 2*d, d)
    def forward(self, x):
        return x + self.net(self.norm(x))


class LearnedGraph(nn.Module):
    def __init__(self, nodes, no_self=False):
        super().__init__(); self.left = nn.Parameter(torch.randn(nodes, 16)*0.1)
        self.right = nn.Parameter(torch.randn(nodes, 16)*0.1); self.no_self = no_self
    def forward(self):
        logits = F.relu(self.left @ self.right.T)
        if self.no_self:
            logits = logits.masked_fill(torch.eye(len(logits), device=logits.device, dtype=torch.bool), -torch.inf)
        return logits.softmax(-1)


class Base(nn.Module):
    def __init__(self, nodes, d):
        super().__init__(); self.nodes, self.d = nodes, d
        self.node = nn.Embedding(nodes, 32); self.tod = nn.Embedding(288, 16); self.dow = nn.Embedding(7, 16)
        self.context_proj = nn.Linear(64, d); self.mask_proj = nn.Linear(4, d)
        self.direct = nn.Linear(48, 12)

    def unpack(self, x, batch):
        assert x.shape[1:] == (12, self.nodes, 9)
        # Own recent and first three own anchors. Column 4 is the extra TOD profile.
        values = x[..., :4].permute(0, 2, 3, 1).contiguous()  # B,N,4,12
        mask = torch.cat([torch.ones_like(x[:, 0, :, :1]), x[:, 0, :, 5:8]], -1)
        origin = batch['abs_idx'].to(x.device).long() + 11
        node = self.node.weight[None].expand(x.shape[0], -1, -1)
        time = torch.cat([self.tod(origin % 288), self.dow((origin // 288) % 7)], -1)
        context = self.context_proj(torch.cat([node, time[:, None].expand(-1, self.nodes, -1)], -1))
        return values, mask, context + self.mask_proj(mask)

    def forward(self, x, batch=None):
        values, mask, context = self.unpack(x, batch)
        deep = self.deep(values, mask, context)
        return (deep + self.direct(values.flatten(-2))).transpose(1, 2)[..., None]


class AxisMixer(nn.Module):
    def __init__(self, d, size, hidden, axis):
        super().__init__(); self.norm = nn.LayerNorm(d); self.axis = axis; self.mix = mlp(size, hidden, size)
    def forward(self, x):
        return x + self.mix(self.norm(x).movedim(self.axis, -1)).movedim(-1, self.axis)


class DecoupleST(Base):
    def __init__(self, nodes):
        super().__init__(nodes, 64); self.value = nn.Linear(1, 64)
        self.period = nn.Parameter(torch.randn(4,64)*0.02); self.position = nn.Parameter(torch.randn(12,64)*0.02)
        self.temporal = AxisMixer(64,12,24,3); self.graph = LearnedGraph(nodes, no_self=True)
        self.net = FF(64); self.reconstruct = nn.Linear(64,64)
        self.local = nn.Sequential(AxisMixer(64,12,24,3), FF(64), AxisMixer(64,12,24,3), FF(64))
        self.net_head = mlp(3072,128,12); self.local_head = mlp(3072,128,12)
    def deep(self, x, mask, ctx):
        h = self.value(x[..., None]) + ctx[:, :, None, None] + self.period[None, None, :, None] + self.position[None, None, None]
        h = self.temporal(h)
        net = self.net(torch.einsum('nm,bmptd->bnptd', self.graph(), h))
        result = self.net_head(net.flatten(-3))
        z = h - self.reconstruct(net)
        result = result + self.local_head(self.local(z).flatten(-3))
        return result

    def forward(self, packet, sid):
        x, mask, ctx = self.unpack(packet, {'abs_idx': sid})
        pred = self.deep(x, mask, ctx)
        pred = pred + self.direct(x.flatten(-2))
        return pred.transpose(1, 2)[..., None]
