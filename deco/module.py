"""Coordination module: a residual mixer over the three variables at one sensor.

mode='cross' lets each variable read the other two; mode='self' feeds copies of the target variable (the target-only module).
The final layers start at zero, so an untrained module returns the forecast unchanged.
"""
import torch
from torch import nn


class MixerBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm_time = nn.LayerNorm(width)
        self.temporal = nn.ModuleList([nn.Sequential(nn.Linear(width, 2*width), nn.GELU(),
                                                    nn.Linear(2*width, width)) for _ in range(3)])
        self.norm_channel = nn.LayerNorm(width)
        self.mix = nn.ModuleList([nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
                                  for _ in range(3)])

    def forward(self, h, mode):
        z = self.norm_time(h)
        h = h + torch.stack([self.temporal[c](z[:, c]) for c in range(3)], 1)
        z = self.norm_channel(h)
        messages = []
        for c in range(3):
            sources = [c] + [s for s in range(3) if s != c] if mode == 'cross' else [c]*3
            messages.append(self.mix[c](z[:, sources].transpose(1, 2)).squeeze(-1))
        return h + torch.stack(messages, 1)


class UnifiedMixer(nn.Module):
    def __init__(self, mode, nodes=307, width=64, blocks=2):
        super().__init__()
        assert mode in ('self', 'cross')
        self.mode = mode
        # 4 history segments x 12 samples + 12 backbone forecasts per channel.
        self.encoders = nn.ModuleList([nn.Linear(60, width) for _ in range(3)])
        self.node = nn.Embedding(nodes, width)
        self.blocks = nn.ModuleList([MixerBlock(width) for _ in range(blocks)])
        self.heads = nn.ModuleList([nn.Linear(width, 12) for _ in range(3)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, history, backbone, node):
        # history B,48,3; backbone B,12,3, both in frozen backbone coordinates.
        x = torch.cat([history, backbone], dim=1).transpose(1, 2)
        h = torch.stack([self.encoders[c](x[:, c]) for c in range(3)], 1)
        h = h + self.node(node)[:, None]
        for block in self.blocks:
            h = block(h, self.mode)
        delta = torch.stack([self.heads[c](h[:, c]) for c in range(3)], -1)
        return backbone + delta, delta
