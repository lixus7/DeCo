"""Coordination module: a residual mixer over the three variables at one sensor.

mode='cross' lets each variable read the other two; mode='self' feeds copies of the target variable (the target-only module).
The final layers start at zero, so an untrained module returns the forecast unchanged.

sensor_embedding=False is DeCo's module: the learned sensor embedding is not added to the encoded variables. The table is still created,
so that every other tensor gets the same initialization as with the embedding; it receives no gradient and is excluded from the parameter
counts of the paper.

zero='histories' / 'forecasts' are the input ablations of DeCo-S: the correction of target variable c is computed from a view of the window
in which, for the two other variables only, the history slots (or the forecast steps) are set to 0 in the module's standardized
coordinates, i.e. to their training mean. The target's own inputs are never changed. The three views (one per target) run as one batch of
3B windows through the unchanged blocks, and target c's correction is read from view c. Layers, initialization and parameter count are
those of the cross module.
"""
import torch
from torch import nn

HIST = 48


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


def zero_views(history, backbone, zero):
    """(3B, 48, 3) histories and (3B, 12, 3) forecasts: block c of B windows is the view of target c."""
    assert zero in ('histories', 'forecasts', None)
    hs, bs = [], []
    for c in range(3):
        h, b = history.clone(), backbone.clone()
        others = [s for s in range(3) if s != c]
        if zero == 'histories':
            h[:, :, others] = 0
        elif zero == 'forecasts':
            b[:, :, others] = 0
        hs.append(h)
        bs.append(b)
    return torch.cat(hs, 0), torch.cat(bs, 0)


class UnifiedMixer(nn.Module):
    def __init__(self, mode, nodes=307, width=64, blocks=2, sensor_embedding=True, zero=None):
        super().__init__()
        assert mode in ('self', 'cross')
        assert zero in (None, 'histories', 'forecasts') and (zero is None or (mode == 'cross' and sensor_embedding))
        self.mode, self.sensor_embedding, self.zero = mode, sensor_embedding, zero
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
        if self.zero is not None:
            return self.forward_views(history, backbone, node)
        x = torch.cat([history, backbone], dim=1).transpose(1, 2)
        h = torch.stack([self.encoders[c](x[:, c]) for c in range(3)], 1)
        if self.sensor_embedding:
            h = h + self.node(node)[:, None]
        for block in self.blocks:
            h = block(h, self.mode)
        delta = torch.stack([self.heads[c](h[:, c]) for c in range(3)], -1)
        return backbone + delta, delta

    def forward_views(self, history, backbone, node):
        n = len(history)
        h, b = zero_views(history, backbone, self.zero)
        x = torch.cat([h, b], dim=1).transpose(1, 2)
        z = torch.stack([self.encoders[c](x[:, c]) for c in range(3)], 1)
        z = z + self.node(node.repeat(3))[:, None]
        for block in self.blocks:
            z = block(z, self.mode)
        delta = torch.stack([self.heads[c](z[c * n:(c + 1) * n, c]) for c in range(3)], -1)
        return backbone + delta, delta
