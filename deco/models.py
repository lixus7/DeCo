"""Forecasters and module factory: DecoupleST SISO and MISO, RAGC, and the coordination module."""
import importlib.util
from types import SimpleNamespace
import torch
from torch import nn
from common import REPO
from decouplest import DecoupleST
from module import UnifiedMixer


class DecoupleSTMISO(DecoupleST):
    def __init__(self, nodes):
        # Retain exactly the initialization of every unchanged SISO tensor.
        super().__init__(nodes)
        self.value = nn.Linear(3, 64)
        self.direct = nn.Linear(144, 12)

    def expand_history(self, history, sid):
        assert history.shape[1:] == (12, self.nodes, 3)
        packet = history.new_zeros((len(history), 12, self.nodes, 9))
        _, mask, ctx = self.unpack(packet, {'abs_idx': sid})
        x = history.new_zeros((len(history), self.nodes, 4, 12, 3))
        x[:, :, 0] = history.permute(0, 2, 1, 3)
        return x, mask, ctx

    def forward(self, history, sid):
        x, mask, ctx = self.expand_history(history, sid)
        h = self.value(x) + ctx[:, :, None, None] + self.period[None, None, :, None] + self.position[None, None, None]
        h = self.temporal(h)
        net = self.net(torch.einsum('nm,bmptd->bnptd', self.graph(), h))
        result = self.net_head(net.flatten(-3))
        z = h - self.reconstruct(net)
        result = result + self.local_head(self.local(z).flatten(-3))
        result = result + self.direct(x.flatten(-3))
        return result.transpose(1, 2)[..., None]


class RAGC(nn.Module):
    """Official SD recipe (4 layers, cosine graph, SSE .1); MISO PeMS adaptation."""
    def __init__(self, nodes, weekday_offset=0):
        super().__init__(); self.weekday_offset = weekday_offset
        args = SimpleNamespace(spatial_dim=64, seq_len=12, input_dim=3, output_dim=1,
            embed_dim=32, pred_len=12, num_layer=4, temp_dim_tid=32, temp_dim_diw=32,
            steps_per_day=288, day_of_week_size=7, cos_gconv=True, softmax_gconv=False,
            geo_gconv=False, sse=True, dropout=False, p=.1)
        spec = importlib.util.spec_from_file_location('ragc_official', REPO/'third_party/ragc_upstream/models/RAGC.py')
        ragc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ragc)
        self.core = ragc.RAGC(args, nodes, None)

    def forward(self, x, sid):
        t = sid[:, None] + torch.arange(12, device=x.device)
        cal = torch.stack([(t % 288).float()/288, ((t//288+self.weekday_offset) % 7).float()/7], -1)
        packet = torch.cat([x, cal[:, :, None].expand(-1, -1, x.shape[2], -1)], -1)
        result = self.core(packet)
        return result[0] if self.training else result


# Module arms. coord: DeCo-S's module (with sensor embedding); coord_nonode: DeCo's module (no sensor embedding); target_only: the
# target-only module; coord_no_hist / coord_no_fc: DeCo-S's module with the other variables' histories / forecasts set to zero.
ARMS = {'coord': dict(mode='cross'), 'coord_nonode': dict(mode='cross', sensor_embedding=False), 'target_only': dict(mode='self'),
        'coord_no_hist': dict(mode='cross', zero='histories'), 'coord_no_fc': dict(mode='cross', zero='forecasts')}


def coordination_module(arm, nodes):
    spec = dict(ARMS[arm])
    return UnifiedMixer(spec.pop('mode'), nodes, 64, 2, **spec)


def details(model, arm, history, base, node):
    p = model(history, base, node)[0]
    return dict(own=p, prediction=p)
