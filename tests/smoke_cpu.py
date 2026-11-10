#!/usr/bin/env python3
"""End-to-end check on synthetic data: every stage of DeCo, DeCo-S, the target-only module, the input ablations and the state analysis.

Runs on CPU in a few minutes with a few batches per stage: 39 forecasters and 3 early-fusion forecasters, their second training phase,
both banks, five module arms, the blend and the traffic-state comparison. It checks that the code runs and that a few defining
properties hold, not the paper's numbers.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def properties():
    """Defining properties of the new module arms and of the embedding swap, on random weights (CPU)."""
    sys.path.insert(0, str(REPO/'deco'))
    import torch
    from models import DecoupleST, DecoupleSTMISO, coordination_module
    from second_phase import embedding_swap, with_embedding_swap
    torch.manual_seed(0)
    nodes, n = 9, 16
    history = torch.randn(n, 48, 3); history[:, 12:] = 0
    backbone = torch.randn(n, 12, 3)
    node = torch.randint(0, nodes, (n,))
    cross = coordination_module('coord', nodes)
    for head in cross.heads:
        torch.nn.init.normal_(head.weight, std=.1)
    with torch.no_grad():
        for arm in ['coord_nonode', 'target_only', 'coord_no_hist', 'coord_no_fc']:
            fresh = coordination_module(arm, nodes)
            assert sum(p.numel() for p in fresh.parameters()) == sum(p.numel() for p in cross.parameters()), arm
            assert torch.equal(fresh(history, backbone, node)[0], backbone), (arm, 'an untrained module must return the forecast')
        nonode = coordination_module('coord_nonode', nodes); nonode.load_state_dict(cross.state_dict())
        moved = nonode(history, backbone, (node + 1) % nodes)[1]
        assert torch.equal(moved, nonode(history, backbone, node)[1]), 'coord_nonode must not depend on the sensor'
        assert not torch.equal(cross(history, backbone, (node + 1) % nodes)[1], cross(history, backbone, node)[1])
        delta = cross(history, backbone, node)[1]
        for arm, part in [('coord_no_hist', 'histories'), ('coord_no_fc', 'forecasts')]:
            view = coordination_module(arm, nodes); view.load_state_dict(cross.state_dict())
            base = view(history, backbone, node)[1]
            for c in range(3):
                others = [s for s in range(3) if s != c]
                h, b = history.clone(), backbone.clone()
                if part == 'histories':
                    h[:, :12, others] += torch.randn(n, 12, 2)
                else:
                    b[:, :, others] += torch.randn(n, 12, 2)
                assert (view(h, b, node)[1][..., c] - base[..., c]).abs().max() <= 1e-5, (arm, c, 'zeroed inputs must not matter')
            plain = coordination_module('coord', nodes); plain.load_state_dict(cross.state_dict())
            assert (plain(history, backbone, node)[1] - delta).abs().max() == 0
        table = torch.randn(50, 4)
        swapped = embedding_swap(table, 0.1)
        assert ((swapped == table).all(1) | (swapped[:, None] == table[None]).all(-1).any(1)).all(), 'swapped rows come from the table'
        for parent, miso, packet in [(DecoupleST, False, torch.zeros(2, 12, nodes, 9)), (DecoupleSTMISO, True, torch.zeros(2, 12, nodes, 3))]:
            packet.normal_()
            if not miso:
                packet[..., 1:] = 0
            sid = torch.tensor([100, 5000])
            model = with_embedding_swap(parent, miso)(nodes)
            ref = parent(nodes); ref.load_state_dict(model.state_dict())
            model.eval(); ref.eval()
            assert torch.equal(model(packet, sid), ref(packet, sid))
            model.train(); ref.train()
            assert (model.train_path(packet, sid, swap=False) - ref(packet, sid)).abs().max() <= 1e-5
    print('module and swap properties: ok')


def main():
    properties()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        T, N = 2000, 6
        rng = np.random.default_rng(0)
        t = np.arange(T)[:, None]
        day = np.sin(2*np.pi*t/288)
        raw = np.stack([100+40*day+rng.normal(0, 5, (T, N)), .08+.03*day+rng.normal(0, .005, (T, N)), 60-5*day+rng.normal(0, 1, (T, N))], -1).astype(np.float32)
        valid = rng.random(raw.shape) > .01
        raw[~valid] = 0
        source = np.where(valid, np.arange(T)[:, None, None], -1)
        np.maximum.accumulate(source, axis=0, out=source)
        causal = np.where(source >= 0, np.take_along_axis(raw, np.maximum(source, 0), 0), 0).astype(np.float32)
        d = tmp/'data/synthetic'
        d.mkdir(parents=True)
        np.save(d/'labels.npy', raw)
        np.save(d/'valid.npy', valid)
        np.save(d/'causal.npy', causal)
        np.save(d/'adj.npy', np.ones((N, N), np.float32))
        levels = dict(S1=dict(speed_le=57., occupancy_ge=.1), S2=dict(speed_le=56., occupancy_ge=.105))
        (tmp/'datasets.json').write_text(json.dumps(dict(synthetic=dict(nodes=N, length=T, train_end=1200, val_end=1600, start_date='2024-01-01',
                                                                        congestion=dict(speed_le=58., occupancy_ge=.09), congestion_levels=levels,
                                                                        threshold_first_target=12))))
        env = dict(os.environ, DECO_DATA=str(tmp/'data'), DECO_PREPARED=str(tmp/'prepared'), DECO_DATASETS=str(tmp/'datasets.json'), DECO_OUTPUT=str(tmp/'results'))
        for command in [['prepare'], ['pipeline', '--smoke', '--ablation'], ['states', '--smoke', '--seeds', '2024', '--levels', 'original', 'S2'], ['thresholds']]:
            subprocess.run([sys.executable, str(REPO/'run.py'), *command, '--dataset', 'synthetic', '--seed', '2024'], env=env, check=True, stdout=subprocess.DEVNULL)
        results = tmp/'results'
        for arm in ['coord', 'target_only', 'coord_no_hist', 'coord_no_fc']:
            done = json.loads((results/f'eval-a12-synthetic-bank2024-s2024-{arm}/COMPLETE.json').read_text())
            for deployment in ['K12', 'FINAL', 'TRANSFER']:
                mae = done['deployments'][deployment]['test']['mae']
                assert len(mae) == 3 and all(np.isfinite(mae)), (arm, deployment, mae)
            print(arm, 'FINAL test MAE', [round(x, 4) for x in done['deployments']['FINAL']['test']['mae']])
        for c in range(3):
            for name in [*(f'r12c-synthetic-2024-f{k:02d}-c{c}' for k in range(12)), f'r12c-synthetic-2024-final-c{c}', f'misoc-synthetic-2024-final-c{c}']:
                record = json.loads((results/name/'COMPLETE.json').read_text())
                assert record['swap_p'] == 0.1 and (results/name/'S.pt').exists(), name
        deco = json.loads((results/'deco-synthetic-2024/COMPLETE.json').read_text())
        mae = deco['metrics']['test']['mae']
        assert len(mae) == 3 and all(np.isfinite(mae)), mae
        main_ = np.load(results/'eval-a12c-synthetic-bank2024-s2024-coord_nonode/test_K12_prediction.npy')
        ids = np.load(results/'deco-synthetic-2024/test_ids.npy')
        early = []
        for c in range(3):
            fids = np.load(results/f'misoc-synthetic-2024-final-c{c}/test_ids.npy')
            early.append(np.load(results/f'misoc-synthetic-2024-final-c{c}/test_prediction.npy')[np.searchsorted(fids, ids)][..., 0])
        expected = (0.8*main_.astype(np.float64) + 0.2*np.stack(early, -1).astype(np.float64)).astype(np.float32)
        assert np.array_equal(np.load(results/'deco-synthetic-2024/test_prediction.npy'), expected), 'blend'
        print('DeCo test MAE', [round(x, 4) for x in mae])
        rows = (results/'states-synthetic-original-S2/STATE_COMPARISONS.csv').read_text().splitlines()
        assert len(rows) > 1, 'no state comparison'
        print('state comparisons:', len(rows) - 1, 'rows')
        print('PASS')


if __name__ == '__main__':
    main()
