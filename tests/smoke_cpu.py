#!/usr/bin/env python3
"""End-to-end check on synthetic data: all 39 forecasters, the bank, both modules and evaluation.

Runs on CPU in a few minutes with a few batches per stage. It checks that the code runs, not the paper's numbers.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
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
        (tmp/'datasets.json').write_text(json.dumps(dict(synthetic=dict(nodes=N, length=T, train_end=1200, val_end=1600, start_date='2024-01-01',
                                                                        congestion=dict(speed_le=58., occupancy_ge=.09)))))
        env = dict(os.environ, DECO_DATA=str(tmp/'data'), DECO_PREPARED=str(tmp/'prepared'), DECO_DATASETS=str(tmp/'datasets.json'), DECO_OUTPUT=str(tmp/'results'))
        for command in [['prepare'], ['pipeline', '--smoke']]:
            subprocess.run([sys.executable, str(REPO/'run.py'), *command, '--dataset', 'synthetic', '--seed', '2024'], env=env, check=True, stdout=subprocess.DEVNULL)
        for arm in ['coord', 'target_only']:
            done = json.loads((tmp/f'results/eval-a12-synthetic-bank2024-s2024-{arm}/COMPLETE.json').read_text())
            for deployment in ['K12', 'FINAL', 'TRANSFER']:
                mae = done['deployments'][deployment]['test']['mae']
                assert len(mae) == 3 and all(np.isfinite(mae)), (arm, deployment, mae)
                print(arm, deployment, 'test MAE', [round(x, 4) for x in mae])
        print('PASS')


if __name__ == '__main__':
    main()
