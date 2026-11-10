"""DeCo's output: the fold average of the corrected forecasts, blended with the early-fusion forecasters.

    Y_DeCo = (1 - w) * (1/K) sum_k [Y_k + g(X, Y_k)] + w * Y_e,    w = 0.2 (fixed)

The first term is the K12 deployment of the coordination module trained on the second-phase bank (module arm coord_nonode, written by
module_train.evaluate as <cohort>_K12_prediction.npy: corrections added in raw units, float64 fold accumulation, one float32 cast).
Y_e stacks the three early-fusion DecoupleST MISO forecasters after their second training phase (one per variable). The blend is
computed in float64 and cast once to float32.
"""
import time

import numpy as np

from common import CHANNELS, dump, finish, metrics, raw_data, sha

WEIGHT = 0.2


def blend(main, other, w=WEIGHT):
    return ((1 - w) * np.asarray(main, np.float64) + w * np.asarray(other, np.float64)).astype(np.float32)


def run(task, out, results, smoke=False, cohorts=('val', 'test')):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ds, w = task['dataset'], task.get('weight', WEIGHT)
    evaluation, fusion = results / task['evaluation'], [results / name for name in task['early_fusion']]
    assert len(fusion) == len(CHANNELS)
    raw = raw_data(ds)
    scores, sources = {}, {}
    for cohort in cohorts:
        ids = np.load(evaluation / f'{cohort}_ids.npy')
        main = np.load(evaluation / f'{cohort}_K12_prediction.npy', mmap_mode='r')
        assert main.shape[0] == len(ids) and main.shape[-1] == len(CHANNELS)
        rows = []
        for folder in fusion:
            fids = np.load(folder / f'{cohort}_ids.npy')
            pos = np.searchsorted(fids, ids)
            assert (pos < len(fids)).all() and np.array_equal(fids[pos], ids), (folder.name, 'early-fusion forecasts miss some origins')
            rows.append((np.load(folder / f'{cohort}_prediction.npy', mmap_mode='r'), pos))
        pred = np.lib.format.open_memmap(out / f'{cohort}_prediction.npy', mode='w+', dtype=np.float32, shape=main.shape)
        for at in range(0, len(ids), 256):
            other = np.stack([a[pos[at:at + 256]][..., 0] for a, pos in rows], -1)
            pred[at:at + 256] = blend(main[at:at + 256], other, w)
        pred.flush()
        np.save(out / f'{cohort}_ids.npy', ids)
        scores[cohort] = metrics(raw, ids, pred, dataset=ds, cohort=cohort)
        del pred
    sources = {p.name: sha(p / 'COMPLETE.json') for p in [evaluation, *fusion]}
    dump(out / 'SOURCES.json', sources)
    return finish(out, task, smoke, started, dict(metrics=scores, weight=w, source_complete_sha256=sources))
