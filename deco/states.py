"""Traffic states (Section V-F of the paper): thresholds, day statistics and paired comparisons by state.

A target is congested when speed <= s and occupancy >= o. The state of a target combines the last observed step and the target step:
free flow (neither congested), onset (only the target step), in congestion (both), recovery (only the last observed step). Targets whose
occupancy or speed is invalid at either step have no state.

Levels: the thresholds are quantiles of the training targets, pooled over sensors, rounded to 0.1 mph and 0.0001 occupancy:
    original  speed q0.35, occupancy q0.65   (configs/datasets.json 'congestion'; PeMS04 keeps its earlier constants 63.9 / 0.0598)
    S1        speed q0.20, occupancy q0.80
    S2        speed q0.05, occupancy q0.95   (the paper's level; 'congestion_strict')
For PeMS03/07 the quantiles use the valid targets at timestamps 12..train_end-1; for PeMS04/08 they use timestamps 2028..train_end-1
(the training segment of the earlier input layout), stored as 'threshold_first_target' in configs/datasets.json.

Statistics: a test window belongs to the day of its first forecast step, (origin + 12) // 288. Per seed and day, n and the sum of absolute
errors are accumulated per mask (include_legal_zero, exclude_zero_targets), state, lead and variable. MAE per seed pools all days; the
point estimate is the mean over seeds; the gain is 100 (1 - candidate / reference). The 95% interval resamples days: 10,000 draws from
np.random.default_rng(20260929), shared by methods and seeds; per draw the seed-level MAEs are averaged.
"""
import csv
import time

import numpy as np

from common import CHANNELS, INPUTS, dump, finish, raw_data, sha, valid_data
from evaluation import states

STATES = ['all', 'onset', 'stays_congested', 'recovery', 'stays_normal']
MASKS = ['include_legal_zero', 'exclude_zero_targets']
QUANTILES = {'original': (0.35, 0.65), 'S1': (0.20, 0.80), 'S2': (0.05, 0.95)}
DRAWS, DRAW_SEED = 10000, 20260929


def thresholds(raw, valid, train_end, first=12, levels=QUANTILES):
    """Quantiles of the valid training targets at timestamps first..train_end-1, pooled over sensors, rounded like the stored constants."""
    fit, ok = np.asarray(raw[first:train_end]), np.asarray(valid[first:train_end])
    speed, occupancy = fit[..., 2][ok[..., 2]].astype(np.float64), fit[..., 1][ok[..., 1]].astype(np.float64)
    return {name: dict(speed_le=round(float(np.quantile(speed, qs)), 1), occupancy_ge=round(float(np.quantile(occupancy, qo)), 4),
                       speed_quantile=qs, occupancy_quantile=qo) for name, (qs, qo) in levels.items()}


def level_constants(dataset, level):
    cfg = INPUTS[dataset]
    if level == 'original':
        return {k: cfg['congestion'][k] for k in ['speed_le', 'occupancy_ge']}
    return {k: cfg['congestion_levels'][level][k] for k in ['speed_le', 'occupancy_ge']}


def day_statistics(raw, valid, ids, prediction, dataset, congestion):
    """n and sums of absolute and squared errors, each (days, mask, state, lead, variable)."""
    day = (ids + 12) // 288
    days = np.unique(day)
    shape = (len(days), 2, 5, 12, 3)
    s = {k: np.zeros(shape, np.int64 if k == 'n' else np.float64) for k in ['n', 'sae', 'sse']}
    for d, value in enumerate(days):
        at = np.flatnonzero(day == value)
        for start in range(0, len(at), 64):
            ix = at[start:start + 64]
            sid = ids[ix]
            truth, masks = states(raw, sid, dataset, valid, congestion)
            truth = truth.astype(np.float64)
            ok = valid[sid[:, None] + 12 + np.arange(12)]
            err = np.asarray(prediction[ix], np.float64) - truth
            assert np.isfinite(err).all()
            for st in range(5):
                for m in range(2):
                    use = ok & masks[st, ..., None]
                    if m:
                        use &= truth != 0
                    s['n'][d, m, st] += use.sum((0, 2))
                    s['sae'][d, m, st] += np.where(use, np.abs(err), 0).sum((0, 2))
                    s['sse'][d, m, st] += np.where(use, err * err, 0).sum((0, 2))
    return days, s


def divide(a, n):
    return np.divide(a, n, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(n)), np.nan), where=n > 0)


def resample_counts(n_days, draws=DRAWS, seed=DRAW_SEED):
    sampled = np.random.default_rng(seed).integers(0, n_days, size=(draws, n_days))
    counts = np.zeros((draws, n_days), np.int64)
    np.add.at(counts, (np.arange(draws)[:, None], sampled), 1)
    return counts


def distribution(sae, n, counts):
    """sae, n: (seeds, days)."""
    per_seed = divide(sae.sum(1), n.sum(1))
    per_day_seed = divide(sae, n)
    boot = divide(counts @ sae.T, counts @ n.T).mean(1)
    return dict(point=float(per_seed.mean()), seed=per_seed, day=per_day_seed.mean(0), boot=boot)


def compare(cand, ref):
    point = 100 * (1 - cand['point'] / ref['point']) if ref['point'] > 0 else float('nan')
    seed_gains = 100 * (1 - divide(cand['seed'], ref['seed']))
    ok = np.isfinite(cand['boot']) & np.isfinite(ref['boot']) & (ref['boot'] > 0)
    gains = 100 * (1 - cand['boot'][ok] / ref['boot'][ok])
    low, high = np.quantile(gains, [.025, .975], method='linear') if len(gains) else (float('nan'), float('nan'))
    dv = np.isfinite(cand['day']) & np.isfinite(ref['day'])
    return dict(gain=float(point), lower=float(low), upper=float(high), seeds_won=int((seed_gains > 0).sum()),
                seeds_lost=int((seed_gains < 0).sum()), days_won=int((dv & (cand['day'] < ref['day'])).sum()), days=int(dv.sum()),
                seed_gains=seed_gains)


# Methods of the paper's state table, all with one forecaster per variable (test split): the single DecoupleST forecasters, the target-only
# module and the coordination module with sensor embedding (DeCo-S), both in the FINAL deployment of the first-phase bank.
METHODS = ['single', 'target_only', 'coord']
PAIRS = [('target_only', 'single'), ('coord', 'single'), ('coord', 'target_only')]
LABEL = {'single': 'single forecasters', 'target_only': 'Target-only', 'coord': 'DeCo-S'}


def load_prediction(results, method, ds, seed, cohort, ids):
    if method == 'single':
        parts = []
        for c in range(3):
            d = results / f'r12-{ds}-{seed}-final-c{c}'
            fids = np.load(d / f'{cohort}_ids.npy')
            pos = np.searchsorted(fids, ids)
            assert np.array_equal(fids[pos], ids)
            parts.append(np.asarray(np.load(d / f'{cohort}_prediction.npy', mmap_mode='r')[pos][..., 0]))
        return np.stack(parts, -1)
    d = results / f'eval-a12-{ds}-bank{seed}-s{seed}-{method}'
    assert np.array_equal(np.load(d / f'{cohort}_ids.npy'), ids)
    return np.load(d / f'{cohort}_FINAL_prediction.npy', mmap_mode='r')


def run(task, out, results, smoke=False):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    ds, seeds, levels, cohort = task['dataset'], task['seeds'], task['levels'], task.get('cohort', 'test')
    raw, valid = raw_data(ds), valid_data(ds)
    ids = np.load(results / f'eval-a12-{ds}-bank{seeds[0]}-s{seeds[0]}-coord' / f'{cohort}_ids.npy')
    constants = {lv: level_constants(ds, lv) for lv in levels}
    stats = {}
    for method in METHODS:
        acc = {lv: {'n': [], 'sae': []} for lv in levels}
        for seed in seeds:
            pred = load_prediction(results, method, ds, seed, cohort, ids)
            for lv in levels:
                days, st = day_statistics(raw, valid, ids, pred, ds, constants[lv])
                acc[lv]['n'].append(st['n'])
                acc[lv]['sae'].append(st['sae'])
            del pred
        for lv in levels:
            stats[(lv, method)] = {k: np.stack(v) for k, v in acc[lv].items()}
    counts = resample_counts(len(days))
    rows, per_seed = [], []
    for lv in levels:
        for mi, mask in enumerate(MASKS):
            for c, variable in enumerate(CHANNELS):
                n_all = stats[(lv, 'single')]['n'][:, :, mi, 0, :, c].sum()
                for si in range(1, 5):
                    n_state = stats[(lv, 'single')]['n'][:, :, mi, si, :, c]
                    dist = {}
                    for method in METHODS:
                        v = stats[(lv, method)]
                        n = v['n'][:, :, mi, si, :, c].sum(-1)
                        if (n.sum(1) > 0).all():
                            dist[method] = distribution(v['sae'][:, :, mi, si, :, c].sum(-1), n, counts)
                            for i, seed in enumerate(seeds):
                                per_seed.append(dict(level=lv, dataset=ds, mask=mask, variable=variable, state=STATES[si], method=LABEL[method],
                                                     seed=seed, mae=float(dist[method]['seed'][i]), targets=int(n[i].sum())))
                    for cand, ref in PAIRS:
                        if cand in dist and ref in dist:
                            r = compare(dist[cand], dist[ref])
                            rows.append(dict(level=lv, dataset=ds, mask=mask, variable=variable, state=STATES[si],
                                             comparison=f'{LABEL[cand]} vs {LABEL[ref]}', speed_le=constants[lv]['speed_le'],
                                             occupancy_ge=constants[lv]['occupancy_ge'], targets=int(n_state.sum()),
                                             share=float(n_state.sum() / n_all), gain_pct=r['gain'], interval_low_pct=r['lower'],
                                             interval_high_pct=r['upper'], seeds_won=r['seeds_won'], seeds_lost=r['seeds_lost'],
                                             days_won=r['days_won'], days=r['days'],
                                             seed_gains_pct=' '.join(f'{g:.4f}' for g in r['seed_gains'])))
    fields = {'STATE_COMPARISONS.csv': ['level', 'dataset', 'mask', 'variable', 'state', 'comparison', 'speed_le', 'occupancy_ge', 'targets', 'share',
                                        'gain_pct', 'interval_low_pct', 'interval_high_pct', 'seeds_won', 'seeds_lost', 'days_won', 'days',
                                        'seed_gains_pct'],
              'STATE_MAE_PER_SEED.csv': ['level', 'dataset', 'mask', 'variable', 'state', 'method', 'seed', 'mae', 'targets']}
    for name, data in [('STATE_COMPARISONS.csv', rows), ('STATE_MAE_PER_SEED.csv', per_seed)]:
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields[name])
            w.writeheader()
            w.writerows(data)
    dump(out / 'THRESHOLDS.json', constants)
    sources = {}
    for seed in seeds:
        for arm in ['target_only', 'coord']:
            p = results / f'eval-a12-{ds}-bank{seed}-s{seed}-{arm}' / 'COMPLETE.json'
            sources[f'{arm}-{seed}'] = sha(p)
        sources[f'single-{seed}'] = [sha(results / f'r12-{ds}-{seed}-final-c{c}' / 'COMPLETE.json') for c in range(3)]
    return finish(out, task, smoke, started, dict(levels=constants, days=len(days), draws=DRAWS, draw_seed=DRAW_SEED,
                                                  source_complete_sha256=sources))
