#!/usr/bin/env python3
"""Run one stage of the DeCo pipeline. Each call does one task and writes to results/<task id>/.

The base configuration for one dataset and seed is 39 forecaster fits (12 folds and one final fit, per variable), one prediction bank,
and one module. `pipeline` runs them in sequence; on a cluster, schedule the single commands instead.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'deco'))
from common import INPUTS, output_root  # noqa: E402

CHANNELS = {'flow': 0, 'occupancy': 1, 'speed': 2}


def forecaster_task(ds, seed, channel, fold=None, architecture='decouplest'):
    name = 'final' if fold is None else f'f{fold:02d}'
    prefix = {'decouplest': 'r12', 'decouplest_miso': 'miso', 'ragc': 'ragc'}[architecture]
    return dict(id=f'{prefix}-{ds}-{seed}-{name}-c{channel}', kind='forecaster', architecture=architecture, dataset=ds, seed=seed,
                channel=channel, fold=fold, variant='full', history='recent12')


def baseline_task(ds, seed, architecture, variant):
    return dict(id=f'ex-{ds}-{seed}-{architecture}-{variant}', kind='forecaster', architecture=architecture, dataset=ds, seed=seed,
                fold=None, variant=variant, history='recent12')


def bank_task(ds, seed):
    return dict(id=f'bank12-{ds}-{seed}', kind='bank', dataset=ds, seed=seed, history='recent12')


def module_task(ds, seed, arm):
    return dict(id=f'a12-{ds}-bank{seed}-s{seed}-{arm}', kind='module', dataset=ds, seed=seed, arm=arm, bank=f'bank12-{ds}-{seed}',
                history='recent12', dual_deployment=True)


def eval_task(ds, seed, arm):
    return dict(id=f'eval-a12-{ds}-bank{seed}-s{seed}-{arm}', kind='module_eval', dataset=ds, seed=seed, arm=arm, bank=f'bank12-{ds}-{seed}',
                training_task=f'a12-{ds}-bank{seed}-s{seed}-{arm}', history='recent12', dual_deployment=True)


def run_forecaster(task, smoke=False, resume=False, microbatch=16):
    import train
    out = output_root()/task['id']
    limits = dict(mode='smoke', max_epochs=1, batch_limit=1) if smoke else dict(mode='formal')
    return train.run(task, out, resume=resume, microbatch=microbatch, **limits)


def run_bank(task, smoke=False):
    from banks import assemble
    out = output_root()/task['id']; out.mkdir(parents=True)
    return assemble(task, out, output_root(), smoke)


def run_module(task, smoke=False, resume=False):
    from module_train import run
    out = output_root()/task['id']; out.mkdir(parents=True, exist_ok=True)
    return run(task, out, output_root()/task['bank'], smoke, resume)


def run_eval(task, smoke=False):
    from module_train import evaluate
    out = output_root()/task['id']; out.mkdir(parents=True)
    return evaluate(task, out, output_root(), smoke)


def pipeline(ds, seed, smoke=False, arms=('coord', 'target_only')):
    for channel in range(3):
        for fold in [*range(12), None]:
            run_forecaster(forecaster_task(ds, seed, channel, fold), smoke)
    run_bank(bank_task(ds, seed), smoke)
    return {arm: (run_module(module_task(ds, seed, arm), smoke), run_eval(eval_task(ds, seed, arm), smoke))[1] for arm in arms}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['prepare', 'forecaster', 'baseline', 'bank', 'module', 'evaluate', 'pipeline'])
    p.add_argument('--dataset', required=True, choices=sorted(INPUTS))
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--variable', choices=sorted(CHANNELS), help='forecaster: the variable to predict')
    p.add_argument('--fold', type=int, help='forecaster: 0-11; omit for the final forecaster')
    p.add_argument('--architecture', default='decouplest', help='forecaster: decouplest, decouplest_miso or ragc; baseline: staeformer, himnet or d2stgnn')
    p.add_argument('--variant', help='baseline: siso_flow, miso_flow, joint, ...')
    p.add_argument('--arm', default='coord', choices=['coord', 'target_only'], help='module: coordination module or target-only module')
    p.add_argument('--microbatch', type=int, default=16, help='forecaster: split each batch of 16 to fit memory; the loss weighting is unchanged')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='a few batches only, to check that the code runs')
    a = p.parse_args()
    if a.command == 'prepare':
        from prepare import prepare
        prepare(a.dataset)
    elif a.command == 'forecaster':
        run_forecaster(forecaster_task(a.dataset, a.seed, CHANNELS[a.variable], a.fold, a.architecture), a.smoke, a.resume, a.microbatch)
    elif a.command == 'baseline':
        run_forecaster(baseline_task(a.dataset, a.seed, a.architecture, a.variant), a.smoke, a.resume, a.microbatch)
    elif a.command == 'bank':
        run_bank(bank_task(a.dataset, a.seed), a.smoke)
    elif a.command == 'module':
        run_module(module_task(a.dataset, a.seed, a.arm), a.smoke, a.resume)
    elif a.command == 'evaluate':
        run_eval(eval_task(a.dataset, a.seed, a.arm), a.smoke)
    else:
        pipeline(a.dataset, a.seed, a.smoke)


if __name__ == '__main__':
    main()
