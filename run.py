#!/usr/bin/env python3
"""Run one stage of the DeCo pipeline. Each call does one task and writes to results/<task id>/.

DeCo for one dataset and seed (Algorithm 1 of the paper):
  1. 39 DecoupleST forecaster fits (12 folds and one final fit, per variable) and 3 early-fusion DecoupleST MISO fits;
  2. a second training phase for each of these 42 forecasters (ten epochs with embedding swap, weights averaged);
  3. a bank of the second-phase forecasts (out-of-fold forecasts for training, fold and final forecasts for validation and test);
  4. the coordination module without sensor embedding (arm coord_nonode), trained on the out-of-fold forecasts;
  5. evaluation: the module corrects every fold forecast and the corrected forecasts are averaged (K12 deployment);
  6. the blend: 0.8 x that average + 0.2 x the early-fusion forecasts.
DeCo-S and the target-only module use the first-phase bank (arms coord and target_only, FINAL deployment); the input ablations of DeCo-S
are arms coord_no_hist and coord_no_fc on the same bank. `pipeline` runs every step in sequence and skips completed tasks; on a cluster,
schedule the single commands instead.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent/'deco'))
from common import INPUTS, output_root  # noqa: E402

CHANNELS = {'flow': 0, 'occupancy': 1, 'speed': 2}
ARMS = ['coord', 'coord_nonode', 'target_only', 'coord_no_hist', 'coord_no_fc']


def forecaster_task(ds, seed, channel, fold=None, architecture='decouplest'):
    name = 'final' if fold is None else f'f{fold:02d}'
    prefix = {'decouplest': 'r12', 'decouplest_miso': 'miso', 'ragc': 'ragc'}[architecture]
    return dict(id=f'{prefix}-{ds}-{seed}-{name}-c{channel}', kind='forecaster', architecture=architecture, dataset=ds, seed=seed,
                channel=channel, fold=fold, variant='full', history='recent12')


def baseline_task(ds, seed, architecture, variant):
    return dict(id=f'ex-{ds}-{seed}-{architecture}-{variant}', kind='forecaster', architecture=architecture, dataset=ds, seed=seed,
                fold=None, variant=variant, history='recent12')


def continue_task(ds, seed, channel, fold=None, architecture='decouplest'):
    """Second training phase of one first-phase forecaster (DecoupleST fold or final fit, or the early-fusion MISO fit)."""
    assert architecture in ('decouplest', 'decouplest_miso') and (fold is None or architecture == 'decouplest')
    source = forecaster_task(ds, seed, channel, fold, architecture)['id']
    name = 'final' if fold is None else f'f{fold:02d}'
    prefix = {'decouplest': 'r12c', 'decouplest_miso': 'misoc'}[architecture]
    return dict(id=f'{prefix}-{ds}-{seed}-{name}-c{channel}', kind='second_phase', architecture=architecture, dataset=ds, seed=seed,
                channel=channel, fold=fold, source=source, history='recent12')


def bank_task(ds, seed, phase=1):
    return dict(id=f'bank12{"c" if phase == 2 else ""}-{ds}-{seed}', kind='bank', dataset=ds, seed=seed, history='recent12',
                forecasters='r12c' if phase == 2 else 'r12')


def module_task(ds, seed, arm, phase=1):
    tag = 'a12c' if phase == 2 else 'a12'
    return dict(id=f'{tag}-{ds}-bank{seed}-s{seed}-{arm}', kind='module', dataset=ds, seed=seed, arm=arm, bank=bank_task(ds, seed, phase)['id'],
                history='recent12', dual_deployment=True)


def eval_task(ds, seed, arm, phase=1):
    m = module_task(ds, seed, arm, phase)
    return dict(id=f'eval-{m["id"]}', kind='module_eval', dataset=ds, seed=seed, arm=arm, bank=m['bank'], training_task=m['id'],
                history='recent12', dual_deployment=True)


def blend_task(ds, seed):
    return dict(id=f'deco-{ds}-{seed}', kind='blend', dataset=ds, seed=seed, evaluation=eval_task(ds, seed, 'coord_nonode', 2)['id'],
                early_fusion=[continue_task(ds, seed, c, None, 'decouplest_miso')['id'] for c in range(3)], weight=0.2)


def states_task(ds, seeds, levels):
    return dict(id=f'states-{ds}-{"-".join(levels)}', kind='states', dataset=ds, seeds=list(seeds), levels=list(levels), cohort='test')


def run_forecaster(task, smoke=False, resume=False, microbatch=16):
    import train
    out = output_root()/task['id']
    limits = dict(mode='smoke', max_epochs=1, batch_limit=1) if smoke else dict(mode='formal')
    return train.run(task, out, resume=resume, microbatch=microbatch, **limits)


def run_continue(task, smoke=False):
    from second_phase import run
    return run(task, output_root()/task['id'], output_root(), smoke)


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


def run_blend(task, smoke=False):
    from blend import run
    return run(task, output_root()/task['id'], output_root(), smoke)


def run_states(task, smoke=False):
    from states import run
    return run(task, output_root()/task['id'], output_root(), smoke)


def done(task):
    return (output_root()/task['id']/'COMPLETE.json').exists()


def pipeline(ds, seed, smoke=False, ablation=False):
    """Every step for one dataset and seed: DeCo-S and the target-only module, then DeCo. Completed tasks are skipped."""
    def step(runner, task):
        if not done(task):
            runner(task, smoke)
    for c in range(3):
        for fold in [*range(12), None]:
            step(run_forecaster, forecaster_task(ds, seed, c, fold))
        step(run_forecaster, forecaster_task(ds, seed, c, None, 'decouplest_miso'))
    step(run_bank, bank_task(ds, seed, 1))
    for arm in ['coord', 'target_only'] + (['coord_no_hist', 'coord_no_fc'] if ablation else []):
        step(run_module, module_task(ds, seed, arm, 1))
        step(run_eval, eval_task(ds, seed, arm, 1))
    for c in range(3):
        for fold in [*range(12), None]:
            step(run_continue, continue_task(ds, seed, c, fold))
        step(run_continue, continue_task(ds, seed, c, None, 'decouplest_miso'))
    step(run_bank, bank_task(ds, seed, 2))
    step(run_module, module_task(ds, seed, 'coord_nonode', 2))
    step(run_eval, eval_task(ds, seed, 'coord_nonode', 2))
    step(run_blend, blend_task(ds, seed))
    return json.loads((output_root()/blend_task(ds, seed)['id']/'COMPLETE.json').read_text())


def thresholds(ds):
    """The congestion thresholds of the rule in deco/states.py, next to the stored constants."""
    from common import raw_data, valid_data
    from states import thresholds as rule
    cfg = INPUTS[ds]
    found = rule(raw_data(ds), valid_data(ds), cfg['train_end'], cfg['threshold_first_target'])
    stored = dict(original=cfg['congestion'], **cfg['congestion_levels'])
    for level, value in found.items():
        print(ds, level, 'rule:', value['speed_le'], value['occupancy_ge'], 'stored:', stored[level]['speed_le'], stored[level]['occupancy_ge'])
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['prepare', 'forecaster', 'baseline', 'continue', 'bank', 'module', 'evaluate', 'blend', 'states',
                                       'thresholds', 'pipeline'])
    p.add_argument('--dataset', required=True, choices=sorted(INPUTS))
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--variable', choices=sorted(CHANNELS), help='forecaster, continue: the variable to predict')
    p.add_argument('--fold', type=int, help='forecaster, continue: 0-11; omit for the final forecaster')
    p.add_argument('--architecture', default='decouplest',
                   help='forecaster: decouplest, decouplest_miso or ragc; continue: decouplest or decouplest_miso; baseline: staeformer, himnet or d2stgnn')
    p.add_argument('--variant', help='baseline: siso_flow, miso_flow, joint, ...')
    p.add_argument('--phase', type=int, choices=[1, 2], default=1,
                   help='bank, module, evaluate: forecasters of the first phase (1) or after the second training phase (2)')
    p.add_argument('--arm', default='coord', choices=ARMS,
                   help='module, evaluate: coord (DeCo-S), coord_nonode (DeCo), target_only, coord_no_hist, coord_no_fc')
    p.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026], help='states: seeds to pool')
    p.add_argument('--levels', nargs='+', default=['S2'], choices=['original', 'S1', 'S2'], help='states: congestion levels')
    p.add_argument('--ablation', action='store_true', help='pipeline: also train the two input-zeroing arms of DeCo-S')
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
    elif a.command == 'continue':
        run_continue(continue_task(a.dataset, a.seed, CHANNELS[a.variable], a.fold, a.architecture), a.smoke)
    elif a.command == 'bank':
        run_bank(bank_task(a.dataset, a.seed, a.phase), a.smoke)
    elif a.command == 'module':
        run_module(module_task(a.dataset, a.seed, a.arm, a.phase), a.smoke, a.resume)
    elif a.command == 'evaluate':
        run_eval(eval_task(a.dataset, a.seed, a.arm, a.phase), a.smoke)
    elif a.command == 'blend':
        run_blend(blend_task(a.dataset, a.seed), a.smoke)
    elif a.command == 'states':
        run_states(states_task(a.dataset, a.seeds, a.levels), a.smoke)
    elif a.command == 'thresholds':
        thresholds(a.dataset)
    else:
        pipeline(a.dataset, a.seed, a.smoke, a.ablation)


if __name__ == '__main__':
    main()
