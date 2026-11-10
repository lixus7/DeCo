"""Checkpoint selection on validation: one training trajectory, one selected epoch per deployment."""
import math

EPS = 1e-8
RULE = {
    'score': 'all validation origins x nodes x 12 leads; valid labels AND raw truth !=0 per channel; '
             'MAE / TRAIN bank scaler std; pooled points, float64 sums; equal channel mean',
    'strict_absolute_improvement': EPS, 'maximum_epoch': 25, 'patience': 8,
    'stop': 'both deployment counters >=8; same epoch candidates, OOF-only training',
    'fallback': 'selected epoch 0; no candidate beats identity by more than 1e-8',
}


def improves(value, best):
    return value < best - EPS


def replay(history, maximum=25):
    """Recompute both selections and the joint stop from every epoch, including 0."""
    if not history or [r['epoch'] for r in history] != list(range(len(history))):
        raise ValueError('Missing/duplicate/noncontiguous epoch candidates')
    if not 1 <= len(history) <= maximum+1:
        raise ValueError('Epoch budget mismatch')
    best = {}; stale = {m: 0 for m in ['K12', 'FINAL']}
    for ep, row in enumerate(history):
        for m in ['K12', 'FINAL']:
            score = row[m]['score']
            if type(score) not in [float, int] or not math.isfinite(score) or score < 0:
                raise ValueError('Invalid VAL score')
            cs = row[m]['mae_normalized']; ns = row[m]['counts']
            if len(cs) != 3 or len(ns) != 3 or not all(type(n) is int and n > 0 for n in ns):
                raise ValueError('Invalid per-channel VAL accounting')
            if not all(math.isfinite(x) and x >= 0 for x in cs) or abs(sum(cs)/3-score) > 1e-14:
                raise ValueError('Score is not the equal channel mean')
            if ep and ns != history[0][m]['counts']:
                raise ValueError('VAL point set changed')
            if not ep or improves(score, best[m]['score']):
                best[m] = dict(epoch=ep, score=score); stale[m] = 0
            else:
                stale[m] += 1
        if row['K12']['counts'] != row['FINAL']['counts']:
            raise ValueError('Deployment VAL masks differ')
        if ep < len(history)-1 and all(v >= 8 for v in stale.values()):
            raise ValueError('Trajectory continued past joint stopping rule')
    if len(history)-1 != maximum and not all(v >= 8 for v in stale.values()):
        raise ValueError('Training stopped before both deployment patience counters expired')
    for m, record in best.items():
        record.update(identity_score=history[0][m]['score'],
                      gain=history[0][m]['score']-record['score'],
                      fallback=record['epoch'] == 0, stale=stale[m])
    return best
