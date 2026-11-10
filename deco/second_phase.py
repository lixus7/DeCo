"""Second training phase of the forecasters: ten further epochs from the selected checkpoint with embedding swap, then weight averaging.

Each DecoupleST forecaster (fold, final and early-fusion MISO) continues from the best.pt of its first-phase fit:
- the first phase's optimizer settings with a fresh Adam state (lr 1e-3, weight decay 1e-4, gradient clipping at 5, batch 16), the
  same training partition, inputs and masks (valid nonzero targets) and the same empty-batch rule;
- in training, each sensor's rows of the node-embedding table and of the two tables of the learned graph (E1, E2) are replaced, together,
  by the rows of a randomly drawn sensor with probability 0.1 at every step (RAGC's SSE regularization);
- ten epochs, no checkpoint selection; the weights of epochs 1-10 are averaged with equal weights (in float64, cast back);
- forecasts come from the averaged weights with the plain class (no swap in eval mode).

Random streams: data order np.random.default_rng(seed + 100000 + 1000 (fold + 1)) and swap draws torch.manual_seed(seed + 200000 +
1000 (fold + 1) + variable) for fold forecasters; default_rng(seed + 100000) and manual_seed(seed + 200000 + variable) for the final
and early-fusion forecasters. Swap draws use the device's generator, so CPU and GPU runs draw different swaps.
"""
import json
import time

import numpy as np
import torch
from torch.nn import functional as F

from common import (DEVICE, INPUTS, causal_data, device_name, dump, finish, raw_data, scaler, seed_all, sha, training_partition,
                    valid_data)
from fold_inputs import InputContexts
from models import DecoupleST, DecoupleSTMISO

SWAP_P = 0.1
EPOCHS = 10


def embedding_swap(table, p):
    """Rows of `table` (N x C) replaced by the rows of random sensors with probability p (RAGC's SSE, same random calls)."""
    n = table.shape[0]
    swap = torch.rand(n, device=table.device) < p
    index = torch.randint(0, n, (n,), device=table.device)
    return torch.where(swap[:, None], table[index], table)


def with_embedding_swap(parent, miso):
    """The forecaster class plus embedding swap in training. No new parameter; in eval mode the output is the parent's."""

    class Swapped(parent):
        def __init__(self, *args):
            super().__init__(*args)
            self.swap_p = SWAP_P

        def tables(self, swap=True):
            node, left, right = self.node.weight, self.graph.left, self.graph.right
            if not swap:
                return node, left, right
            return embedding_swap(torch.cat([node, left, right], 1), self.swap_p).split([node.shape[1], left.shape[1], right.shape[1]], 1)

        def train_path(self, first, sid, swap=True):
            """Base.unpack, DecoupleST.deep (or the MISO forward) and LearnedGraph.forward with the swapped tables."""
            node, left, right = self.tables(swap)
            b = first.shape[0]
            if miso:
                x = first.new_zeros((b, self.nodes, 4, 12, 3))
                x[:, :, 0] = first.permute(0, 2, 1, 3)
                mask = first.new_zeros((b, self.nodes, 4))
                mask[..., 0] = 1
                value, direct = self.value(x), self.direct(x.flatten(-3))
            else:
                x = first[..., :4].permute(0, 2, 3, 1).contiguous()
                mask = torch.cat([torch.ones_like(first[:, 0, :, :1]), first[:, 0, :, 5:8]], -1)
                value, direct = self.value(x[..., None]), self.direct(x.flatten(-2))
            origin = sid.to(first.device).long() + 11
            time_ = torch.cat([self.tod(origin % 288), self.dow((origin // 288) % 7)], -1)
            ctx = self.context_proj(torch.cat([node[None].expand(b, -1, -1), time_[:, None].expand(-1, self.nodes, -1)], -1)) + self.mask_proj(mask)
            h = self.temporal(value + ctx[:, :, None, None] + self.period[None, None, :, None] + self.position[None, None, None])
            adj = F.relu(left @ right.T).masked_fill(torch.eye(self.nodes, device=h.device, dtype=torch.bool), -torch.inf).softmax(-1)
            net = self.net(torch.einsum('nm,bmptd->bnptd', adj, h))
            result = self.net_head(net.flatten(-3)) + self.local_head(self.local(h - self.reconstruct(net)).flatten(-3)) + direct
            return result.transpose(1, 2)[..., None]

        def forward(self, first, sid):
            return self.train_path(first, sid) if self.training and self.swap_p > 0 else super().forward(first, sid)

    return Swapped


def seeds(task):
    """(data-order seed, swap seed) of one continuation."""
    s, c, k = task['seed'], task['channel'], task['fold']
    if k is None:
        return s + 100000, s + 200000 + c
    return s + 100000 + 1000 * (k + 1), s + 200000 + 1000 * (k + 1) + c


def run(task, out, results, smoke=False):
    """Continue the first-phase forecaster results/<task['source']>; write S.pt and its forecasts in the forecaster-run format."""
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    seed_all(task['seed'])
    ds, ch, fold, miso = task['dataset'], task['channel'], task['fold'], task['architecture'] == 'decouplest_miso'
    src = results / task['source']
    source = json.loads((src / 'COMPLETE.json').read_text())
    assert source['task']['architecture'] == task['architecture'] and source['task']['channel'] == ch and source['task']['fold'] == fold
    assert source['task']['seed'] == task['seed'] and source['task']['dataset'] == ds and source['smoke'] == bool(smoke)
    best = torch.load(src / 'best.pt', map_location='cpu', weights_only=False)
    n = INPUTS[ds]['nodes']
    raw, valid = raw_data(ds), valid_data(ds)
    part = training_partition(ds, 'recent12', fold)
    mean, std = np.asarray(best['mean'], np.float32), np.asarray(best['std'], np.float32)
    m, s = scaler(raw, part['scaler_rows'], valid)
    assert np.array_equal(m, mean) and np.array_equal(s, std), 'the first-phase scaler is not reproduced'
    contexts = InputContexts(raw, valid, causal_data(ds), part, mean, std, fold)
    plain = DecoupleSTMISO if miso else DecoupleST
    swapped = with_embedding_swap(plain, miso)

    def x(sid, role):
        h = contexts.batch(sid, role)
        if miso:
            return torch.as_tensor(h, device=DEVICE)
        packet = np.zeros((len(sid), 12, n, 9), np.float32)
        packet[..., 0] = h[..., ch]
        return torch.as_tensor(packet, device=DEVICE)

    def target(sid):
        ix = sid[:, None] + 12 + np.arange(12)
        yraw = np.asarray(raw[ix][..., ch], np.float32)
        mask = (yraw != 0) & np.asarray(valid[ix][..., ch])
        return torch.as_tensor((yraw - mean[ch]) / std[ch], device=DEVICE), torch.as_tensor(mask, device=DEVICE)

    @torch.no_grad()
    def val_score(model, selection):
        model.eval()
        err = cnt = 0.0
        for i in range(0, len(selection), 16):
            sid = selection[i:i + 16]
            y, mask = target(sid)
            p = model(x(sid, 'selection'), torch.as_tensor(sid, device=DEVICE))[..., 0]
            err += float(((p - y).abs() * mask).double().sum())
            cnt += float(mask.sum())
        return err / cnt

    @torch.no_grad()
    def forecasts(model, ids):
        model.eval()
        result = np.empty((len(ids), 12, n, 1), np.float32)
        for i in range(0, len(ids), 16):
            sid = ids[i:i + 16]
            p = model(x(sid, 'forecast'), torch.as_tensor(sid, device=DEVICE))[..., 0]
            result[i:i + 16, ..., 0] = (p * np.float32(std[ch]) + np.float32(mean[ch])).cpu().numpy()
        return result

    def build(cls, state):
        model = cls(n).to(DEVICE)
        model.load_state_dict(state)
        return model

    # Checks before training: the selected checkpoint's inner validation score is reproduced (the smoke trainer scores the first 16
    # inner-validation windows), and the swap class equals the plain class in eval mode and without swaps.
    selection = part['val'][:16] if smoke else part['val']
    first = build(plain, best['model'])
    v0 = val_score(first, selection)
    assert abs(v0 - best['val']) <= 1e-5 * best['val'], ('inner validation score of best.pt not reproduced', v0, best['val'])
    probe = np.load(src / 'test_ids.npy')[:2]
    with torch.no_grad():
        xb, sb = x(probe, 'forecast'), torch.as_tensor(probe, device=DEVICE)
        base = first(xb, sb)
        check = build(swapped, best['model'])
        check.eval()
        assert torch.equal(check(xb, sb), base), 'the swap class in eval mode differs from the plain class'
        check.train()
        assert (check.train_path(xb, sb, swap=False) - base).abs().max() <= 1e-5, 'the training path without swaps differs'
    del first, check

    model = build(swapped, best['model'])
    order_seed, swap_seed = seeds(task)
    torch.manual_seed(swap_seed)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    rng = np.random.default_rng(order_seed)
    train_ids = part['train'][:64] if smoke else part['train']
    epochs = 1 if smoke else EPOCHS
    states, log, skipped = [], [], 0
    for ep in range(1, epochs + 1):
        tick = time.monotonic()
        model.train()
        order = rng.permutation(train_ids)
        accum, steps = 0.0, 0
        for i in range(0, len(order), 16):
            sid = order[i:i + 16]
            y, mask = target(sid)
            count = mask.sum()
            if int(count) == 0:
                skipped += 1          # no defined objective: no forward, no optimizer step, no swap draw
                continue
            opt.zero_grad(set_to_none=True)
            p = model(x(sid, 'train'), torch.as_tensor(sid, device=DEVICE))[..., 0]
            loss = ((p - y).abs() * mask).sum() / count
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            opt.step()
            accum += float(loss)
            steps += 1
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        log.append(dict(epoch=ep, train_loss=accum / max(steps, 1), steps=steps, seconds=time.monotonic() - tick))
        print(task['id'], log[-1], flush=True)
    averaged = {k: (sum(st[k].double() for st in states) / len(states)).to(states[0][k].dtype) for k in states[0]}
    final = build(plain, averaged)
    for cohort in ['val', 'test', 'held']:
        path = src / f'{cohort}_ids.npy'
        if not path.exists():
            continue
        ids = np.load(path)
        np.save(out / f'{cohort}_ids.npy', ids)
        np.save(out / f'{cohort}_prediction.npy', forecasts(final, ids))
    inner_val = val_score(final, selection)
    torch.save(dict(model=averaged, mean=best['mean'], std=best['std'], task=task, source_task=best['task'], epochs_averaged=list(range(1, epochs + 1))),
               out / 'S.pt')
    dump(out / 'HISTORY.json', log)
    return finish(out, task, smoke, started, dict(source_complete_sha256=sha(src / 'COMPLETE.json'), source_best_epoch=best.get('epoch'),
                                                  inner_val_score=inner_val, source_inner_val_score=float(best['val']), skipped_empty_batches=skipped,
                                                  epochs=epochs, swap_p=SWAP_P, order_seed=order_seed, swap_seed=swap_seed,
                                                  training_seconds=sum(r['seconds'] for r in log), device=device_name()))
