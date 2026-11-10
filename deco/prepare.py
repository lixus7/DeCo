"""Build the input fills, fold partitions, scaler rows and the D2STGNN walk matrix for one dataset."""
import json
import numpy as np
from common import DATA, INPUTS, PREPARED, causal_data, dump, raw_data, scaler, split_ids, valid_data
from fold_inputs import allowed_scaler_rows, label_mask, refill, source_indices, time_partition


def prepare(ds):
    cfg = INPUTS[ds]; raw = raw_data(ds); valid = valid_data(ds); ids = split_ids(ds)
    out = PREPARED/ds; out.mkdir(parents=True, exist_ok=True)
    # The shared causal input must be the forward fill of the valid values, with zeros before the first one.
    np.testing.assert_array_equal(refill(raw, valid, np.zeros(cfg['length'], bool)), causal_data(ds))
    part = time_partition(ids, cfg, None); np.savez(out/'part_final.npz', **part)
    mean, std = scaler(raw, part['scaler_rows'], valid); np.savez(out/'scaler_final.npz', mean=mean, std=std)
    records = [dict(fold=None, **{k: len(v) for k, v in part.items()}, mean=mean.tolist(), std=std.tolist())]
    for k in range(12):
        part = time_partition(ids, cfg, k)
        held = label_mask(part['held'], len(raw)); blocked = held | label_mask(part['val'], len(raw))
        part['scaler_rows'] = allowed_scaler_rows(part, source_indices(valid, blocked), blocked)
        assert part['scaler_rows'].max() < cfg['train_end']
        mean, std = scaler(raw, part['scaler_rows'], valid)
        np.savez(out/f'part_f{k:02d}.npz', **part); np.savez(out/f'scaler_f{k:02d}.npz', mean=mean, std=std)
        records.append(dict(fold=k, **{key: len(v) for key, v in part.items()}, mean=mean.tolist(), std=std.tolist()))
    adj = np.load(DATA/ds/'adj.npy').astype(np.float32)
    np.save(out/'walk.npy', adj/np.maximum(adj.sum(1, keepdims=True), 1))
    dump(out/'PARTITIONS.json', records)
    return records
