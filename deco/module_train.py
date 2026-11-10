"""One OOF trajectory, two validation selections, then evaluation of the K12, FINAL and TRANSFER deployments.

A run that pauses (SIGTERM, SIGUSR1 or the wall-time limit) and resumes keeps its updates, checkpoints and predictions, but the recorded
seconds of the interrupted epoch cover only the part after the resume, so COMPLETE.json training_seconds is then too low.
"""
import json
import os
import time
import numpy as np
import torch
from common import CONFIG, DEVICE, device_name, dump, finish, metrics, peak_gib, reset_peak, seed_all, sha
from selection import RULE, improves, replay
from models import coordination_module, details
from module_data import Data

DEPLOYMENTS = ('K12', 'FINAL', 'TRANSFER')


@torch.no_grad()
def predict_raw(model, arm, data, flat, mode, identity=False):
    """Residuals are applied in raw units. Identity returns the teacher exactly.

    Averaging is after the nonlinear module. Float64 fold accumulation followed
    by one float32 cast is also used for the identity K12 cache in banks.py.
    """
    if mode not in ['K12', 'FINAL']:
        raise ValueError(mode)
    row, node = flat//data.n, flat % data.n
    h, _, nodes, _, _ = data.batch(flat)
    total = None
    folds = range(12) if mode == 'K12' else ['FINAL']
    for k in folds:
        source = data.bank.final[row-data.bank.n_oof, :, node] if k == 'FINAL' else data.bank.folds[k, row-data.bank.n_oof, :, node]
        raw = np.asarray(source, np.float32)
        if identity:
            value = raw.astype(np.float64)
        else:
            base = torch.as_tensor((raw-data.bank.mean)/data.bank.std, device=h.device)
            prediction = details(model, arm, h, base, nodes)['prediction']
            correction = (prediction-base).cpu().numpy().astype(np.float64)
            value = raw.astype(np.float64)+correction*data.bank.std.astype(np.float64)
        total = value if total is None else total+value
    return (total/(12 if mode == 'K12' else 1)).astype(np.float32)


def pooled_val(prediction, truth, std, valid=None):
    """Unrounded scientific score; no per-day/node averaging and no TEST input."""
    mask = truth != 0
    if valid is not None: mask &= valid
    counts = mask.sum((0, 1), dtype=np.int64)
    sums = ((np.abs(prediction.astype(np.float64)-truth.astype(np.float64))/std.astype(np.float64))*mask).sum((0, 1), dtype=np.float64)
    return sums, counts


@torch.no_grad()
def select(model, arm, data, points, identity=False):
    model.eval(); result = {}
    for mode in ['K12', 'FINAL']:
        sums = np.zeros(3, np.float64); counts = np.zeros(3, np.int64)
        for i in range(0, len(points), 8192):
            flat = points[i:i+8192]; row, node = flat//data.n, flat % data.n
            sid = data.bank.ids[row]
            truth = data.labels[sid[:, None]+12+np.arange(12), node[:, None]]
            pred = predict_raw(model, arm, data, flat, mode, identity)
            a, n = pooled_val(pred, truth, data.bank.std, data.valid[sid[:, None]+12+np.arange(12), node[:, None]]); sums += a; counts += n
        if not (counts > 0).all():
            raise ValueError('Empty masked VAL channel')
        mae = sums/counts
        result[mode] = dict(score=float(mae.mean()), mae_normalized=mae.tolist(), counts=counts.tolist())
    return result


def load_model(folder, epoch, arm, nodes):
    state = torch.load(folder/'epochs'/f'epoch{epoch:03d}.pt', map_location=DEVICE, weights_only=False)
    model = coordination_module(arm, nodes).to(DEVICE); model.load_state_dict(state['model']); model.eval()
    return model


def save_predictions(task, out, training, data, selection, cohorts, smoke=False):
    result = {}
    for label in DEPLOYMENTS:
        selected = selection['selected']['FINAL' if label == 'FINAL' else 'K12']
        mode = 'K12' if label == 'K12' else 'FINAL'
        ep = selected['epoch']; model = load_model(training, ep, task['arm'], data.n)
        result[label] = {}
        for cohort in cohorts:
            points = data.points(cohort, smoke); rows = np.unique(points//data.n); ids = data.bank.ids[rows]
            path = out/f'{cohort}_{label}_prediction.npy'
            pred = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(rows), 12, data.n, 3))
            for i in range(0, len(points), 8192):
                flat = points[i:i+8192]; row, node = flat//data.n, flat % data.n
                pred[np.searchsorted(rows, row), :, node] = predict_raw(model, task['arm'], data, flat, mode, ep == 0)
            pred.flush()
            idpath = out/f'{cohort}_ids.npy'
            if not idpath.exists():
                np.save(idpath, ids)
            else:
                np.testing.assert_array_equal(np.load(idpath), ids)
            result[label][cohort] = metrics(data.raw, ids, pred, dataset=task['dataset'], cohort=cohort)
            if label == 'K12':
                alias=out/f'{cohort}_prediction.npy'
                if alias.exists():
                    if not alias.samefile(path):alias.unlink();os.link(path,alias)
                else:os.link(path,alias)
            del pred
        del model
    return result



def run(task, out, bank_path, smoke=False, resume=False, pause_after_steps=None, wall_seconds=None):
    import signal
    import train as trainer
    from train import atomic_save,rng_state,restore_rng,Paused
    trainer.STOP=False;signal.signal(signal.SIGTERM,trainer.request_stop);signal.signal(signal.SIGUSR1,trainer.request_stop)
    start=time.monotonic();seed_all(task['seed']);data=Data(task,bank_path);bank=data.bank
    if not smoke:bank.check_ids()
    assert task['history']=='recent12' and bank.final is not None
    model=coordination_module(task['arm'],data.n).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
    fit=data.points('train',smoke);val=data.points('val',smoke)
    identity=dict(task=task,smoke=smoke,config_sha256=sha(CONFIG),bank_complete_sha256=sha(bank_path/'COMPLETE.json'))
    history=[];best={};stale={'K12':0,'FINAL':0};ep=0;cursor=0;order=None;steps=0;accum=0.;batches=0
    if resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert saved['identity']==identity
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        ep,cursor,order,steps,history,best,stale,accum,batches=[saved[k] for k in ['epoch','cursor','order','steps','history','best','stale','accum','batches']]
        restore_rng(saved['rng'])
    else:
        assert not (out/'last.pt').exists() and not (out/'COMPLETE.json').exists()
        (out/'epochs').mkdir()
        h,base,node,_,_=data.batch(val[:16],'FINAL')
        torch.testing.assert_close(details(model,task['arm'],h,base,node)['prediction'],base,rtol=0,atol=0)
    def checkpoint():
        atomic_save(out/'last.pt',dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),
            epoch=ep,cursor=cursor,order=order,steps=steps,history=history,best=best,stale=stale,accum=accum,batches=batches))
    maximum=2 if smoke else 25;reset_peak()
    while ep<=maximum and not (ep and all(n>=8 for n in stale.values())):
        tick=time.monotonic()
        if ep:
            model.train()
            if order is None:
                gen=torch.Generator(device=DEVICE).manual_seed(901000+task['seed']*100+ep)
                order=fit[torch.randperm(len(fit),generator=gen,device=DEVICE).cpu().numpy()]
                if smoke:order=order[:1024]
                cursor=0;accum=0.;batches=0
            while cursor<len(order):
                points=order[cursor:cursor+512];h,base,node,y,mask=data.batch(points)
                opt.zero_grad(set_to_none=True);err=(details(model,task['arm'],h,base,node)['prediction']-y).abs()
                count=mask.sum((0,1));assert torch.all(count>0)
                loss=((err*mask).sum((0,1))/count).mean();assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
                cursor+=len(points);steps+=1;batches+=1;accum+=loss.item()
                pause=trainer.STOP or (pause_after_steps is not None and steps>=pause_after_steps) or (wall_seconds and time.monotonic()-start>wall_seconds-180)
                if steps%200==0 or pause:checkpoint()
                if pause:
                    dump(out/'PAUSED.json',dict(epoch=ep,cursor=cursor,steps=steps));raise Paused()
        # If interrupted during long dual-VAL scoring, resume from these weights without another optimizer update.
        checkpoint()
        values=select(model,task['arm'],data,val,identity=ep==0)
        state=dict(model=model.state_dict(),task=task,epoch=ep,mean=bank.mean,std=bank.std)
        atomic_save(out/'epochs'/f'epoch{ep:03d}.pt',state)
        for mode in ['K12','FINAL']:
            score=values[mode]['score']
            if not ep or improves(score,best[mode]['score']):best[mode]=dict(epoch=ep,score=score);stale[mode]=0
            else:stale[mode]+=1
        row=dict(epoch=ep,**values,loss=accum/batches if batches else None,seconds=time.monotonic()-tick)
        history.append(row);dump(out/'HISTORY.json',history);print(task['id'],row,flush=True)
        ep+=1;order=None;cursor=0;checkpoint()
    selected=replay(history,maximum)
    selection=dict(rule=RULE,dataset=task['dataset'],seed=task['seed'],arm=task['arm'],history=history,selected=selected,
        smoke=smoke,test_evaluated=False,bank_scaler_std=bank.std.tolist(),bank_sample_ids_sha256=sha(bank_path/'sample_id.npy'),
        bank_complete_sha256=sha(bank_path/'COMPLETE.json'),val_origins=len(np.unique(val//data.n)),nodes=data.n,horizon=12)
    dump(out/'SELECTION.json',selection)
    for mode,filename in [('K12','best.pt'),('FINAL','best_FINAL.pt')]:
        state=torch.load(out/'epochs'/f'epoch{selected[mode]["epoch"]:03d}.pt',map_location='cpu',weights_only=False)
        state.update(best_epoch=selected[mode]['epoch'],own_best_epoch=selected[mode]['epoch'],own_best=None,selection_mode=mode)
        atomic_save(out/filename,state)
    saved=save_predictions(task,out,out,data,selection,['val'],smoke)
    return finish(out,task,smoke,start,dict(metrics=saved['K12'],deployments=saved,selected=selected,test_evaluated=False,
        parameters=sum(p.numel() for p in model.parameters()),training_seconds=sum(x['seconds'] for x in history),
        cuda_peak_gib=peak_gib(),device=device_name(),checkpoint_reload_checked=True))

def evaluate(task, out, results, smoke=False):
    tick = time.monotonic()
    data = Data(task, results/task['bank']); folder = results/task['training_task']
    selection = json.loads((folder/'SELECTION.json').read_text())
    if selection['smoke'] != smoke or selection['rule'] != RULE:
        raise ValueError('Wrong selection receipt')
    if not smoke and replay(selection['history']) != selection['selected']:
        raise ValueError('Selection mismatch')
    saved = save_predictions(task, out, folder, data, selection, ['val', 'test'], smoke)
    return finish(out, task, smoke, tick, dict(metrics=saved['K12'], deployments=saved,
        selected=selection['selected'], selection_sha256=sha(folder/'SELECTION.json'),
        test_evaluated=True, selection_recomputed_on_test=False))
