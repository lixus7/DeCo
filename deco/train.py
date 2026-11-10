"""Shared 200/30 trainer with validity masks and exact optimizer-boundary continuation."""
import os,time,random,signal,datetime
import numpy as np
import torch
from common import *
from models import DecoupleST,DecoupleSTMISO,RAGC
from external_models import Baseline,channels
from fold_inputs import InputContexts

class Paused(Exception):pass
STOP=False
def request_stop(*_):
    global STOP
    STOP=True

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])

def atomic_save(path,value):
    tmp=Path(str(path)+f'.tmp.{os.getpid()}');torch.save(value,tmp);tmp.replace(path)

def build(task):
    ds=task['dataset'];N=INPUTS[ds]['nodes'];arch=task['architecture']
    if arch=='decouplest':return DecoupleST(N)
    if arch=='decouplest_miso':return DecoupleSTMISO(N)
    if arch=='ragc':return RAGC(N,datetime.date.fromisoformat(INPUTS[ds]['start_date']).weekday())
    return Baseline(arch,ds,task['variant'])

def forward(model,task,x,sid,y=None,valid=None):
    arch=task['architecture']
    if arch=='decouplest':
        packet=x.new_zeros((len(x),12,x.shape[2],9));packet[...,0]=x[...,task['channel']]
        return model(packet,sid)
    if arch in ['ragc','decouplest_miso']:return model(x,sid)
    return model(x,sid,y,valid)

def run(task,out,*,mode='formal',resume=False,max_epochs=None,batch_limit=None,pause_after_steps=None,microbatch=16,wall_seconds=None):
    global STOP
    STOP=False;seed_all(task['seed']);started=time.monotonic()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert mode in ['formal','timing','smoke']
    assert 1<=microbatch<=16 and 16%microbatch==0
    ds=task['dataset'];raw=raw_data(ds);valid=valid_data(ds);causal=causal_data(ds)
    part=training_partition(ds,'recent12',task.get('fold'));ids=split_ids(ds)
    mean,std=scaler(raw,part['scaler_rows'],valid)
    fill_started=time.monotonic();contexts=InputContexts(raw,valid,causal,part,mean,std,task.get('fold'));fill_seconds=time.monotonic()-fill_started
    dump(out/'INPUT_CONTEXT.json',dict(contexts.audit,fill_setup_seconds=fill_seconds))
    outc=[task['channel']] if task['architecture'] in ['decouplest','decouplest_miso','ragc'] else channels(task['variant'])[1]
    model=build(task).to(DEVICE)
    def batch(sids,input_role='train'):
        x=torch.as_tensor(contexts.batch(sids,input_role),device=DEVICE)
        indices=sids[:,None]+12+np.arange(12)
        yr=np.asarray(raw[indices][...,outc],np.float32);v=valid[indices][...,outc]
        y=torch.as_tensor((yr-mean[outc])/std[outc],device=DEVICE)
        return x,torch.as_tensor(sids,device=DEVICE),y,torch.as_tensor(v,device=DEVICE),torch.as_tensor(v&(yr!=0),device=DEVICE)
    # Initialize lazy D2 diffusion parameters before Adam construction, as in 04/08.
    model.eval()
    with torch.no_grad():
        b=batch(part['train'][:min(2,microbatch)]);forward(model,task,*b[:4])
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    data_rng=np.random.default_rng(task['seed']);best=float('inf');stale=0;ep=1;cursor=0;order=None;log=[];steps=0;skipped=[]
    accum=np.zeros(len(outc),np.float64);counts=np.zeros(len(outc),np.int64);epoch_seconds=0.
    identity=dict(task=task,config_sha256=sha(CONFIG),mode=mode,microbatch=microbatch,
                  max_epochs=max_epochs,batch_limit=batch_limit)
    if resume:
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert state['identity']==identity
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for value in opt.state.values():
            for k,v in value.items():
                if torch.is_tensor(v):value[k]=v.to(DEVICE)
        ep,cursor,order,log,best,stale,steps=[state[k] for k in ['epoch','cursor','order','history','best','stale','steps']]
        accum,counts,epoch_seconds=state['accum'],state['counts'],state['epoch_seconds']
        data_rng.bit_generator.state=state['data_rng'];restore_rng(state['rng']);skipped=state.get('skipped',[])
        dump(out/'SKIPPED_BATCHES.json',skipped)
    elif (out/'last.pt').exists() or (out/'COMPLETE.json').exists():raise FileExistsError('Use explicit resume; do not overwrite an attempt')
    else:
        np.savez(out/'PARTITION.npz',**part)
        dump(out/'PARTITION.json',{k:dict(n=len(v),first=int(v[0]) if len(v) else None,last=int(v[-1]) if len(v) else None) for k,v in part.items()})
    def checkpoint():
        atomic_save(out/'last.pt',dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),data_rng=data_rng.bit_generator.state,
            epoch=ep,cursor=cursor,order=order,history=log,best=best,stale=stale,steps=steps,accum=accum,counts=counts,epoch_seconds=epoch_seconds,skipped=skipped))
        dump(out/'SKIPPED_BATCHES.json',skipped)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGUSR1,request_stop)
    maximum=200 if max_epochs is None else max_epochs;reset_peak()
    while ep<=maximum and stale<30:
        tick=time.monotonic();model.train()
        if order is None:
            order=data_rng.permutation(part['train']);cursor=0;accum[:]=0;counts[:]=0;epoch_seconds=0.
            if batch_limit:order=order[:16*batch_limit]
        while cursor<len(order):
            sids=order[cursor:cursor+16]
            # Counts belong to the full logical batch, so microbatches do not change loss weighting.
            ix=sids[:,None]+12+np.arange(12)
            yraw=raw[ix][...,outc];m=valid[ix][...,outc]&(yraw!=0);den=m.sum((0,1,2))
            if not (den>0).any():
                # No defined objective: do not forward, backpropagate, or advance Adam/RNG.
                skipped.append(dict(epoch=ep,cursor=int(cursor),batch_index=int(cursor//16),
                    origins=[int(s) for s in sids],target_counts=[int(n) for n in den],steps=steps))
                cursor+=len(sids)
                sync();epoch_seconds+=time.monotonic()-tick;tick=time.monotonic();checkpoint()
                if STOP or (pause_after_steps is not None and steps>=pause_after_steps) or (wall_seconds and time.monotonic()-started>wall_seconds-180):
                    dump(out/'PAUSED.json',dict(epoch=ep,cursor=cursor,steps=steps,reason='empty-target batch checkpoint',at=time.time()));raise Paused()
                continue
            assert (den>0).all(),'Partially empty output channel'
            denominator=torch.as_tensor(den,device=DEVICE)
            opt.zero_grad(set_to_none=True)
            for at in range(0,len(sids),microbatch):
                x,sid,y,v,mask=batch(sids[at:at+microbatch]);pred=forward(model,task,x,sid,y,v)
                err=(pred-y).abs();sums=(err*mask).sum((0,1,2));loss=(sums/denominator).mean()
                assert torch.isfinite(loss),'Nonfinite objective';loss.backward()
                accum+=sums.detach().double().cpu().numpy();counts+=mask.sum((0,1,2)).cpu().numpy()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
            if hasattr(model,'batches_seen'):model.batches_seen.add_(1)
            cursor+=len(sids);steps+=1
            if steps%25==0 or STOP or (pause_after_steps is not None and steps>=pause_after_steps) or (wall_seconds and time.monotonic()-started>wall_seconds-180):
                sync();epoch_seconds+=time.monotonic()-tick;tick=time.monotonic();checkpoint()
                if STOP or (pause_after_steps is not None and steps>=pause_after_steps) or (wall_seconds and time.monotonic()-started>wall_seconds-180):
                    dump(out/'PAUSED.json',dict(epoch=ep,cursor=cursor,steps=steps,reason='bounded checkpoint continuation',at=time.time()));raise Paused()
        sync();train_seconds=epoch_seconds+time.monotonic()-tick;tick=time.monotonic()
        model.eval();err=np.zeros(len(outc),np.float64);cnt=np.zeros(len(outc),np.int64)
        selection=part['val'][:16] if batch_limit else part['val']
        with torch.no_grad():
            for at in range(0,len(selection),microbatch):
                x,sid,y,v,mask=batch(selection[at:at+microbatch],'selection');pred=forward(model,task,x,sid)
                err+=((pred-y).abs()*mask).double().sum((0,1,2)).cpu().numpy();cnt+=mask.sum((0,1,2)).cpu().numpy()
        assert (cnt>0).all();va=float((err/cnt).mean());assert np.isfinite(va)
        if va<best:
            best=va;stale=0;atomic_save(out/'best.pt',dict(model=model.state_dict(),task=task,mean=mean,std=std,epoch=ep,val=va))
        else:stale+=1
        sync();val_seconds=time.monotonic()-tick
        row=dict(epoch=ep,train_normalized_masked_mae=float((accum/counts).mean()),val_normalized_masked_mae=va,
            train_seconds=train_seconds,val_seconds=val_seconds,seconds=train_seconds+val_seconds,steps=steps)
        log.append(row);dump(out/'HISTORY.json',log);print(task['id'],row,flush=True)
        ep+=1;cursor=0;order=None;epoch_seconds=0.;checkpoint()
    state=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['model']);model.eval()
    if mode=='timing':
        receipt=dict(task=task,full_epoch=batch_limit is None,epochs=len(log),train_origins=len(part['train']),val_origins=len(part['val']),
            history=log,device=device_name(),peak_allocated_gib=peak_gib(),
            microbatch=microbatch,effective_batch=16,
            parameters=sum(p.numel() for p in model.parameters()),fill_setup_seconds=fill_seconds,mode=mode,seconds=time.monotonic()-started)
        dump(out/'TIMING.json',receipt);return receipt
    from evaluation import score,write_csv
    metrics_out={};strata=[]
    for cohort,origins in {**{k:ids[k] for k in ['val','test']},**({'held':part['held']} if len(part['held']) else {})}.items():
        sids=origins[:4] if mode=='smoke' else origins
        path=out/f'{cohort}_prediction.npy';pred=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(sids),12,raw.shape[1],len(outc)))
        with torch.no_grad():
            for at in range(0,len(sids),microbatch):
                x,sid,_,_,_=batch(sids[at:at+microbatch],'forecast');value=forward(model,task,x,sid)
                pred[at:at+microbatch]=value.cpu().numpy()*std[outc]+mean[outc]
        pred.flush();np.save(out/f'{cohort}_ids.npy',sids)
        if cohort!='held':
            summary,rows=score(raw,sids,pred,outc,ds,cohort,dict(task=task['id']))
            metrics_out[cohort]=summary;strata+=rows
        del pred
    write_csv(out/'STATE_METRICS.csv',strata)
    return finish(out,task,mode!='formal',started,dict(metrics=metrics_out,best_epoch=state['epoch'],epochs_run=len(log),
        training_seconds=sum(r['seconds'] for r in log),parameters=sum(p.numel() for p in model.parameters()),
        cuda_peak_gib=peak_gib(),device=device_name(),microbatch=microbatch,effective_batch=16,
        missing_teacher_forcing='HimNet invalid labels use model autoregression'))
