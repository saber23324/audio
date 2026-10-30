"""Actual new Stage1 warm-start adaptation using the verified legacy phase4 loss."""
from __future__ import annotations
import argparse,hashlib,json,random,time,shutil
from collections import defaultdict
import numpy as np
import torch
from utility_gate.common import seed_all
from .common import ROOT,REPO,VISION_ROUTER,read_jsonl,write_json,append_jsonl,sha256,journal,status,train_rows,read_config,now
from .runtime import Runtime
from .features import load_rgb

def module_hash(module):
    digest=hashlib.sha256()
    for name,tensor in module.state_dict().items():
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

@torch.inference_mode()
def evaluate(runtime,router,rows,epoch):
    runtime.model.model.eval();router.eval();out=[];group=defaultdict(list)
    for r in rows:group[r['source_id']].append(r)
    for source,qs in group.items():
        data=load_rgb(qs[0]);visual=data['visual'].to(runtime.device).float()
        for name,route in [('legacy_stage1_rgb64',runtime.vision_router),('retrained_stage1_rgb64',router),('full_vision196',None)]:
            z=runtime.vision_latents(visual,route)[0] if route is not None else runtime.full_vision(visual)
            for row in qs:
                value=runtime.score(row,z)
                record={**row,**value,'strategy':name,'epoch':epoch,'source_role':'stage1_val','first_gold_token_nll':value['gold_token_nll'][0]}
                out.append(record)
        print('stage1 val',epoch,source,flush=True)
    path=ROOT/f'qa_eval/stage1_val_epoch{epoch}.jsonl'
    if path.exists():raise FileExistsError('preserve completed validation epoch')
    from utility_gate.common import write_jsonl
    write_jsonl(path,out)
    result={}
    for name in sorted({r['strategy'] for r in out}):
        a=[r for r in out if r['strategy']==name]
        result[name]={'n_questions':len(a),'n_sources':len(group),'accuracy':float(np.mean([r['correct'] for r in a])),
            'gold_string_nll':float(np.mean([r['gold_nll'] for r in a])),'first_gold_token_nll':float(np.mean([r['first_gold_token_nll'] for r in a]))}
    return result

def run(resume=None):
    if (ROOT/'locks/stage1_retrained_lock.json').exists() or (ROOT/'locks/pretest_lock.json').exists():raise ValueError('Stage1 fitting closed after freeze')
    cfg=read_config()['model_training']['stage1'];seed_all(SEED:=1234)
    rows=train_rows('stage1_train');val=train_rows('stage1_val');r=Runtime();router=r.training_router()
    r.audio_encoder.cache_dir=None
    if hasattr(r.model.model,'gradient_checkpointing_enable'):r.model.model.gradient_checkpointing_enable()
    if any(getattr(m,'p',0)>0 for m in r.model.model.modules() if isinstance(m,torch.nn.Dropout)):raise ValueError('legacy decoder checkpoint mode assumes zero dropout')
    params=[p for p in router.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['eps'])
    directory=ROOT/'models/stage1_vision64_retrained';start_epoch=0;completed=set();step=0
    if resume:
        ckpt=torch.load(resume,map_location='cpu',weights_only=False)
        if ckpt['training_manifest_sha256']!=sha256(ROOT/'manifests/stage1_train.jsonl'):raise ValueError('resume role drift')
        router.load_state_dict(ckpt['audio_bottleneck'],strict=True);optimizer.load_state_dict(ckpt['optimizer']);step=ckpt['global_step'];start_epoch=ckpt['epoch']
        completed=set(ckpt.get('completed_source_ids',[]));torch.set_rng_state(ckpt['rng_torch']);np.random.set_state(ckpt['rng_numpy']);random.setstate(ckpt['rng_python'])
        if ckpt.get('rng_cuda'):torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
    elif (directory/'epoch_1.pt').exists():raise FileExistsError('use explicit checkpoint resume')
    status('M1','running',mode=cfg['mode'],train_questions=len(rows),val_questions=len(val))
    frozen_before=module_hash(r.model)
    initial={n:p.detach().cpu().clone() for n,p in router.state_dict().items()}
    write_json(ROOT/'audit/stage1_actual_configuration.json',{'original_checkpoint_sha256':sha256(VISION_ROUTER),'stage1_active_trainable_parameters':sum(p.numel() for p in params),
        'all_adapter_parameters':sum(p.numel() for p in router.parameters()),'trainable_names':[n for n,p in router.named_parameters() if p.requires_grad],
        'frozen_large_model_parameter_count':sum(p.numel() for p in r.model.parameters()),'frozen_model_state_sha256_before':frozen_before,
        'objective_source_sha256':sha256(REPO/'streamingtom/train_adbt_videomme.py'),'audio_used_in_stage1_training':False,
        'configuration':cfg,'development_manifest_sha256':read_config()['development_manifest_sha256']})
    old_args=torch.load(VISION_ROUTER,map_location='cpu',weights_only=False)['args']
    new_args={**old_args,'output_dir':str(directory),'resume':'','init_adapter':str(VISION_ROUTER),'wandb':False,'dataset_name':'MUSIC-AVQA official train, new source-disjoint local roles','learning_rate':cfg['learning_rate'],'epochs':cfg['epochs'],'seed':SEED}
    group=defaultdict(list)
    for row in rows:group[row['source_id']].append(row)
    history_path=ROOT/'logs/stage1_history.json'
    history=json.loads(history_path.read_text()) if resume and history_path.exists() else []
    started=time.perf_counter()
    def save(path,epoch,source_done):
        if path.exists():raise FileExistsError(path)
        payload={'audio_bottleneck':router.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'global_step':step,'args':new_args,
            'completed_source_ids':list(source_done),'training_manifest_sha256':sha256(ROOT/'manifests/stage1_train.jsonl'),
            'validation_manifest_sha256':sha256(ROOT/'manifests/stage1_val.jsonl'),'warm_start_sha256':sha256(VISION_ROUTER),
            'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),'rng_cuda':torch.cuda.get_rng_state_all(),
            'mode':'warm_start_adaptation','trainable_names':[n for n,p in router.named_parameters() if p.requires_grad]}
        torch.save(payload,path)
    for epoch in range(start_epoch,cfg['epochs']):
        order=list(group);random.Random(SEED+epoch).shuffle(order);losses=[];source_done=set(completed) if epoch==start_epoch else set()
        for i,source in enumerate(order):
            if source in source_done:continue
            data=load_rgb(group[source][0]);visual=data['visual'].to(r.device,dtype=torch.float32).clone().detach()
            for row in group[source]:
                router.train();r.model.model.train();optimizer.zero_grad(set_to_none=True)
                loss,z,p=r.first_token_loss(row,visual,router)
                if not torch.isfinite(loss):raise ValueError('nonfinite Stage1 loss')
                loss.backward()
                active_grads={n:float(v.grad.float().norm()) for n,v in router.named_parameters() if v.grad is not None}
                if set(active_grads)-{'query_generator.query_tokens','visual_key.weight'} or any(v.grad is not None for v in r.model.parameters()):raise AssertionError('gradient escaped allowed bottleneck')
                grad=float(torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'],error_if_nonfinite=True))
                if grad==0:raise AssertionError('Stage1 received zero gradient')
                optimizer.step();step+=1;losses.append(float(loss.detach()))
                append_jsonl(ROOT/'logs/stage1_train_steps.jsonl',{'epoch':epoch+1,'step':step,'source_id':source,'question_id':row['question_id'],
                    'first_answer_token_CE':float(loss.detach()),'grad_norm_before_clip':grad,'active_grad_norms':active_grads,'frozen_backbone_gradient_count':0,
                    'frames':len(visual),'tokens_per_frame':z.shape[1],'learning_rate':optimizer.param_groups[0]['lr']})
            source_done.add(source)
            if (i+1)%32==0:save(directory/f'progress_epoch{epoch+1}_sources{i+1}.pt',epoch,source_done)
            print('stage1 train',epoch+1,i+1,len(order),'step',step,'source',source,flush=True)
        save(directory/f'epoch_{epoch+1}.pt',epoch+1,[])
        metrics=evaluate(r,router,val,epoch+1)
        history.append({'epoch':epoch+1,'train_first_token_CE_mean':float(np.mean(losses)),'validation':metrics,'checkpoint':str(directory/f'epoch_{epoch+1}.pt'),'global_step':step})
        write_json(ROOT/'logs/stage1_history.json',history)
    selected=min(history,key=lambda h:(h['validation']['retrained_stage1_rgb64']['first_gold_token_nll'],h['epoch']))
    target=directory/'selected.pt'
    if target.exists():raise FileExistsError(target)
    shutil.copy2(selected['checkpoint'],target)
    frozen_after=module_hash(r.model)
    if frozen_before!=frozen_after:raise AssertionError('frozen model state changed during Stage1 fitting')
    changed=[n for n,v in router.state_dict().items() if not torch.equal(v.detach().cpu(),initial[n])]
    if not changed or set(changed)-{'query_generator.query_tokens','visual_key.weight'}:raise AssertionError('new training did not update exactly allowed weights')
    write_json(ROOT/'locks/stage1_retrained_lock.json',{'created_utc':now(),'selected_epoch':selected['epoch'],'selected_checkpoint':str(target),'checkpoint_sha256':sha256(target),
        'mode':cfg['mode'],'warm_start_checkpoint_sha256':sha256(VISION_ROUTER),'optimizer_steps':step,'changed_parameter_names':changed,
        'frozen_backbone_state_sha_before':frozen_before,'frozen_backbone_state_sha_after':frozen_after,'frozen_backbone_gradients':0,
        'fit_manifest_sha256':sha256(ROOT/'manifests/stage1_train.jsonl'),'validation_manifest_sha256':sha256(ROOT/'manifests/stage1_val.jsonl'),
        'selection_rule':cfg['selection'],'elapsed_s':time.perf_counter()-started,'complete_history_sha256':sha256(ROOT/'logs/stage1_history.json')})
    status('M1','completed',optimizer_steps=step,selected_epoch=selected['epoch'],mode=cfg['mode']);journal('M1 Stage1 really trained and frozen',optimizer_steps=step,selected_epoch=selected['epoch'],changed=changed)
    print('STAGE1 FROZEN',selected['epoch'],step,changed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume');a=p.parse_args();run(a.resume)
