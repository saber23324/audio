"""Freeze Gate A, actually train only the new R-A adapter, including forced exposure."""
import argparse,copy,json,random,time,hashlib,shutil
from collections import defaultdict
import numpy as np
import torch
from utility_gate.common import seed_all
from utility_gate.adaptive_grounded64.train_vision_router import module_hash
from .common import ROOT,config,training_rows,read_jsonl,write_jsonl,append_jsonl,write_json,sha256,status,journal,now,V1
from .runtime import Runtime
from .features import load
from .fusion import GroundedFusionRouter
from .policy import GateBundle

@torch.inference_mode()
def validate(r,fusion,bundle,rows,epoch):
    r.model.model.eval();fusion.eval();groups=defaultdict(list);records=[]
    for row in rows:groups[row['source_id']].append(row)
    for source,qs in groups.items():
        d=load(qs[0]);v=d['visual'].to(r.device).float();g=d['visual_grounded'].to(r.device).float();z=d['V1'].to(r.device);f0=d['F0'].to(r.device)
        mask,_=bundle.mask(d);selected=torch.tensor(mask,device=r.device);available=(d['audio_valid']&torch.tensor(d['map_valid'])).to(r.device)
        za=fusion(v,g,z,available);zg=fusion(v,g,z,selected&available)
        for name,zz in [('B0_V1',z),('B1_F0',f0),('B2_R_always',za),('B3_GA_R',zg)]:
            for row in qs:records.append({**row,**r.score(row,zz),'strategy':name,'epoch':epoch,'Gate_A_mask':mask,'tokens_per_frame':64,
                'trained_Router_called_for_selected_only':name=='B3_GA_R','Gate_weights_updated':False})
    p=ROOT/f'results/router_val_epoch{epoch}.jsonl'
    if p.exists():raise FileExistsError('preserve completed Router validation')
    write_jsonl(p,records);summary={}
    for name in ('B0_V1','B1_F0','B2_R_always','B3_GA_R'):
        a=[x for x in records if x['strategy']==name];summary[name]={'qa':len(a),'accuracy':float(np.mean([x['correct'] for x in a])),'gold_nll':float(np.mean([x['gold_nll'] for x in a]))}
    return summary

def run(resume=None):
    gate_lock_path=ROOT/'locks/gate_a_lock.json'
    if not gate_lock_path.exists():raise ValueError('Gate A must be trained selected and frozen before Router')
    if (ROOT/'locks/fusion_router_lock.json').exists() or (ROOT/'pretest_lock.json').exists():raise ValueError('Router fitting already closed')
    gate_hash_before=sha256(gate_lock_path);gate_lock=json.loads(gate_lock_path.read_text());gate_weights_before={p:sha256(p) for p in gate_lock['all_candidate_weights_sha256']}
    cfg=config()['Router_training'];seed_all(1234);r=Runtime();bundle=GateBundle()
    query=r.new_router.query_generator.query_tokens;key=r.new_router.visual_key.weight
    fusion=GroundedFusionRouter(query_init=query,key_init=key,key_dim=cfg['key_dim'],rank=cfg['value_rank']).to(r.device)
    r.attach_fusion(fusion);r.model.model.gradient_checkpointing_enable()
    if any(m.p>0 for m in r.model.model.modules() if isinstance(m,torch.nn.Dropout)):raise ValueError('frozen decoder expected zero dropout')
    params=list(fusion.parameters());optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay']);directory=ROOT/'models/fusion_router'
    rows=training_rows('train');val=training_rows('val');groups=defaultdict(list)
    for row in rows:groups[row['source_id']].append(row)
    model_hash_before=module_hash(r.model);V1_before=sha256(V1);initial=copy.deepcopy({k:v.cpu() for k,v in fusion.state_dict().items()})
    start_epoch=0;completed=set();step=0;history=[]
    if resume:
        p=torch.load(resume,map_location='cpu',weights_only=False)
        if p['frozen_gate_lock_sha256']!=gate_hash_before:raise ValueError('resume would alter Gate A')
        fusion.load_state_dict(p['state_dict']);optimizer.load_state_dict(p['optimizer']);start_epoch=p['epoch'];completed=set(p.get('completed_sources',[]));step=p['step']
        if (ROOT/'logs/fusion_learning_history.json').exists():history=json.loads((ROOT/'logs/fusion_learning_history.json').read_text())
    elif list(directory.glob('epoch*.pt')):raise FileExistsError('use explicit resume, never overwrite completed checkpoints')
    def save(path,epoch,done):
        if path.exists():raise FileExistsError(path)
        torch.save({'kind':'gate_first_grounded_fusion_RA','spec':fusion.spec,'state_dict':fusion.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch,'step':step,'completed_sources':sorted(done),'configuration':cfg,'frozen_gate_lock_sha256':gate_hash_before,
            'frozen_Gate_main_sha256':gate_lock['gate_a_best_sha256'],'frozen_V1_sha256':V1_before,'train_manifest_sha256':config()['development_manifest_sha256']['train'],
            'seed':1234,'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()},path)
    # Actual zero-init identity forward and current-model QA before optimization.
    first=load(rows[0]);z=first['V1'].to(r.device);v=first['visual'].to(r.device).float();g=first['visual_grounded'].to(r.device).float()
    with torch.no_grad():zero_identity=torch.equal(fusion(v,g,z,torch.ones(4,dtype=torch.bool,device=r.device)),z)
    if not zero_identity and not resume:raise AssertionError('R-A zero initialization is not V1 identity')
    if not resume:write_json(ROOT/'audit/fusion_initialization.json',{'actual_zero_init_Z_identity':zero_identity,'parameters_trainable':sum(p.numel() for p in params),
        'parameter_names':[n for n,p in fusion.named_parameters() if p.requires_grad],'backbone_parameters_trainable':sum(p.numel() for p in r.model.parameters() if p.requires_grad),
        'Gate_A_frozen_before_Router_training':True,'from_zero_residual_not_F0':True})
    logpath=ROOT/'logs/fusion_train_steps.jsonl' if not resume else ROOT/f'logs/fusion_train_steps_resume_{start_epoch}.jsonl'
    selected_updates=forced_updates=zero_grad_samples=0
    if resume:
        seen_steps={}
        for record in read_jsonl(ROOT/'logs/fusion_train_steps.jsonl'):
            if record.get('step',step+1)<=step:seen_steps[record['step']]=record
        selected_updates=sum(sum(x['gate_selected_on']) for x in seen_steps.values());forced_updates=sum(sum(x['forced_on_exposure']) for x in seen_steps.values())
    start=time.perf_counter();status('M3','running',Gate_A_main_ON_rate=gate_lock['selected_main']['metrics']['on_rate'],resume_checkpoint=resume)
    for epoch in range(start_epoch,cfg['epochs']):
        order=list(groups);random.Random(1234+epoch).shuffle(order);losses=[];done=set(completed) if epoch==start_epoch else set()
        for i,source in enumerate(order):
            if source in done:continue
            d=load(groups[source][0]);v=d['visual'].to(r.device).float().clone().detach();g=d['visual_grounded'].to(r.device).float().clone().detach();base=d['V1'].to(r.device).clone().detach()
            gate_mask,scores=bundle.mask(d);quality=(d['audio_valid']&torch.tensor(d['map_valid'])).to(r.device);chosen=torch.tensor(gate_mask,device=r.device)&quality
            forced=quality&~chosen;exposure=chosen|forced
            for row in groups[source]:
                fusion.eval();r.model.model.eval()
                with torch.no_grad():pure_z=fusion(v,g,base,chosen);pure_loss=float(r.first_token_loss_z(row,pure_z))
                if not bool(exposure.any()):
                    append_jsonl(logpath,{'source_id':source,'question_id':row['question_id'],'epoch':epoch+1,'skipped':'no valid map/audio','pure_GA_CE':pure_loss});continue
                fusion.train();r.model.model.train();optimizer.zero_grad(set_to_none=True)
                z=fusion(v,g,base,exposure);loss=r.first_token_loss_z(row,z)
                if not torch.isfinite(loss):raise ValueError('nonfinite Router CE')
                loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,cfg['clip'],error_if_nonfinite=True))
                # Individual confidently fitted samples can legitimately have
                # zero fp16 input gradient. Retain them and verify actual
                # nonzero updates over the complete run, not on every QA.
                zero_grad_samples+=int(norm==0)
                if any(p.grad is not None for m in (r.model,r.vision_router,r.audio_router,r.new_router) for p in m.parameters()):raise AssertionError('gradient escaped new Router whitelist')
                grad_names={n:float(p.grad.float().norm()) for n,p in fusion.named_parameters() if p.grad is not None};optimizer.step();step+=1;losses.append(float(loss.detach()))
                selected_updates+=int(chosen.sum());forced_updates+=int(forced.sum())
                append_jsonl(logpath,{'epoch':epoch+1,'step':step,'source_id':source,'question_id':row['question_id'],'zero_gradient_sample':norm==0,
                    'pure_GA_CE':pure_loss,'training_exposure_CE':float(loss.detach()),'gate_selected_on':chosen.cpu().tolist(),'forced_on_exposure':forced.cpu().tolist(),
                    'training_exposure_mask':exposure.cpu().tolist(),'Gate_ON_not_forced_at_validation_or_test':True,'gradient_norm':norm,'gradient_names_and_norms':grad_names,
                    'frozen_backbone_gradient_count':0,'frozen_Gate_gradient_count':0,'tokens_per_frame':64,'residual_diagnostics':fusion.last_diagnostics})
            done.add(source)
            if (i+1)%64==0:save(directory/f'progress_epoch{epoch+1}_source{i+1}.pt',epoch,done)
            print('Router ACTUAL train',epoch+1,i+1,len(groups),'step',step,'selected/forced',selected_updates,forced_updates,flush=True)
        save(directory/f'epoch{epoch+1}.pt',epoch+1,[]);metrics=validate(r,fusion,bundle,val,epoch+1)
        history.append({'epoch':epoch+1,'steps':step,'train_CE_mean':float(np.mean(losses)),'validation':metrics,'checkpoint':str(directory/f'epoch{epoch+1}.pt')});write_json(ROOT/'logs/fusion_learning_history.json',history)
        print('Router ACTUAL validation',epoch+1,metrics,flush=True)
    selected=min(history,key=lambda x:(-x['validation']['B2_R_always']['accuracy'],x['validation']['B2_R_always']['gold_nll'],x['epoch']))
    shutil.copy2(selected['checkpoint'],directory/'selected.pt');model_hash_after=module_hash(r.model)
    if model_hash_before!=model_hash_after or sha256(V1)!=V1_before or sha256(gate_lock_path)!=gate_hash_before:raise AssertionError('frozen model or Gate lock changed')
    if any(sha256(p)!=s for p,s in gate_weights_before.items()):raise AssertionError('Router training changed frozen Gate weights')
    changed=[n for n,v in fusion.state_dict().items() if not torch.equal(v.detach().cpu(),initial[n])]
    if not changed or not any(n.startswith('output.') for n in changed):raise AssertionError('new Router did not train')
    write_json(ROOT/'locks/fusion_router_lock.json',{'created_utc':now(),'selected_epoch':selected['epoch'],'selected_checkpoint':str(directory/'selected.pt'),'selected_sha256':sha256(directory/'selected.pt'),
        'Gate_A_lock_sha256_before_after':gate_hash_before,'Gate_weights_SHA_unchanged':True,'backbone_state_sha_before':model_hash_before,'backbone_state_sha_after':model_hash_after,
        'V1_sha256_before_after':V1_before,'optimizer_steps':step,'changed_parameters':changed,'parameters_trainable':sum(p.numel() for p in params),
        'gradient_update_window_counts':{'gate_selected_on':selected_updates,'forced_on_exposure':forced_updates},'training_time_exposure_is_not_deployment_Gate_ON':True,
        'elapsed_s':time.perf_counter()-start,'resume_from':resume,'zero_gradient_samples_recorded':zero_grad_samples,'selection':'legal val B2 always-on real EM then Gold NLL; trained epochs only',
        'Gate_u_not_relabelled_for_new_Router':'G_A remains trained for fixed F0; deployment with RF is frozen transfer'})
    status('M3','completed',optimizer_steps=step,selected_epoch=selected['epoch'],forced_exposure_updates=forced_updates);journal('frozen-Gate Router actually trained and locked',optimizer_steps=step,selected_epoch=selected['epoch'],changed_parameters=changed)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume');a=p.parse_args();run(a.resume)
