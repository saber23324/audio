"""Actual matched forced-ON R_REAL/R_DUP/R_ZERO fits, old model immutable."""
import argparse,json,copy,time,random,shutil
import numpy as np
import torch
from .common import ROOT,V1,config,rows,seed_all,write_json,write_jsonl,append_jsonl,sha256,journal,status
from .features import load
from .runtime import Runtime,new_fusion,fuse_sequence

@torch.inference_mode()
def validate(r,m,variant,epoch):
    r.model.eval();m.eval();records=[]
    for row in rows():
        d=load(row);v=d['visual'].to(r.device).float();g=(d['visual'] if variant=='R_DUP' else d['zero'] if variant=='R_ZERO' else d['real']).to(r.device).float();base=d['base'].to(r.device);z=fuse_sequence(m,v,g,base);assert z.shape==(4,64,3584)
        records.append({**row,**r.score(row,z),'strategy':variant,'epoch':epoch,'visual_tokens_actual':256,'tokens_per_frame':64})
    out=ROOT/f'results/router/{variant}_epoch{epoch}.jsonl'
    if out.exists():raise FileExistsError(out)
    write_jsonl(out,records);return {role:{'qa':len(a),'correct':sum(x['correct'] for x in a),'accuracy':float(np.mean([x['correct'] for x in a])),'gold_nll':float(np.mean([x['gold_nll'] for x in a]))} for role in ('train','val') if (a:=[x for x in records if x['role']==role])}

def run(variant):
    cfg=config()['Router'];directory=ROOT/f'checkpoints/{variant}';directory.mkdir(exist_ok=True)
    if list(directory.glob('*.pt')):raise FileExistsError('retain partial training; explicit new recovery required')
    r=Runtime();seed_all(1234);m=new_fusion(r);r.model.model.gradient_checkpointing_enable();params=list(m.parameters());opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};frozen_versions={n:p._version for n,p in r.model.named_parameters()};v1_initial={n:p.detach().cpu().clone() for n,p in r.new_router.named_parameters()}
    assert sum(p.numel() for p in params)==696320 and not any(p.requires_grad for p in r.model.parameters())
    first=load(rows('train')[0]);v=first['visual'].to(r.device);g=first['real'].to(r.device);base=first['base'].to(r.device)
    with torch.inference_mode():
        z=m(v,g,base);identity=torch.equal(z,base);off=m(v,torch.full_like(g,float('nan')),base,torch.zeros(4,dtype=torch.bool,device=r.device));assert torch.equal(off,base)
        a=r.score(rows('train')[0],z);b=r.score(rows('train')[0],base)
        assert identity and a['generated_ids']==b['generated_ids']
    write_json(ROOT/f'audit/{variant}_identity.json',{'zero_residual_Z_exact':identity,'generated_IDs_exact':True,'off_ignores_NaN_aux':True,'shape':list(z.shape),'active_parameters':696320,'frozen_backbone_parameters_trainable':0,'initial_parameter_shapes':{k:list(x.shape) for k,x in initial.items()}})
    step=0;history=[];started=time.perf_counter();zero_grad=0
    def save(path,epoch):
        if path.exists():raise FileExistsError(path)
        torch.save({'kind':'imu_direct_transfer_fusion','spec':m.spec,'state_dict':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'step':step,'variant':variant,'config':cfg,'seed':1234,'V1_SHA':sha256(V1),'manifest_SHA':config()['manifest_sha256'],'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()},path)
    save(directory/'initial_untrained.pt',0)
    for epoch in range(1,cfg['epochs']+1):
        order=rows('train');random.Random(1234+epoch).shuffle(order);losses=[]
        for row in order:
            d=load(row);v=d['visual'].to(r.device).float().clone().detach();g=(d['visual'] if variant=='R_DUP' else d['zero'] if variant=='R_ZERO' else d['real']).to(r.device).float().clone().detach();base=d['base'].to(r.device).clone().detach();m.train();r.model.model.train();opt.zero_grad(set_to_none=True);z=m(v,g,base);loss=r.first_loss(row,z)
            if not torch.isfinite(loss):raise ValueError('nonfinite full-vocabulary first answer CE')
            loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));zero_grad+=int(gn==0)
            assert not any(p.grad is not None for p in r.model.parameters()) and not any(p.grad is not None for p in r.new_router.parameters())
            opt.step();step+=1;losses.append(float(loss.detach()));append_jsonl(ROOT/f'logs/{variant}_train_steps.jsonl',{'epoch':epoch,'step':step,'qa_id':row['qa_id'],'source_id':row['source_id'],'first_answer_CE':losses[-1],'gradient_norm':gn,'Gate_in_training':False,'exposure_mask':[True]*4,'new_Router_only_optimizer':True,'frozen_gradient_count':0,'tokens_per_frame':64})
            if step%16==0:print('ACTUAL ROUTER UPDATE',variant,epoch,step,round(losses[-1],5),flush=True)
        r.model.model.eval();m.eval();save(directory/f'epoch{epoch}.pt',epoch);metrics=validate(r,m,variant,epoch);history.append({'epoch':epoch,'step':step,'mean_train_CE':float(np.mean(losses)),'QA':metrics,'checkpoint':str(directory/f'epoch{epoch}.pt')});write_json(ROOT/f'logs/{variant}_history.json',history);print('ROUTER FULL QA',variant,epoch,metrics,flush=True)
    selected=min(history,key=lambda h:(-h['QA']['val']['accuracy'],h['QA']['val']['gold_nll'],h['epoch']));shutil.copy2(selected['checkpoint'],directory/'selected.pt');changed=[k for k,p in m.state_dict().items() if not torch.equal(p.cpu(),initial[k])]
    assert changed and 'output.weight' in changed and frozen_versions=={n:p._version for n,p in r.model.named_parameters()}
    assert all(torch.equal(p.cpu(),v1_initial[n]) for n,p in r.new_router.named_parameters())
    write_json(ROOT/f'locks/{variant}.json',{'checkpoint':str(directory/'selected.pt'),'sha256':sha256(directory/'selected.pt'),'selected_epoch':selected['epoch'],'selected_val':selected['QA']['val'],'optimizer_steps':step,'trained_epochs':cfg['epochs'],'zero_grad_samples':zero_grad,'changed_keys':changed,'params':696320,'frozen_foundation_versions_unchanged':True,'frozen_V1_exact':True,'history':history,'elapsed_s':time.perf_counter()-started,'training_init':'new zero-output residual, V1 query/key slice; same seed all variants','loss':'first answer full-vocabulary CE, not KD','frozen_before_Gate_Utility':True})
    journal('actual matched IMU Fusion Router fit complete',variant=variant,steps=step,selected_epoch=selected['epoch'],changed=changed)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['R_REAL','R_DUP','R_ZERO']);a=p.parse_args();run(a.variant)
