"""Only optimizer in this experiment: new lightweight Gate parameters."""
import argparse,copy,hashlib,json,time,random
import numpy as np
import torch
from utility_gate.features import GateFeatures,load_feature_payload
from utility_gate.gate_first_native.gate import freeze_proxy,proxy_scores
from .common import ROOT,SEED,config,train_rows,read_jsonl,write_json,append_jsonl,sha256,journal,status
from .features import load
from .enumerate_masks import groups,table
from .gate_models import FixedRouterGate,log_policy,target_distribution

def dataset(role,proxy):
    gs=groups(role);keys=('visual','audio','context','valid_audio');a={k:[] for k in keys};cheap=[];nll=[];rr=[];meta=[]
    for source,qs in gs.items():
        d=load(qs[0]);f=load_feature_payload(d['features']);ac=load_feature_payload(d['acoustic_features'])
        for k in keys:a[k].append(getattr(f,k))
        cheap.append(ac.audio);rr.append(proxy_scores(proxy,f)[0]);rows=read_jsonl(table(role,source))
        nll.append([float(np.mean([r['gold_nll'] for r in rows if r['policy_code']==m])) for m in range(16)])
        meta.append({'source_id':source,'n_qa':len(qs),'quality_mask':d['audio_valid'].tolist()})
    return {**{k:torch.stack(v) for k,v in a.items()},'cheap':torch.stack(cheap),'r':torch.stack(rr),'nll':torch.tensor(nll),'meta':meta}
def batch(d,idx,family,device):
    f=GateFeatures(d['visual'][idx].reshape(-1,7168), (d['cheap'] if family=='acoustic' else d['audio'])[idx].flatten(0,1),d['context'][idx].reshape(-1,4),d['valid_audio'][idx].reshape(-1))
    return f.to(device),d['r'][idx].to(device)
def run():
    if not (ROOT/'locks/policy_label_lock.json').exists():raise ValueError('actual16 labels not complete')
    cfg=config()['train'];torch.set_num_threads(4);proxy=freeze_proxy();tr=dataset('train',proxy);va=dataset('val',proxy);del proxy
    device='cuda' if torch.cuda.is_available() else 'cpu';catalog=[];start=time.perf_counter()
    for family in cfg['families']:
        for lam in cfg['lambdas']:
            for temp in cfg['temperatures']:
                name=f'{family}_l{lam:g}_t{temp:g}';directory=ROOT/'models'/name;done=directory/'completed.json'
                if done.exists():catalog.append(json.loads(done.read_text()));continue
                if directory.exists():raise RuntimeError('partial fit requires a new explicit resume namespace; retain failures')
                directory.mkdir();random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
                m=FixedRouterGate(family,cfg['hidden'],cfg['dropout']).to(device);opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
                initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};whitelist=[n for n,p in m.named_parameters() if p.requires_grad]
                if {id(p) for g in opt.param_groups for p in g['params']}!={id(p) for p in m.parameters()}:raise AssertionError('optimizer mismatch')
                qtr=target_distribution(tr['nll'],tr['valid_audio'],lam,temp);qva=target_distribution(va['nll'],va['valid_audio'],lam,temp)
                best=float('inf');candidates=[];losses=[];steps=0;grad_nonzero=0;fitstart=time.perf_counter();generator=torch.Generator().manual_seed(SEED)
                fval,rval=batch(va,torch.arange(len(va['meta'])),family,device)
                for epoch in range(1,cfg['epochs']+1):
                    m.train();total=0.;gns=[]
                    for idx in torch.randperm(len(tr['meta']),generator=generator).split(cfg['batch_size_sources']):
                        f,r=batch(tr,idx,family,device);q=qtr[idx].to(device);lp=log_policy(m,f,r);loss=-(q*lp).sum(-1).mean()
                        if not torch.isfinite(loss):raise ValueError('nonfinite sequence CE')
                        opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5.));opt.step();steps+=1;grad_nonzero+=gn>0;gns.append(gn);total+=float(loss)*len(idx)
                    m.eval()
                    with torch.inference_mode():vl=float(-(qva.to(device)*log_policy(m,fval,rval)).sum(-1).mean())
                    record={'model':name,'epoch':epoch,'train_sequence_CE':total/len(tr['meta']),'val_sequence_CE':vl,'mean_gradient_norm_before_clip':float(np.mean(gns)),'optimizer_steps':steps}
                    losses.append(record);append_jsonl(ROOT/f'train/{name}.jsonl',record)
                    improved=vl<best;best=min(best,vl)
                    if epoch in [x for x in cfg['checkpoint_candidates'] if isinstance(x,int)] or improved:
                        cp=directory/f'epoch{epoch}.pt';torch.save({'kind':'fixed_router_sequence_gate','spec':m.spec,'state_dict':m.state_dict(),'optimizer_state':opt.state_dict(),'epoch':epoch,'lambda':lam,'temperature':temp,'seed':SEED,
                            'target_protocol_sha256':sha256(ROOT/'locks/policy_label_lock.json'),'train_manifest_sha256':config()['development_manifest_sha256']['train'],
                            'trainable_whitelist':whitelist,'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':generator.get_state()},cp)
                        candidates.append({'path':str(cp),'epoch':epoch,'val_distill_loss':vl,'validation_best_at_save':improved})
                    print('GATE TRAIN',name,epoch,record['train_sequence_CE'],vl,flush=True)
                bestcp=min(candidates,key=lambda x:x['val_distill_loss'])
                allowed=[x for x in candidates if x['epoch'] in [1,10,20,30] or x['path']==bestcp['path']]
                changed=[k for k,v in m.state_dict().items() if not torch.equal(v.cpu(),initial[k])]
                summary={'model':name,'family':family,'lambda':lam,'temperature':temp,'parameters':sum(p.numel() for p in m.parameters()),'epochs':cfg['epochs'],'optimizer_steps':steps,
                    'nonzero_gradient_steps':grad_nonzero,'changed_state_keys':changed,'only_new_gate_optimizer':True,'Router_optimizer_count':0,'foundation_gradients':False,
                    'train_sources':len(tr['meta']),'val_sources':len(va['meta']),'source_equal_weighted':True,'elapsed_s':time.perf_counter()-fitstart,'fit_device':device,
                    'initial_train_CE':losses[0]['train_sequence_CE'],'final_train_CE':losses[-1]['train_sequence_CE'],'best_val_distillation':bestcp,'candidate_checkpoints':allowed,
                    'checkpoint_sha256':{x['path']:sha256(x['path']) for x in candidates},'trainable_whitelist':whitelist}
                if not changed or grad_nonzero==0:raise AssertionError('Gate did not actually train')
                write_json(done,summary);catalog.append(summary);del m,opt
    write_json(ROOT/'train/catalog.json',catalog);status('M2','trained',models=len(catalog),elapsed_s=time.perf_counter()-start);journal('actual Gate-only complete sequence distillation',models=len(catalog),Router_training=False)
if __name__=='__main__':run()
