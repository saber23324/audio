import json,copy,time
import numpy as np
import torch
import torch.nn.functional as F
from .common import ROOT,config,rows,read_jsonl,write_json,append_jsonl,sha256,seed_all,journal,status
from .features import load
from .gate import IMUGate,IMUFeatures

def dataset(role):
    out={k:[] for k in ('visual','imu','context','valid')};loss=[];correct=[];meta=[]
    for row in rows(role):
        d=load(row)
        for k in out:out[k].append(d['features'][k])
        u=read_jsonl(ROOT/'results/utility'/(row['qa_id'].replace(':','_')+'.jsonl'));assert len(u)==16;loss.append([next(x['gold_nll'] for x in u if x['policy_code']==m) for m in range(16)]);correct.append([next(x['correct'] for x in u if x['policy_code']==m) for m in range(16)]);meta.append({'qa_id':row['qa_id'],'source_id':row['source_id']})
    return {**{k:torch.stack(v) for k,v in out.items()},'nll':torch.tensor(loss),'correct':torch.tensor(correct,dtype=torch.bool),'meta':meta}
def batch(d,idx,device):return IMUFeatures(*(d[k][idx].flatten(0,1).to(device) for k in ('visual','imu','context','valid')))
def objective(lp,nll,spec,temp,correct=None):
    ks=lp.new_tensor([m.bit_count() for m in range(16)]);scores=nll+float(spec['lambda'])*ks[None]
    if spec['name'] in ('ranking','classification'):
        if correct is None:raise ValueError('accuracy targets require actual all16 greedy correctness')
        scores=2*(~correct).float()+nll/(1+nll)  # lexicographic correctness, monotone bounded NLL tie-break
    if spec['loss']=='hard_trajectory':return F.nll_loss(lp,scores.argmin(-1))
    if spec['loss']=='soft_trajectory':return -(torch.softmax(-scores/temp,-1)*lp).sum(-1).mean()
    # True pairwise ranking of complete trajectory utility, no additive
    # per-window labels. Stabilize unidentifiable all-tie samples with CE.
    diff=scores[:,:,None]-scores[:,None,:];pref=diff<-1e-5;weight=(-diff).clamp(min=0,max=5)*pref;rank=-F.logsigmoid(lp[:,:,None]-lp[:,None,:]);den=weight.sum((1,2));pair=(rank*weight).sum((1,2))/den.clamp_min(1e-8);distill=-(torch.softmax(-scores/temp,-1)*lp).sum(-1)
    return torch.where(den>0,pair+.1*distill,distill).mean()
def run():
    lock=ROOT/'locks/utility_lock.json'
    if not lock.exists():raise ValueError('all16 actual complete QA must be sealed first')
    cfg=config()['Gate'];tr=dataset('train');va=dataset('val');device='cuda' if torch.cuda.is_available() else 'cpu';catalog=[];cp=json.loads((ROOT/'locks/R_REAL.json').read_text());before=sha256(cp['checkpoint'])
    for spec in cfg['candidates']:
        directory=ROOT/f'checkpoints/G_{spec["name"]}';directory.mkdir(exist_ok=True)
        if list(directory.glob('*.pt')):raise FileExistsError('partial Gate fit must be preserved; explicit recovery required')
        seed_all(1234);model=IMUGate(extended=not spec['name'].startswith('cost')).to(device);initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);steps=0;nonzero=0;best=float('inf');checkpoints=[];start=time.perf_counter();rng=torch.Generator().manual_seed(1234);fv=batch(va,torch.arange(len(va['meta'])),device)
        for epoch in range(1,cfg['epochs']+1):
            model.train();losses=[]
            for idx in torch.randperm(len(tr['meta']),generator=rng).split(cfg['batch_size_episodes']):
                f=batch(tr,idx,device);lp=model.log_policy(f);loss=objective(lp,tr['nll'][idx].to(device),spec,cfg['temperature'],tr['correct'][idx].to(device));assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));opt.step();steps+=1;nonzero+=gn>0;losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():vl=float(objective(model.log_policy(fv),va['nll'].to(device),spec,cfg['temperature'],va['correct'].to(device)))
            improved=vl<best;best=min(best,vl);record={'epoch':epoch,'optimizer_steps':steps,'train_loss':float(np.mean(losses)),'val_loss':vl,'nonzero_gradient_steps':nonzero,'Router_updated':False};append_jsonl(ROOT/f'logs/G_{spec["name"]}_train.jsonl',record)
            if improved or epoch in cfg['checkpoint_epochs']:
                path=directory/f'epoch{epoch}.pt';torch.save({'kind':'direct_transfer_IMU_Gate','spec':model.spec,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'steps':steps,'objective':spec,'seed':1234,'R_REAL_SHA':before,'utility_lock_SHA':sha256(lock),'rng_torch':torch.get_rng_state()},path);checkpoints.append({'path':str(path),'sha256':sha256(path),'epoch':epoch,'val_training_objective':vl})
            print('ACTUAL GATE UPDATE',spec['name'],epoch,steps,round(record['train_loss'],6),round(vl,6),flush=True)
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[k])];assert changed and nonzero>0;allowed=[c for c in checkpoints if c['epoch'] in cfg['checkpoint_epochs'] or c==min(checkpoints,key=lambda c:c['val_training_objective'])]
        catalog.append({'name':spec['name'],'objective':spec,'steps':steps,'epochs':cfg['epochs'],'parameters':sum(p.numel() for p in model.parameters()),'nonzero_gradient_steps':nonzero,'changed_keys':changed,'only_Gate_trainable':True,'Router_optimizer_count':0,'selected_candidate_checkpoints':allowed,'all_checkpoints':checkpoints,'elapsed_s':time.perf_counter()-start})
    assert sha256(cp['checkpoint'])==before;write_json(ROOT/'locks/Gate_training_catalog.json',catalog);status('M4','trained_Gates',models=len(catalog),optimizer_steps=sum(c['steps'] for c in catalog));journal('all mandatory new Gate objectives actually trained on new R_REAL all16 QA utility',models=len(catalog),steps=sum(c['steps'] for c in catalog))
if __name__=='__main__':run()
