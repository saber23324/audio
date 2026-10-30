"""Select using real all16 QA on fresh validation, never confirmation labels."""
import json
from collections import defaultdict
import numpy as np
import torch
from utility_gate.features import take_features
from .common import ROOT,config,write_json,write_jsonl,read_jsonl,journal,sha256,status
from .gate_train import dataset,batch
from .gate_models import load_gate,history
from utility_gate.gate_first_native.gate import freeze_proxy
from .enumerate_masks import table

@torch.inference_mode()
def choose(model,f,r,threshold,cap=None):
    B=len(f.visual)//4;h=model.encode(f).reshape(B,4,-1);rv=r.reshape(B,4,1);valid=f.valid_audio.reshape(B,4);past=[0]*B;counts=[0]*B;codes=[0]*B;probs=[]
    for t in range(4):
        hist=torch.tensor([history(p,t,model.extended) for p in past],device=h.device)
        p=model.logits(h[:,t],hist,rv[:,t]).sigmoid();probs.append(p.cpu().tolist())
        for i in range(B):
            on=bool(valid[i,t]) and float(p[i])>threshold and (cap is None or counts[i]<cap)
            if on:past[i]|=1<<t;codes[i]=past[i];counts[i]+=1
    return codes,probs
def actual_metrics(codes,data,qa,cost,spec):
    rows=[];ks=[];ons=[];costs=[]
    for i,(meta,p) in enumerate(zip(data['meta'],codes)):
        selected=qa[meta['source_id']][int(p)];rows.extend(selected);k=int(p).bit_count();ks.append(k);ons.append([(p>>t)&1 for t in range(4)])
        if spec['kind'] in ('off','on'):costs.append(cost[spec['kind']])
        else:costs.append(cost[f"{spec['family']}_k{k}"])
    base=[x for s in qa for x in qa[s][0]];baseline={(x['source_id'],x['question_id']):x for x in base}
    return {'qa':len(rows),'sources':len(codes),'correct':sum(x['correct'] for x in rows),'accuracy':float(np.mean([x['correct'] for x in rows])),
        'gold_nll':float(np.mean([x['gold_nll'] for x in rows])),'mean_k':float(np.mean(ks)),'ON_rate':float(np.mean(ks)/4),
        'k_distribution':{str(k):ks.count(k) for k in range(5)},'ON_per_time':np.mean(ons,axis=0).tolist(),'estimated_frontend_s_from_pretest_real_cost':float(np.mean(costs)),
        'Rescue_vs_V1':sum(x['correct'] and not baseline[x['source_id'],x['question_id']]['correct'] for x in rows),
        'Harm_vs_V1':sum(not x['correct'] and baseline[x['source_id'],x['question_id']]['correct'] for x in rows),
        'evaluation':'causal chosen mask retrieves that exact actual complete VLM generation from legal all16; selected policies rerun online independently'}
def run():
    if (ROOT/'locks/selection_lock.json').exists():raise FileExistsError('already selected; do not retroactively tune')
    cfg=config();proxy=freeze_proxy();data=dataset('val',proxy);del proxy
    qa={m['source_id']:{p:[x for x in read_jsonl(table('val',m['source_id'])) if x['policy_code']==p] for p in range(16)} for m in data['meta']}
    cost=json.loads((ROOT/'timing/development_calibration.json').read_text())['mean_frontend_s'];catalog=json.loads((ROOT/'train/catalog.json').read_text());candidates=[]
    for kind,p in [('off',0),('on',15)]:
        codes=[sum((1<<t) for t,ok in enumerate(m['quality_mask']) if ok) if p else 0 for m in data['meta']];spec={'kind':kind}
        candidates.append({'id':kind,'spec':spec,'lambda':0.,'metrics':actual_metrics(codes,data,qa,cost,spec),'val_codes':codes})
    for entry in catalog:
        for cp in entry['candidate_checkpoints']:
            m=load_gate(cp['path'])[0];f,r=batch(data,torch.arange(len(data['meta'])),entry['family'],'cpu')
            for threshold in cfg['train']['thresholds']:
                codes,_=choose(m,f,r,float(threshold));spec={'kind':'learned','family':entry['family'],'checkpoint':cp['path'],'checkpoint_sha256':sha256(cp['path']),'threshold':threshold,
                    'lambda':entry['lambda'],'temperature':entry['temperature'],'epoch':cp['epoch']}
                candidates.append({'id':entry['model']+f'_e{cp["epoch"]}_p{threshold:g}','spec':spec,'lambda':entry['lambda'],'metrics':actual_metrics(codes,data,qa,cost,spec),'val_codes':codes})
    def key(x):
        a=x['metrics'];return (-a['accuracy'],a['gold_nll'],a['estimated_frontend_s_from_pretest_real_cost'],x['id'])
    acc=min((x for x in candidates if x['lambda']==0),key=key)
    eligible=[x for x in candidates if x['metrics']['accuracy']>=acc['metrics']['accuracy']-cfg['selection']['accuracy_margin']-1e-12 and x['metrics']['gold_nll']<=acc['metrics']['gold_nll']+cfg['selection']['NLL_margin']+1e-12]
    efficient=min(eligible,key=lambda x:(x['metrics']['estimated_frontend_s_from_pretest_real_cost'],x['metrics']['mean_k'],-x['metrics']['accuracy'],x['id']))
    selected={'accuracy_first':acc,'cost_aware':efficient}
    for family in cfg['train']['families']:
        selected[family]=min((x for x in candidates if x['spec'].get('family')==family and x['lambda']==0),key=key)
    bestfixed=min(candidates[:2],key=key)
    lock={'data_protocol_sha256':sha256(ROOT/'data/protocol.json'),'fit_scope':'fresh val only; all16 actual full-QA; source-level Gates; no confirmation predictions exist',
        'candidate_count':len(candidates),'distinct_selected_val_masks':len({tuple(x['val_codes']) for x in candidates}),
        'selected':selected,'best_fixed_endpoint_from_validation':bestfixed['spec'],'bootstrap_seed':cfg['bootstrap']['seed'],
        'val_source_order':[x['source_id'] for x in data['meta']],'random_probability':{k:selected[k]['metrics']['ON_rate'] for k in ('accuracy_first','cost_aware')},
        'degen_policy_note':'all-off/on are valid; selected constant or threshold0/1 may be a degenerate Gate. No artificial nonzero ON target.',
        'source_code_sha256':sha256(__file__)}
    write_jsonl(ROOT/'validation/candidates.jsonl',candidates);write_json(ROOT/'locks/selection_lock.json',lock);status('M3','selected',main={k:v['id'] for k,v in selected.items()});journal('two policies selected before confirmation',candidate_count=len(candidates),selection_sha256=sha256(ROOT/'locks/selection_lock.json'));print({k:{'id':v['id'],'metrics':v['metrics']} for k,v in selected.items()},flush=True)
if __name__=='__main__':run()
