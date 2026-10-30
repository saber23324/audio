"""Real complete-mixture validation QA; thresholds never target activation rate."""
import argparse,json
from collections import defaultdict
import numpy as np
import torch
from utility_gate.common import write_jsonl
from .common import ROOT,train_rows,read_jsonl,read_config,write_json,sha256,status,journal,now
from .features import load_rgb
from .prepare_av import load_av,grounded_path
from .fuser64 import fuse,dependencies
from .runtime import Runtime,append_step_features
from .online_policy import GateBundle,candidates,decide

def plans():
    # Fitted on train waveform energy only; never reads val/test labels.
    sources={r['source_id']:r for r in train_rows('gate_train')};energies=[]
    for row in sources.values():
        a=load_av(row);w=a['windows'].float();energies.extend(w.square().mean(1).sqrt()[a['audio_valid']].tolist())
    return candidates(read_config()['policy_selection'],float(np.median(energies)))

@torch.inference_mode()
def run(shard=0,shards=1):
    rows=train_rows('gate_val');group=defaultdict(list)
    for row in rows:group[row['source_id']].append(row)
    policy=plans();r=Runtime();r.load_new_router();gates=GateBundle();beta=dependencies()['beta']
    directory=ROOT/'qa_eval/val_policy';directory.mkdir(parents=True,exist_ok=True)
    for index,(source,qs) in enumerate(list(group.items())[shard::shards]):
        path=directory/f'{source}.jsonl'
        if path.exists():continue
        rgb=load_rgb(qs[0]);a=load_av(qs[0]);ground=torch.load(grounded_path(source),map_location='cpu',weights_only=False)
        z,_=r.vision_latents(rgb['visual']);zg,_=r.vision_latents(ground['visual_grounded']);action=fuse(z,zg,on=True,beta=beta)
        masks={name:[] for name in policy};scores=[]
        for i in range(len(z)):
            f=append_step_features(rgb['visual'],a['audio'],a['audio_valid'],rgb['timestamps'],i)
            s=gates.scores(f,float(a['windows'][i].float().square().mean().sqrt()));scores.append(s)
            for name,p in policy.items():masks[name].append(decide(p,s,bool(a['audio_valid'][i])))
        seen={};records=[]
        for name,mask in masks.items():
            key=tuple(mask)
            if key in seen:predictions,original=seen[key]
            else:
                mixed=z.clone();m=torch.tensor(mask,device=r.device);mixed[m]=action[m]
                predictions=[r.score(row,mixed) for row in qs];original=name;seen[key]=(predictions,name)
            for row,value in zip(qs,predictions):
                records.append({**row,**value,'strategy':name,'policy':policy[name],'on_count':sum(mask),'window_count':len(mask),
                    'on_mask':mask,'real_inference_source_strategy':original,'identical_token_result_reused':original!=name,
                    'tokens_per_frame':64,'evaluation_mode':'offline_development_cached_predicted_views'})
        write_jsonl(path,records)
        write_jsonl(directory/f'{source}.scores.jsonl',[{'source_id':source,'frame_index':i,'timestamp_s':float(rgb['timestamps'][i]),**s} for i,s in enumerate(scores)])
        print('policy complete-mixture val',shard,index+1,source,'distinct actual forwards',len(seen)*len(qs),flush=True)
    journal('complete mixture policy validation shard done',shard=shard,shards=shards)

def combine():
    policy=plans();rows=train_rows('gate_val');sources=sorted({r['source_id'] for r in rows});out=[]
    for source in sources:
        a=read_jsonl(ROOT/f'qa_eval/val_policy/{source}.jsonl')
        if {x['strategy'] for x in a}!=set(policy):raise ValueError('incomplete validation matrix')
        out.extend(a)
    summary={}
    for name,p in policy.items():
        a=[x for x in out if x['strategy']==name];unique={x['source_id']:x for x in a};visual=[x for x in a if x['modality_type']=='Visual']
        summary[name]={'policy':p,'n_qa':len(a),'accuracy':float(np.mean([x['correct'] for x in a])),
            'gold_nll':float(np.mean([x['gold_nll'] for x in a])),'visual_accuracy':float(np.mean([x['correct'] for x in visual])) if visual else None,
            'on_rate':sum(x['on_count'] for x in unique.values())/sum(x['window_count'] for x in unique.values()),
            'on_count':sum(x['on_count'] for x in unique.values()),'windows':sum(x['window_count'] for x in unique.values())}
    base=summary['V1']['visual_accuracy']
    def choose(prefix):
        eligible=[(n,v) for n,v in summary.items() if n.startswith(prefix) and (base is None or v['visual_accuracy']>=base-.02)]
        if not eligible:raise ValueError('all-off must be an admissible policy')
        def tie(item):
            n,v=item;p=v['policy'];t=float(p.get('tau_u',p.get('tau_r',p.get('tau_energy','inf'))))
            return (-v['accuracy'],v['gold_nll'],-t,-float(p.get('tau_r','inf')),n)
        return min(eligible,key=tie)
    selected={name:choose(prefix) for name,prefix in [('G1','u_'),('G2','dual_'),('R_ONLY','r_'),('ENERGY','energy_')]}
    p=selected['G1'][1]['on_rate']
    locked={'V1':policy['V1'],'F0':policy['F0'],'G0':policy['G0'],**{n:v[1]['policy'] for n,v in selected.items()},
        'G3_seed1234':{'kind':'random','probability':p,'seed':1234},'G3_seed2234':{'kind':'random','probability':p,'seed':2234},'G3_seed3234':{'kind':'random','probability':p,'seed':3234}}
    write_json(ROOT/'locks/policy_lock.json',{'created_utc':now(),'policies':locked,'selected_validation_names':{k:v[0] for k,v in selected.items()},
        'selected_validation_results':{k:v[1] for k,v in selected.items()},'random_probability_from_G1_validation':p,
        'selection_objective':read_config()['policy_selection']['objective'],'activation_rate_target':None,
        'all_off_permitted':True,'all_on_permitted':True,'action_dependency':dependencies()['fingerprint'],
        'validation_predictions_sha256':{source:sha256(ROOT/f'qa_eval/val_policy/{source}.jsonl') for source in sources}})
    write_json(ROOT/'qa_eval/validation_policy_summary.json',summary);write_jsonl(ROOT/'qa_eval/validation_policy_all.jsonl',out)
    status('M5','development_selected_pending_pretest_lock',G1_validation_on_rate=p,selected={n:v[0] for n,v in selected.items()});print('POLICIES SELECTED',selected,'randomp',p,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--combine',action='store_true');a=p.parse_args()
    if a.combine:combine()
    else:run(a.shard,a.shards)

