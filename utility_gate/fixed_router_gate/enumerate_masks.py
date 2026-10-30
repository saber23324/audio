"""All16 complete R_A/V1 policies, actual greedy QA and Gold scoring."""
import argparse,hashlib
from collections import defaultdict
import numpy as np
import torch
from .common import ROOT,RA,V1,train_rows,read_jsonl,write_jsonl,write_json,sha256,journal,status
from .runtime import Runtime,qa_prompt
from .features import load

def bits(code):return [bool(code&(1<<t)) for t in range(4)]
def code(mask):return sum(int(x)<<i for i,x in enumerate(mask))
def table(role,source):return ROOT/f'labels/{role}/{source}.jsonl'
def groups(role):
    g=defaultdict(list)
    for q in train_rows(role):g[q['source_id']].append(q)
    return g
def run(role,shard=0,shards=1):
    if not (ROOT/'locks/M0_audit_lock.json').exists():raise ValueError('M0 not passed')
    runtime=Runtime();directory=ROOT/f'labels/{role}';directory.mkdir(exist_ok=True)
    for i,(source,qs) in enumerate(list(groups(role).items())[shard::shards]):
        target=table(role,source)
        if target.exists():
            if len(read_jsonl(target))!=16*len(qs):raise ValueError('incomplete grid')
            continue
        d=load(qs[0]);base=d['V1'].to(runtime.device);enhanced=d['R_A'].to(runtime.device);records=[]
        for policy in range(16):
            mask=torch.tensor(bits(policy),device=runtime.device)&d['audio_valid'].to(runtime.device).bool();z=base.clone();z[mask]=enhanced[mask]
            for row in qs:
                result=runtime.score(row,z)
                if result['visual_tokens']!=256:raise ValueError('not256 visual tokens')
                records.append({**row,**result,'policy_code':policy,'requested_mask':bits(policy),'on_mask':mask.tolist(),'mask_code':code(mask.tolist()),
                    'k_extra_images':int(mask.sum()),'encoded_image_count':4+int(mask.sum()),'source_quality_mask':d['audio_valid'].tolist(),
                    'timestamps_s':[2.,4.,6.,8.],'tokens_per_frame':64,'R_A_sha256':d['R_A_sha256'],'V1_sha256':d['V1_sha256'],
                    'prompt_sha256':hashlib.sha256(qa_prompt(runtime,row).numpy().tobytes()).hexdigest(),'valid_flag':True,
                    'actual_full_QA':True,'frontends_cached_for_legal_enumeration':True,'encoded_image_count_semantics':'logical deployment calls, cached fronts are not actual timing',
                    'utility_not_sum_of_single_windows':True})
        write_jsonl(target,records);print('ACTUAL16',role,shard,i+1,source,len(records),flush=True)
    journal('all16 actual full QA shard',role=role,shard=shard,shards=shards)
def combine():
    summaries={};policies=[]
    for role in ('train','val'):
        nqa=0;quads=defaultdict(int);bestk=defaultdict(int);oraclek=defaultdict(list);interactions=[];oracle=[]
        for source,qs in groups(role).items():
            a=read_jsonl(table(role,source));by={p:{r['question_id']:r for r in a if r['policy_code']==p} for p in range(16)}
            if any(set(x)!=set(q['question_id'] for q in qs) for x in by.values()):raise ValueError('incomplete16')
            mean=np.array([np.mean([r['gold_nll'] for r in by[p].values()]) for p in range(16)]);acc=[np.mean([r['correct'] for r in by[p].values()]) for p in range(16)]
            d=load(qs[0]);eligible=[p for p in range(16) if all(not b or bool(ok) for b,ok in zip(bits(p),d['audio_valid']))]
            best=min(eligible,key=lambda p:(mean[p],p.bit_count(),p));bestk[best.bit_count()]+=1
            for p in range(16):policies.append({'role':role,'source_id':source,'policy_code':p,'eligible':p in eligible,'k':p.bit_count(),'mean_gold_nll':float(mean[p]),'accuracy':float(acc[p]),'n_qa':len(qs)})
            interactions.append(float((mean[15]-mean[0])-sum(mean[1<<t]-mean[0] for t in range(4))))
            for q in qs:
                key=q['question_id'];v=by[0][key]['correct'];r=by[15][key]['correct'];quads['BothCorrect' if v and r else 'R_A_only' if r else 'V1_only' if v else 'BothWrong']+=1;nqa+=1
                oracle.append(any(by[p][key]['correct'] for p in eligible))
                for k in range(5):
                    ps=[p for p in eligible if p.bit_count()==k]
                    if ps:oraclek[k].append({'correct_upper':any(by[p][key]['correct'] for p in ps),'min_nll':min(by[p][key]['gold_nll'] for p in ps)})
        summaries[role]={'sources':len(groups(role)),'qa':nqa,'actual_QA_generations':16*nqa,'endpoint_quadrants':dict(quads),'oracle_all16_accuracy':float(np.mean(oracle)),
            'source_min_NLL_oracle_k_distribution':dict(bestk),'interaction_abs_mean_NLL':float(np.mean(np.abs(interactions))),
            'interaction_nonzero_sources':sum(abs(x)>1e-6 for x in interactions),
            'oracle_at_k':{k:{'qa_eligible':len(a),'accuracy_upper':float(np.mean([x['correct_upper'] for x in a])),'min_gold_nll':float(np.mean([x['min_nll'] for x in a]))} for k,a in oraclek.items()}}
    write_jsonl(ROOT/'policy_mask_results.jsonl',policies);write_json(ROOT/'labels/summary.json',summaries)
    write_json(ROOT/'locks/policy_label_lock.json',{'summary':summaries,'R_A':sha256(RA),'V1':sha256(V1),'files':{str(p):sha256(p) for p in (ROOT/'labels').rglob('*.jsonl')}})
    status('M1','completed',**summaries);print(summaries,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['train','val']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--combine',action='store_true');a=p.parse_args();combine() if a.combine else run(a.role,a.shard,a.shards)
