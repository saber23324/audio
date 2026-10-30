"""Exhaust all16 four-arrival policies using actual complete QA generation."""
import argparse,json
from collections import defaultdict
import numpy as np
import torch
from .common import ROOT,training_rows,read_jsonl,write_jsonl,write_json,sha256,status,journal,V1
from .features import load
from .runtime import Runtime

def bits(code):return [(code>>i)&1==1 for i in range(4)]
def code(mask):return sum(int(on)<<i for i,on in enumerate(mask))
def table(role,source):return ROOT/f'labels/policies/{role}/{source}.jsonl'

def run(role,shard=0,shards=1):
    if not (ROOT/'locks/M0_audit_lock.json').exists():raise ValueError('M0 audit must finish before phase A')
    rows=training_rows(role);groups=defaultdict(list)
    for row in rows:groups[row['source_id']].append(row)
    r=Runtime();directory=ROOT/f'labels/policies/{role}';directory.mkdir(parents=True,exist_ok=True)
    for i,(source,qs) in enumerate(list(groups.items())[shard::shards]):
        path=table(role,source)
        if path.exists():continue
        d=load(qs[0]);base=d['V1'].to(r.device);ground=d['F0'].to(r.device);records=[]
        for policy in range(16):
            mask=torch.tensor(bits(policy),device=r.device);z=base.clone();z[mask]=ground[mask]
            for row in qs:
                result=r.score(row,z)
                records.append({**row,**result,'policy_code':policy,'on_mask':bits(policy),'timestamps_s':[2.,4.,6.,8.],
                    'tokens_per_frame':64,'base_model_sha256':d['V1_checkpoint_sha256'],'fixed_F0_beta':.25,
                    'actual_full_mixed_generation':True,'utility_not_sum_of_single_windows':True})
        write_jsonl(path,records);print('actual16 policies',role,shard,i+1,source,len(records),flush=True)
    journal('actual exhaustive policy shard done',role=role,shard=shard,shards=shards)

def combine():
    summary={}
    for role in ('train','val'):
        rows=training_rows(role);groups=defaultdict(list)
        for row in rows:groups[row['source_id']].append(row)
        windows=[];policies=[];perQA=[]
        for source,qs in groups.items():
            a=read_jsonl(table(role,source));by={p:{r['question_id']:r for r in a if r['policy_code']==p} for p in range(16)}
            if any(set(v)!=set(r['question_id'] for r in qs) for v in by.values()):raise ValueError('incomplete actual full-policy grid')
            u=[]
            for p in range(16):
                values=[by[0][q['question_id']]['gold_nll']-by[p][q['question_id']]['gold_nll'] for q in qs];u.append(float(np.median(values)))
                policies.append({'source_id':source,'role':role,'policy_code':p,'median_native_QA_NLL_utility':u[-1],'n_qa':len(qs),
                    'accuracy':float(np.mean([r['correct'] for r in by[p].values()])),'mean_gold_nll':float(np.mean([r['gold_nll'] for r in by[p].values()]))})
            for t in range(4):
                for past in range(1<<t):
                    pairs=[]
                    for future in range(1<<(3-t)):
                        p0=past+(future<<(t+1));pairs.append((p0,p0+(1<<t)))
                    marginal=float(np.mean([u[p1]-u[p0] for p0,p1 in pairs]))
                    windows.append({'source_id':source,'role':role,'frame_index':t,'timestamp_s':2.*(t+1),'past_mask_code':past,
                        'conditional_marginal_utility':marginal,'tie':abs(marginal)<=.001,'actual_policy_pairs':pairs,
                        'target_definition':'mean over enumerated future choices of source-median full-policy utility difference; no future model inputs'})
            for q in qs:
                qid=q['question_id'];v=by[0][qid];f=by[15][qid]
                quad='both_correct' if v['correct'] and f['correct'] else 'F0_only_correct' if f['correct'] else 'V1_only_correct' if v['correct'] else 'both_wrong'
                perQA.append({'source_id':source,'question_id':qid,'role':role,'quadrant':quad,'V1_correct':v['correct'],'F0_correct':f['correct'],
                    'oracle_endpoints_correct':bool(v['correct'] or f['correct']),'oracle_all16_correct':any(by[p][qid]['correct'] for p in range(16)),
                    'oracle_policy_min_nll':min(range(16),key=lambda p:by[p][qid]['gold_nll']),'V1_gold_nll':v['gold_nll'],'F0_gold_nll':f['gold_nll']})
        write_jsonl(ROOT/f'labels/conditional_marginals_{role}.jsonl',windows);write_jsonl(ROOT/f'labels/source_policies_{role}.jsonl',policies)
        write_jsonl(ROOT/f'labels/current_endpoint_oracles_{role}.jsonl',perQA)
        summary[role]={'sources':len(groups),'qa':len(rows),'actual_full_QA_generations':len(rows)*16,'conditional_window_states':len(windows),
            'ties':sum(r['tie'] for r in windows),'positive_sources':len({r['source_id'] for r in windows if r['conditional_marginal_utility']>.001}),
            'negative_sources':len({r['source_id'] for r in windows if r['conditional_marginal_utility']<-.001}),
            'endpoint_quadrants':{s:sum(r['quadrant']==s for r in perQA) for s in ('both_correct','F0_only_correct','V1_only_correct','both_wrong')},
            'oracle_all16_accuracy':float(np.mean([r['oracle_all16_correct'] for r in perQA]))}
    write_json(ROOT/'locks/policy_label_lock.json',{'V1_sha256':sha256(V1),'beta':.25,'timestamps':[2.,4.,6.,8.],'code_sha256':sha256(__file__),
        'files':{str(p):sha256(p) for p in (ROOT/'labels').glob('*.jsonl')},'summary':summary})
    status('M1','completed',policy_grid=summary);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['train','val']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--combine',action='store_true');a=p.parse_args()
    combine() if a.combine else run(a.role,a.shard,a.shards)

