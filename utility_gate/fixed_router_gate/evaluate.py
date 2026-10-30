"""Actually isolated complete QA inference for prelocked policies and controls."""
import argparse,json,hashlib,time
from collections import defaultdict
import torch
from .common import ROOT,RA,V1,config,train_rows,read_jsonl,write_jsonl,journal,sha256
from .runtime import Runtime,Localizer,qa_prompt
from .online_inference import Controllers,execute
def variants():
    lock=json.loads((ROOT/'locks/selection_lock.json').read_text());s=lock['selected'];acc=s['accuracy_first']['spec'];cost=s['cost_aware']['spec']
    out={'T0_V1':{'spec':{'kind':'off'}},'T1_RA_always':{'spec':{'kind':'on'}},'T2_old_gate':{'spec':{'kind':'old'}},
        'T3_accuracy':{'spec':acc},'T4_cost':{'spec':cost},'T6_visual':{'spec':s['visual']['spec']}}
    for label,key in [('accuracy','accuracy_first'),('cost','cost_aware')]:
        spec=s[key]['spec']
        for seed in config()['selection']['random_seeds']:out[f'T5_random_{label}_{seed}']={'spec':{'kind':'random','probability':lock['random_probability'][key],'seed':seed}}
        for kind,tid in [('duplicate','T7'),('sham','T8'),('zero','T9')]:out[f'{tid}_{label}_{kind}']={'spec':spec,'view':kind,'same_mask_as':f'T{3 if label=="accuracy" else 4}_{label}'}
        for k in range(5):out[f'T10_{label}_cap{k}']={'spec':spec,'cap':k}
    for family in ('audio_r','audio_no_r','acoustic','history'):out[f'ABL_{family}']={'spec':s[family]['spec']}
    # Quantity diagnostic: first K already-arrived valid windows, never future topK.
    for k in range(5):out[f'BUDGET_prefix{k}']={'spec':{'kind':'on'},'cap':k}
    return out
def prepare_controllers(controllers,specs):
    from utility_gate.gate_first_native.policy import GateBundle
    from utility_gate.gate_first_native.gate import freeze_proxy
    for v in specs.values():
        spec=v['spec']
        if 'checkpoint' in spec:controllers.model(spec['checkpoint'])
    controllers.old=GateBundle();controllers.proxy=freeze_proxy()
def run(role,shard=0,shards=1):
    if role=='val':rows=train_rows('val')
    else:
        from .locks import guard
        guard(role);rows=read_jsonl(ROOT/f'data/{role}.jsonl')
    gs=defaultdict(list)
    for row in rows:gs[row['source_id']].append(row)
    runtime=Runtime();loc=Localizer(runtime.device);controllers=Controllers();specs=variants();prepare_controllers(controllers,specs)
    directory=ROOT/('validation' if role=='val' else 'confirmation')/role;directory.mkdir(parents=True,exist_ok=True)
    # Warm-up kept in a separate artifact, never included as a scored result.
    first=next(iter(gs.values()))[0];execute(runtime,loc,first,{'kind':'on'},controllers)
    for i,(source,qs) in enumerate(list(gs.items())[shard::shards]):
        target=directory/f'{source}.jsonl'
        if target.exists():
            rows_existing=read_jsonl(target)
            if len(rows_existing)!=len(qs)*len(specs) or {x['strategy'] for x in rows_existing}!=set(specs):raise ValueError('partial source artifact')
            continue
        results=[];actions=[];masks={}
        # Rotate order without looking at scores to reduce timing/order bias.
        ordered=list(specs);shift=int(hashlib.sha256(source.encode()).hexdigest()[:8],16)%len(ordered);ordered=ordered[shift:]+ordered[:shift]
        for name in ordered:
            v=specs[name];z,meta,trace=execute(runtime,loc,qs[0],v['spec'],controllers,v.get('view','real'),v.get('cap'));masks[name]=meta['on_mask']
            for a in trace:actions.append({**a,'strategy':name,'role':role,'view':v.get('view','real')})
            for row in qs:
                score=runtime.score(row,z)
                if score['visual_tokens']!=256:raise AssertionError('different LLM budget')
                results.append({**row,**score,**meta,'strategy':name,'controller':v['spec'],'view':v.get('view','real'),'cap':v.get('cap'),
                    'R_A_sha256':sha256(RA),'V1_sha256':sha256(V1),'prompt_sha256':hashlib.sha256(qa_prompt(runtime,row).numpy().tobytes()).hexdigest(),
                    'pipeline_total_s':meta['frontend_wall_s']+score['llm_total_s'],'pipeline_ttft_s':meta['frontend_wall_s']+score['llm_ttft_s'],
                    'pipeline_excludes_gold_scoring':True,'actual_complete_generation':True,'GT_used':False,'window_count':4,
                    'source_frontend_amortized_across_QA':True,'rerun_independent_policy_frontend':True})
            print('ACTUAL QA',role,shard,i+1,source,name,sum(x['correct'] for x in results if x['strategy']==name),'/',len(qs),'k',meta['k_extra_images'],flush=True)
        for name,v in specs.items():
            if v.get('same_mask_as') and masks[name]!=masks[v['same_mask_as']]:raise AssertionError('second-view control changed selection mask')
        write_jsonl(target,results);write_jsonl(target.with_suffix('.actions.jsonl'),actions)
    journal('full conditional QA inference shard complete',role=role,shard=shard,shards=shards,variants=len(specs))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',required=True,choices=['val','confirmation_music','confirmation_valor']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.role,a.shard,a.shards)
