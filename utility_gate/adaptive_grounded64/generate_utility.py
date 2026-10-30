"""Actual add-one-grounded interventions: no whole-video utility replication."""
import argparse,json,time
from collections import defaultdict
import numpy as np
import torch
from utility_gate.common import write_jsonl
from .common import ROOT,REPO,train_rows,read_config,sha256,fingerprint,write_json,journal,status
from .features import load_rgb
from .prepare_av import grounded_path,load_av
from .fuser64 import fuse,dependencies
from .runtime import Runtime

def label_dependencies():
    return {'action':dependencies(),'label_code_sha256':sha256(__file__),
        'inference_adapter_sha256':sha256(REPO/'utility_gate/adaptive_grounded64/runtime.py')}

def run(role,shard=0,shards=1):
    if role not in ('gate_train','gate_val'):raise ValueError('only legal development labels permitted')
    rows=train_rows(role);groups=defaultdict(list)
    for row in rows:groups[row['source_id']].append(row)
    dep=label_dependencies();dep_key=fingerprint(dep);beta=dep['action']['beta'];r=Runtime();r.load_new_router()
    directory=ROOT/f'labels/per_source/{role}';directory.mkdir(parents=True,exist_ok=True)
    for j,(source,qs) in enumerate(list(groups.items())[shard::shards]):
        path=directory/f'{source}.jsonl'
        if path.exists():
            from .common import read_jsonl
            existing=read_jsonl(path)
            if not existing or any(x['label_dependency_key']!=dep_key for x in existing):raise ValueError('utility labels stale; cannot reuse')
            continue
        rgb=load_rgb(qs[0]);g=torch.load(grounded_path(source),map_location='cpu',weights_only=False)
        z,p=r.vision_latents(rgb['visual']);zg,pg=r.vision_latents(g['visual_grounded']);action=fuse(z,zg,on=True,beta=beta)
        baseline={row['question_id']:r.score(row,z) for row in qs};records=[]
        for index,t in enumerate(rgb['timestamps']):
            intervention=z.clone();intervention[index]=action[index]
            for row in qs:
                base=baseline[row['question_id']];altered=r.score(row,intervention)
                utility=float(base['gold_nll']-altered['gold_nll'])
                records.append({'source_id':source,'question_id':row['question_id'],'dataset':row['dataset'],'source_role':role,
                    'frame_index':index,'timestamp_s':float(t),'question':row['question'],'answer_text':row['answer_text'],
                    'qa_type':row.get('qa_type',row.get('question_type')), 'base_gold_nll':base['gold_nll'],'add_one_gold_nll':altered['gold_nll'],
                    'utility_add_one':utility,'base_correct':base['correct'],'add_one_correct':altered['correct'],
                    'correctness_delta':int(altered['correct'])-int(base['correct']),
                    'base_generated_ids':base['generated_ids'],'add_one_generated_ids':altered['generated_ids'],
                    'map_valid':bool(g['map_valid'][index]),'uncertain':abs(utility)<=.001,
                    'only_intervened_frame':index,'all_other_frames':'RGB_new_stage1',
                    'tokens_per_frame':64,'label_dependency_key':dep_key,'action_dependency':dep['action']['fingerprint']})
            print('utility',role,shard,j+1,source,index+1,len(z),flush=True)
        write_jsonl(path,records)
    write_json(ROOT/f'labels/worker_{role}_{shard}.json',{'role':role,'shard':shard,'shards':shards,'label_dependencies':dep,'dependency_key':dep_key,'completed':True})
    journal('real add-one utility shard complete',role=role,shard=shard,shards=shards)

def combine():
    from .common import read_jsonl
    dep=label_dependencies();key=fingerprint(dep)
    summary={}
    for role in ('gate_train','gate_val'):
        rows=train_rows(role);sources=sorted({x['source_id'] for x in rows});records=[];aggregated=[]
        for source in sources:
            a=read_jsonl(ROOT/f'labels/per_source/{role}/{source}.jsonl')
            if any(r['label_dependency_key']!=key for r in a):raise ValueError('dependency mismatch')
            expected_qa={r['question_id'] for r in rows if r['source_id']==source};group=defaultdict(list)
            for r in a:group[r['frame_index']].append(r)
            for index,values in group.items():
                if {r['question_id'] for r in values}!=expected_qa:raise ValueError('incomplete intervention QA set')
                u=float(np.median([r['utility_add_one'] for r in values]))
                aggregated.append({'source_id':source,'frame_index':index,'timestamp_s':values[0]['timestamp_s'],
                    'source_role':role,'utility_target':u,'utility_mean':float(np.mean([r['utility_add_one'] for r in values])),
                    'n_qa':len(values),'correctness_delta_sum':sum(r['correctness_delta'] for r in values),'uncertain':abs(u)<=.001,
                    'label_dependency_key':key,'action_dependency':dep['action']['fingerprint']})
            records.extend(a)
        suffix=role.removeprefix('gate_');write_jsonl(ROOT/f'labels/utility_add_one_{suffix}.jsonl',records)
        write_jsonl(ROOT/f'labels/utility_aggregated_{suffix}.jsonl',aggregated)
        targets=np.array([r['utility_target'] for r in aggregated])
        summary[role]={'sources':len(sources),'QA_frame_interventions':len(records),'windows':len(aggregated),
            'beneficial_gt_001':int((targets>.001).sum()),'harmful_lt_minus001':int((targets<-.001).sum()),
            'uncertain':int((abs(targets)<=.001).sum()),'quantiles':dict(zip(['min','q25','median','q75','max'],np.quantile(targets,[0,.25,.5,.75,1]).tolist()))}
    write_json(ROOT/'locks/utility_label_lock.json',{'label_dependencies':dep,'dependency_key':key,
        'files':{str(p):sha256(p) for p in sorted((ROOT/'labels').glob('utility_*.jsonl'))},'distribution':summary})
    status('M3','completed',distribution=summary);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['gate_train','gate_val']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--combine',action='store_true');a=p.parse_args()
    if a.combine:combine()
    else:run(a.role,a.shard,a.shards)

