import argparse,json,hashlib,random
import torch
from utility_gate.gate_first_native.fusion import load_fusion
from .common import ROOT,rows,write_jsonl,journal
from .features import load
from .runtime import Runtime
from .evaluate import policies
from .gate import load_gate
from .pipeline import run as pipeline

@torch.inference_mode()
def run(shard=0,shards=1):
    r=Runtime();models={name:load_fusion(json.loads((ROOT/f'locks/{name}.json').read_text())['checkpoint'],r.device)[0] for name in ('R_REAL','R_DUP','R_ZERO')};selection=json.loads((ROOT/'locks/validation_selection.json').read_text())['selected'];gates={name:load_gate(spec['checkpoint'],r.device)[0] for name,spec in selection.items()};val=rows('val');allrows=rows();jobs=[]
    for index in range(shard,len(val),shards):
        row=val[index]
        for strategy,router,aux,mask,spec in policies(row,index):jobs.append((index,strategy,router,aux,mask,spec,4))
        for name,spec in selection.items():
            for cap in range(4):jobs.append((index,f'BUDGET_{name}_cap{cap}','R_REAL','real',None,spec,cap))
    random.Random(1234+shard).shuffle(jobs)
    # One warmup before random measured jobs. No feature/plot cache is retained.
    first=val[shard];pipeline(r,models['R_REAL'],first,forced_mask=[False]*4,cap=4)
    for index,strategy,router,aux,mask,spec,cap in jobs:
        row=val[index];target=ROOT/'results/cost'/(row['qa_id'].replace(':','_')+'_'+strategy+'.json')
        if target.exists():continue
        donors=[q for q in allrows if q['role']==row['role'] and q['source_id']!=row['source_id'] and q['class_id']!=row['class_id']];donor=min(donors,key=lambda q:hashlib.sha256((row['qa_id']+q['qa_id']).encode()).hexdigest());gate=None
        if spec is not None:
            key='accuracy' if strategy.startswith('T2_accuracy') or strategy.startswith('BUDGET_accuracy') else 'cost';gate=gates[key]
        result=pipeline(r,models.get(router,models['R_REAL']),row,aux='real' if aux=='none' else aux,gate=gate,threshold=spec['threshold'] if spec else .5,forced_mask=mask,cap=cap,donor=donor)
        from .common import write_json
        write_json(target,{**row,**result,'strategy':strategy,'router':router,'cap':cap,'development_only':True,'model_loading_not_timed':True,'gold_not_in_Gate_features':True});print('ACTUAL PAID PIPELINE',shard,row['qa_id'],strategy,result['k'],round(result['pipeline_generation_s'],4),flush=True)
    journal('paid causal original-video-decode pipeline cost shard completed',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.shard,a.shards)
