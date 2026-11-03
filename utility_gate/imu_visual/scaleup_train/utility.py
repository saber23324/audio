"""All sixteen actual complete-QA interventions for the fixed trained R_REAL."""
import argparse,json
import numpy as np
import torch
from utility_gate.gate_first_native.fusion import load_fusion
from .common import ROOT,rows,bits,write_json,write_jsonl,sha256,journal,status
from .features import load
from .runtime import Runtime,fuse_sequence

@torch.inference_mode()
def run(shard=0,shards=1):
    locks={v:json.loads((ROOT/f'locks/{v}.json').read_text()) for v in ('R_REAL','R_DUP','R_ZERO')};cp=locks['R_REAL'];assert sha256(cp['checkpoint'])==cp['sha256'];r=Runtime();m,_=load_fusion(cp['checkpoint'],r.device);assert not any(p.requires_grad for p in m.parameters())
    for row in [q for q in rows() if q['role'] in ('train','val')][shard::shards]:
        target=ROOT/'results/utility'/(row['qa_id'].replace(':','_')+'.jsonl')
        if target.exists():continue
        d=load(row);v=d['visual'].to(r.device).float();g=d['real'].to(r.device).float();base=d['base'].to(r.device);on=fuse_sequence(m,v,g,base);out=[]
        for code in range(16):
            mask=torch.tensor(bits(code),device=r.device,dtype=torch.bool);z=torch.where(mask[:,None,None],on,base);score=r.score(row,z);assert score['visual_tokens']==256
            out.append({**row,**score,'policy_code':code,'mask':mask.cpu().tolist(),'k':int(mask.sum()),'frozen_R_REAL_SHA':cp['sha256'],'actual_complete_QA_generation':True,'sum_single_window_utility':False,'offline_feature_cache_not_pipeline_cost':True})
        write_jsonl(target,out);print('ACTUAL ALL16 COMPLETE QA',row['qa_id'],flush=True)
    assert sha256(cp['checkpoint'])==cp['sha256'];journal('trained fixed R_REAL actual all16 utility shard complete',shard=shard,shards=shards,router_SHA=cp['sha256'])
def seal():
    manifest=rows();interactions=[];hashes={};summary={}
    for role in ('train','val'):
        selected=[r for r in manifest if r['role']==role]
        for row in selected:
            p=ROOT/'results/utility'/(row['qa_id'].replace(':','_')+'.jsonl');a=read(p);assert len(a)==16 and sorted(x['policy_code'] for x in a)==list(range(16));x={r['policy_code']:r for r in a};loss=np.array([x[m]['gold_nll'] for m in range(16)]);interactions.append({'qa_id':row['qa_id'],'role':role,'all_on_vs_sum_single_interaction_NLL':float((loss[15]-loss[0])-sum(loss[1<<t]-loss[0] for t in range(4)))});hashes[str(p)]=sha256(p)
        summary[role]={'qa':len(selected),'actual_complete_generations':16*len(selected),'sessions':len({r['session_id'] for r in selected})}
    write_jsonl(ROOT/'results/utility_interactions.jsonl',interactions);write_json(ROOT/'locks/utility_lock.json',{'router_SHA':json.loads((ROOT/'locks/R_REAL.json').read_text())['sha256'],'all_actual16_files':hashes,'roles':summary,'interaction_abs_mean':float(np.mean([abs(x['all_on_vs_sum_single_interaction_NLL']) for x in interactions]))});status('M4','true_utility_ready',**summary);journal('all16 true Utility checked complete and frozen before Gate optimizers',**summary)
def read(path):
    from .common import read_jsonl
    return read_jsonl(path)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--seal',action='store_true');a=p.parse_args();seal() if a.seal else run(a.shard,a.shards)
