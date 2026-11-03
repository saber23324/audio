import argparse,json
import torch
from utility_gate.gate_first_native.fusion import load_fusion
from .common import ROOT,rows,write_jsonl,journal
from .features import load
from .runtime import Runtime,fuse_sequence
from .gate import load_gate,from_payload,choose

def policies(row,index):
    selected=json.loads((ROOT/'locks/validation_selection.json').read_text())['selected'];random=json.loads((ROOT/'locks/random_schedules.json').read_text());out=[('T0_V1','V1','none',[False]*4,None),('T1_REAL_always','R_REAL','real',[True]*4,None),('T8_DUP_trained_always','R_DUP','dup',[True]*4,None),('T9_ZERO_trained_always','R_ZERO','zero',[True]*4,None)];d=load(row);f=from_payload(d['features'])
    for name,spec in selected.items():
        model,_=load_gate(spec['checkpoint']);mask,scores=choose(model,f,spec['threshold'])
        for control,aux in [('', 'real'),('_same_mask_DUP','dup'),('_same_mask_ZERO','zero'),('_same_mask_WRONG','wrong')]:out.append((f'T2_{name}'+control,'R_REAL',aux,mask,spec))
        for seed in (1234,2234,3234):out.append((f'T4_random_{name}_{seed}','R_REAL','real',random['masks'][f'{name}_{seed}'][index],None))
    return out
@torch.inference_mode()
def run(shard=0,shards=1):
    r=Runtime();models={}
    for name in ('R_REAL','R_DUP','R_ZERO'):models[name]=load_fusion(json.loads((ROOT/f'locks/{name}.json').read_text())['checkpoint'],r.device)[0]
    val=rows('val')
    for index in range(shard,len(val),shards):
        row=val[index];path=ROOT/'results/eval'/(row['qa_id'].replace(':','_')+'.jsonl')
        if path.exists():continue
        d=load(row);v=d['visual'].to(r.device).float();base=d['base'].to(r.device);records=[]
        for name,router,aux,mask,spec in policies(row,index):
            selected=torch.tensor(mask,device=r.device,dtype=torch.bool)
            if router=='V1':z=base
            else:
                g=(d['visual'] if aux=='dup' else d[aux]).to(r.device).float();z=fuse_sequence(models[router],v,g,base,selected)
            score=r.score(row,z);assert score['visual_tokens']==256
            records.append({**row,**score,'strategy':name,'mask':mask,'k':sum(mask),'ON_rate':sum(mask)/4,'router':router,'aux':aux,'checkpoint_policy':spec,'full_QA_rerun':True,'not_independent_confirmation':True,'input_cost_feature_cached_for_QA_validation':True,'visual_tokens_actual':256})
        write_jsonl(path,records);print('ACTUAL FROZEN POLICY QA',row['qa_id'],len(records),flush=True)
    journal('frozen trained Router/Gate validation full QA reruns completed',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.shard,a.shards)
