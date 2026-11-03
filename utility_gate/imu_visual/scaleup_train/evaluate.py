import argparse,json
import torch
from utility_gate.gate_first_native.fusion import load_fusion
from .common import ROOT,rows,write_jsonl,journal
from .features import load
from .runtime import Runtime,fuse_sequence
from .gate import load_gate,from_payload,choose

def policies(row,index,device="cpu"):
    selected=json.loads((ROOT/'locks/validation_selection.json').read_text())['selected'];random={'masks':json.loads((ROOT/'locks/cohort_random_schedules.json').read_text())['schedules'][row['role']]};out=[('T0_V1','V1','none',[False]*4,None),('T1_REAL_always','R_REAL','real',[True]*4,None),('T8_DUP_trained_always','R_DUP','dup',[True]*4,None),('T9_ZERO_trained_always','R_ZERO','zero',[True]*4,None)];d=load(row);f=from_payload(d['features'],device)
    for name,spec in selected.items():
        model,_=load_gate(spec['checkpoint'],device);mask,scores=choose(model,f,spec['threshold']);spec={**spec,'cached_gate_scores':scores}
        for control,aux in [('', 'real'),('_same_mask_DUP','dup'),('_same_mask_ZERO','zero'),('_same_mask_WRONG','wrong')]:out.append((f'T2_{name}'+control,'R_REAL',aux,mask,spec))
        for seed in (1234,2234,3234):out.append((f'T4_random_{name}_{seed}','R_REAL','real',random['masks'][f'{name}_{seed}'][index],None))
    for name,aux in [('R_REAL_scale025','real'),('R_REAL_scale050','real'),('R_REAL_scratch','real'),('R_DUP_scratch','dup')]:out.append((name+'_always',name,aux,[True]*4,None))
    for variant,aux in [('R_REAL','real'),('R_DUP','dup'),('R_ZERO','zero')]:out.append(('Old_'+variant+'_always','Old_'+variant,aux,[True]*4,None))
    out.append(('T10_REAL_wrong_always','R_REAL','wrong',[True]*4,None))
    for k in range(5):out.append((f'Fixed_k_{k}','R_REAL','real',[i<k for i in range(4)],None))
    # Fixed predeclared physical rule, no future normalization/source labels.
    stats=f.imu;valid=f.valid
    cheap=[bool(valid[i]) and (float(stats[i,9:12].mean())>.20) for i in range(4)]
    out.append(('Cheap_gyro_std_0p20','R_REAL','real',cheap,None))
    old=ROOT.parent/'imu_gate_router_transfer_20261030_v3'
    oldspec=json.loads((old/'locks/validation_selection.json').read_text())['selected']
    for name,spec in oldspec.items():
        model,_=load_gate(spec['checkpoint'],device);mask,_=choose(model,f,spec['threshold'])
        out.append((f'Old_v3_gate_{name}','R_REAL','real',mask,None))
    return out
@torch.inference_mode()
def run(shard=0,shards=1,role="val"):
    r=Runtime();models={}
    for name in ('R_REAL','R_DUP','R_ZERO','R_REAL_scale025','R_REAL_scale050','R_REAL_scratch','R_DUP_scratch'):models[name]=load_fusion(json.loads((ROOT/f'locks/{name}.json').read_text())['checkpoint'],r.device)[0]
    for name in ('R_REAL','R_DUP','R_ZERO'):models['Old_'+name]=load_fusion(ROOT.parent/'imu_gate_router_transfer_20261030_v3'/f'checkpoints/{name}/selected.pt',r.device)[0]
    val=rows(role)
    for index in range(shard,len(val),shards):
        row=val[index];path=ROOT/'results/eval'/(row['qa_id'].replace(':','_')+'.jsonl')
        if path.exists():continue
        d=load(row);v=d['visual'].to(r.device).float();base=d['base'].to(r.device);records=[]
        for name,router,aux,mask,spec in policies(row,index,r.device):
            selected=torch.tensor(mask,device=r.device,dtype=torch.bool)
            if router=='V1':z=base
            else:
                g=(d['visual'] if aux=='dup' else d[aux]).to(r.device).float();z=fuse_sequence(models[router],v,g,base,selected)
            score=r.score(row,z);assert score['visual_tokens']==256
            records.append({**row,**score,'strategy':name,'mask':mask,'k':sum(mask),'ON_rate':sum(mask)/4,'router':router,'aux':aux,'checkpoint_policy':spec,'full_QA_rerun':True,'not_independent_confirmation':role!='holdout','input_cost_feature_cached_for_QA_validation':True,'visual_tokens_actual':256})
        write_jsonl(path,records);print('ACTUAL FROZEN POLICY QA',row['qa_id'],len(records),flush=True)
    journal('frozen trained Router/Gate validation full QA reruns completed',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--role',choices=['val','holdout'],default='val');a=p.parse_args();run(a.shard,a.shards,a.role)
