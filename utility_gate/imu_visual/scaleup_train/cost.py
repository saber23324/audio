import argparse,json,hashlib,random
import torch
from utility_gate.gate_first_native.fusion import load_fusion
from .common import ROOT,rows,write_jsonl,journal
from .features import load
from .runtime import Runtime
from .evaluate import policies
from .gate import load_gate
from .pipeline import run as pipeline
from .native_paid import run as native_pipeline

class CheapRule(torch.nn.Module):
    def step(self,f,past_actions,index):
        return (f.imu[:,9:12].mean(-1)>.20).float()*20-10


@torch.inference_mode()
def run(shard=0,shards=1,role="val"):
    r=Runtime();models={name:load_fusion(json.loads((ROOT/f'locks/{name}.json').read_text())['checkpoint'],r.device)[0] for name in ('R_REAL','R_DUP','R_ZERO','R_REAL_scale025','R_REAL_scale050','R_REAL_scratch','R_DUP_scratch')};selection=json.loads((ROOT/'locks/validation_selection.json').read_text())['selected'];gates={name:load_gate(spec['checkpoint'],r.device)[0] for name,spec in selection.items()};val=rows(role);allrows=rows();jobs=[]
    for name in ('R_REAL','R_DUP','R_ZERO'):models['Old_'+name]=load_fusion(ROOT.parent/'imu_gate_router_transfer_20261030_v3'/f'checkpoints/{name}/selected.pt',r.device)[0]
    for index in range(shard,len(val),shards):
        row=val[index]
        for strategy,router,aux,mask,spec in policies(row,index,r.device):jobs.append((index,strategy,router,aux,mask,spec,4))
        for strategy,aux in [('N0_RGB','none'),('N1_RGB_IMU','real')]:jobs.append((index,strategy,'NATIVE',aux,[aux=='real']*4,None,4))
        for name,spec in selection.items():
            for cap in range(4):jobs.append((index,f'BUDGET_{name}_cap{cap}','R_REAL','real',None,spec,cap))
    random.Random(1234+shard).shuffle(jobs)
    # One warmup before random measured jobs. No feature/plot cache is retained.
    if shard>=len(val):return
    first=rows('val')[0];pipeline(r,models['R_REAL'],first,forced_mask=[False]*4,cap=4)
    for index,strategy,router,aux,mask,spec,cap in jobs:
        row=val[index];target=ROOT/'results/cost'/(row['qa_id'].replace(':','_')+'_'+strategy+'.json')
        if target.exists():continue
        donors=[q for q in allrows if q['role']==row['role'] and q['source_id']!=row['source_id'] and q['class_id']!=row['class_id']];donor=min(donors,key=lambda q:hashlib.sha256((row['qa_id']+q['qa_id']).encode()).hexdigest());gate=None
        if spec is not None:
            key='accuracy' if strategy.startswith('T2_accuracy') or strategy.startswith('BUDGET_accuracy') else 'cost';gate=gates[key]
        from .pipeline import decode_rgb
        import os
        os.environ.pop('IMU_SCALEUP_CACHE_ONLY',None)
        for t in row['timestamps_s']:decode_rgb(row,t)
        os.environ['IMU_SCALEUP_CACHE_ONLY']='1'
        if strategy=='Cheap_gyro_std_0p20':gate=CheapRule().to(r.device)
        actual={'Fusion':0,'neural_decision':0}
        def fusion_hook(*args):actual['Fusion']+=1
        def decision_hook(*args):actual['neural_decision']+=1
        handles=[models.get(router,models['R_REAL']).register_forward_hook(fusion_hook)]
        neural=gate is not None and hasattr(gate,'core')
        if neural:handles.append(gate.core.decision.register_forward_hook(decision_hook))
        try:
            result=native_pipeline(r,row,aux) if router=='NATIVE' else pipeline(r,models.get(router,models['R_REAL']),row,aux='real' if aux=='none' else aux,gate=gate,threshold=spec['threshold'] if spec else .5,forced_mask=mask,cap=cap,donor=donor)
        finally:
            for handle in handles:handle.remove()
        assert actual['Fusion']==(0 if router=='NATIVE' else result['k'])
        assert actual['neural_decision']==(result['counts']['Gate'] if neural else 0)
        result.update(actual_module_forward_hooks=actual,neural_Gate=neural)

        from .common import write_json
        write_json(target,{**row,**result,'strategy':strategy,'router':router,'cap':cap,'development_only':role!='holdout','model_loading_not_timed':True,'gold_not_in_Gate_features':True});print('ACTUAL PAID PIPELINE',shard,row['qa_id'],strategy,result['k'],round(result['pipeline_generation_s'],4),flush=True)
    journal('paid causal original-video-decode pipeline cost shard completed',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--role',choices=['val','holdout'],default='val');a=p.parse_args();run(a.shard,a.shards,a.role)
