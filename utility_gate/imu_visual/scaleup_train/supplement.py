"""Fixed-k random, paid neural Gate, and native full-pipeline diagnostics."""
import argparse,json,random
import numpy as np
import torch
from utility_gate.gate_first_native.fusion import load_fusion
from utility_gate.runtime import clocked,synchronize
from utility_gate.grounded_dual_image.native import prepare
from utility_gate.imu_visual.render_imu_plot import render_imu_plot
from .common import ROOT,rows,write_json,sha256,journal
from .runtime import Runtime,prompt
from .data import sensor_window
from .gate import load_gate
from . import pipeline as p

@torch.inference_mode()
def native(r,row,aux):
    before=dict(r.calls);synchronize();torch.cuda.reset_peak_memory_stats();start=__import__('time').perf_counter();images=[];timings={'video_decode':0.,'IMU_crop_stats':0.,'plot_render':0.,'native_prepare':0.};renders=0;cfg=json.loads((ROOT/'locks/IMU_PLOT_V1.json').read_text())
    for t in row['timestamps_s']:
        (rgb,actual,index),s=clocked(lambda:p.decode_rgb(row,t));timings['video_decode']+=s;images.append(rgb)
        if aux=='none':continue
        if aux=='dup':images.append(rgb);continue
        w,s=clocked(lambda:sensor_window(row,t));timings['IMU_crop_stats']+=s
        (image,meta),s=clocked(lambda:render_imu_plot(w,cfg));timings['plot_render']+=s;renders+=1;images.append(np.array(image))
    ids=prompt(r,row,True,[1 if aux=='none' else 2]*4);(embeds,views,meta),elapsed=clocked(lambda:prepare(r,ids,images));timings['native_prepare']=elapsed;score=r.answer(ids,views,row['answer_text'],prepared=embeds,mcqa=True);synchronize();wall=__import__('time').perf_counter()-start;hooks={k:r.calls[k]-before[k] for k in before};assert hooks['SigLIP_images']==len(images) and hooks['V1']==0 and hooks['R_IMU']==0
    return {**score,'k':0 if aux=='none' else 4,'mask':[aux!='none']*4,'ON_rate':0. if aux=='none' else 1.,'stage_seconds':timings,'counts':{'render':renders,'extra_encode':0 if aux=='none' else 4,'Gate':0,'Fusion':0},'actual_forward_hooks':hooks,'pipeline_generation_s':wall-score['nll_evaluation_extra_s'],'pipeline_TTFT_derived_s':wall-score['nll_evaluation_extra_s']-score['llm_total_s']+score['llm_ttft_s'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'feature_cache':False,'plot_cache':False,'native_uncompressed_reference':True,'input_media':'actual compressed video decoded; network/model loading excluded','aux':aux}

@torch.inference_mode()
def run(shard=0,shards=1):
    r=Runtime();router=load_fusion(json.loads((ROOT/'locks/R_REAL.json').read_text())['checkpoint'],r.device)[0];r.calls['R_IMU']=0;r.calls['Gate_decision']=0
    def bump(name):r.calls[name]+=1
    router.register_forward_pre_hook(lambda m,args:bump('R_IMU'));gates={};catalog=json.loads((ROOT/'locks/Gate_training_catalog.json').read_text())
    for g in catalog:
        cp=min(g['all_checkpoints'],key=lambda c:c['val_training_objective']);gates[g['name']]=load_gate(cp['path'],r.device)[0];gates[g['name']].core.decision.register_forward_pre_hook(lambda m,args:bump('Gate_decision'))
    val=rows('val');directory=ROOT/'results/supplement';directory.mkdir(exist_ok=True);jobs=[]
    for i in range(shard,len(val),shards):
        for k in (1,2,3):
            for seed in (1234,2234,3234):jobs.append((i,f'FIXED_K{k}_RANDOM_{seed}','random',k,seed))
        for g in gates:jobs.append((i,'PAID_GATE_'+g,'gate',4,g))
        for name,aux in [('N0_native_pipeline','none'),('N1_native_pipeline','real'),('N2_native_pipeline','dup')]:jobs.append((i,name,'native',4,aux))
    random.Random(1234+shard).shuffle(jobs);p.run(r,router,val[shard],forced_mask=[False]*4);native(r,val[shard],'real')
    for i,name,kind,k,key in jobs:
        row=val[i];target=directory/(row['qa_id'].replace(':','_')+'_'+name+'.json')
        if target.exists():continue
        if kind=='native':out=native(r,row,key)
        elif kind=='gate':
            out=p.run(r,router,row,gate=gates[key],threshold=.5);assert out['actual_forward_hooks']['Gate_decision']==out['counts']['Gate']
        else:
            mask=np.zeros(4,dtype=bool);mask[np.random.default_rng(int(key)+10000*i).choice(4,k,replace=False)]=True;out=p.run(r,router,row,forced_mask=mask.tolist(),cap=k);assert out['k']==k
        if kind!='native':assert out['actual_forward_hooks']['R_IMU']==out['k']
        write_json(target,{**row,**out,'strategy':name,'diagnostic':'fixed-k random / threshold0.5 paid Gate / native cost; no policy retuning or independent-test claim','seed':key if kind=='random' else None,'max_extra_budget':k,'raw_QA_rerun':True,'development_only':True});print('ADDITIONAL FULL PIPELINE',shard,row['qa_id'],name,round(out['pipeline_generation_s'],4),flush=True)
    journal('mandatory nondegenerate fixed-budget, paid neural Gate and native cost diagnostics completed',shard=shard,shards=shards)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1);x=a.parse_args();run(x.shard,x.shards)
