import argparse,json,hashlib,time
import numpy as np
import torch
from PIL import Image
from utility_gate.imu_visual.causal_imu_window import cheap_stats
from utility_gate.imu_visual.render_imu_plot import render_imu_plot
from utility_gate.grounded_dual_image.native import prepare
from .common import ROOT,rows,feature_path,write_json,write_jsonl,sha256,journal,status
from .data import sensor_window
from .runtime import Runtime,prompt

def cheap_features(visual,stats,valid):
    # Exact old current RGB + strictly previous RGB mean interface (7168).
    v=torch.stack([frame.float().mean(0) for frame in visual]);previous=torch.stack([torch.stack([v[j] for j in range(i)]).mean(0) if i else torch.zeros_like(v[0]) for i in range(len(v))]);idx=torch.arange(4,dtype=torch.float32);times=torch.tensor([2.,4.,6.,8.]);dt=torch.tensor([0.,2.,2.,2.]);past_valid=torch.tensor([float(valid[:i].float().mean()) if i else 0. for i in range(4)])
    return {'visual':torch.cat((v,previous),-1),'imu':torch.as_tensor(stats,dtype=torch.float32),'valid':valid.bool(),'context':torch.stack((times.log1p(),dt.log1p(),past_valid,(idx+1).log1p()),-1)}
def load(row):return torch.load(feature_path(row),map_location='cpu',weights_only=False)

@torch.inference_mode()
def run(shard=0,shards=1,holdout=False):
    if holdout:assert (ROOT/'locks/pretest_lock.json').exists(),'sealed cohort cannot be scored before policy lock'
    allrows=rows();selected=[q for q in allrows if (q['role']=='holdout')==holdout];r=Runtime();cfg=json.loads((ROOT/'locks/IMU_PLOT_V1.json').read_text())
    for i,row in enumerate(selected[shard::shards]):
        target=feature_path(row);npth=ROOT/'results/native'/(row['qa_id'].replace(':','_')+'.jsonl')
        if target.exists() and (row['role'] in ('train','holdout') or npth.exists()):continue
        before=dict(r.calls);fr=json.loads((ROOT/f'data/frames_{row["session_id"]}.json').read_text());rgb=[np.array(Image.open(fr[str(t)]['path']).convert('RGB')) for t in row['timestamps_s']]
        donors=[q for q in allrows if q['role']==row['role'] and q['source_id']!=row['source_id'] and q['class_id']!=row['class_id']];donor=min(donors,key=lambda q:hashlib.sha256((row['qa_id']+q['qa_id']).encode()).hexdigest());charts={k:[] for k in (['real','zero'] if row['role']=='train' else ['real','zero','wrong'])};metadata={k:[] for k in charts};stats=[];valid=[]
        for t,td in zip(row['timestamps_s'],donor['timestamps_s']):
            w=sensor_window(row,t);valid.append(w.valid);stats.append(cheap_stats(w));dw=sensor_window(donor,td) if 'wrong' in charts else w
            for name,ww in [('real',w),('zero',w),('wrong',dw)]:
                if name not in charts:continue
                image,meta=render_imu_plot(ww,cfg,zero=name=='zero');charts[name].append(np.array(image));metadata[name].append(meta)
        v=torch.stack([r.visual_frame(f) for f in rgb]);base=r.vision_latents(v)
        aux={k:torch.stack([r.visual_frame(f) for f in fs]) for k,fs in charts.items()};f={k:x.cpu() for k,x in cheap_features(v,np.array(stats),torch.tensor(valid)).items()};payload={'qa_id':row['qa_id'],'visual':v.cpu(),'base':base.cpu(),**{k:value.cpu() for k,value in aux.items()},'features':f,'plot_metadata':metadata,'donor_qa_id':donor['qa_id'],'RGB_capture_times':[fr[str(t)]['capture_s'] for t in row['timestamps_s']],'V1_SHA':sha256(__import__('utility_gate.imu_visual.scaleup_train.common',fromlist=['V1']).V1),'frozen_model':True,'no_audio_frontend':True}
        if not target.exists():
            temporary=target.with_suffix('.pt.inprogress');torch.save(payload,temporary);temporary.replace(target)
        results=[]
        for name,view in ([('N0_RGB','rgb'),('N1_RGB_IMU','real')] if row['role']=='val' else []):
            if holdout and view=='v1':continue  # Formal compressed holdout V1 runs only in paid primary pass.
            hooks=dict(r.calls);start=time.perf_counter()
            if view=='v1':score=r.score(row,base);native_tokens=256
            else:
                ims=[]
                for t in range(4):
                    if view=='only':ims.append(charts['real'][t]);continue
                    ims.append(rgb[t])
                    if view!='rgb':ims.append(rgb[t] if view=='dup' else charts[view][t])
                counts=[2]*4 if view in ('real','dup','zero','wrong') else [1]*4;ids=prompt(r,row,True,counts,only=view=='only');embeds,views,meta=prepare(r,ids,ims);score=r.answer(ids,views,row['answer_text'],prepared=embeds,mcqa=True);native_tokens=score['visual_tokens']
            results.append({**row,**score,'strategy':name,'tokens_per_frame':64 if view=='v1' else None,'visual_tokens_actual':native_tokens,'actual_forward_hooks':{n:r.calls[n]-hooks[n] for n in hooks},'elapsed_s':time.perf_counter()-start,'new_model_training':False,'feature_encoding_not_in_this_condition_time':True})
        if results:write_jsonl(npth,results)
        print('FROZEN CACHE + REQUIRED NATIVE',shard,i+1,row['qa_id'],flush=True)
    journal('frozen train caches and complete val/holdout native RGB/IMU references; prior extra diagnostics preserved as subsets',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--holdout',action='store_true');a=p.parse_args();
    if a.holdout:assert (ROOT/'locks/pretest_lock.json').exists()
    run(a.shard,a.shards,a.holdout)
