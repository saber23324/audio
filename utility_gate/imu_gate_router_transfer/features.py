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
    v=visual.float().mean(1);previous=torch.stack([v[:i].mean(0) if i else torch.zeros_like(v[0]) for i in range(len(v))]);idx=torch.arange(4,dtype=torch.float32);times=torch.tensor([2.,4.,6.,8.]);dt=torch.tensor([0.,2.,2.,2.]);past_valid=torch.tensor([float(valid[:i].float().mean()) if i else 0. for i in range(4)])
    return {'visual':torch.cat((v,previous),-1),'imu':torch.as_tensor(stats,dtype=torch.float32),'valid':valid.bool(),'context':torch.stack((times.log1p(),dt.log1p(),past_valid,(idx+1).log1p()),-1)}
def load(row):return torch.load(feature_path(row),map_location='cpu',weights_only=False)

@torch.inference_mode()
def run(shard=0,shards=1):
    allrows=rows();r=Runtime();cfg=json.loads((ROOT/'locks/IMU_PLOT_V1.json').read_text())
    for i,row in enumerate(allrows[shard::shards]):
        target=feature_path(row);npth=ROOT/'results/native'/(row['qa_id'].replace(':','_')+'.jsonl')
        if target.exists() and npth.exists():continue
        before=dict(r.calls);fr=json.loads((ROOT/f'data/frames_{row["session_id"]}.json').read_text());rgb=[np.array(Image.open(fr[str(t)]['path']).convert('RGB')) for t in row['timestamps_s']]
        donors=[q for q in allrows if q['role']==row['role'] and q['source_id']!=row['source_id'] and q['class_id']!=row['class_id']];donor=min(donors,key=lambda q:hashlib.sha256((row['qa_id']+q['qa_id']).encode()).hexdigest());charts={'real':[],'zero':[],'wrong':[]};metadata={k:[] for k in charts};stats=[];valid=[]
        for t,td in zip(row['timestamps_s'],donor['timestamps_s']):
            w=sensor_window(row,t);valid.append(w.valid);stats.append(cheap_stats(w));dw=sensor_window(donor,td)
            for name,ww in [('real',w),('zero',w),('wrong',dw)]:
                image,meta=render_imu_plot(ww,cfg,zero=name=='zero');charts[name].append(np.array(image));metadata[name].append(meta)
        v=torch.stack([r.visual_frame(f) for f in rgb]);base=r.vision_latents(v)
        aux={k:torch.stack([r.visual_frame(f) for f in fs]) for k,fs in charts.items()};f=cheap_features(v.cpu(),np.array(stats),torch.tensor(valid));payload={'qa_id':row['qa_id'],'visual':v.cpu(),'base':base.cpu(),'real':aux['real'].cpu(),'zero':aux['zero'].cpu(),'wrong':aux['wrong'].cpu(),'features':f,'plot_metadata':metadata,'donor_qa_id':donor['qa_id'],'RGB_capture_times':[fr[str(t)]['capture_s'] for t in row['timestamps_s']],'V1_SHA':sha256(__import__('utility_gate.imu_gate_router_transfer.common',fromlist=['V1']).V1),'frozen_model':True,'no_audio_frontend':True}
        torch.save(payload,target)
        results=[]
        for name,view in [('N0_RGB','rgb'),('N1_RGB_IMU','real'),('N2_RGB_DUP','dup'),('N3_RGB_ZERO','zero'),('N4_RGB_WRONG','wrong'),('N5_V1_64','v1'),('N6_IMU_only_diagnostic','only')]:
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
        write_jsonl(npth,results);print('FEATURES + NATIVE FULL QA',shard,i+1,row['qa_id'],flush=True)
    journal('frozen per-frame RGB/IMU features and all native endpoints actually evaluated',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.shard,a.shards)
