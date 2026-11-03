"""Post-primary chart-only controls; never used for selecting a new model."""
import argparse,json,hashlib,time
import numpy as np
import torch
from utility_gate.grounded_dual_image.native import prepare
from .common import ROOT,read_jsonl,write_jsonl,journal
from .native_imu_probe import prompt,source_window
from .render_imu_plot import render_imu_plot
from .runtime import VisionRuntime

@torch.inference_mode()
def run(shard=0,shards=1):
    rows=read_jsonl(ROOT/'M1_data/utd_probe_manifest.jsonl');cfg=json.loads((ROOT/'M2_frozen_imu_plots/utd_renderer_lock.json').read_text())['config'];r=VisionRuntime()
    out=ROOT/'M2_frozen_imu_plots/utd_post_primary_controls';out.mkdir(exist_ok=True)
    journal('post-primary diagnostic controls start; original108x25 results already inspected; not preregistered confirmation',shard=shard)
    for row in rows[shard::shards]:
        target=out/(row['source_id']+'.jsonl')
        if target.exists():continue
        candidates=[q for q in rows if q['class_id']!=row['class_id'] and q['imu_sensor_positions']==row['imu_sensor_positions'] and q['participant_id']!=row['participant_id']]
        donor=min(candidates,key=lambda q:hashlib.sha256((row['source_id']+q['source_id']).encode()).hexdigest());results=[]
        for name,view,W in [('A_prior_no_image','prior',2.),('A_plot_zero_W2','zero',2.),('A_plot_wrong_W2','wrong',2.),('A_plot_zero_W4','zero',4.),('A_plot_wrong_W4','wrong',4.)]:
            before=dict(r.calls);start=time.perf_counter();meta=None
            if view=='prior':
                ids=prompt(r,row,0,only=True);score=r.answer(ids,[],row['answer_text'],mcqa=True)
            else:
                w=source_window(donor if view=='wrong' else row,W);chart,meta=render_imu_plot(w,cfg,zero=view=='zero');ids=prompt(r,row,1,only=True)
                embeds,views,native=prepare(r,ids,[np.asarray(chart)]);score=r.answer(ids,views,row['answer_text'],prepared=embeds,mcqa=True)
            results.append({**row,**score,'variant':name,'post_primary_development_diagnostic':True,'not_preregistered_confirmation':True,'plot_metadata':meta,'donor_source':donor['source_id'] if view=='wrong' else None,'actual_forward_hooks':{k:r.calls[k]-before[k] for k in before},'measured_eval_s':time.perf_counter()-start,'no_training':True,'no_audio_used':True})
        write_jsonl(target,results);print('APPENDIX',row['source_id'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.shard,a.shards)
