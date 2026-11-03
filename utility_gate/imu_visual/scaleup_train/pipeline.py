"""Real causal paid pipeline: compressed video decode -> cheap Gate -> optional plot."""
import json,time
import numpy as np
import torch
import av
from utility_gate.runtime import clocked,synchronize
from utility_gate.imu_visual.intake import SOURCES
from utility_gate.imu_visual.causal_imu_window import cheap_stats
from utility_gate.imu_visual.render_imu_plot import render_imu_plot
from .common import ROOT,rows
from .data import sensor_window
from .range_io import RangeFile
from .gate import IMUFeatures

def decode_rgb(row,t):
    # Seeking uses CFR presentation index only after recovering the original
    # camera clock; selection is always by capture_time<=t.
    capture=np.load(row['RGB_clock_path'])['times'];index=int(np.searchsorted(capture,t,side='right')-1)
    if index<0:raise ValueError('no arrived camera frame')
    entry=next(x for x in json.loads((ROOT/'data/video_index.json').read_text()) if x['name'].split('/')[0]==row['session_id'])
    with av.open(RangeFile(SOURCES['holo_video'],entry['size'],entry['offset_data'])) as c:
        stream=c.streams.video[0];rate=float(stream.average_rate);c.seek(max(0,int(index/rate/stream.time_base)),stream=stream,backward=True);selected=None
        for f in c.decode(video=0):
            n=int(round(float(f.pts*f.time_base)*rate))
            if n>index:break
            selected=(f.to_ndarray(format='rgb24'),float(capture[n]),n)
            if n==index:break
    if selected is None or selected[2]!=index:raise ValueError('exact causal decoded frame missing')
    assert selected[1]<=t;return selected

def one_feature(current,past,stats,valid,past_valid,index,device):
    prior=torch.stack(past).mean(0) if past else torch.zeros_like(current);context=torch.tensor([[np.log1p(2*(index+1)),np.log1p(0 if index==0 else 2),sum(past_valid)/max(index,1),np.log1p(index+1)]],device=device,dtype=torch.float32)
    return IMUFeatures(torch.cat((current,prior))[None].to(device),torch.as_tensor(stats,device=device,dtype=torch.float32)[None],context,torch.tensor([valid],device=device))

def safe_sensor(row,t):
    try:return sensor_window(row,t)
    except (ValueError,KeyError,OSError,IndexError):return None

@torch.inference_mode()
def run(r,router,row,aux='real',gate=None,threshold=.5,forced_mask=None,cap=4,donor=None):
    cfg=json.loads((ROOT/'locks/IMU_PLOT_V1.json').read_text());before=dict(r.calls);synchronize()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();history=[];past_valid=[];zs=[];mask=[];probabilities=[];perwindow=[];counts={'render':0,'extra_encode':0,'Gate':0,'Fusion':0};timing={k:0. for k in ('video_decode','RGB_vision','V1','IMU_crop_stats','Gate','plot_render','extra_vision','Fusion')}
    past_actions=0
    for index,t in enumerate(row['timestamps_s']):
        (frame,actual,index_abs),elapsed=clocked(lambda:decode_rgb(row,t));timing['video_decode']+=elapsed
        v,elapsed=clocked(lambda:r.visual_frame(frame));timing['RGB_vision']+=elapsed
        z,elapsed=clocked(lambda:r.vision_latents(v[None]));timing['V1']+=elapsed;base=z[0];current=v.float().mean(0);w=None;score=None;valid=True
        budget_exhausted=sum(mask)>=cap
        need_gate=gate is not None and threshold not in (0.,1.) and not budget_exhausted
        if need_gate:
            def crop():
                ww=safe_sensor(row,t);return ww,cheap_stats(ww) if ww is not None else np.zeros(27,np.float32)
            (w,stats),elapsed=clocked(crop);timing['IMU_crop_stats']+=elapsed;valid=w is not None and w.valid;f=one_feature(current,history,stats,valid,past_valid,index,r.device);logit,elapsed=clocked(lambda:gate.step(f,past_actions,index));timing['Gate']+=elapsed;counts['Gate']+=1;score=float(logit.sigmoid());on=valid and score>threshold
        elif gate is not None:on=threshold==0. and not budget_exhausted
        else:on=bool(forced_mask[index]) if forced_mask is not None else False
        on=on and not budget_exhausted
        if forced_mask is not None and gate is not None:
            if on!=bool(forced_mask[index]):raise AssertionError('online causal Gate mask differs from locked cheap-feature validation mask')
        if on:
            if w is None:
                w,elapsed=clocked(lambda:safe_sensor(row,t));timing['IMU_crop_stats']+=elapsed;valid=w is not None and w.valid
            if not valid:on=False
        if on:
            if aux=='dup':second=frame
            else:
                def render():
                    ww=sensor_window(donor,donor['timestamps_s'][index]) if aux=='wrong' else w
                    image,meta=render_imu_plot(ww,cfg,zero=aux=='zero');return np.array(image),meta
                (second,plot_meta),elapsed=clocked(render);timing['plot_render']+=elapsed;counts['render']+=1
            g,elapsed=clocked(lambda:r.visual_frame(second));timing['extra_vision']+=elapsed;counts['extra_encode']+=1
            zz,elapsed=clocked(lambda:router(v[None],g[None],base[None]));timing['Fusion']+=elapsed;counts['Fusion']+=1;z=zz[0];past_actions|=1<<index
        else:z=base
        mask.append(on);zs.append(z);history.append(current);past_valid.append(valid);probabilities.append(score);perwindow.append({'relative_time_s':2*(index+1),'source_capture_request_s':t,'RGB_actual_capture_s':actual,'RGB_frame_index':index_abs,'ON':on,'score':score,'valid_IMU':valid if w is not None else None,'Gate_before_plot_and_extra_encoder':True,'budget_exhausted':budget_exhausted,'output_tokens':z.shape[0]})
    latents=torch.stack(zs);assert latents.shape==(4,64,3584);answer=r.score(row,latents);synchronize();wall=time.perf_counter()-started;hooks={k:r.calls[k]-before[k] for k in before}
    assert hooks['SigLIP_images']==4+sum(mask) and hooks['V1']==4 and counts['Fusion']==sum(mask) and counts['extra_encode']==sum(mask)
    assert counts['render']==(0 if aux=='dup' else sum(mask))
    return {**answer,'mask':mask,'scores':probabilities,'k':sum(mask),'ON_rate':sum(mask)/4,'per_window':perwindow,'stage_seconds':timing,'counts':counts,'actual_forward_hooks':hooks,'pipeline_generation_s':wall-answer['nll_evaluation_extra_s'],'gold_scoring_extra_s':answer['nll_evaluation_extra_s'],'measured_including_gold_s':wall,'pipeline_TTFT_derived_s':wall-answer['nll_evaluation_extra_s']-answer['llm_total_s']+answer['llm_ttft_s'],'peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,'feature_cache':False,'plot_cache':False,'input_media':'original compressed video via verified stored archive byte ranges; video decode paid, network acquisition/model loading excluded','aux':aux}
