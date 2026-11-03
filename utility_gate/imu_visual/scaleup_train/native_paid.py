"""Original native image layout, per-image B1 encoding, actual paid references."""
import json,time
import numpy as np
import torch
from utility_gate.runtime import clocked,synchronize
from utility_gate.grounded_dual_image.native import prepare
from utility_gate.imu_visual.render_imu_plot import render_imu_plot
from .common import ROOT
from .pipeline import decode_rgb
from .data import sensor_window
from .runtime import prompt

@torch.inference_mode()
def prepare_b1(r,ids,images):
    original=r.model.encode_images;batches=[]
    def encode(images,*args,**kwargs):
        parts=[]
        for i in range(len(images)):
            batches.append(len(images[i:i+1]));parts.append(original(images[i:i+1],*args,**kwargs))
        return torch.cat(parts,0)
    r.model.encode_images=encode
    try:result=prepare(r,ids,images)
    finally:r.model.encode_images=original
    assert batches and set(batches)=={1}
    return (*result,batches)

@torch.inference_mode()
def run(r,row,aux):
    before=dict(r.calls);synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter();images=[];timing={k:0. for k in ['video_decode','IMU_crop_stats','plot_render','native_prepare']};cfg=json.loads((ROOT/'locks/IMU_PLOT_V1.json').read_text());per=[];on=aux=='real'
    for i,t in enumerate(row['timestamps_s']):
        (rgb,actual,index),elapsed=clocked(lambda:decode_rgb(row,t));timing['video_decode']+=elapsed;images.append(rgb)
        if on:
            w,elapsed=clocked(lambda:sensor_window(row,t));timing['IMU_crop_stats']+=elapsed
            (image,_),elapsed=clocked(lambda:render_imu_plot(w,cfg));timing['plot_render']+=elapsed;images.append(np.array(image))
        per.append({'relative_time_s':2*(i+1),'source_capture_request_s':t,'RGB_actual_capture_s':actual,'RGB_frame_index':index,'ON':on,'output_tokens':1460 if on else 730,'native_uncompressed':True})
    ids=prompt(r,row,True,[2 if on else 1]*4);(embeds,views,meta,batches),elapsed=clocked(lambda:prepare_b1(r,ids,images));timing['native_prepare']+=elapsed;answer=r.answer(ids,views,row['answer_text'],prepared=embeds,mcqa=True);synchronize();wall=time.perf_counter()-started;hooks={k:r.calls[k]-before[k] for k in before};assert hooks['SigLIP_images']==len(images) and hooks['V1']==0
    return {**answer,'k':4 if on else 0,'mask':[on]*4,'ON_rate':float(on),'per_window':per,'stage_seconds':timing,'counts':{'render':4 if on else 0,'extra_encode':4 if on else 0,'Gate':0,'Fusion':0},'actual_forward_hooks':hooks,'pipeline_generation_s':wall-answer['nll_evaluation_extra_s'],'gold_scoring_extra_s':answer['nll_evaluation_extra_s'],'pipeline_TTFT_derived_s':wall-answer['nll_evaluation_extra_s']-answer['llm_total_s']+answer['llm_ttft_s'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'feature_cache':False,'plot_cache':False,'native_uncompressed_reference':True,'native_encoder_batch_sizes':batches,'input_media':'actual original compressed RGB decoded; strict causal IMU; network/model load excluded','aux':aux}
