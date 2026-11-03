"""Actual frozen recording-level IMU plots/controls; no Router/Gate training."""
import argparse,copy,json,hashlib,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import av
from PIL import Image
from utility_gate.grounded_dual_image.native import prepare
from utility_gate.runtime import clocked
from relevance_gate.media import frame_at_or_before
from .common import ROOT,DATA,read_jsonl,write_json,write_jsonl,sha256,journal
from .causal_imu_window import causal_window
from .render_imu_plot import render_imu_plot
from .runtime import VisionRuntime

SYSTEM=('You are answering a question about a completed observed recording. Camera images show the scene. An IMU chart, when present, shows acceleration and angular velocity of the sensor location indicated in the chart. Its horizontal axis is time and its measurements are not activity labels. Use only the movement evidence supported by the observations. Answer in the required format.')
ONLY_SYSTEM=('You are answering a question about an IMU recording shown as a chart. The chart shows acceleration and angular velocity at the indicated sensor location. Its horizontal axis is time and its measurements are not activity labels. Use the plotted movement evidence. Answer in the required format.')
def prompt(r,row,n,style='sensor_instructions',only=False,numeric=None):
    from llava.conversation import conv_templates
    from llava.constants import DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    conv=copy.deepcopy(conv_templates['qwen_1_5']);system='You are a helpful assistant.' if style=='neutral' else ONLY_SYSTEM if only else SYSTEM
    conv.system='<|im_start|>system\n'+system;body='\n'.join([DEFAULT_IMAGE_TOKEN]*n)
    if numeric:body+='\nIMU measurements (time in seconds, acceleration m/s^2, angular velocity deg/s):\n'+numeric
    body+='\n'+row['question']+'\n'+'\n'.join(row['options'])+'\nRespond only with the uppercase option letter A, B, C, or D.'
    conv.append_message(conv.roles[0],body);conv.append_message(conv.roles[1],None)
    return tokenizer_image_token(conv.get_prompt(),r.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt')
def source_window(row,W=2.,shift=0.):
    d=np.load(row['imu_path']);t=float(d['times'][-1])-shift
    return causal_window(d['times'],d['values'],t,row['imu_channels'],row['imu_unit_map'],row['imu_sensor_positions'],W,50.,row['sensor_clock'])
def variant_list():
    # Small, predeclared renderer/prompt sensitivity; no model search.
    base=[('I0_native_RGB','rgb','P1',2.,'sensor_instructions'),('I0_V1_64','v1','P1',2.,'sensor_instructions'),('I1_plot','only','P1',2.,'sensor_instructions'),('I2_real','real','P1',2.,'sensor_instructions'),('I3_duplicate','duplicate','P1',2.,'sensor_instructions'),('I4_zero','zero','P1',2.,'sensor_instructions'),('I5_wrong_source','wrong','P1',2.,'sensor_instructions'),('I6_past_shift','shift','P1',2.,'sensor_instructions'),('I7_numeric_text','numeric','P1',2.,'neutral')]
    for kind in ('P1','P2'):
        for style in ('sensor_instructions','neutral'):
            for view in ('only','real','zero','wrong'):
                name=f'S_{kind}_{style}_{view}'
                if kind=='P1' and style=='sensor_instructions':continue
                base.append((name,view,kind,2.,style))
    base.extend([(f'S_W4_{view}',view,'P1',4.,'sensor_instructions') for view in ('only','real','zero','wrong')])
    return base
@torch.inference_mode()
def run(shard=0,shards=1,limit=None):
    rows=read_jsonl(ROOT/'M1_data/utd_probe_manifest.jsonl');cfg=json.loads((ROOT/'M2_frozen_imu_plots/utd_renderer_lock.json').read_text());r=VisionRuntime()
    outdir=ROOT/'M2_frozen_imu_plots/utd';outdir.mkdir(exist_ok=True);images=DATA/'utd/plots';images.mkdir(exist_ok=True)
    for index,row in enumerate(rows[shard::shards][:limit]):
        target=outdir/(row['source_id']+'.jsonl')
        if target.exists():continue
        with av.open(row['video_path']) as c:
            framesall=[f for f in c.decode(video=0)];duration=float(framesall[-1].pts*framesall[-1].time_base)
        stamps=[duration*x for x in (.25,.5,.75,1.)];frames=[frame_at_or_before(row['video_path'],t)[0] for t in stamps]
        # Wrong donor from another source, same placement, different official class.
        candidates=[q for q in rows if q['class_id']!=row['class_id'] and q['imu_sensor_positions']==row['imu_sensor_positions'] and q['participant_id']!=row['participant_id']]
        donor=min(candidates,key=lambda q:hashlib.sha256((row['source_id']+q['source_id']).encode()).hexdigest());results=[];render_meta=[]
        cache={}
        for name,view,kind,W,style in variant_list():
            before=dict(r.calls);begin=time.perf_counter();native=True;ims=list(frames);window=source_window(donor if view=='wrong' else row,W,.5 if view=='shift' else 0.);chart=None;metadata=None
            if view not in ('rgb','v1','duplicate','numeric'):
                key=(kind,W,view)
                if key not in cache:
                    config={**cfg['config'],'kind':kind};chart,metadata=render_imu_plot(window,config,zero=view=='zero');cache[key]=(chart,metadata)
                chart,metadata=cache[key];ims=[np.asarray(chart)] if view=='only' else ims+[np.asarray(chart)]
                if index<3 and shard==0:
                    chart.save(images/f'{row["source_id"]}_{name}.png')
                render_meta.append({'source_id':row['source_id'],'variant':name,'donor_source':donor['source_id'] if view=='wrong' else None,**metadata})
            elif view=='duplicate':ims=ims+[frames[-1]]
            elif view=='numeric':ims=[]
            numeric=None
            if view=='numeric':numeric='\n'.join(' '.join([f'{t-window.decision_time:.2f}']+[f'{x:.3f}' for x in v]) for t,v in zip(window.times,window.values))
            ids=prompt(r,row,len(ims),style,only=view in ('only','numeric'),numeric=numeric)
            if view=='v1':
                vv=torch.stack([r.visual_frame(f) for f in frames]);z=r.vision_latents(vv);ids=prompt(r,row,1,style);score=r.answer(ids,[z],row['answer_text'],mcqa=True);meta={'native_total_visual_tokens':256,'actual_image_count':4,'path':'fixedV1_64'}
            elif not ims:score=r.answer(ids,[],row['answer_text'],mcqa=True);meta={'native_total_visual_tokens':0,'actual_image_count':0,'path':'numeric_text_alternative'}
            else:
                (embeds,views,meta),front=clocked(lambda:prepare(r,ids,ims));score=r.answer(ids,views,row['answer_text'],prepared=embeds,mcqa=True);meta['frontend_including_native_prepare_s']=front
            pred=row['option_class_ids']['ABCD'.index(score['text'])] if score['text'] in 'ABCD' and len(score['text'])==1 else None
            results.append({**row,**score,'variant':name,'predicted_class_id':pred,'plot_kind':kind,'window_s':W,'prompt_style':style,'prompt_sha256':hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                'native_metadata':meta,'actual_forward_hooks':{k:r.calls[k]-before[k] for k in before},'total_measured_eval_s':time.perf_counter()-begin,
                'rendering_shared_cached_within_source_for_development':True,'cost_not_isolated_pipeline':True,'plot_metadata':metadata,'wrong_donor_source':donor['source_id'] if view=='wrong' else None,
                'RGB_frame_times_s':stamps,'RGB_observed_end_s':duration,'strict_streaming_sync_claim':False,'no_audio_used':True,'no_training':True})
        write_jsonl(target,results);write_jsonl(target.with_suffix('.plots.jsonl'),render_meta);print('UTD FROZEN PROBE',shard,index+1,row['source_id'],len(results),flush=True)
    journal('real frozen UTD recording-level plot probe complete',shard=shard,shards=shards,strict_sync_claim=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int);a=p.parse_args();run(a.shard,a.shards,a.limit)
