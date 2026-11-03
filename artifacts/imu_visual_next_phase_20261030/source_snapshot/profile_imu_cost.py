"""Uncached resident-model native reference cost; no trained IMU Router claim."""
import json,time,random
import numpy as np
import torch
from PIL import Image
from utility_gate.runtime import clocked,synchronize
from utility_gate.grounded_dual_image.native import prepare
from .common import ROOT,read_jsonl,write_jsonl,journal
from .native_imu_probe import prompt
from .holo_probe import window
from .render_imu_plot import render_imu_plot
from .runtime import VisionRuntime

@torch.inference_mode()
def run():
    r=VisionRuntime();all_rows=read_jsonl(ROOT/'M1_data/holo_probe_manifest.jsonl');rows=list({x['source_id']:x for x in reversed(all_rows)}.values());cfg=json.loads((ROOT/'M2_frozen_imu_plots/holo_renderer_lock.json').read_text())['config'];out=ROOT/'M6_pipeline_cost/resident_native_cost.jsonl'
    if out.exists():raise FileExistsError('cost output already exists')
    # Warm every native shape and V1 before randomized measured runs.
    selected=[]
    for row in rows:
        for repeat in range(3):
            for k in range(5):selected.append((row,repeat,'native',k))
            selected.append((row,repeat,'V1',0))
    random.Random(1234).shuffle(selected);warm=rows[0];records=[]
    for trial,(row,repeat,path,k) in enumerate([(warm,-1,'native',k) for k in range(5)]+[(warm,-1,'V1',0)]+selected):
        before=dict(r.calls);synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();fr=json.loads((ROOT/f'M1_data/holo_frames_{row["session_id"]}.json').read_text())
        (frames,read_s)=clocked(lambda:[np.array(Image.open(fr[str(t)]['path']).convert('RGB')) for t in row['timestamps_s']]);render_s=0.;charts=[]
        # First-k is a causal, fixed schedule for cost only; no QA-tuned selector.
        for t in row['timestamps_s'][:k]:
            (image,meta),elapsed=clocked(lambda:render_imu_plot(window(row,t),cfg));charts.append(np.array(image));render_s+=elapsed
        if path=='V1':
            vv,vision_s=clocked(lambda:torch.stack([r.visual_frame(f) for f in frames]));z,resample_s=clocked(lambda:r.vision_latents(vv));ids=prompt(r,row,1);score=r.answer(ids,[z],row['answer_text'],mcqa=True)
        else:
            ids=prompt(r,row,len(frames)+k);(embeds,views,meta),vision_s=clocked(lambda:prepare(r,ids,frames+charts));resample_s=0.;score=r.answer(ids,views,row['answer_text'],prepared=embeds,mcqa=True)
        synchronize();total=time.perf_counter()-start;hooks={n:r.calls[n]-before[n] for n in before}
        assert hooks['SigLIP_images']==4+k and len(charts)==k
        if k==0:assert render_s==0
        if path=='V1':assert score['visual_tokens']==256 and hooks['V1']==4
        else:assert score['visual_tokens']==730*(4+k) and hooks['V1']==0
        entry={'source_id':row['source_id'],'qa_id':row['qa_id'],'repeat':repeat,'path':path,'k':k,'schedule':'first-k fixed causal cost diagnostic, no learned Gate','plot_render_count':len(charts),'actual_forward_hooks':hooks,'RGB_PNG_read_s':read_s,'plot_render_s':render_s,'vision_and_native_prepare_s':vision_s,'V1_resample_s':resample_s,'pipeline_generation_s':total-score['nll_evaluation_extra_s'],'evaluation_including_gold_s':total,'gold_scoring_extra_s':score['nll_evaluation_extra_s'],'llm_prefill_s':score['llm_prefill_s'],'llm_ttft_s':score['llm_ttft_s'],'llm_total_s':score['llm_total_s'],'visual_tokens':score['visual_tokens'],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'correct':score['correct'],'gold_nll':score['gold_nll'],'decoded_video_cache':'already decoded PNG input; PNG read paid; network/video stream decode excluded explicitly','model_initialization_excluded':True,'feature_cache':False,'render_cache':False,'gate_overhead_s':None,'trained_R_IMU':False}
        if repeat>=0:records.append(entry)
        print('COST',trial,path,k,round(entry['pipeline_generation_s'],4),flush=True)
    write_jsonl(out,records);journal('actual resident native IMU cost curves completed',measurements=len(records),warmups=6,checkpoint_training=False)
if __name__=='__main__':run()
