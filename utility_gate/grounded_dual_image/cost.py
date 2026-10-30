"""Direct feature-uncached developer-cohort latency, without Gold teacher passes."""
from __future__ import annotations
import json,time
from collections import defaultdict
import numpy as np
import torch
from utility_gate.runtime import clocked,synchronize
from utility_gate.common import seed_all
from .common import ROOT,read_jsonl,append_jsonl,journal
from .runtime import QARuntime,prompt,S1
from .grounder import FrozenLocalizer,render,sham
from .prompts import native_prompt
from .native import prepare,feature_mix
from .media import raw_windows

def main():
    from relevance_gate.media import frame_at_or_before
    seed_all(1234);r=QARuntime();g=FrozenLocalizer(r.device)
    cfg=json.loads((ROOT/'config/selected_config.json').read_text());alpha=cfg['overlay_alpha'];mixalpha=cfg['feature_mix_alpha']
    rows=read_jsonl(ROOT/'manifests/music_dev_available.jsonl');unique={}
    for row in rows:unique.setdefault(row['source_video_id'],row)
    rows=list(unique.values())[:16]
    output=ROOT/'qa_eval/direct_pipeline_cost_dev.jsonl'
    if output.exists():raise FileExistsError(output)
    # Two repeats, fixed SHA-ordered developer sources, all outcomes retained.
    for repeat in range(2):
      for row in rows:
       for strategy in ('V64','V196','A64','D2_RGB_RGB','D4_RGB_Grounded_guided','D6_RGB_Sham_guided','D11_equal64_feature_mix'):
        torch.cuda.reset_peak_memory_stats();synchronize();start=time.perf_counter();timing={}
        ts=np.array([.125,.375,.625,.875],np.float32)*row['duration_s']
        frames,timing['decode_s']=clocked(lambda:np.stack([frame_at_or_before(row['video_path'],float(t))[0] for t in ts]))
        needs_map=strategy in ('D4_RGB_Grounded_guided','D6_RGB_Sham_guided','D11_equal64_feature_mix')
        if strategy=='A64':
            # Clear the old runtime's tiny PCM memoization for a measured fresh
            # decode. Model weights and operating-system disk caches stay warm.
            r.strict_audio.clear()
            (windows,valid,reasons,rms),timing['audio_decode_s']=clocked(lambda:r.causal_windows(row['video_path'],ts))
        if needs_map:
            (windows,valid,reasons,traces),timing['map_raw_audio_decode_and_resample_s']=clocked(lambda:raw_windows(row['video_path'],ts))
            pred=g.predict(frames,windows.numpy(),valid);timing.update(pred['timing'])
            def rendering():
                hs=[sham(h,row['source_id'],i) if strategy=='D6_RGB_Sham_guided' else h for i,h in enumerate(pred['maps'])]
                return np.stack([render(f,h,v,alpha) for f,h,v in zip(frames,hs,pred['valid'])])
            grounded,timing['render_s']=clocked(rendering)
        prep=None
        if strategy in ('V64','V196','A64','D11_equal64_feature_mix'):
            visual,timing['rgb_visual_s']=clocked(lambda:r.extractor.encode_visual(r.model,r.processor,frames,4).float())
            times=torch.tensor(ts,device=r.device)
            if strategy=='V196':views=[r.full_vision(visual)]
            elif strategy=='A64':
                aud,timing['beats_s']=clocked(lambda:r.audio_encoder.encode_windows(windows,r.device).to(r.device).float())
                (z,p),timing['router_s']=clocked(lambda:r.audio_router(visual,aud,times));views=[z]
            else:
                (z,p),timing['router_s']=clocked(lambda:r.vision_router(visual,None,times));views=[z]
                if needs_map:
                    vg,timing['extra_visual_s']=clocked(lambda:r.extractor.encode_visual(r.model,r.processor,grounded,4).float())
                    (zg,pg),timing['extra_router_s']=clocked(lambda:r.vision_router(vg,None,times));views=[feature_mix(z,zg,mixalpha)]
            ids=prompt(r.tokenizer,row['question'],S1 if needs_map else 'You are a helpful assistant.')
        else:
            second=frames if strategy=='D2_RGB_RGB' else grounded
            images=[f for pair in zip(frames,second) for f in pair]
            ids=native_prompt(r.tokenizer,row['question'],ts,system='S2');prep,views,native=prepare(r,ids,images);timing.update(native)
        synchronize();before_llm=time.perf_counter()-start
        result=r.answer(ids,views,None,prepared=prep,compute_nll=False)
        synchronize();end_to_end=time.perf_counter()-start
        append_jsonl(output,{'source_id':row['source_id'],'question_id':row['question_id'],'repeat':repeat,'strategy':strategy,**result,
            'stage_timing':timing,'direct_pipeline_end_to_end_s':end_to_end,'direct_pipeline_ttft_s':before_llm+result['llm_ttft_s'],
            'feature_cache_used':False,'raw_audio_memoization_used':False,'model_weights_warm':True,'disk_cache':'OS-managed, not forcibly flushed',
            'localizer_resident_in_all_evaluation_strategies':True,'gold_teacher_pass_included':False})
       print('pipeline cost',repeat,row['source_video_id'],flush=True)
    journal('direct uncached-feature pipeline timings completed',sources=16,repeats=2,conditions=7,teacher_nll_included=False)

if __name__=='__main__':main()
