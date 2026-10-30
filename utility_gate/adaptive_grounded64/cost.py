"""Isolated uncached complete pipeline, gold-free generation timing on fixed dev sources."""
import json,time
from collections import defaultdict
import numpy as np
import torch
from utility_gate.runtime import clocked,synchronize
from utility_gate.grounded_dual_image.media import raw_windows
from relevance_gate.media import frame_at_or_before
from .common import ROOT,train_rows,write_json,journal,read_config,sha256
from .features import times
from .fuser64 import fuse,dependencies
from .runtime import Runtime,qa_prompt,append_step_features
from .localizer import Localizer,render
from .gate_model import load_new,MODES
from .online_policy import decide

@torch.inference_mode()
def run():
    runtime=Runtime();runtime.load_new_router();localizer=Localizer(runtime.device);gate,_=load_new(MODES[0])
    policy=json.loads((ROOT/'locks/policy_lock.json').read_text())['policies']['G1'];beta=dependencies()['beta'];cfg=read_config()['fuser']
    rows=train_rows('gate_val');sources={r['source_id']:r for r in reversed(rows)}
    chosen=[sources[s] for s in sorted(sources)[:4]];records=[]
    for row in chosen:
        for name in ('V1','G1','F0','G4'):
            clocked(lambda:torch.empty(1,device=runtime.device));torch.cuda.reset_peak_memory_stats();synchronize();start=time.perf_counter()
            history=[];vv=[];aa=[];valid=[];components=defaultdict(float);on_count=0;stamps=times(row['duration_s'])
            for i,t in enumerate(stamps):
                (frame,pt),sec=clocked(lambda:frame_at_or_before(row['video_path'],float(t)));components['frame_decode_s']+=sec
                v,sec=clocked(lambda:runtime.visual_frame(frame));components['RGB_SigLIP_projector_pool_s']+=sec;v=v.half().float();vv.append(v.cpu())
                (z,p),sec=clocked(lambda:runtime.new_router(v[None],None));components['RGB_router_s']+=sec;z=z[0]
                on=False
                if name!='V1':
                    (w,aok,why,trace),sec=clocked(lambda:raw_windows(row['video_path'],[float(t)]));components['audio_raw_crop_decode_resample_s']+=sec
                    if name in ('G1','G4'):
                        a,sec=clocked(lambda:runtime.audio_encoder.encode_windows(w,runtime.device).float().cpu());components['BEATs_s_including_OFF']+=sec;aa.append(a[0]);valid.append(bool(aok[0]))
                        f=append_step_features(torch.stack(vv),torch.stack(aa),valid,stamps[:i+1],i)
                        prediction,sec=clocked(lambda:gate.forward_features(f));components['new_G1_gate_s']+=sec
                        on=decide(policy,{'u':float(prediction.utility_delta[0,0])},bool(aok[0]))
                    else:on=bool(aok[0])
                    if on:
                        on_count+=1
                        if name=='G4':image=frame
                        else:
                            prediction,sec=clocked(lambda:localizer.predict([frame],w,aok));components['localizer_total_s']+=sec
                            image,sec=clocked(lambda:render(frame,prediction['maps'][0],prediction['valid'][0],alpha=cfg['overlay_alpha'],area=cfg['highlight_area']));components['render_s']+=sec
                        vground,sec=clocked(lambda:runtime.visual_frame(image));components['second_SigLIP_projector_pool_s']+=sec
                        (zg,pg),sec=clocked(lambda:runtime.new_router(vground[None].half().float(),None));components['second_router_s']+=sec
                        z=fuse(z,zg[0],on=True,beta=beta)
                history.append(z)
            synchronize();frontend=time.perf_counter()-start
            generated=runtime.answer(qa_prompt(runtime,row),[torch.stack(history)],None,compute_nll=False)
            synchronize();total=time.perf_counter()-start
            records.append({'source_id':row['source_id'],'strategy':name,'frames':len(stamps),'on_count':on_count,'on_rate':on_count/len(stamps),
                'frontend_s':frontend,'frontend_components_s':dict(components),'LLM_prefill_s':generated['llm_prefill_s'],
                'LLM_TTFT_s':generated['llm_ttft_s'],'pipeline_TTFT_s':frontend+generated['llm_ttft_s'],
                'LLM_total_s':generated['llm_total_s'],'pipeline_total_s':total,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'visual_tokens':generated['visual_tokens'],'generated_ids':generated['generated_ids'],'QA_scoring_pass_included':False,
                'input_features_cached':False,'GPU':torch.cuda.get_device_name(),'GPU_index_visible':0})
            print('isolated cost',row['source_id'],name,total,flush=True)
    write_json(ROOT/'qa_eval/pipeline_cost_summary.json',{'protocol':'first4 lexicographically fixed legal dev source IDs; all inference uncached; one GPU; four strategies sequential; no playback waiting or gold teacher scoring',
        'records':records,'summary':{name:{k:float(np.mean([r[k] for r in records if r['strategy']==name])) for k in ('frontend_s','LLM_prefill_s','LLM_TTFT_s','pipeline_TTFT_s','LLM_total_s','pipeline_total_s','peak_allocated_bytes','on_rate')} for name in ('V1','G1','F0','G4')},
        'cached_confirmation':'multi-strategy shared base frontend timings are separately recorded in raw QA; these are not isolated speedup claims'})
    journal('isolated uncached whole pipeline measured',sources=[r['source_id'] for r in chosen])

if __name__=='__main__':run()

