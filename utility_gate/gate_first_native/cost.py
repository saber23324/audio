"""Isolated, feature-uncached complete pipeline timing for the three fixed components."""
import time,json
from collections import defaultdict
import numpy as np
import torch
from relevance_gate.media import frame_at_or_before
from utility_gate.runtime import clocked,synchronize
from utility_gate.grounded_dual_image.media import raw_windows
from utility_gate.grounded_dual_image.native import prepare
from .common import ROOT,training_rows,write_json,journal,config
from .runtime import Runtime,Localizer,append_step_features,native_prompt,qa_prompt
from .policy import GateBundle
from .fusion import load_fusion
from .evaluate import second_view,control_kind

@torch.inference_mode()
def measure(r,localizer,bundle,fusion,row,name):
    native=name.startswith('C');gated=name in ('A_MAIN_GA','B3_GA_R','C2_GA_native','C3_GA_sham','C4_GA_dup','C_DEPLOY')
    stamps=config()['timestamps_s'];vv=[];aa=[];valid=[];views=[];counts=[];zz=[];past=0;components=defaultdict(float);ons=0;localizer_calls=0;image_count=0
    start_counts=dict(r.router_calls);torch.cuda.reset_peak_memory_stats();synchronize();start=time.perf_counter()
    for i,t in enumerate(stamps):
        (frame,pt),sec=clocked(lambda:frame_at_or_before(row['video_path'],float(t)));components['RGB_decode_s']+=sec
        need_audio=name not in ('A0_V1','C0_RGB_RGB')
        w=ok=None;available=True
        if need_audio:
            (w,ok,reason,trace),sec=clocked(lambda:raw_windows(row['video_path'],[float(t)]));components['raw_audio_crop_decode_resample_s']+=sec;available=bool(ok[0])
        v=None;base=None
        if gated or not native:
            v,sec=clocked(lambda:r.visual_frame(frame));v=v.half().float();components['original_SigLIP_projector_pool196_s']+=sec;vv.append(v.cpu())
        if gated:
            a,sec=clocked(lambda:r.audio_encoder.encode_windows(w,r.device).float().cpu());components['BEATs_s_including_OFF']+=sec;aa.append(a[0]);valid.append(available)
            f=append_step_features(torch.stack(vv),torch.stack(aa),valid,stamps[:i+1],i)
            output,sec=clocked(lambda:bundle.one(f,past,i));components['selected_Gate_A_s']+=sec;on=output['on']
            if on:past|=1<<i
        else:on=available and name not in ('A0_V1','C0_RGB_RGB')
        if not native:
            (z,p),sec=clocked(lambda:r.new_router(v[None],None));base=z[0];components['frozen_V1_Router_s']+=sec
        image=frame;mapinfo={'map_valid':True,'localizer_s':0.,'render_s':0.,'localizer_called':False}
        if on:
            image,mapinfo=second_view(r,localizer,frame,w,ok,row['source_id'],i,control_kind(name));components['conditional_Localizer_s']+=mapinfo['localizer_s'];components['conditional_render_s']+=mapinfo.get('render_s',0.);localizer_calls+=int(mapinfo['localizer_called'])
        ons+=int(on)
        if native:
            ims=[frame] if name=='C_DEPLOY' and not on else [frame,image];counts.append(len(ims));image_count+=len(ims)
            dummy={'dataset':'music_avqa','question':'Describe the visible observation.'};ids=native_prompt(r,dummy,[t],[len(ims)])
            (e,vs,meta),sec=clocked(lambda:prepare(r,ids,ims));components['native_SigLIP_projector_multimage_prepare_s']+=sec;views.extend(vs)
        else:
            z=base
            if on and mapinfo['map_valid']:
                vg,sec=clocked(lambda:r.visual_frame(image));components['conditional_second_SigLIP_projector_s']+=sec;vg=vg.half().float()
                if name.startswith('B'):
                    zf,sec=clocked(lambda:fusion(v[None],vg[None],base[None],torch.ones(1,dtype=torch.bool,device=r.device)));z=zf[0];components['new_Fusion_Router_s']+=sec
                else:
                    (zg,pg),sec=clocked(lambda:r.new_router(vg[None],None));components['grounded_fixed_Router_s']+=sec;z=base+.25*(zg[0]-base)
            zz.append(z)
    synchronize();frontend=time.perf_counter()-start
    if native:
        ids=native_prompt(r,row,stamps,counts)
        delta={k:r.router_calls[k]-start_counts[k] for k in start_counts}
        if any(delta.values()):raise AssertionError('native isolated cost invoked a Router')
    else:ids=qa_prompt(r,row);views=[torch.stack(zz)]
    generated=r.answer(ids,views,None,compute_nll=False)
    synchronize();total=time.perf_counter()-start
    return {'source_id':row['source_id'],'role':'val','strategy':name,'frontend_s':frontend,'frontend_components_s':dict(components),
        'LLM_prefill_s':generated['llm_prefill_s'],'LLM_TTFT_s':generated['llm_ttft_s'],'pipeline_TTFT_s':frontend+generated['llm_ttft_s'],
        'LLM_total_s':generated['llm_total_s'],'pipeline_total_s':total,'visual_tokens':generated['visual_tokens'],
        'actual_image_count':image_count if native else None,'on_count':ons,'on_rate':ons/4,'Localizer_calls':localizer_calls,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'input_feature_cache':False,'Gold_scoring_included':False,'model_loading_included':False,'GPU':torch.cuda.get_device_name()}

def run():
    start=time.perf_counter();r=Runtime();localizer=Localizer(r.device);bundle=GateBundle();fusion=load_fusion(device=r.device)[0];r.attach_fusion(fusion);init=time.perf_counter()-start
    rows=training_rows('val');unique={x['source_id']:x for x in reversed(rows)};chosen=[unique[k] for k in sorted(unique)[:3]]
    names=['A0_V1','A1_F0','A_MAIN_GA','B2_R_always','B3_GA_R','B8_dupR_always','B9_shamR_always','C0_RGB_RGB','C1_always_ground','C2_GA_native','C3_GA_sham','C4_GA_dup','C_DEPLOY'];out=[]
    for row in chosen:
        for name in names:
            result=measure(r,localizer,bundle,fusion,row,name);out.append(result);print('isolated full pipeline',row['source_id'],name,result['pipeline_total_s'],result['on_rate'],result['visual_tokens'],flush=True)
    write_json(ROOT/'results/pipeline_cost.json',{'records':out,'model_initialization_s_separate':init,
        'protocol':'first3 lexicographic legal val sources, fixed order, models resident, uncached input features, no Gold teacher scoring, compute time excludes waiting for video playback; descriptive small sample',
        'summary':{n:{k:float(np.mean([r[k] for r in out if r['strategy']==n])) for k in ('frontend_s','LLM_prefill_s','LLM_TTFT_s','pipeline_TTFT_s','LLM_total_s','pipeline_total_s','visual_tokens','on_rate','peak_allocated_bytes')} for n in names}})
    journal('all three lines isolated actual pipeline cost measured',sources=[r['source_id'] for r in chosen])

if __name__=='__main__':run()
