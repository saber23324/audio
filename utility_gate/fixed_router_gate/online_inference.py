"""Causal conditional front end: Gate runs before any second-view computation."""
import hashlib,json,time
from collections import defaultdict
import numpy as np
import torch
from relevance_gate.media import frame_at_or_before
from utility_gate.features import take_features
from utility_gate.runtime import clocked,synchronize
from utility_gate.grounded_dual_image.media import raw_windows
from utility_gate.gate_first_native.policy import GateBundle,rng
from utility_gate.gate_first_native.gate import freeze_proxy,proxy_scores
from .common import ROOT,config
from .runtime import render,sham,append_step_features
from .features import acoustic_stats
from .gate_models import load_gate

class Controllers:
    def __init__(self):self.models={};self.old=None;self.proxy=None
    def model(self,path):
        if path not in self.models:self.models[path]=load_gate(path)[0]
        return self.models[path]
    @torch.inference_mode()
    def one(self,spec,f,cheap,past,index,available,draw):
        kind=spec['kind'];out={'on':False,'prob_on':None,'logit_on':None,'r_proxy':None}
        if not available:return {**out,'quality_forced_off':True}
        if kind=='off':return out
        if kind=='on':return {**out,'on':True}
        if kind=='random':return {**out,'on':bool(draw<spec['probability']),'prob_on':spec['probability']}
        if kind=='old':
            if self.old is None:self.old=GateBundle()
            old=self.old.one(f,past,index);return {**out,**old,'logit_on':old['u']}
        m=self.model(spec['checkpoint']);r=None
        if m.use_r:
            if self.proxy is None:self.proxy=freeze_proxy()
            r=proxy_scores(self.proxy,f)[0][:,None];out['r_proxy']=float(r[0,0])
        value=float(m.forward_features(cheap if m.family=='acoustic' else f,past,index,r)[0]);prob=float(torch.sigmoid(torch.tensor(value)))
        on=prob>float(spec.get('threshold',.5))
        if kind=='prefix':on=index<int(spec['K'])
        return {**out,'on':on,'prob_on':prob,'logit_on':value}

def needs_beats(spec):return spec['kind']=='old' or spec.get('family') in ('audio_r','audio_no_r','history')

@torch.inference_mode()
def execute(runtime,loc,row,spec,controllers,view='real',cap=None):
    """No source/question/gold is passed to the learned network. Append-only Z."""
    synchronize();start=time.perf_counter();visual_before=runtime.visual_encoder_calls;local_before=loc.forward_calls;vv=[];aa=[];cc=[];available=[];history=[];actions=[];cost=defaultdict(float);past=0;k=0;calls=0
    random=rng(row['source_id'],int(spec.get('seed',1234)));times=[2.,4.,6.,8.]
    for i,t in enumerate(times):
        stepstart=time.perf_counter();(frame,pt),ds=clocked(lambda:frame_at_or_before(row['video_path'],t));cost['RGB_decode_s']+=ds
        if pt>t+1e-6:raise AssertionError('future RGB')
        v,vs=clocked(lambda:runtime.visual_frame(frame).half().float());cost['primary_SigLIP_projector_pool_s']+=vs;vv.append(v.cpu())
        (z,p),rs=clocked(lambda:runtime.new_router(v[None],None));cost['V1_resampling_s']+=rs;z=z[0]
        exhausted=cap is not None and k>=cap
        skip=spec['kind']=='off' or exhausted
        w=torch.zeros(1,32000);ok=torch.tensor([False]);why=['not_requested'];trace=[None];a=torch.zeros(1,96,768)
        if not skip:
            (w,ok,why,trace),ads=clocked(lambda:raw_windows(row['video_path'],[t]));cost['raw_native_audio_crop_resample_s']+=ads
            if needs_beats(spec):
                a,bes=clocked(lambda:runtime.audio_encoder.encode_windows(w,runtime.device).float().cpu());cost['prepaid_BEATs_s']+=bes
        aa.append(a[0]);cc.append(acoustic_stats(w)[0]);available.append(bool(ok[0]))
        info={'on':False,'prob_on':None,'logit_on':None,'r_proxy':None}
        if not skip:
            def decision():
                f=append_step_features(torch.stack(vv),torch.stack(aa),available,times[:i+1],i)
                cheap=append_step_features(torch.stack(vv),torch.stack(cc)[:,None],available,times[:i+1],i)
                return controllers.one(spec,f,cheap,past,i,bool(ok[0]),float(random.random()))
            info,gs=clocked(decision);cost['Gate_feature_proxy_and_forward_s']+=gs
        on=bool(info['on']);localizer_called=False;mapgood=None;hsha=None;origin=None;extra=False
        # The ON decision is final before this block. Never encode second view OFF.
        if on:
            image=frame;origin='RGB_DUPLICATE'
            if view!='duplicate':
                window=torch.zeros_like(w) if view=='zero' else w
                pred,ls=clocked(lambda:loc.predict([frame],window,ok));cost['conditional_EZVSL_s']+=ls;calls+=1;localizer_called=True
                heat=pred['maps'][0];mapgood=bool(pred['valid'][0])
                if view=='sham':heat=sham(heat,row['source_id'],i)
                image,rends=clocked(lambda:render(frame,heat,mapgood));cost['conditional_render_s']+=rends;hsha=hashlib.sha256(heat.tobytes()).hexdigest();origin='EZVSL_'+view
            vg,es=clocked(lambda:runtime.visual_frame(image).half().float());cost['conditional_second_SigLIP_projector_pool_s']+=es;extra=True;k+=1
            rz,fs=clocked(lambda:runtime.fusion(v[None],vg[None],z[None]));cost['conditional_R_A_s']+=fs;z=rz[0];past|=1<<i
        if z.shape!=(64,3584):raise ValueError('64 tokens/frame interface broken')
        # Clone prevents any later window from rewriting history.
        history.append(z.detach().cpu().clone())
        actions.append({'source_id':row['source_id'],'frame_index':i,'timestamp_s':t,'frame_pts_s':pt,'gate_on':on,**info,
            'actual_second_encoder_called':extra,'localizer_called':localizer_called,'map_valid':mapgood,'map_origin':origin,'map_sha256':hsha,
            'audio_available':bool(ok[0]),'audio_quality_reason':why[0],'raw_audio_trace':trace[0],
            'cap':cap,'cap_already_exhausted':exhausted,'encoded_extra_so_far':k,'Gate_before_Localizer_and_second_encoder':True,
            'LLM_visual_tokens':64,'history_append_only':True,'QA_inputs_to_Gate':False,'step_wall_s':time.perf_counter()-stepstart})
    synchronize();frontend=time.perf_counter()-start
    if k!=sum(x['actual_second_encoder_called'] for x in actions) or calls!=sum(x['localizer_called'] for x in actions):raise AssertionError('physical call count differs from k')
    if runtime.visual_encoder_calls-visual_before!=4+k or loc.forward_calls-local_before!=calls:raise AssertionError('instrumented encoder/localizer calls differ from reported count')
    meta={'on_mask':[x['on'] for x in actions],'k_extra_images':k,'encoded_image_count':4+k,'actual_second_encoder_calls':k,'actual_localizer_calls':calls,
        'visual_tokens':256,'tokens_per_frame':64,'frontend_components_s':dict(cost),'frontend_wall_s':frontend,'frontend_components_total_s':sum(cost.values()),
        'cache_policy':'no RGB/BEATs/second-view feature cache across policy runs; model resident; OS page cache uncontrolled; loaded controllers reused',
        'playback_wait_included':False,'model_loading_included':False,'shared_frontend_across_policies':False}
    return torch.stack(history).to(runtime.device),meta,actions
