"""Actual causal A/B and native Router-bypass C inference with matched controls."""
import argparse,json,hashlib,time
from collections import defaultdict
import numpy as np
import torch
from relevance_gate.media import frame_at_or_before
from utility_gate.runtime import clocked
from utility_gate.grounded_dual_image.media import raw_windows
from utility_gate.grounded_dual_image.native import prepare
from .common import ROOT,config,training_rows,read_jsonl,write_jsonl,write_json,journal,sha256
from .runtime import Runtime,Localizer,render,sham,append_step_features,native_prompt,NATIVE_SYSTEM
from .policy import GateBundle,rng
from .fusion import load_fusion

def variants(stage):
    if stage=='AB':
        out=['V0','V2','Stage3_real','Stage3_null','A0_V1','A1_F0','A2_u','A3_dual','A_MAIN_GA','A5_old','A_vision','A_energy','A_cosine',
            'B0_V1','B1_F0','B2_R_always','B3_GA_R','B4_GA_F0','B5_GA_dupR','B6_GA_shamR','B8_dupR_always','B9_shamR_always','B10_zeroR_always']
        return out+[f'A4_random{s}' for s in (1234,2234,3234)]+[f'B7_randomR{s}' for s in (1234,2234,3234)]
    return ['C0_RGB_RGB','C1_always_ground','C2_GA_native','C3_GA_sham','C4_GA_dup','C6_always_sham','C7_old_native','C9_GA_zero','C10_always_zero','C_DEPLOY']+[f'C5_random{s}' for s in (1234,2234,3234)]

def policy_decisions(bundle,f,past,index,rms,random_draws,control):
    results={n:bundle.one(f,past[n],index,n) for n in ('main','u','dual','vision')};main=results['main'];old=bundle.lock['frozen_old_rule'];available=bool(f.valid_audio[0])
    accepted={'main':main['on'],'u':results['u']['on'],'dual':results['dual']['on'],'vision':results['vision']['on'],
        'old':available and main['old_u']>float(old['tau_u']) and main['r_proxy']>float(old['tau_r']),
        'energy':available and rms>float(control['energy']['selected']['threshold']),
        'cosine':available and main['cosine_proxy']>float(control['cosine']['selected']['threshold'])}
    p=bundle.lock['random_probability_from_actual_main_val_ON']
    accepted.update({f'random{s}':available and random_draws[s]<p for s in (1234,2234,3234)})
    for which in ('main','u','dual','vision'):
        if accepted[which]:past[which]|=1<<index
    return accepted,results

def action_for(name,a,available):
    if name in ('V0','V2','Stage3_real','Stage3_null','A0_V1','B0_V1','C0_RGB_RGB'):return False
    if name in ('A1_F0','B1_F0','B2_R_always','B8_dupR_always','B9_shamR_always','B10_zeroR_always','C1_always_ground','C6_always_sham','C10_always_zero'):return available
    if name=='A2_u':return a['u']
    if name=='A3_dual':return a['dual']
    if name=='A5_old' or name=='C7_old_native':return a['old']
    if name in ('A_vision','A_energy','A_cosine'):return a[name[2:]]
    if name.startswith('A4_random') or name.startswith('C5_random'):return a['random'+name.split('random')[1]]
    if name.startswith('B7_randomR'):return a['random'+name.split('randomR')[1]]
    return a['main']

def control_kind(name):
    if name in ('B5_GA_dupR','B8_dupR_always','C0_RGB_RGB','C4_GA_dup'):return 'duplicate'
    if name in ('B6_GA_shamR','B9_shamR_always','C3_GA_sham','C6_always_sham'):return 'sham'
    if name in ('B10_zeroR_always','C9_GA_zero','C10_always_zero'):return 'zero'
    return 'real'

def second_view(r,localizer,frame,windows,valid,source,index,kind):
    if kind=='duplicate':return frame,{'map_origin':'RGB_DUPLICATE','map_valid':True,'localizer_s':0.,'map_sha256':None,'localizer_called':False}
    inputs=torch.zeros_like(windows) if kind=='zero' else windows
    pred,sec=clocked(lambda:localizer.predict([frame],inputs,valid));h=pred['maps'][0];good=bool(pred['valid'][0])
    if kind=='sham':h=sham(h,source,index)
    image,rs=clocked(lambda:render(frame,h,good,alpha=.2,area=.2))
    return image,{'map_origin':'EZVSL_ZERO_WAVEFORM_CONTROL' if kind=='zero' else 'EZVSL_SPATIAL_ROLL_SHAM' if kind=='sham' else 'EZVSL_PRED_CAUSAL',
        'map_valid':good,'localizer_s':sec,'render_s':rs,'map_sha256':hashlib.sha256(h.tobytes()).hexdigest(),'localizer_called':True}

def source_run(r,localizer,bundle,fusion,source,qs,stage,path):
    names=variants(stage);native=stage=='C';cfg=config();frames=[];vv=[];aa=[];valids=[];hist={n:[] for n in names};counts={n:[] for n in names}
    oncounts=defaultdict(int);calls=defaultdict(int);extra=defaultdict(int);native_meta=defaultdict(list);cost={n:defaultdict(float) for n in names};actions=[]
    past={n:0 for n in ('main','u','dual','vision')};streams={s:rng(source,s) for s in (1234,2234,3234)}
    controls=json.loads((ROOT/'locks/nonlearned_controls_lock.json').read_text());router_before=dict(r.router_calls);stamp=cfg['timestamps_s']
    zero_audio=None
    if not native:zero_audio,_=clocked(lambda:r.audio_encoder.encode_windows(torch.zeros(1,32000),r.device).to(r.device).float())
    for i,t in enumerate(stamp):
        (frame,pt),ds=clocked(lambda:frame_at_or_before(qs[0]['video_path'],float(t)))
        if pt>t+1e-6:raise AssertionError('future RGB')
        v,ves=clocked(lambda:r.visual_frame(frame));v=v.half().float();frames.append(frame);vv.append(v.cpu())
        (w,ok,why,trace),ads=clocked(lambda:raw_windows(qs[0]['video_path'],[float(t)]))
        a,bes=clocked(lambda:r.audio_encoder.encode_windows(w,r.device).float().cpu());aa.append(a[0]);valids.append(bool(ok[0]))
        f=append_step_features(torch.stack(vv),torch.stack(aa),valids,stamp[:i+1],i)
        (decisions,scores),gs=clocked(lambda:policy_decisions(bundle,f,past,i,float(w.square().mean().sqrt()),{s:streams[s].random() for s in streams},controls))
        mainmask=decisions['main']
        zbase=None;pv=None;bs=0.
        if not native:
            (zb,pv),bs=clocked(lambda:r.new_router(v[None],None));zbase=zb[0]
        for name in names:
            on=bool(action_for(name,decisions,bool(ok[0])));kind=control_kind(name);mapmeta={'map_origin':None,'map_valid':None,'localizer_s':0.,'localizer_called':False,'map_sha256':None};ese=0.;rs=0.
            if native:
                second=frame
                if on:second,mapmeta=second_view(r,localizer,frame,w,ok,source,i,kind);calls[name]+=int(mapmeta['localizer_called'])
                ims=[frame] if name=='C_DEPLOY' and not on else [frame,second]
                # Native preparation uses a query-free neutral placeholder only
                # to recover actual image spans; real questions arrive after4 observations.
                dummy={'dataset':'music_avqa','question':'Describe the visible observation.'}
                ids=native_prompt(r,dummy,[t],[len(ims)])
                (embeds,views,meta),ese=clocked(lambda:prepare(r,ids,ims))
                hist[name].extend(x.cpu().clone() for x in views);counts[name].append(len(ims));native_meta[name].append(meta)
                tokens=sum(len(x) for x in views);extra[name]+=1
            else:
                if name=='V0':
                    (z0,p),rs=clocked(lambda:r.vision_router(v[None],None));z=z0[0]
                elif name=='V2':z=v
                elif name in ('Stage3_real','Stage3_null'):
                    av=torch.stack(aa).to(r.device) if name=='Stage3_real' else zero_audio.expand(i+1,-1,-1)
                    (z0,p),rs=clocked(lambda:r.audio_router(torch.stack(vv).to(r.device),av,torch.tensor(stamp[:i+1],device=r.device)));z=z0[-1]
                elif on:
                    image,mapmeta=second_view(r,localizer,frame,w,ok,source,i,kind);calls[name]+=int(mapmeta['localizer_called'])
                    if mapmeta['map_valid']:
                        vg,ese=clocked(lambda:r.visual_frame(image));vg=vg.half().float();extra[name]+=1
                        if name.startswith('B') and name not in ('B1_F0','B4_GA_F0'):
                            z0,rs=clocked(lambda:fusion(v[None],vg[None],zbase[None],torch.ones(1,dtype=torch.bool,device=r.device)));z=z0[0]
                        else:
                            (zg,pg),rs=clocked(lambda:r.new_router(vg[None],None));z=zbase+.25*(zg[0]-zbase)
                    else:z=zbase
                else:z=zbase
                if z.shape!=(196 if name=='V2' else 64,3584):raise AssertionError('fixed64 budget violated')
                hist[name].append(z.detach().cpu().clone());tokens=len(z);counts[name].append(1)
            oncounts[name]+=int(on);c=cost[name];c['frame_decode_s']+=ds
            if native:
                c['actual_native_encoder_prepare_s']+=ese
                if name not in ('C0_RGB_RGB','C1_always_ground','C6_always_sham','C10_always_zero'):
                    c['Gate_RGB_encoder196_s']+=ves;c['BEATs_s_even_when_OFF']+=bes;c['all_deployable_gate_variants_shared_s']+=gs
                if name!='C0_RGB_RGB':c['raw_audio_crop_decode_resample_s']+=ads
            else:
                c['RGB_encoder196_s']+=ves;c['RGB_V1_router_s']+=bs;c['conditional_second_encoder_s']+=ese;c['branch_Router_s']+=rs
                if name not in ('V0','V2','A0_V1','B0_V1'):c['raw_audio_crop_decode_resample_s']+=ads
                if name not in ('V0','V2','A0_V1','B0_V1','A1_F0','B1_F0','B2_R_always','B8_dupR_always','B9_shamR_always','B10_zeroR_always'):
                    c['BEATs_s_even_when_OFF']+=bes;c['all_deployable_gate_variants_shared_s']+=gs
            c['conditional_localizer_s']+=mapmeta['localizer_s'];c['render_s']+=mapmeta.get('render_s',0.)
            actions.append({'source_id':source,'strategy':name,'stage':stage,'frame_index':i,'timestamp_s':t,'frame_pts_s':pt,'audio_available':bool(ok[0]),
                'audio_quality_reason':why[0],'raw_audio_trace':trace[0],'gate_on':on,'main_Gate_A_on':mainmask,'main_u':scores['main']['u'],'r_proxy':scores['main']['r_proxy'],
                'best_u_only':scores['u']['u'],'best_dual_u':scores['dual']['u'],'vision_only_u':scores['vision']['u'],
                **mapmeta,'actual_second_encoder_called':bool(native or ese>0),'llm_visual_tokens_this_frame':tokens,
                'native_image_count':len(ims) if native else None,'history_append_only':True,'Gate_did_not_read_QA_or_source_id':True,
                'visual_token_path':'native_original_multimage_730_actual_meter' if native else 'native196_to64' if name!='V2' else 'full196_reference',
                'policy_transfer_scope':'GateA trained for fixed F0, frozen transfer to RF/native where applicable','GT_used':False})
    router_delta={k:r.router_calls[k]-router_before[k] for k in router_before}
    if native and any(router_delta.values()):raise AssertionError('native Gate-only invoked a64-token Router')
    records=[]
    for name in names:
        if native:views=[z.to(r.device) for z in hist[name]]
        else:
            z=torch.stack(hist[name]).to(r.device);z=r.full_vision(z) if name=='V2' else z;views=[z]
        for row in qs:
            if native:
                ids=native_prompt(r,row,stamp,counts[name]);result=r.answer(ids,views,row['answer_text'],mcqa=row['dataset']!='music_avqa')
            else:result=r.score(row,views[0])
            records.append({**row,**result,'stage':stage,'strategy':name,'on_count':oncounts[name],'window_count':4,'timestamps_s':stamp,
                'tokens_per_frame':None if native else 196 if name=='V2' else 64,'actual_native_images_per_frame':counts[name] if native else None,
                'actual_localizer_calls':calls[name],'actual_second_encoder_calls':extra[name],'frontend_components_s':dict(cost[name]),
                'shared_frontend_across_policies':True,'raw_timings_are_not_isolated_pipeline_latency':True,'native_router_call_deltas':router_delta if native else None,
                'native_metadata':native_meta[name] if native else None,'system_prompt_text':NATIVE_SYSTEM if native else 'unchanged native short-answer / original ABCD S0',
                'history_never_rewritten':True,'GT_used':False,'native_Gate_zero_shot_transfer':native,'original_Gate_weights_changed':False})
        print('actual QA',stage,source,name,sum(r['correct'] for r in records if r['strategy']==name),'/',len(qs),flush=True)
    write_jsonl(path,records);write_jsonl(path.with_suffix('.actions.jsonl'),actions)
    return records,actions

def run(role,stage,shard=0,shards=1):
    if role=='val':rows=training_rows('val')
    else:
        from .locks import guard
        guard(role);rows=read_jsonl(ROOT/f'manifests/{role}.jsonl')
    if not (ROOT/'locks/fusion_router_lock.json').exists():raise ValueError('Router must be trained before staged A/B/C QA')
    groups=defaultdict(list)
    for row in rows:groups[row['source_id']].append(row)
    r=Runtime();localizer=Localizer(r.device);bundle=GateBundle();fusion=load_fusion(device=r.device)[0] if stage=='AB' else None
    if fusion is not None:r.attach_fusion(fusion)
    directory=ROOT/f'results/{role}_{stage}';directory.mkdir(parents=True,exist_ok=True)
    for i,(source,qs) in enumerate(list(groups.items())[shard::shards]):
        path=directory/f'{source}.jsonl'
        if path.exists():
            if {x['strategy'] for x in read_jsonl(path)}!=set(variants(stage)):raise ValueError('partial source output')
            continue
        source_run(r,localizer,bundle,fusion,source,qs,stage,path);print('DONE',role,stage,shard,i+1,len(groups),source,flush=True)
    journal('real staged QA shard complete',role=role,stage=stage,shard=shard,shards=shards)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',required=True);p.add_argument('--stage',choices=['AB','C'],required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.role,a.stage,a.shard,a.shards)
