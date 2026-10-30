"""Causal conditional-grounding evaluation with append-only 64-token histories."""
import argparse,json,time,hashlib
from collections import defaultdict
import numpy as np
import torch
from relevance_gate.media import frame_at_or_before
from utility_gate.runtime import clocked
from utility_gate.grounded_dual_image.media import raw_windows
from utility_gate.common import write_jsonl
from .common import ROOT,read_jsonl,write_json,read_config,sha256,journal
from .features import times
from .runtime import Runtime,append_step_features
from .localizer import Localizer,render,sham
from .fuser64 import fuse,dependencies
from .online_policy import GateBundle,decide,random_stream

def strategies():
    policies=json.loads((ROOT/'locks/policy_lock.json').read_text())['policies']
    return {'V0':{'kind':'off'},'V2':{'kind':'off'},'A0':{'kind':'off'},'A1':{'kind':'off'},**policies,
        'G4':{'same_mask':'G1','control':'duplicate'},'G5':{'same_mask':'G1','control':'sham'},'G6':{'same_mask':'G1','control':'zero_map'},
        'G7':{'kind':'on','method_identical_to':'F0'}}

def run_source(runtime,localizer,gates,source,questions,policies,output):
    """Gate precedes each branch's Localizer/extra encode; no QA is passed to it."""
    row=questions[0];stamps=times(row['duration_s']);beta=dependencies()['beta'];cfg=read_config()['fuser']
    visual=[];audio=[];valids=[];histories={name:[] for name in policies};actions=[];timing=defaultdict(float)
    streams={name:random_stream(source,p['seed']) for name,p in policies.items() if p.get('kind')=='random'}
    null,ns=clocked(lambda:runtime.audio_encoder.encode_windows(torch.zeros(1,32000),runtime.device).float().cpu())
    timing['null_BEATs_control_once_s']=ns
    on_counts=defaultdict(int);localizer_calls=defaultdict(int);extra_calls=defaultdict(int);cost={n:defaultdict(float) for n in policies}
    for i,t in enumerate(stamps):
        (frame,pts),ds=clocked(lambda:frame_at_or_before(row['video_path'],float(t)))
        if pts>float(t)+1e-6:raise AssertionError('future visual frame')
        v,vs=clocked(lambda:runtime.visual_frame(frame));v=v.cpu().half().float()
        (windows,available,reasons,traces),ads=clocked(lambda:raw_windows(row['video_path'],[float(t)]))
        a,bes=clocked(lambda:runtime.audio_encoder.encode_windows(windows,runtime.device).float().cpu())
        visual.append(v);audio.append(a[0]);valids.append(bool(available[0]));current_v=torch.stack(visual);current_a=torch.stack(audio)
        f=append_step_features(current_v,current_a,valids,stamps[:i+1],i)
        scores,gs=clocked(lambda:gates.scores(f,float(windows[0].square().mean().sqrt())))
        (zbase,pbase),bs=clocked(lambda:runtime.new_router(v[None].to(runtime.device),None))
        zbase=zbase[0]
        decisions={}
        for name,p in policies.items():
            if p.get('same_mask'):continue
            draw=streams[name].random() if name in streams else None
            decisions[name]=decide(p,scores,valids[-1],draw)
        for name in ('G4','G5','G6'):decisions[name]=decisions['G1']
        for name,p in policies.items():
            on=decisions[name];extra_s=0.;map_s=0.;map_valid=None;origin=None;map_digest=None
            if name=='V0':
                (zz,pp),rs=clocked(lambda:runtime.vision_router(v[None].to(runtime.device),None));zout=zz[0];cost[name]['router_s']+=rs
            elif name=='V2':zout=v.to(runtime.device)
            elif name in ('A0','A1'):
                aa=current_a if name=='A0' else null.expand(len(current_a),-1,-1)
                (zz,pp),rs=clocked(lambda:runtime.audio_router(current_v.to(runtime.device),aa.to(runtime.device),torch.as_tensor(stamps[:i+1],device=runtime.device)))
                zout=zz[-1];cost[name]['router_s']+=rs
            elif on:
                control=p.get('control')
                if control=='duplicate':image=frame;origin='RGB_DUPLICATE';map_valid=True
                else:
                    map_audio=torch.zeros_like(windows) if control=='zero_map' else windows
                    # Deliberate null-audio intervention: retain source availability;
                    # do not let quality rejection trivially turn this into duplicate.
                    prediction,map_s=clocked(lambda:localizer.predict([frame],map_audio,available))
                    localizer_calls[name]+=1;h=prediction['maps'][0];map_valid=bool(prediction['valid'][0])
                    if control=='sham':h=sham(h,source,i)
                    origin='EZVSL_ZERO_WAVEFORM_CONTROL' if control=='zero_map' else 'EZVSL_SPATIAL_ROLL_SHAM' if control=='sham' else 'EZVSL_PRED_CAUSAL'
                    map_digest=hashlib.sha256(np.ascontiguousarray(h).tobytes()).hexdigest()
                    image=render(frame,h,map_valid,alpha=cfg['overlay_alpha'],area=cfg['highlight_area'])
                vg,extra_s=clocked(lambda:runtime.visual_frame(image));extra_calls[name]+=1
                (zground,pg),rs=clocked(lambda:runtime.new_router(vg[None].half().float(),None))
                zout=fuse(zbase,zground[0],on=True,beta=beta);cost[name]['router_s']+=bs+rs
            else:zout=zbase;cost[name]['router_s']+=bs
            expected=196 if name=='V2' else 64
            if zout.shape!=(expected,3584):raise AssertionError('visual budget violated')
            # Immutable snapshots: later arrivals append, never replace an index.
            histories[name].append(zout.detach().cpu().clone());on_counts[name]+=int(on)
            cost[name]['frame_decode_s']+=ds;cost[name]['RGB_encoder_s']+=vs
            if name not in ('V0','V1','V2'):
                cost[name]['audio_raw_crop_decode_s']+=ads;cost[name]['BEATs_s']+=bes
            if name not in ('V0','V1','V2','A0','A1','F0','G7'):cost[name]['gate_bundle_s']+=gs
            cost[name]['conditional_localizer_s']+=map_s;cost[name]['conditional_extra_visual_encoder_s']+=extra_s
            actions.append({'source_id':source,'frame_index':i,'timestamp_s':float(t),'frame_pts_s':pts,'strategy':name,
                'gate_on':on,'audio_available':valids[-1],'audio_quality_reason':reasons[0],
                'audio_window_s':[max(0,float(t)-2),float(t)],'raw_audio_trace':traces[0],
                **scores,'locked_policy':p,'map_origin':origin,'map_valid':map_valid,'map_sha256':map_digest,
                'actual_localizer_called':on and p.get('control')!='duplicate','actual_extra_encoder_called':on,
                'llm_visual_tokens_this_frame':expected,'history_append_only':True,'history_length':len(histories[name]),
                'question_or_answer_or_type_input':False,'localizer_checkpoint_sha256':localizer.lock['checkpoint_sha256']})
        if any(decisions[n]!=decisions['G1'] for n in ('G4','G5','G6')):raise AssertionError('control mask mismatch')
    records=[]
    for name in policies:
        if name=='A1':
            # The synthetic control is the same known zero waveform at every
            # arrival; its genuine BEATs encoding was computed once above.
            cost[name]['BEATs_s']=0.;cost[name]['audio_raw_crop_decode_s']=0.
            cost[name]['actual_zero_waveform_BEATs_once_s']=ns
        zz=torch.stack(histories[name]).to(runtime.device)
        if name=='V2':zz=runtime.full_vision(zz)
        for q in questions:
            prediction=runtime.score(q,zz)
            records.append({**q,**prediction,'strategy':name,'tokens_per_frame':196 if name=='V2' else 64,
                'window_count':len(stamps),'on_count':on_counts[name],'actual_localizer_calls':localizer_calls[name],
                'actual_extra_encoder_calls':extra_calls[name],'frontend_components_s':dict(cost[name]),
                'shared_base_RGB_and_BEATs_across_strategies':True,'grounded_features_cached':False,
                'null_BEATs_identical_window_reuse':name=='A1',
                'conditional_execution':True,'action_dependency':dependencies_cached,
                'history_never_rewritten':True})
        print('QA',source,name,sum(x['correct'] for x in records if x['strategy']==name),'/',len(questions),flush=True)
    write_jsonl(output,records);write_jsonl(output.with_suffix('.actions.jsonl'),actions)

dependencies_cached=None
def run(role,shard=0,shards=1,development_controls=False):
    global dependencies_cached
    if development_controls:
        from .common import train_rows
        if role!='gate_val':raise ValueError('development control role')
        rows=train_rows(role)
    else:
        from .locks import guard_confirmation
        guard_confirmation(role);rows=read_jsonl(ROOT/f'manifests/{role}.jsonl')
    groups=defaultdict(list)
    for row in rows:groups[row['source_id']].append(row)
    dependencies_cached=dependencies()['fingerprint'];policies=strategies();runtime=Runtime();runtime.load_new_router();localizer=Localizer(runtime.device);gates=GateBundle()
    directory=ROOT/f'qa_eval/{role}_online';directory.mkdir(parents=True,exist_ok=True)
    for i,(source,qs) in enumerate(list(groups.items())[shard::shards]):
        path=directory/f'{source}.jsonl'
        if path.exists():
            old=read_jsonl(path)
            if {x['strategy'] for x in old}!=set(policies) or any(x['action_dependency']!=dependencies_cached for x in old):raise ValueError('partial/stale evaluation source')
            continue
        run_source(runtime,localizer,gates,source,qs,policies,path)
        print('DONE SOURCE',role,shard,i+1,len(groups),source,flush=True)
    journal('causal conditional actual QA shard completed',role=role,shard=shard,shards=shards,development_controls=development_controls)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--development-controls',action='store_true');a=p.parse_args();run(a.role,a.shard,a.shards,a.development_controls)

