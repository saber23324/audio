"""Frozen native multi-image controls; confirmation requires an immutable lock."""
from __future__ import annotations
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from utility_gate.common import seed_all
from utility_gate.runtime import clocked
from .common import ROOT,REPO,read_jsonl,write_json,append_jsonl,sha256,journal
from .runtime import QARuntime,prompt,S0,S1
from .prompts import native_prompt,PROMPTS
from .grounder import FrozenLocalizer,render,sham,save_maps
from .native import prepare,feature_mix
from .media import raw_windows,raw_window

STRATEGIES=['D0_RGB_legacy','D0_RGB_neutral','D1_OverlayOnly','D2_RGB_RGB','D3_RGB_Grounded_neutral',
    'D4_RGB_Grounded_guided','D5_Grounded_RGB_guided','D6_RGB_Sham_guided','D11_equal64_feature_mix']

def guard(role):
    if 'confirm' not in role:return
    path=ROOT/'pretest_lock.json'
    if not path.exists():raise ValueError('confirmation is forbidden before pretest lock')
    lock=json.loads(path.read_text())
    if sha256(ROOT/f'manifests/{role}_available.jsonl')!=lock['manifests'][role]:raise ValueError('confirmation manifest drift')
    for name,digest in lock['code_sha256'].items():
        if sha256(REPO/name)!=digest:raise ValueError('inference code drift: '+name)
    if sha256(ROOT/'config/selected_config.json')!=lock['selected_config_sha256']:raise ValueError('selected configuration drift')
    if sha256(ROOT/'config/localizer_lock.json')!=lock['localizer_lock_sha256']:raise ValueError('localizer lock drift')
    for path,digest in lock['model_checkpoints'].items():
        if sha256(path)!=digest:raise ValueError('frozen checkpoint drift: '+path)

@torch.inference_mode()
def fixtures(runtime,localizer,row,data,prediction,grounded):
    from .runtime import insert_views
    import wave
    # A real GPU single image and dual image repeated through native preparation.
    report={}
    for n in (1,2):
        frames=[data['frames'][0]] if n==1 else [data['frames'][0],grounded[0]]
        ids=native_prompt(runtime.tokenizer,'What is visible in these images?',[float(data['timestamps'][0])],system='S1' if n==1 else 'S2',single=n==1)
        e,v,m=prepare(runtime,ids,frames);a=runtime.answer(ids,v,'instrument',prepared=e)
        e2,v2,m2=prepare(runtime,ids,frames);b=runtime.answer(ids,v2,'instrument',prepared=e2)
        if not torch.equal(e,e2) or a['generated_ids']!=b['generated_ids']:raise AssertionError('GPU native repeat fixture')
        report[str(n)]={'embedding_exact':True,'generated_ids':a['generated_ids'],'generation_exact':True,**m}
    # Raw waveform future mutation at the file level; strict media helper must
    # discard all samples after t BEFORE resampling, then frozen localizer agrees.
    directory=ROOT/'audit/fixtures';directory.mkdir(exist_ok=True)
    rng=np.random.default_rng(1234);prefix=rng.integers(-2000,2001,32000,dtype=np.int16)
    windows=[]
    for i in (0,1):
        path=directory/f'future_audio_{i}.wav'
        pcm=np.concatenate((prefix,rng.integers(-30000,30001,32000,dtype=np.int16)))
        with wave.open(str(path),'wb') as out:out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(pcm.tobytes())
        windows.append(raw_window(path,2.)[0])
    if not torch.equal(*windows):raise AssertionError('future audio leaked through raw truncation')
    maps=[]
    for window in windows:maps.append(localizer.predict(data['frames'][:1],window[None],torch.tensor([True])))
    if not np.array_equal(maps[0]['maps'],maps[1]['maps']):raise AssertionError('future mutation changed frozen map')
    views=[]
    for pred in maps:
        im=render(data['frames'][0],pred['maps'][0],True)
        visual=runtime.extractor.encode_visual(runtime.model,runtime.processor,im[None],1).float()
        z,_=runtime.vision_router(visual,None,data['timestamps'][:1].to(runtime.device));views.append(z)
    if not torch.equal(*views):raise AssertionError('future mutation changed Z')
    report['causal_future_mutation']={'raw_windows_exact':True,'predicted_maps_exact':True,'Stage1_Z_exact':True,'future_visual_frames_used':False}
    report['mixture_alpha_zero_exact']=torch.equal(feature_mix(data['V64'],data['V64']*2,0),data['V64'])
    write_json(ROOT/'audit/real_gpu_native_fixtures_rawcrop_v1.json',report)
    print('real GPU fixtures passed',flush=True)

def baseline_call(runtime,row,data,output,completed):
    ids=prompt(runtime.tokenizer,row['question'])
    for strategy in ('V64','V196','A64','A64-null'):
        if (row['question_id'],strategy) in completed:continue
        result=runtime.answer(ids,[data[strategy].to(runtime.device)],row['answer_text'],mcqa=row['dataset']!='music_avqa')
        append_jsonl(output,{**row,**result,'strategy':strategy,'system_prompt':'S0','timestamps_s':data['timestamps'].tolist(),
            'frame_pts_s':data['frame_pts_s'],'audio_valid':data['audio_valid'].tolist(),'audio_reasons':data['audio_reasons'],
            'map_origin':'NO_MAP','encoding_timing':data['timing'],'oracle':False})

def run(role,shard=0,shards=1,alpha=.2,mix_alpha=.25,candidates=False,fixture=False,baselines=False):
    guard(role);seed_all(1234)
    if 'confirm' in role:
        cfg=json.loads((ROOT/'config/selected_config.json').read_text());alpha=cfg['overlay_alpha'];mix_alpha=cfg['feature_mix_alpha']
        if candidates:raise ValueError('no parameter candidates on confirmation')
    rows=read_jsonl(ROOT/f'manifests/{role}_available.jsonl');sources=list(dict.fromkeys(r['source_video_id'] for r in rows))
    sources=set(sources[shard::shards]);rows=[r for r in rows if r['source_video_id'] in sources]
    target=ROOT/f'qa_eval/dual_{role}_shard{shard}.jsonl'
    done={(r['question_id'],r['strategy']) for r in read_jsonl(target)} if target.exists() else set()
    baseline_target=ROOT/f'qa_eval/baseline_{role}_shard{shard}.jsonl'
    bd={(r['question_id'],r['strategy']) for r in read_jsonl(baseline_target)} if baseline_target.exists() else set()
    runtime=QARuntime();localizer=FrozenLocalizer(runtime.device)
    group=defaultdict(list)
    for row in rows:group[row['source_video_id']].append(row)
    variants=STRATEGIES+(['D4_candidateB_alpha035','D11_candidateB_mix050'] if candidates else [])
    for i,(source,questions) in enumerate(group.items()):
        if all((r['question_id'],s) in done for r in questions for s in variants):continue
        cache=ROOT/f'features/{source}.pt'
        if cache.exists():data=torch.load(cache,map_location='cpu',weights_only=False)
        else:
            data=runtime.encode_music(questions[0]);torch.save(data,cache)
        (map_windows,map_valid,map_quality,audio_traces),map_audio_decode_s=clocked(lambda:raw_windows(questions[0]['video_path'],data['timestamps']))
        prediction=localizer.predict(data['frames'],map_windows.numpy(),map_valid)
        prediction['audio_traces']=audio_traces;prediction['timing']['map_raw_audio_decode_and_resample_s']=map_audio_decode_s
        save_maps(source,questions[0],data,prediction,alpha)
        grounded=np.stack([render(f,h,v,alpha) for f,h,v in zip(data['frames'],prediction['maps'],prediction['valid'])])
        shifted=np.stack([render(f,sham(h,questions[0]['source_id'],j),v,alpha) for j,(f,h,v) in enumerate(zip(data['frames'],prediction['maps'],prediction['valid']))])
        zg_visual,grounded_encoder_s=clocked(lambda:runtime.extractor.encode_visual(runtime.model,runtime.processor,grounded,4).float())
        (zg,pg),grounded_router_s=clocked(lambda:runtime.vision_router(zg_visual,None,data['timestamps'].to(runtime.device)))
        zrgb=data['V64'].to(runtime.device);mixed=feature_mix(zrgb,zg,mix_alpha)
        if not torch.equal(feature_mix(zrgb,zg,0),zrgb):raise AssertionError('alpha=0 baseline equivalence')
        if fixture:
            fixtures(runtime,localizer,questions[0],data,prediction,grounded);fixture=False
        for row in questions:
            if baselines:baseline_call(runtime,row,data,baseline_target,bd)
            for strategy in variants:
                if (row['question_id'],strategy) in done:continue
                torch.cuda.reset_peak_memory_stats()
                prep=None;metadata={};native_frames=None
                if strategy=='D0_RGB_legacy':
                    ids=prompt(runtime.tokenizer,row['question'],S0);views=[zrgb];system='S0'
                elif strategy=='D0_RGB_neutral':
                    ids=prompt(runtime.tokenizer,row['question'],S1);views=[zrgb];system='S1'
                elif strategy=='D1_OverlayOnly':
                    ids=prompt(runtime.tokenizer,row['question'],S1);views=[zg];system='S1'
                elif strategy=='D11_equal64_feature_mix':
                    ids=prompt(runtime.tokenizer,row['question'],S1);views=[mixed];system='S1'
                elif strategy=='D11_candidateB_mix050':
                    ids=prompt(runtime.tokenizer,row['question'],S1);views=[feature_mix(zrgb,zg,.5)];system='S1'
                else:
                    swapped=strategy=='D5_Grounded_RGB_guided';system='S2_swapped' if swapped else 'S3' if strategy=='D3_RGB_Grounded_neutral' else 'S2'
                    second=data['frames'] if strategy=='D2_RGB_RGB' else shifted if strategy=='D6_RGB_Sham_guided' else grounded
                    if strategy=='D4_candidateB_alpha035':
                        second=np.stack([render(f,h,v,.35) for f,h,v in zip(data['frames'],prediction['maps'],prediction['valid'])])
                    native_frames=[]
                    for rgb,other in zip(data['frames'],second):native_frames.extend((other,rgb) if swapped else (rgb,other))
                    ids=native_prompt(runtime.tokenizer,row['question'],data['timestamps'],system=system,swapped=swapped)
                    prep,views,metadata=prepare(runtime,ids,native_frames)
                result=runtime.answer(ids,views,row['answer_text'],prepared=prep,mcqa=row['dataset']!='music_avqa')
                append_jsonl(target,{**row,**result,'strategy':strategy,'system_prompt':system,'system_prompt_text':PROMPTS[system],
                    'timestamps_s':data['timestamps'].tolist(),'frame_pts_s':data['frame_pts_s'],'map_origin':'NO_MAP' if strategy.startswith('D0_') or strategy=='D2_RGB_RGB' else 'EZVSL_PRED',
                    'oracle':False,'audio_valid':map_valid.tolist(),'audio_router_audio_valid':data['audio_valid'].tolist(),
                    'map_audio_preprocessing':'rawcrop_v1','map_audio_traces':audio_traces,'map_audio_quality':map_quality,
                    'map_valid':prediction['valid'],'map_reasons':prediction['reasons'],
                    'overlay_alpha':.35 if strategy=='D4_candidateB_alpha035' else alpha,'feature_mix_alpha':.5 if strategy=='D11_candidateB_mix050' else mix_alpha,
                    'encoding_timing':data['timing'],'localizer_timing':prediction['timing'],'grounded_extra_visual_encoder_s':grounded_encoder_s,
                    'grounded_extra_resampler_s':grounded_router_s,'native_input':metadata,'same_time_view_pairs':native_frames is not None,
                    'sham_semantics':'deterministic location shift, identical heatmap histogram and highlighted pixel area; not verified semantic negative' if strategy=='D6_RGB_Sham_guided' else None,
                    'value_mixture_scope':'same frozen Stage1 query bank applied independently to RGB and derived RGB; 64/frame output; values from two different images' if strategy.startswith('D11') else None})
        print(f'{role} shard{shard} source{i+1}/{len(group)} {source}',flush=True)
    journal('M3/M4 frozen dual-image evaluation completed',role=role,shard=shard,qa=len(rows),sources=len(group),alpha=alpha,mix_alpha=mix_alpha,candidates=candidates)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',default='music_dev_pilot');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--alpha',type=float,default=.2);p.add_argument('--mix-alpha',type=float,default=.25);p.add_argument('--candidates',action='store_true');p.add_argument('--fixture',action='store_true');p.add_argument('--baselines',action='store_true');a=p.parse_args();run(a.role,a.shard,a.shards,a.alpha,a.mix_alpha,a.candidates,a.fixture,a.baselines)
