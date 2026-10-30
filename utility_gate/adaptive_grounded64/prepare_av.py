"""Offline development-only predicted views and causal features, independent of Stage1."""
import argparse
import numpy as np
import torch
from .common import ROOT,PRIOR,train_rows,sha256,fingerprint,journal,read_config
from .features import load_rgb
from .runtime import Runtime,gate_features
from .localizer import Localizer,render
from utility_gate.features import feature_payload

def av_path(source):return ROOT/f'features/av/{source}.pt'
def grounded_path(source):return ROOT/f'features/development_grounded/{source}.pt'

def key(row,rgb,localizer):
    cfg=read_config()['fuser']
    return fingerprint({'media':row['media_sha256'],'RGB':rgb['feature_cache_key'],
        'localizer':localizer.lock['checkpoint_sha256'],'map_policy':'batch1_rawcrop_2s_leftpad1s',
        'alpha':cfg['overlay_alpha'],'area':cfg['highlight_area'],'color':cfg['color_rgb'],
        'source_code_sha256':sha256(__file__)})

def load_av(row):
    p=av_path(row['source_id']);d=torch.load(p,map_location='cpu',weights_only=False)
    if d['media_sha256']!=row['media_sha256']:raise ValueError('AV cache media drift')
    return d

def run(shard=0,shards=1):
    rows=train_rows('gate_train')+train_rows('gate_val');unique={r['source_id']:r for r in rows}
    runtime=Runtime();localizer=Localizer(runtime.device);cfg=read_config()['fuser']
    for i,row in enumerate(list(unique.values())[shard::shards]):
        rgb=load_rgb(row);ap=av_path(row['source_id']);gp=grounded_path(row['source_id']);expected=key(row,rgb,localizer)
        if ap.exists():av=load_av(row)
        else:
            av=runtime.audio_features(row['video_path'],rgb['timestamps'])
            av.update(source_id=row['source_id'],media_sha256=row['media_sha256'],
                feature_payload=feature_payload(gate_features(rgb['visual'],av['audio'],av['audio_valid'],rgb['timestamps'])))
            torch.save(av,ap.with_suffix('.part'));ap.with_suffix('.part').rename(ap)
        if gp.exists():
            old=torch.load(gp,map_location='cpu',weights_only=False)
            if old['dependency_key']!=expected:raise ValueError('development grounded cache stale')
            continue
        grounded=[];maps=[];native=[];valid=[];reasons=[];timings=[]
        for j,frame in enumerate(rgb['frames']):
            prediction=localizer.predict([frame],av['windows'][j:j+1],av['audio_valid'][j:j+1])
            h=prediction['maps'][0];ok=prediction['valid'][0]
            image=render(frame,h,ok,alpha=cfg['overlay_alpha'],area=cfg['highlight_area'])
            # Preserve exact RGB identity on unavailable/no-map frames.
            v=runtime.visual_frame(image).cpu().half() if ok else rgb['visual'][j]
            grounded.append(v);maps.append(h);native.append(prediction['native_logits'][0]);valid.append(ok);reasons.append(prediction['reasons'][0]);timings.append(prediction['timing'])
        data={'source_id':row['source_id'],'media_sha256':row['media_sha256'],'dependency_key':expected,
            'visual_grounded':torch.stack(grounded),'maps':maps,'native_logits':native,'map_valid':valid,'map_reasons':reasons,
            'timestamps':rgb['timestamps'],'localizer_sha256':localizer.lock['checkpoint_sha256'],
            'map_origin':'EZVSL_PRED_CAUSAL','GT_used':False,'timings':timings,'source_role':row['role']}
        torch.save(data,gp.with_suffix('.part'));gp.with_suffix('.part').rename(gp)
        print('av-grounded dev',shard,i+1,row['source_id'],sum(valid),'/',len(valid),flush=True)
    journal('development causal AV and predicted views cached',shard=shard,shards=shards)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.shard,a.shards)

