"""Versioned frozen RGB caches; no audio enters the Stage1 training feature file."""
from __future__ import annotations
import argparse,time,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from relevance_gate.media import frame_at_or_before
from utility_gate.runtime import clocked
from .runtime import Runtime
from .common import ROOT,read_jsonl,sha256,fingerprint,write_json,journal,read_config,train_rows

def times(duration):
    # Future duration only terminates the stream. Earlier observation times
    # remain2,4,... regardless of final duration; cap is first32 arrivals.
    return np.arange(2.,min(float(duration),64.)+1e-8,2.,dtype=np.float32)

def cache_path(source):return ROOT/f'features/rgb/{source}.pt'

def make_rgb(runtime,row):
    stamps=times(row['duration_s'])
    if not len(stamps):raise ValueError('no scheduled visual observation')
    frames=[];pts=[];vs=[];decode_s=0.;vision_s=0.
    for t in stamps:
        (frame,pt),ds=clocked(lambda:frame_at_or_before(row['video_path'],float(t)))
        v,es=clocked(lambda:runtime.visual_frame(frame));frames.append(frame);pts.append(pt);vs.append(v.cpu().half());decode_s+=ds;vision_s+=es
    key=fingerprint({'media':row['media_sha256'],'vision':runtime.extractor.metadata,'schedule':'2s_first32','native196':True})
    return {'source_id':row['source_id'],'source_video_id':row['source_video_id'],'dataset':row['dataset'],'media_sha256':row['media_sha256'],
        'timestamps':torch.tensor(stamps),'frame_pts_s':pts,'frames':frames,'visual':torch.stack(vs),
        'feature_cache_key':key,'encoder_metadata':runtime.extractor.metadata,'audio_input_used':False,
        'timing':{'frame_decode_s':decode_s,'SigLIP_projector_pool_s':vision_s},'source_role':row['role']}

def load_rgb(row):
    path=cache_path(row['source_id']);d=torch.load(path,map_location='cpu',weights_only=False)
    if d['media_sha256']!=row['media_sha256'] or d['source_id']!=row['source_id'] or d['audio_input_used'] is not False:raise ValueError('wrong frozen RGB cache')
    if d['visual'].shape[1:]!=(196,3584) or len(d['timestamps'])>32:raise ValueError('visual interface changed')
    return d

def run(role,shard=0,shards=1):
    if role.startswith('confirmation'):
        from .locks import guard_confirmation
        guard_confirmation(role)
        rows=read_jsonl(ROOT/f'manifests/{role}.jsonl')
    else:
        # RGB prewarming can precede protocol fitting. Only explicit original
        # train-derived source manifests are readable here.
        if role not in ('stage1_train','stage1_val'):raise ValueError(role)
        rows=read_jsonl(ROOT/f'manifests/{role}.jsonl')
        if any(r['role']!=role or r['source_split']!='train' for r in rows):raise ValueError('illegal Stage1 feature role')
    unique={}
    for r in rows:unique.setdefault(r['source_id'],r)
    runtime=Runtime()
    if shard==0 and role=='stage1_val':print(runtime.legacy_regression(),flush=True)
    for i,row in enumerate(list(unique.values())[shard::shards]):
        path=cache_path(row['source_id'])
        if path.exists():load_rgb(row);continue
        data=make_rgb(runtime,row);temporary=path.with_suffix('.part');torch.save(data,temporary);temporary.rename(path)
        print(role,shard,i+1,row['source_id'],len(data['timestamps']),flush=True)
    journal('frozen RGB-only feature prewarm complete',role=role,shard=shard,shards=shards)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.role,a.shard,a.shards)
