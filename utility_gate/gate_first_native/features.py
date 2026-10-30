"""Four-arrival train/val cache; actual new policy targets are never reused."""
import argparse,json
import numpy as np
import torch
from utility_gate.features import feature_payload
from .common import ROOT,PREV,training_rows,sha256,fingerprint,journal
from .runtime import Runtime,gate_features

def path(source):return ROOT/f'features/development/{source}.pt'
def load(row):
    d=torch.load(path(row['source_id']),map_location='cpu',weights_only=False)
    if d['media_sha256']!=row['media_sha256'] or d['timestamps'].tolist()!=[2.,4.,6.,8.]:raise ValueError('stale/incorrect feature cache')
    return d

def run(shard=0,shards=1):
    rows=training_rows('train')+training_rows('val');unique={r['source_id']:r for r in rows};r=Runtime()
    if shard==0:print(r.legacy_fixture(),flush=True)
    for i,row in enumerate(list(unique.values())[shard::shards]):
        target=path(row['source_id'])
        if target.exists():load(row);continue
        rgb=torch.load(PREV/f'features/rgb/{row["source_id"]}.pt',map_location='cpu',weights_only=False)
        ground=torch.load(PREV/f'features/development_grounded/{row["source_id"]}.pt',map_location='cpu',weights_only=False)
        if rgb['media_sha256']!=row['media_sha256'] or ground['media_sha256']!=row['media_sha256'] or rgb['timestamps'][:4].tolist()!=[2.,4.,6.,8.]:raise ValueError('prior frozen visual cache mismatch')
        v=rgb['visual'][:4];vg=ground['visual_grounded'][:4];ts=rgb['timestamps'][:4]
        av=r.audio_features(row['video_path'],ts)
        features=feature_payload(gate_features(v,av['audio'],av['audio_valid'],ts))
        z,p=r.vision_latents(v);zg,pg=r.vision_latents(vg);f0=z+.25*(zg-z)
        value={'source_id':row['source_id'],'media_sha256':row['media_sha256'],'role':row['role'],'timestamps':ts,
            'frames':rgb['frames'][:4],'frame_pts_s':rgb['frame_pts_s'][:4],'visual':v,'visual_grounded':vg,
            'maps':ground['maps'][:4],'map_valid':ground['map_valid'][:4],**av,'features':features,
            'V1':z.cpu(),'P_V1':p.cpu(),'F0':f0.cpu(),'P_ground':pg.cpu(),
            'V1_checkpoint_sha256':sha256(__import__('utility_gate.gate_first_native.common',fromlist=['V1']).V1),
            'visual_cache_reuse':'same frozen SigLIP/projector and predicted maps at2/4/6/8; no old Utility targets reused',
            'prior_RGB_cache_sha256':sha256(PREV/f'features/rgb/{row["source_id"]}.pt'),'prior_ground_cache_sha256':sha256(PREV/f'features/development_grounded/{row["source_id"]}.pt')}
        torch.save(value,target.with_suffix('.part'));target.with_suffix('.part').rename(target)
        print('new four-frame cache',shard,i+1,row['source_id'],flush=True)
    journal('four-arrival development caches complete',shard=shard,shards=shards)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.shard,a.shards)

