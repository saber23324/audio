"""Only legal development caches; confirmation never takes this code path."""
import argparse
import numpy as np
import torch
from relevance_gate.media import frame_at_or_before
from utility_gate.features import feature_payload
from .common import ROOT,PREV,V1,RA,train_rows,sha256,journal
from .runtime import Runtime,Localizer,render,gate_features

def path(source):return ROOT/f'features/{source}.pt'
def load(row):
    d=torch.load(path(row['source_id']),map_location='cpu',weights_only=False)
    if d['media_sha256']!=row['media_sha256'] or d['timestamps'].tolist()!=[2.,4.,6.,8.] or d['R_A_sha256']!=sha256(RA):raise ValueError('cache identity mismatch')
    return d
def acoustic_stats(w):
    w=torch.as_tensor(w).float();spec=torch.fft.rfft(w).abs();freq=torch.linspace(0,1,spec.shape[-1],device=w.device)
    energy=w.square().mean(-1).sqrt();absolute=w.abs().mean(-1);zcr=(w[...,1:]*w[...,:-1]<0).float().mean(-1)
    return torch.stack((energy,absolute,zcr,w.abs().amax(-1),torch.log1p(energy*100), (spec*freq).sum(-1)/spec.sum(-1).clamp_min(1e-8)),-1)
def run(shard=0,shards=1):
    rows=train_rows('train')+train_rows('val');unique={r['source_id']:r for r in rows};runtime=Runtime();loc=Localizer(runtime.device)
    if shard==0:runtime.legacy_fixture()
    for i,row in enumerate(list(unique.values())[shard::shards]):
        out=path(row['source_id'])
        if out.exists():load(row);continue
        if row['role']=='train':
            old=PREV/f'features/development/{row["source_id"]}.pt';d=torch.load(old,map_location='cpu',weights_only=False)
            if d['media_sha256']!=row['media_sha256']:raise ValueError('old visual bytes differ')
            d={k:d[k] for k in ('source_id','media_sha256','role','timestamps','frames','frame_pts_s','visual','visual_grounded','maps','map_valid','windows','audio','audio_valid','audio_reasons','audio_traces')}
            d['visual_cache_provenance']={'path':str(old),'sha256':sha256(old),'reused_only_frozen_frontend':True,'old_F0_labels_or_latents_used':False}
        else:
            times=torch.tensor([2.,4.,6.,8.]);frames=[];pts=[];vv=[];gg=[];maps=[];good=[]
            av=runtime.audio_features(row['video_path'],times)
            for t,w,ok in zip(times,av['windows'],av['audio_valid']):
                frame,pt=frame_at_or_before(row['video_path'],float(t))
                if pt>float(t)+1e-6:raise ValueError('future frame')
                frames.append(frame);pts.append(pt);vv.append(runtime.visual_frame(frame).half().cpu())
                pred=loc.predict([frame],w[None],torch.tensor([bool(ok)]));maps.append(pred['maps'][0]);good.append(pred['valid'][0])
                gg.append(runtime.visual_frame(render(frame,pred['maps'][0],pred['valid'][0])).half().cpu())
            d={'source_id':row['source_id'],'role':row['role'],'media_sha256':row['media_sha256'],'timestamps':times,
               'frames':np.stack(frames),'frame_pts_s':pts,'visual':torch.stack(vv),'visual_grounded':torch.stack(gg),'maps':np.stack(maps),'map_valid':good,**av}
        v=d['visual'].float().to(runtime.device);g=d['visual_grounded'].float().to(runtime.device)
        z,p=runtime.vision_latents(v);ra=[]
        for t in range(4):ra.append(runtime.fusion(v[t:t+1],g[t:t+1],z[t:t+1])[0].cpu())
        d.update(V1=z.cpu(),P_V1=p.cpu(),R_A=torch.stack(ra),R_A_sha256=sha256(RA),V1_sha256=sha256(V1),
                 features=feature_payload(gate_features(d['visual'],d['audio'],d['audio_valid'],d['timestamps'])),
                 acoustic_features=feature_payload(gate_features(d['visual'],acoustic_stats(d['windows'])[:,None],d['audio_valid'],d['timestamps'])))
        torch.save(d,out.with_suffix('.part'));out.with_suffix('.part').rename(out)
        print('DEV cache',shard,i+1,row['source_id'],flush=True)
    journal('development frozen cache complete',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();run(a.shard,a.shards)

