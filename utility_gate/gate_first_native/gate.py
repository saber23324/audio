"""Tiny causal policy-conditioned utility model; independent old correspondence proxy."""
import json
import numpy as np
import torch
from torch import nn
from utility_gate.dual_head.model import DualHeadGate,load_dual_head
from utility_gate.features import take_features
from .common import ROOT,OLD_GATE,OLD_RULE,sha256

class PolicyGate(nn.Module):
    def __init__(self,mode,scale):
        super().__init__();base=DualHeadGate();self.mode=mode;self.scale=float(scale)
        self.use_audio=mode!='vision_rank';self.use_r=mode=='dual_rank';self.classification=mode=='u_classification'
        self.visual_projection=base.visual_projection;self.audio_projection=base.audio_projection;self.comparison=base.comparison
        if not self.use_audio:self.audio_projection.requires_grad_(False)
        self.utility_head=nn.Sequential(nn.Linear(64+2+int(self.use_r),32),nn.GELU(),nn.Linear(32,1))
    def encode(self,features):
        features.validate();v=self.visual_projection(features.visual.detach())
        a=self.audio_projection(features.audio.detach()) if self.use_audio else torch.zeros_like(v)
        return self.comparison(torch.cat((v,a,v-a,v*a,features.context.detach()),-1))
    def utility(self,h,history,r=None,raw=False):
        x=[h,history]
        if self.use_r:
            if r is None:raise ValueError('frozen proxy feature required')
            x.append(r)
        out=self.utility_head(torch.cat(x,-1)).squeeze(-1)
        if self.classification:return out if raw else out.sigmoid()
        return out*self.scale
    def forward_features(self,features,history,r=None,raw=False):return self.utility(self.encode(features),history,r,raw)

def history(past,index):
    if index<0 or past<0:raise ValueError('nonnegative causal history required')
    n=(int(past)&((1<<index)-1)).bit_count();return [n/max(index,1),float(np.log1p(n))]

def load_gate(path):
    p=torch.load(path,map_location='cpu',weights_only=True);m=PolicyGate(p['mode'],p['scale']);m.load_state_dict(p['state_dict'],strict=True)
    return m.eval().requires_grad_(False),p

def freeze_proxy():return load_dual_head(OLD_GATE)[0]
@torch.inference_mode()
def proxy_scores(model,f):
    o=model.forward_features(f);v=model.visual_projection(f.visual);a=model.audio_projection(f.audio)
    return o.correspondence_logit[:,0],o.utility_delta[:,0],torch.nn.functional.cosine_similarity(v,a,dim=-1)

def accept(rule,u,r,available,random_draw=None):
    if not available:return False
    kind=rule['kind']
    if kind=='off':return False
    if kind=='on':return True
    if kind=='random':return random_draw<rule['probability']
    if kind=='u':return u>float(rule['tau_u'])
    if kind=='dual':return u>float(rule['tau_u']) and r>float(rule['tau_r'])
    if kind=='r':return r>float(rule['tau_r'])
    raise ValueError(kind)

@torch.inference_mode()
def choose_mask(model,features,r_scores,rule):
    h=model.encode(features);past=0;mask=[];scores=[]
    for t in range(len(h)):
        u=float(model.utility(h[t:t+1],torch.tensor([history(past,t)]),r_scores[t:t+1,None]))
        on=accept(rule,u,float(r_scores[t]),bool(features.valid_audio[t]));mask.append(on);scores.append(u)
        if on:past|=1<<t
    return mask,scores

