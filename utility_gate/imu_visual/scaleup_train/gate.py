"""Audited Audio Gate architecture; auxiliary input is cheap causal IMU only."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from utility_gate.features import GateFeatures
from utility_gate.fixed_router_gate.gate_models import FixedRouterGate,history

@dataclass(frozen=True)
class IMUFeatures:
    visual:torch.Tensor
    imu:torch.Tensor
    context:torch.Tensor
    valid:torch.Tensor
    def validate(self):
        if self.visual.ndim!=2 or self.visual.shape[-1]!=7168 or self.imu.shape!=(len(self.visual),27) or self.context.shape!=(len(self.visual),4) or self.valid.shape!=(len(self.visual),):raise ValueError('gold-free [7168,27,4,bool] feature contract')
        if self.valid.dtype!=torch.bool or not all(torch.isfinite(x).all() for x in (self.visual,self.imu,self.context)):raise ValueError('malformed cheap Gate features')
    def legacy(self):
        self.validate();return GateFeatures(self.visual,self.imu,self.context,self.valid)
    def to(self,device):return IMUFeatures(*(x.to(device) for x in (self.visual,self.imu,self.context,self.valid)))

class IMUGate(nn.Module):
    def __init__(self,extended=True,hidden=64,dropout=.1):
        super().__init__();self.spec={'extended':extended,'hidden':hidden,'dropout':dropout}
        self.core=FixedRouterGate('acoustic',hidden,dropout)
        # Only modality-specific input width changes:12 cheap acoustic ->27
        # cheap IMU. No audio encoder or historical correspondence proxy.
        self.core.audio_projection=nn.Sequential(nn.LayerNorm(27,elementwise_affine=False),nn.Linear(27,hidden),nn.GELU())
        self.core.extended=extended
        self.core.decision=nn.Sequential(nn.Linear(hidden+(4 if extended else 2),32),nn.GELU(),nn.Linear(32,1))
    def encode(self,f):return self.core.encode(f.legacy())
    def step(self,f,past,index):
        h=self.encode(f);hist=h.new_tensor([history(past,index,self.core.extended)]);return self.core.logits(h,hist)
    def log_policy(self,f):
        if len(f.visual)%4:raise ValueError('four arrived observations per QA episode')
        batch=len(f.visual)//4;h=self.encode(f).reshape(batch,4,-1);valid=f.valid.reshape(batch,4);bits=h.new_tensor([[(m>>t)&1 for t in range(4)] for m in range(16)]);result=h.new_zeros(batch,16)
        for t in range(4):
            hist=h.new_tensor([history(m,t,self.core.extended) for m in range(16)])[None].expand(batch,-1,-1);logits=self.core.logits(h[:,t,None].expand(-1,16,-1),hist);term=-F.binary_cross_entropy_with_logits(logits,bits[None,:,t].expand(batch,-1),reduction='none');result+=torch.where(valid[:,t,None],term,torch.where(bits[None,:,t].bool(),-1e4,0.))
        return result
def from_payload(payload,device='cpu'):
    return IMUFeatures(*(payload[k].to(device) for k in ('visual','imu','context','valid')))
def take(f,t):return IMUFeatures(*(x[t:t+1] for x in (f.visual,f.imu,f.context,f.valid)))
@torch.inference_mode()
def choose(model,f,threshold=.5,cap=4):
    past=0;out=[];scores=[]
    for t in range(4):
        score=float(model.step(take(f,t),past,t).sigmoid());on=bool(f.valid[t]) and sum(out)<cap and score>threshold;out.append(on);scores.append(score)
        if on:past|=1<<t
    return out,scores
def load_gate(path,device='cpu'):
    p=torch.load(path,map_location='cpu',weights_only=False);m=IMUGate(**p['spec']);m.load_state_dict(p['state_dict'],strict=True);return m.to(device).eval().requires_grad_(False),p
