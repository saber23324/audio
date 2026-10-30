"""Tiny new causal policies; input schema contains no QA or second image."""
import torch
from torch import nn
import torch.nn.functional as F
from utility_gate.features import GateFeatures

FAMILIES=('audio_r','audio_no_r','visual','acoustic','history')
def history(past,index,extended=False):
    if not 0<=index<4 or past<0:raise ValueError('invalid causal history')
    p=int(past)&((1<<index)-1);n=p.bit_count();h=[n/max(index,1),n/4]
    if extended:h.extend([float(bool(p&(1<<(index-j)))) if index>=j else 0. for j in (1,2)])
    return h
class FixedRouterGate(nn.Module):
    def __init__(self,family,hidden=64,dropout=.1):
        super().__init__()
        if family not in FAMILIES:raise ValueError(family)
        self.spec=dict(family=family,hidden=hidden,dropout=dropout);self.family=family
        self.use_audio=family!='visual';self.use_r=family in ('audio_r','history');self.extended=family=='history'
        self.visual_projection=nn.Sequential(nn.LayerNorm(7168,elementwise_affine=False),nn.Linear(7168,hidden),nn.GELU())
        if self.use_audio:self.audio_projection=nn.Sequential(nn.LayerNorm(12 if family=='acoustic' else 1536,elementwise_affine=False),nn.Linear(12 if family=='acoustic' else 1536,hidden),nn.GELU())
        width=hidden*4+4 if self.use_audio else hidden+4
        self.comparison=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Dropout(dropout))
        self.decision=nn.Sequential(nn.Linear(hidden+(4 if self.extended else 2)+int(self.use_r),32),nn.GELU(),nn.Linear(32,1))
    def encode(self,f):
        if not isinstance(f,GateFeatures):raise TypeError('typed gold-free features required')
        f.validate();v=self.visual_projection(f.visual.detach())
        if self.use_audio:
            a=self.audio_projection(f.audio.detach());x=torch.cat((v,a,v-a,v*a,f.context.detach()),-1)
        else:x=torch.cat((v,f.context.detach()),-1)
        return self.comparison(x)
    def logits(self,h,hist,r=None):
        xs=[h,hist]
        if self.use_r:
            if r is None:raise ValueError('frozen proxy feature required')
            xs.append(r)
        return self.decision(torch.cat(xs,-1)).squeeze(-1)
    def forward_features(self,f,past,index,r=None):
        h=self.encode(f);hist=torch.tensor([history(past,index,self.extended)],device=h.device,dtype=h.dtype)
        return self.logits(h,hist,r)

def log_policy(model,f,r):
    """[B,16] complete probabilities; features arrive per t, past actions only."""
    if len(f.visual)%4:raise ValueError('four arrived frames per source')
    B=len(f.visual)//4;h=model.encode(f).reshape(B,4,-1);rv=r.reshape(B,4,1);valid=f.valid_audio.reshape(B,4)
    masks=torch.tensor([[(m>>t)&1 for t in range(4)] for m in range(16)],device=h.device,dtype=h.dtype)
    result=h.new_zeros(B,16)
    for t in range(4):
        hist=torch.tensor([history(m,t,model.extended) for m in range(16)],device=h.device,dtype=h.dtype)[None].expand(B,-1,-1)
        logits=model.logits(h[:,t,None].expand(-1,16,-1),hist,rv[:,t,None].expand(-1,16,-1))
        term=-F.binary_cross_entropy_with_logits(logits,masks[None,:,t].expand(B,-1),reduction='none')
        term=torch.where(valid[:,t,None],term,torch.where(masks[None,:,t].bool(),-1e4,0.))
        result=result+term
    return result
def target_distribution(nll,valid,lam,temperature):
    masks=torch.tensor([[(m>>t)&1 for t in range(4)] for m in range(16)],device=nll.device,dtype=nll.dtype)
    eligible=(~masks.bool()[None]|valid[:,None,:]).all(-1)
    scores=-(nll+float(lam)*masks.sum(-1))/float(temperature);scores=scores.masked_fill(~eligible,-torch.inf)
    return scores.softmax(-1)
def load_gate(path,device='cpu'):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p['kind']!='fixed_router_sequence_gate':raise ValueError('not a new Gate checkpoint')
    m=FixedRouterGate(**p['spec']);m.load_state_dict(p['state_dict'],strict=True)
    return m.to(device).eval().requires_grad_(False),p

