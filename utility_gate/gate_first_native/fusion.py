"""R-A: new zero-output low-rank64-slot cross-view residual; V1 remains frozen."""
import json
import torch
from torch import nn
import torch.nn.functional as F
from .common import ROOT,sha256

class GroundedFusionRouter(nn.Module):
    def __init__(self,query_init=None,key_init=None,dim=3584,key_dim=128,rank=32):
        super().__init__();self.spec={'dim':dim,'key_dim':key_dim,'rank':rank}
        self.query=nn.Parameter(torch.randn(1,64,key_dim)*.02)
        self.key=nn.Linear(dim,key_dim,bias=False);self.value=nn.Linear(dim,rank,bias=False);self.output=nn.Linear(rank,dim,bias=False)
        with torch.no_grad():
            if query_init is not None:self.query.copy_(query_init.detach()[...,:key_dim])
            if key_init is not None:self.key.weight.copy_(key_init.detach()[:key_dim])
            self.output.weight.zero_()
        self.last_diagnostics={}
    def forward(self,rgb,ground,base,on=None,alpha=1.):
        if rgb.ndim!=3 or rgb.shape[1:]!=(196,self.spec['dim']) or ground.shape!=rgb.shape or base.shape!=(len(rgb),64,self.spec['dim']):raise ValueError('expected native196+196 and64-slot base')
        if on is None:on=torch.ones(len(rgb),dtype=torch.bool,device=rgb.device)
        if on.dtype!=torch.bool or on.shape!=(len(rgb),):raise ValueError('explicit window mask required')
        if alpha==0 or not bool(on.any()):return base
        if not torch.isfinite(rgb[on]).all() or not torch.isfinite(ground[on]).all():raise ValueError('nonfinite active visual input')
        values=torch.cat((rgb[on],ground[on]),1);q=F.normalize(self.query.expand(len(values),-1,-1),dim=-1);k=F.normalize(self.key(values),dim=-1)
        p=torch.softmax((q@k.transpose(-1,-2))/.05,dim=-1);delta=self.output(p@self.value(values))
        result=base.clone();result[on]=base[on]+float(alpha)*delta
        self.last_diagnostics={'residual_to_base_norm':float(delta.detach().norm()/base[on].detach().norm().clamp_min(1e-8)),
            'mean_attention_mass_on_grounded_view':float(p.detach()[...,196:].sum(-1).mean()),'output_tokens_per_frame':64,
            'native_RGB_convex_hull_claim':False,'value_origin':'projected RGB and predicted-grounded visual features only'}
        return result

def load_fusion(path=None,device='cpu'):
    path=path or ROOT/'models/fusion_router/selected.pt';p=torch.load(path,map_location='cpu',weights_only=False)
    model=GroundedFusionRouter(**p['spec']);model.load_state_dict(p['state_dict'],strict=True)
    return model.to(device).eval().requires_grad_(False),p

