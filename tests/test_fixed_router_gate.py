import numpy as np
import pytest
import torch
from utility_gate.features import GateFeatures,load_feature_payload
from utility_gate.fixed_router_gate.gate_models import FixedRouterGate,history,log_policy,target_distribution,FAMILIES
from utility_gate.fixed_router_gate.online_inference import Controllers
from utility_gate.fixed_router_gate.selection import choose
from utility_gate.fixed_router_gate.common import train_rows
from utility_gate.adaptive_grounded64.runtime import append_step_features

def fixture(B=2,audio=1536):
    torch.manual_seed(7)
    return GateFeatures(torch.randn(B*4,7168),torch.randn(B*4,audio),torch.randn(B*4,4),torch.ones(B*4,dtype=torch.bool))
@pytest.mark.parametrize('family',FAMILIES)
def test_complete_sequence_distribution_normalized_and_gate_only_gradients(family):
    m=FixedRouterGate(family,dropout=0).eval();f=fixture(audio=12 if family=='acoustic' else 1536);r=torch.randn(2,4)
    lp=log_policy(m,f,r);assert lp.shape==(2,16)
    assert torch.allclose(lp.exp().sum(-1),torch.ones(2),atol=1e-6)
    nll=torch.randn(2,16);q=target_distribution(nll,torch.ones(2,4,dtype=torch.bool),.1,.25)
    loss=-(q*lp).sum(-1).mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert any(p.grad.abs().sum()>0 for p in m.parameters())
@pytest.mark.parametrize('invalid',[(0,),(1,3),(0,1,2,3)])
def test_availability_is_hard_zero_without_nans(invalid):
    m=FixedRouterGate('audio_r',dropout=0).eval();f=fixture(B=1)
    for i in invalid:f.valid_audio[i]=False
    p=log_policy(m,f,torch.zeros(1,4)).exp();assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(),torch.tensor(1.),atol=1e-6)
    for code in range(16):
        if any(code&(1<<i) for i in invalid):assert p[0,code]==0
    q=target_distribution(torch.zeros(1,16),f.valid_audio.reshape(1,4),0,1)
    assert torch.allclose(p.sum(),q.sum())
def test_joint_target_retains_nonadditive_qa_truth():
    # Each single intervention helps, all-on is harmful. Full16 labels retain it.
    nll=torch.full((1,16),8.);nll[0,0]=3
    for t in range(4):nll[0,1<<t]=1
    nll[0,15]=10
    q=target_distribution(nll,torch.ones(1,4,dtype=torch.bool),0,.25)
    assert q[0,1]>q[0,15]*1000 and q[0,1]>q[0,0]
def test_no_future_features_actions_or_question_fields():
    v=torch.randn(4,196,3584);a=torch.randn(4,96,768);ts=torch.tensor([2.,4.,6.,8.]);valid=torch.ones(4,dtype=torch.bool)
    f=append_step_features(v,a,valid,ts,1);v[2:]=float('nan');a[2:]=float('inf');g=append_step_features(v,a,valid,ts,1)
    assert all(torch.equal(getattr(f,k),getattr(g,k)) for k in ('visual','audio','context','valid_audio'))
    for p in range(4):assert history(p,2,True)==history(p|12,2,True)
    with pytest.raises(ValueError):load_feature_payload({'visual':f.visual,'audio':f.audio,'context':f.context,'valid_audio':f.valid_audio,'question':'leak'})
def test_visual_gate_ignores_all_audio_values():
    m=FixedRouterGate('visual',dropout=0).eval();f=fixture(1);g=GateFeatures(f.visual,f.audio*1000,f.context,f.valid_audio)
    assert torch.equal(log_policy(m,f,torch.zeros(1,4)),log_policy(m,g,torch.randn(1,4)))
@pytest.mark.parametrize('cap',range(5))
def test_online_cap_uses_past_counter_no_future_ranking(cap):
    m=FixedRouterGate('history',dropout=0).eval();f=fixture(2)
    code,_=choose(m,f,torch.zeros(2,4),-1,cap=cap)
    assert code==[(1<<cap)-1]*2
def test_confirm_test_roles_are_rejected_before_manifest_read():
    for role in ('test','confirmation_music','confirmation_valor'):
        with pytest.raises(ValueError):train_rows(role)
def test_invalid_audio_rejected_before_any_gate_read():
    c=Controllers();out=c.one({'kind':'learned','checkpoint':'does_not_exist'},None,None,0,0,False,0.)
    assert out['on'] is False and not c.models
def test_model_scale_and_parameter_ownership():
    for family in FAMILIES:
        m=FixedRouterGate(family);assert sum(p.numel() for p in m.parameters())<600000
        assert all(p.requires_grad for p in m.parameters())
        assert not any('router' in n.lower() or 'beats' in n.lower() for n,p in m.named_parameters())
