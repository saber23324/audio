import json,inspect
from pathlib import Path
import numpy as np
import pytest
import torch
from utility_gate.gate_first_native.common import ROOT,training_rows,sha256
from utility_gate.gate_first_native.fusion import GroundedFusionRouter
from utility_gate.gate_first_native.gate import history,accept
from utility_gate.gate_first_native.policy_labels import bits,code
from utility_gate.gate_first_native.runtime import append_step_features

def test_policy_grid_is_complete_not_additive_fake():
    assert len({tuple(bits(i)) for i in range(16)})==16
    for p in range(16):assert code(bits(p))==p
    source=inspect.getsource(__import__('utility_gate.gate_first_native.policy_labels',fromlist=['run']).run)
    assert 'r.score(row,z)' in source and 'for policy in range(16)' in source

@pytest.mark.parametrize('index',[0,1,2,3])
def test_future_choice_bits_do_not_enter_history(index):
    past=(1<<index)-1;assert history(past,index)==history(past|(1<<30),index)
    assert history(past,index)[0] in (0.,1.)

@pytest.mark.parametrize('batch',[1,3])
def test_new_RA_zero_init_and_off_are_identity(batch):
    m=GroundedFusionRouter(dim=17,key_dim=8,rank=4);rgb=torch.randn(batch,196,17);g=torch.randn_like(rgb);base=torch.randn(batch,64,17)
    assert torch.equal(m(rgb,g,base),base)
    with torch.no_grad():m.output.weight.fill_(.1)
    assert m(rgb,g,base,torch.zeros(batch,dtype=torch.bool)) is base
    assert m(rgb,g,base,torch.ones(batch,dtype=torch.bool),alpha=0) is base
    assert m(rgb,g,base).shape==base.shape

def test_forced_training_exposure_has_gradient_without_forcing_deployment():
    m=GroundedFusionRouter(dim=17,key_dim=8,rank=4);v=torch.randn(4,196,17);g=torch.randn_like(v);base=torch.randn(4,64,17)
    deployed=torch.zeros(4,dtype=torch.bool);forced=~deployed
    assert m(v,g,base,deployed) is base
    loss=(m(v,g,base,forced)-torch.ones_like(base)).square().mean();loss.backward()
    assert m.output.weight.grad.norm()>0

def test_future_inactive_values_do_not_change_current_RF():
    m=GroundedFusionRouter(dim=17,key_dim=8,rank=4)
    with torch.no_grad():m.output.weight.fill_(.01)
    v=torch.randn(3,196,17);g=torch.randn_like(v);base=torch.randn(3,64,17);on=torch.tensor([True,False,False])
    a=m(v,g,base,on)[0];v[1:]=float('nan');g[1:]=float('inf');b=m(v,g,base,on)[0]
    assert torch.equal(a,b)

def test_current_features_are_future_invariant():
    v=torch.randn(4,196,3584);a=torch.randn(4,96,768);valid=torch.ones(4,dtype=torch.bool);ts=torch.tensor([2.,4.,6.,8.])
    first=append_step_features(v,a,valid,ts,1);v[2:]=float('nan');a[2:]=float('inf');second=append_step_features(v,a,valid,ts,1)
    for n in ('visual','audio','context','valid_audio'):assert torch.equal(getattr(first,n),getattr(second,n))

@pytest.mark.parametrize('role',['confirmation_music','confirmation_valor','test','../val'])
def test_test_gold_never_accessible_to_fitting_loaders(role):
    with pytest.raises(ValueError,match='forbidden'):training_rows(role)

@pytest.mark.parametrize('threshold,expected',[('-inf',True),('inf',False),(0.,False)])
def test_no_preset_activation_rate_and_strict_threshold_ties(threshold,expected):
    assert accept({'kind':'u','tau_u':threshold},0.,0.,True)==expected
    assert not accept({'kind':'u','tau_u':threshold},999.,999.,False)

def test_actual_staged_training_and_frozen_boundaries():
    a=json.loads((ROOT/'locks/gate_a_lock.json').read_text());r=json.loads((ROOT/'locks/fusion_router_lock.json').read_text())
    assert a['fitting_before_Router']
    assert r['optimizer_steps']==1968 and r['parameters_trainable']==696320
    assert r['Gate_A_lock_sha256_before_after']==sha256(ROOT/'locks/gate_a_lock.json')
    assert r['backbone_state_sha_before']==r['backbone_state_sha_after']
    assert set(r['changed_parameters'])=={'query','key.weight','value.weight','output.weight'}
    assert r['gradient_update_window_counts']['forced_on_exposure']>0

def test_actual_large_model_identity_native_bypass_and_future_checks():
    d=json.loads((ROOT/'audit/integration_checks.json').read_text())
    assert d['completed'] and all(d['identity'].values())
    assert all(v==0 for v in d['native_Router_forward_counts'].values())
    assert d['actual_native_visual_tokens']==5840 and d['actual_native_images']==8
    assert d['future_current_Gate_output_exact'] and d['future_raw_audio_and_predicted_maps_exact']

def test_all_native_val_controls_have_matched_format_and_no_Router():
    ps=list((ROOT/'results/val_C').glob('*.jsonl'));preds=[]
    for p in ps:
        if '.actions.' not in p.name:preds.extend(json.loads(s) for s in p.read_text().splitlines())
    assert len({r['source_id'] for r in preds})==48
    for r in preds:
        assert all(v==0 for v in r['native_router_call_deltas'].values())
        if r['strategy']!='C_DEPLOY':assert r['visual_tokens']==5840 and r['actual_native_images_per_frame']==[2]*4
    assert len({r['system_prompt_text'] for r in preds})==1

def test_same_GA_mask_is_used_by_fixed_and_trained_Router():
    for p in (ROOT/'results/val_AB').glob('*.actions.jsonl'):
        data=[json.loads(s) for s in p.read_text().splitlines()];by={}
        for r in data:by.setdefault(r['strategy'],{})[r['frame_index']]=r['gate_on']
        assert by['A_MAIN_GA']==by['B3_GA_R']==by['B4_GA_F0']==by['B5_GA_dupR']==by['B6_GA_shamR']

