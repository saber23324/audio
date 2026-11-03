"""Regressions for new scale-up isolation and paid-network boundaries."""
import os
import numpy as np
import torch
import pytest

def test_scaleup_parent_family_groups_parts_tasks_and_aliases():
    from utility_gate.imu_visual.scaleup_train.common import parent_group
    assert parent_group('holo:R066-15July-ATV-Part1')==parent_group('R066-15July-Circuit-Breaker-part2')
    assert parent_group('R0027-12-GoPro')==parent_group('R27-12July-GoPro')
    assert parent_group('z017-june20-take1')==parent_group('Z017-june20-take2')
    assert parent_group('R066-15July-ATV')!=parent_group('R067-15July-ATV')

def test_accuracy_gate_target_prioritizes_actual_generation_over_nll():
    from utility_gate.imu_visual.scaleup_train.train_gate import objective
    logits=torch.zeros(1,16,requires_grad=True);lp=logits.log_softmax(-1)
    nll=torch.full((1,16),.01);nll[0,7]=10.
    correct=torch.zeros(1,16,dtype=torch.bool);correct[0,7]=True
    loss=objective(lp,nll,{'name':'classification','loss':'hard_trajectory','lambda':0.},.25,correct)
    loss.backward()
    assert logits.grad.argmin().item()==7

def test_native_adapter_encodes_one_image_and_restores_method(monkeypatch):
    from types import SimpleNamespace
    from utility_gate.imu_visual.scaleup_train import native_paid as m
    class Encoder(torch.nn.Module):
        def __init__(self):super().__init__();self.batches=[]
        def encode_images(self,x):self.batches.append(len(x));return x*2
    r=SimpleNamespace(model=Encoder());original=r.model.encode_images
    monkeypatch.setattr(m,'prepare',lambda r,ids,images:(r.model.encode_images(images),None,{}))
    pixels=torch.arange(12).reshape(4,3);result=m.prepare_b1(r,None,pixels)
    assert torch.equal(result[0],pixels*2) and result[3]==[1]*4 and r.model.encode_images==original
    def fail(*args):raise ValueError('simulated native failure')
    monkeypatch.setattr(m,'prepare',fail)
    with pytest.raises(ValueError):m.prepare_b1(r,None,pixels)
    assert r.model.encode_images==original

def test_scaleup_paid_range_cannot_fetch_network(tmp_path,monkeypatch):
    from utility_gate.imu_visual.scaleup_train import range_io as m
    monkeypatch.setattr(m,'DATA',tmp_path);monkeypatch.setattr(m,'OLD_DATA',tmp_path/'old')
    monkeypatch.setenv('IMU_SCALEUP_CACHE_ONLY','1')
    monkeypatch.setattr(m.requests,'get',lambda *a,**k:pytest.fail('network during measured cost'))
    with pytest.raises(FileNotFoundError):m.get_range('https://example.invalid/test',0,128)

def test_scaleup_router_validation_does_not_read_sealed_features(tmp_path,monkeypatch):
    from utility_gate.imu_visual.scaleup_train import train_router as m
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results/router').mkdir(parents=True)
    qq=[{'qa_id':role,'role':role} for role in ['train','val','holdout']]
    monkeypatch.setattr(m,'rows',lambda:qq)
    accessed=[]
    def load(row):
        accessed.append(row['role']);assert row['role']!='holdout'
        x=torch.zeros(4,196,3584);return {'visual':x,'real':x,'base':torch.zeros(4,64,3584)}
    monkeypatch.setattr(m,'load',load)
    class R:
        device='cpu';model=torch.nn.Identity()
        def score(self,row,z):return {'correct':True,'gold_nll':1.0}
    class Fusion(torch.nn.Module):
        def forward(self,v,g,base):assert len(v)==1;return base
    result=m.validate(R(),Fusion(),'R_REAL',1)
    assert accessed==['train','val'] and set(result)=={'train','val'}

def test_scaleup_cheap_rule_is_current_physical_stat_only():
    from utility_gate.imu_visual.scaleup_train.cost import CheapRule
    from utility_gate.imu_visual.scaleup_train.gate import IMUFeatures
    f=IMUFeatures(torch.zeros(1,7168),torch.zeros(1,27),torch.zeros(1,4),torch.ones(1,dtype=torch.bool))
    rule=CheapRule();assert rule.step(f,0,0).sigmoid().item()<.5
    f.imu[:,9:12]=.21
    assert rule.step(f,0,0).sigmoid().item()>.5

def test_scaleup_sensor_held_original_lower_bound(tmp_path):
    from utility_gate.imu_visual.scaleup_train.data import sensor_window
    p=tmp_path/'s.npz';t=np.arange(2.,4.001,1/60);original=np.column_stack((t-.02,t))
    np.savez(p,times=t,values=np.ones((len(t),6)),original_times=original)
    w=sensor_window({'imu_path':str(p)},4.)
    assert w.valid and w.times.min()>=2.02 and w.times.max()<=4.
