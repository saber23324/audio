import inspect,json,wave
from dataclasses import fields
import numpy as np
import pytest
import torch
from utility_gate.adaptive_grounded64.fuser64 import fuse
from utility_gate.adaptive_grounded64.online_policy import decide,random_stream
from utility_gate.adaptive_grounded64.features import times
from utility_gate.adaptive_grounded64.runtime import append_step_features
from utility_gate.adaptive_grounded64.common import train_rows,ROOT,fingerprint
from utility_gate.features import GateFeatures

@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('reason',['off','beta0','unavailable','duplicate'])
def test_exact_fuser_identity(batch,reason):
    z=torch.randn(batch,64,17);g=torch.randn_like(z)
    kwargs={'on':True,'beta':.25,'available':True}
    if reason=='off':kwargs['on']=False
    if reason=='beta0':kwargs['beta']=0
    if reason=='unavailable':kwargs['available']=False
    if reason=='duplicate':g=z.clone()
    got=fuse(z,g,**kwargs);assert torch.equal(got,z)
    if reason!='duplicate':assert got is z

def test_fuser_weight_and_budget():
    z=torch.ones(2,64,17);g=z*3
    assert torch.equal(fuse(z,g,on=True,beta=.25),z*1.5)
    with pytest.raises(ValueError):fuse(torch.zeros(65,17))
    with pytest.raises(ValueError):fuse(z,on=True,beta=.5)

@pytest.mark.parametrize('tau,expected',[('-inf',True),('inf',False),(0,False),(-.01,True)])
def test_policy_all_on_off_and_strict_tie(tau,expected):
    p={'kind':'u','u_key':'u','tau_u':tau}
    assert decide(p,{'u':0},True)==expected
    assert decide(p,{'u':0},False) is False

def test_random_control_uses_fixed_dev_probability_not_video_topk():
    a=random_stream('source',1234);b=random_stream('source',1234)
    p={'kind':'random','probability':.37}
    draws=[a.random() for _ in range(100)]
    assert draws==[b.random() for _ in range(100)]
    on=[decide(p,{},True,x) for x in draws]
    assert on==[x<.37 for x in draws]
    assert [decide({'kind':'random','probability':0},{},True,x) for x in draws]==[False]*100
    assert [decide({'kind':'random','probability':1},{},True,x) for x in draws]==[True]*100

def test_fixed_arrivals_do_not_respace_with_future_duration():
    assert np.array_equal(times(8),times(60)[:4]);assert len(times(999))==32
    assert np.array_equal(times(5),[2.,4.]);assert not len(times(1.9))

def test_future_injection_cannot_change_current_gate_features():
    v=torch.randn(9,196,3584);a=torch.randn(9,96,768);valid=torch.ones(9,dtype=torch.bool);t=torch.arange(1,10.)*2
    original=append_step_features(v,a,valid,t,3)
    v[4:]=float('nan');a[4:]=float('inf');valid[4:]=False;t[4:]=99999
    modified=append_step_features(v,a,valid,t,3)
    for key in ('visual','audio','context','valid_audio'):assert torch.equal(getattr(original,key),getattr(modified,key))
    assert {f.name for f in fields(GateFeatures)}=={'visual','audio','context','valid_audio'}

@pytest.mark.parametrize('role',['confirmation_music','confirmation_valor','test','val','../confirmation_music'])
def test_training_loader_cannot_open_confirmation_answers(role):
    with pytest.raises(ValueError,match='forbidden'):train_rows(role)

def test_native_raw_crop_removes_both_outside_boundaries(tmp_path):
    from utility_gate.grounded_dual_image.media import raw_window
    rng=np.random.default_rng(5);middle=rng.integers(-10000,10000,96000,dtype=np.int16);results=[]
    for i in range(2):
        path=tmp_path/f'{i}.wav';x=np.r_[rng.integers(-20000,20000,96000,dtype=np.int16),middle,rng.integers(-20000,20000,96000,dtype=np.int16)]
        with wave.open(str(path),'wb') as out:out.setnchannels(1);out.setsampwidth(2);out.setframerate(48000);out.writeframes(x.tobytes())
        w,valid,reason,trace=raw_window(path,4.)
        assert valid and reason=='ok';assert trace['first_raw_sample_s']>=2 and trace['last_raw_sample_interval_end_s']<=4
        results.append(w)
    assert torch.equal(*results)

def test_cache_dependency_closure_distinguishes_all_action_changes():
    base={'stage1':'a','localizer':'b','beta':.25,'fuser':'F1'};key=fingerprint(base)
    for field,value in [('stage1','c'),('localizer','d'),('beta',.5),('fuser','F2')]:assert fingerprint({**base,field:value})!=key

def test_deployable_policy_never_accepts_question_answer_or_future():
    assert set(inspect.signature(decide).parameters)=={'policy','scores','available','draw'}
    source=inspect.getsource(decide)
    assert 'topk' not in source.lower() and 'sort' not in source.lower()

