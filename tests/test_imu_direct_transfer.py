import torch
import numpy as np
import pytest
from utility_gate.imu_gate_router_transfer.gate import IMUGate,IMUFeatures,choose
from utility_gate.imu_gate_router_transfer.features import cheap_features
from utility_gate.gate_first_native.fusion import GroundedFusionRouter

def fixture():
    torch.manual_seed(7)
    return IMUFeatures(torch.randn(4,7168),torch.randn(4,27),torch.randn(4,4),torch.ones(4,dtype=torch.bool))
def test_same_RA_class_zero_identity_64_and_off_invalid_aux():
    m=GroundedFusionRouter();v=torch.randn(2,196,3584);b=torch.randn(2,64,3584);g=torch.randn_like(v)
    assert sum(p.numel() for p in m.parameters())==696320
    assert torch.equal(m(v,g,b),b)
    assert torch.equal(m(v,torch.full_like(g,float('nan')),b,torch.zeros(2,dtype=torch.bool)),b)
    with pytest.raises(ValueError):m(v,torch.full_like(g,float('nan')),b)
def test_only_new_router_parameters_receive_and_apply_gradient():
    m=GroundedFusionRouter();v=torch.randn(1,196,3584);b=torch.randn(1,64,3584);g=torch.randn_like(v);old={n:p.clone() for n,p in m.named_parameters()};opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
    m(v,g,b).square().mean().backward();assert m.output.weight.grad.norm()>0;opt.step();assert not torch.equal(m.output.weight,old['output.weight'])
def test_gate_exact_transferred_counts_and_16_joint_distribution():
    for extended,count in [(False,479489),(True,479553)]:
        m=IMUGate(extended).eval();assert sum(p.numel() for p in m.parameters())==count
        lp=m.log_policy(fixture());assert lp.shape==(1,16);assert torch.allclose(lp.exp().sum(-1),torch.ones(1),atol=1e-6)
def test_gate_prefix_future_changes_do_not_affect_earlier_actions():
    m=IMUGate().eval();f=fixture();before,_=choose(m,f,.5);g=IMUFeatures(f.visual.clone(),f.imu.clone(),f.context.clone(),f.valid.clone());g.visual[2:]*=1000;g.imu[2:]*=-99
    after,_=choose(m,g,.5);assert before[:2]==after[:2]
    for cap in range(5):assert sum(choose(m,f,0.,cap)[0])<=cap
def test_missing_sensors_off_and_features_reject_labels():
    f=fixture();g=IMUFeatures(f.visual,f.imu,f.context,torch.zeros(4,dtype=torch.bool));assert choose(IMUGate().eval(),g,0)[0]==[False]*4
    with pytest.raises(TypeError):IMUFeatures(f.visual,f.imu,f.context,f.valid,answer='A')
def test_current_and_past_RGB_features_never_read_future():
    v=torch.randn(4,196,3584);s=np.zeros((4,27));valid=torch.ones(4,dtype=torch.bool);f=cheap_features(v,s,valid);v[3].fill_(float('nan'));g=cheap_features(v,s,valid)
    assert torch.equal(f['visual'][:3],g['visual'][:3]) and f['visual'].shape==(4,7168)

def test_held_original_samples_must_also_satisfy_window_lower_boundary(tmp_path):
    from utility_gate.imu_gate_router_transfer.data import sensor_window
    times=np.arange(101)/50.;orig=np.column_stack((times-.01,times));path=tmp_path/'async.npz';np.savez(path,times=times,values=np.ones((101,6)),original_times=orig)
    w=sensor_window({'imu_path':str(path)},2.)
    assert w.valid and w.times[0]==.02 and w.times[-1]==2.

def test_canonical_sequence_executes_only_selected_frame_batch_one():
    from utility_gate.imu_gate_router_transfer.runtime import fuse_sequence
    m=GroundedFusionRouter();v=torch.randn(4,196,3584);g=torch.randn_like(v);base=torch.randn(4,64,3584);g[1:]=float('nan');seen=[]
    m.register_forward_pre_hook(lambda module,args:seen.append(len(args[0])))
    z=fuse_sequence(m,v,g,base,[True,False,False,False])
    assert seen==[1] and torch.equal(z[1:],base[1:]) and z.shape==(4,64,3584)

@pytest.mark.parametrize('valid',[False,True])
def test_actual_pipeline_gate_precedes_render_and_extra_encode(tmp_path,monkeypatch,valid):
    import json
    from types import SimpleNamespace
    from utility_gate.imu_gate_router_transfer import pipeline as p
    (tmp_path/'locks').mkdir();(tmp_path/'locks/IMU_PLOT_V1.json').write_text('{}');monkeypatch.setattr(p,'ROOT',tmp_path)
    monkeypatch.setattr(p,'decode_rgb',lambda row,t:(np.zeros((8,8,3),np.uint8),t,0));monkeypatch.setattr(p,'safe_sensor',lambda row,t:SimpleNamespace(valid=valid,channels=range(6)));monkeypatch.setattr(p,'cheap_stats',lambda w:np.zeros(27,np.float32))
    renders=[]
    def render(*args,**kw):
        assert valid;renders.append(1);return np.zeros((8,8,3),np.uint8),{}
    monkeypatch.setattr(p,'render_imu_plot',render)
    class Fake:
        device='cpu'
        def __init__(self):self.calls={'SigLIP_images':0,'V1':0}
        def visual_frame(self,frame):self.calls['SigLIP_images']+=1;return torch.zeros(196,3584)
        def vision_latents(self,v):self.calls['V1']+=1;return torch.zeros(1,64,3584)
        def score(self,row,z):
            assert z.shape==(4,64,3584);return {'nll_evaluation_extra_s':0.,'llm_total_s':0.,'llm_ttft_s':0.,'correct':True,'gold_nll':1.,'visual_tokens':256}
    class G:
        def step(self,*args):return torch.tensor(100.)
    calls=[]
    def fusion(v,g,b):calls.append(1);return b
    r=Fake();out=p.run(r,fusion,{'timestamps_s':[2.,4.,6.,8.]},gate=G());assert out['k']==4*valid and len(renders)==4*valid and len(calls)==4*valid and out['actual_forward_hooks']['SigLIP_images']==4+4*valid
