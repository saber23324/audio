"""Small new utility regressors and a read-only correspondence proxy."""
from pathlib import Path
import torch
from utility_gate.dual_head.model import DualHeadGate,load_dual_head
from .common import REPO,ROOT,sha256

OLD_GATE=REPO/'artifacts/utility_gate/dual_head_proxy_20261028_v1/model/dual_combined/gate.pt'
OLD_RULE=OLD_GATE.with_name('rule.json')
MODES=('new_utility_only','new_utility_on_frozen_combined_corr_trunk')

def make_model(mode,scale):
    if mode not in MODES:raise ValueError(mode)
    if mode==MODES[0]:
        model=DualHeadGate(utility_scale=scale)
        model.correspondence_head.requires_grad_(False);model.decision_head.requires_grad_(False)
    else:
        model,_=load_dual_head(OLD_GATE)
        model.spec['utility_scale']=float(scale)
        model.utility_head.reset_parameters();model.utility_head.requires_grad_(True)
    return model

def load_new(mode,device='cpu'):
    path=ROOT/f'models/new_utility_gate/{mode}/gate.pt'
    model,payload=load_dual_head(path,device=device)
    if payload['provenance']['action_dependency']!=__import__('utility_gate.adaptive_grounded64.fuser64',fromlist=['dependencies']).dependencies()['fingerprint']:
        raise ValueError('new Gate trained for a different frozen action')
    return model,payload

