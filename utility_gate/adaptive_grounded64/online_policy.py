"""Independent current-arrival threshold decisions. No video ranking or target rate."""
import json,hashlib
import numpy as np
import torch
from utility_gate.dual_head.model import load_dual_head
from .gate_model import load_new,MODES,OLD_GATE,OLD_RULE

def threshold(x):return float(x)

def decide(policy,scores,available,draw=None):
    if not available:return False
    kind=policy['kind']
    if kind=='off':return False
    if kind=='on':return True
    if kind=='random':
        if draw is None:raise ValueError('pre-arrival independent random draw required')
        return bool(draw<policy['probability'])
    if kind=='u':return scores[policy['u_key']]>threshold(policy['tau_u'])
    if kind=='r':return scores['r_proxy']>threshold(policy['tau_r'])
    if kind=='dual':return scores['r_proxy']>threshold(policy['tau_r']) and scores[policy['u_key']]>threshold(policy['tau_u'])
    if kind=='energy':return scores['rms']>threshold(policy['tau_energy'])
    raise ValueError(kind)

def random_stream(source,seed):
    # Source only seeds an independent control, never enters any learned model.
    digest=hashlib.sha256(f'{seed}:{source}:adaptive_random_control'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'big'))

class GateBundle:
    def __init__(self):
        self.old,_=load_dual_head(OLD_GATE)
        self.new,_=load_new(MODES[0]);self.dual,_=load_new(MODES[1])
    @torch.inference_mode()
    def scores(self,features,rms=0.):
        if len(features.visual)!=1:raise ValueError('one causal arrival required')
        old=self.old.forward_features(features);new=self.new.forward_features(features);dual=self.dual.forward_features(features)
        return {'old_u':float(old.utility_delta[0,0]),'u':float(new.utility_delta[0,0]),
            'dual_u':float(dual.utility_delta[0,0]),'r_proxy':float(old.correspondence_logit[0,0]),'rms':float(rms)}

def candidates(config,train_energy_median):
    old=json.loads(OLD_RULE.read_text());r_old=old['selection_corr_threshold'];u_old=old['selection_utility_threshold']
    out={'V1':{'kind':'off'},'F0':{'kind':'on'},'G0':{'kind':'dual','u_key':'old_u','tau_r':r_old,'tau_u':u_old}}
    for u in config['u_threshold_candidates']:
        out[f'u_{u}']={'kind':'u','u_key':'u','tau_u':u}
    for r in config['r_threshold_candidates']:
        r=r_old if r=='old_corr' else r
        out[f'r_{r}']={'kind':'r','tau_r':r}
        for u in config['u_threshold_candidates']:
            out[f'dual_{r}_{u}']={'kind':'dual','u_key':'dual_u','tau_r':r,'tau_u':u}
    for t in ['-inf',train_energy_median,'inf']:
        out[f'energy_{t}']={'kind':'energy','tau_energy':t}
    return out

