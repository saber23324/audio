"""Fixed F1 fusion with exact bypass and explicit dependency closure."""
import json
import torch
from .common import ROOT,sha256,fingerprint,read_config,PRIOR

def fuse(base,grounded=None,*,on=False,beta=0.25,available=True):
    if base.shape[-2]!=64:raise ValueError('expected native 64-query slots')
    if not on or not available or beta==0:return base
    if grounded is None or grounded.shape!=base.shape:raise ValueError('aligned same-bank grounded slots required')
    if not 0<=beta<=1:raise ValueError('invalid fixed mixture')
    return base+beta*(grounded-base)

def dependencies(beta=None):
    stage=json.loads((ROOT/'locks/stage1_retrained_lock.json').read_text())
    if sha256(stage['selected_checkpoint'])!=stage['checkpoint_sha256']:raise ValueError('new Stage1 changed; labels invalid')
    if beta is None:
        fixed=json.loads((ROOT/'locks/fuser_lock.json').read_text());beta=fixed['beta']
    localizer=json.loads((PRIOR/'config/localizer_lock.json').read_text())
    if sha256(localizer['checkpoint_path'])!=localizer['checkpoint_sha256']:raise ValueError('localizer changed; labels invalid')
    closure={'new_stage1':stage['checkpoint_sha256'],'localizer':localizer['checkpoint_sha256'],
        'beta':float(beta),'fuser':'F1_same_bank_slot_mix','fuser_code':sha256(__file__),
        'render':read_config()['fuser'],'causal_audio':'rawcrop_v1_2s_then_leftpad1s','frame_policy':read_config()['frame_policy']}
    return {**closure,'fingerprint':fingerprint(closure)}

