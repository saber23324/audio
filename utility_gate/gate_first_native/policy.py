"""Frozen phase-A gate selected for F0; later Router/native use is transfer."""
import json,hashlib
import numpy as np
import torch
from utility_gate.features import take_features
from .common import ROOT,sha256
from .gate import load_gate,freeze_proxy,proxy_scores,history,accept

class GateBundle:
    def __init__(self):
        self.lock=json.loads((ROOT/'locks/gate_a_lock.json').read_text());self.proxy=freeze_proxy();self.models={}
        for name,entry in [('main',self.lock['selected_main']),('u',self.lock['best_u']),('dual',self.lock['best_dual']),('vision',self.lock['vision_gate'])]:self.models[name]=load_gate(entry['checkpoint'])[0]
        if sha256(self.lock['gate_a_best_checkpoint'])!=self.lock['gate_a_best_sha256']:raise ValueError('frozen Gate A changed')
    def rule(self,name):return self.lock['selected_main' if name=='main' else 'best_u' if name=='u' else 'best_dual' if name=='dual' else 'vision_gate']['rule']
    @torch.inference_mode()
    def one(self,features,past,index,which='main'):
        if len(features.visual)!=1:raise ValueError('only arrived current/past state permitted')
        r,old,cos=proxy_scores(self.proxy,features);model=self.models[which]
        u=float(model.forward_features(features,torch.tensor([history(past,index)]),r[:,None]))
        on=accept(self.rule(which),u,float(r[0]),bool(features.valid_audio[0]))
        return {'on':on,'u':u,'r_proxy':float(r[0]),'old_u':float(old[0]),'cosine_proxy':float(cos[0])}
    def mask(self,data,which='main'):
        from utility_gate.features import load_feature_payload
        f=load_feature_payload(data['features']);past=0;mask=[];scores=[]
        for t in range(4):
            o=self.one(take_features(f,slice(t,t+1)),past,t,which);mask.append(o['on']);scores.append(o)
            if o['on']:past|=1<<t
        return mask,scores

def rng(source,seed):return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{seed}:gatefirst:{source}:random'.encode()).digest()[:8],'big'))
