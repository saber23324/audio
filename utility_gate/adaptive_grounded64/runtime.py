"""Frozen project runtime; training uses only an independent new Stage1 copy."""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
from utility_gate.grounded_dual_image.runtime import QARuntime,prompt
from utility_gate.runtime import load_router,clocked,synchronize
from utility_gate.gate2_interfaces import assemble_anchor_features
from utility_gate.features import take_features,feature_payload,load_feature_payload
from utility_gate.grounded_dual_image.media import raw_windows
from .common import ROOT,REPO,VISION_ROUTER,sha256,write_json,read_config

def qa_prompt(runtime,row):
    if row['dataset']=='music_avqa':return prompt(runtime.tokenizer,row['question'])
    from streamingtom.train_adbt_videomme import make_prompt
    return make_prompt(row,runtime.tokenizer)

class Runtime(QARuntime):
    def __init__(self):
        super().__init__()
        self.audio_encoder.cache_dir=None
        self.new_router=None
        self.metadata.update(experiment='adaptive_grounded64',generation={'do_sample':False,'max_new_tokens':16},
            training_objective_verified='legacy Stage1 phase4 FIRST answer-token full-vocabulary CE, no teacher KD')

    def load_new_router(self,path=None):
        path=path or ROOT/'models/stage1_vision64_retrained/selected.pt'
        self.new_router=load_router(path,self.model.config.hidden_size,self.device)
        return self.new_router

    @torch.inference_mode()
    def visual_frame(self,frame):
        return self.extractor.encode_visual(self.model,self.processor,np.asarray(frame)[None],1).float()[0]

    @torch.inference_mode()
    def score(self,row,z):
        self.model.model.eval()
        return self.answer(qa_prompt(self,row),[z.to(self.device)],row['answer_text'],mcqa=row['dataset']!='music_avqa')

    @torch.inference_mode()
    def vision_latents(self,visual,router=None):
        router=router or self.new_router
        zs=[];ps=[]
        for v in visual:
            z,p=router(v[None].to(self.device,dtype=torch.float32),None)
            zs.append(z[0]);ps.append(p[0])
        return torch.stack(zs),torch.stack(ps)

    @torch.inference_mode()
    def audio_features(self,path,timestamps):
        (windows,valid,reasons,traces),decode_s=clocked(lambda:raw_windows(path,timestamps))
        audio,encode_s=clocked(lambda:self.audio_encoder.encode_windows(windows,self.device).float().cpu())
        return {'windows':windows,'audio':audio,'audio_valid':valid,'audio_reasons':reasons,'audio_traces':traces,
            'timing':{'raw_native_crop_decode_resample_s':decode_s,'BEATs_s':encode_s}}

    def training_router(self):
        router=load_router(VISION_ROUTER,self.model.config.hidden_size,self.device)
        allowed={'query_generator.query_tokens','visual_key.weight'}
        for name,p in router.named_parameters():p.requires_grad_(name in allowed)
        if {n for n,p in router.named_parameters() if p.requires_grad}!=allowed:raise ValueError('active Stage1 parameter audit changed')
        return router.train()

    def first_token_loss(self,row,visual,router):
        from streamingtom.train_adbt_videomme import build_inputs
        # This is the audited original loss/alignment; only prompt adaptation
        # is needed for MUSIC's native open-answer task. No audio/teacher loss.
        visual=visual.to(self.device,dtype=torch.float32).clone().detach()
        z,p=router(visual,None)
        embeds,labels,mask=build_inputs(self.model,self.tokenizer,qa_prompt(self,row),row['answer_text'],z)
        output=self.model.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False,return_dict=True)
        shifted=labels[:,1:];h=output.last_hidden_state[:,:-1]
        a=shifted.ne(-100);logits=self.model.lm_head(h[a])
        loss=F.cross_entropy(logits[:1].float(),shifted[a][:1])
        return loss,z,p

    @torch.inference_mode()
    def legacy_regression(self):
        values={}
        for name,router in [('m0-vision-20261022',self.vision_router),('m0-gate-disabled-final-20261022',self.audio_router)]:
            f=torch.load(REPO/'artifacts/relevance_gate'/name/'fixture.pt',map_location='cpu',weights_only=True)
            v=torch.cat([r['visual'] for r in f['chunks']]).to(self.device)
            a=f['audio_info']['embeddings'].to(self.device) if router.stage==3 else None
            z,p=router(v,a,f['timestamps'].to(self.device))
            generation=self.generate(f['prompt_ids'],z)
            equal={'P':torch.equal(p.cpu(),torch.cat([r['attention'] for r in f['chunks']])),
                'Z':torch.equal(z.cpu(),torch.cat([r['latents'] for r in f['chunks']])),
                'generated_ids':generation['generated_ids']==f['generated_ids'].reshape(-1).tolist()}
            if not all(equal.values()):raise AssertionError(name)
            values[name]={**equal,'actual_generated_ids':generation['generated_ids']}
        write_json(ROOT/'trace/legacy_disabled_regression.json',values);return values

def gate_features(visual,audio,valid,timestamps):
    # Caller passes only the arrived prefix. The old trained proxy expects
    # current means plus strictly earlier history, not future/full-video means.
    visual=torch.as_tensor(visual).float();audio=torch.as_tensor(audio).float()
    return assemble_anchor_features(visual.mean(1)[None],audio.mean(1)[None],torch.as_tensor(valid).bool()[None],torch.as_tensor(timestamps).float()[None])

def append_step_features(visual,audio,valid,timestamps,index):
    f=gate_features(visual[:index+1],audio[:index+1],valid[:index+1],timestamps[:index+1])
    return take_features(f,slice(index,index+1))
