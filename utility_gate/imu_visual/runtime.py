"""Frozen visual-only backbone, deliberately no audio frontend in this phase."""
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from utility_gate.grounded_dual_image.runtime import QARuntime
from utility_gate.runtime import load_router
from .common import V1

class VisionRuntime(QARuntime):
    def __init__(self):
        from relevance_gate.feature_cache import bind_legacy_imports
        bind_legacy_imports()
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from streamingtom.train_adbt_videomme import encode_visual
        pretrained='lmms-lab/llava-onevision-qwen2-7b-ov';start=time.perf_counter()
        _,self.model,self.processor,_=load_pretrained_model(pretrained,None,get_model_name_from_path(pretrained),device_map='auto',multimodal=True,local_files_only=True,attn_implementation='flash_attention_2')
        self.model.eval().requires_grad_(False);self.device=next(self.model.get_vision_tower().parameters()).device
        self.tokenizer=AutoTokenizer.from_pretrained(pretrained,local_files_only=True);self.encode_visual=encode_visual;self.new_router=load_router(V1,self.model.config.hidden_size,self.device)
        self.calls={'SigLIP':0,'SigLIP_images':0,'projector':0,'V1':0};self.hooks=[]
        def vision(m,args):
            self.calls['SigLIP']+=1;x=args[0];self.calls['SigLIP_images']+=len(x)
        self.hooks.append(self.model.get_vision_tower().register_forward_pre_hook(vision))
        self.hooks.append(self.model.get_model().mm_projector.register_forward_pre_hook(lambda m,args:self._count('projector')))
        self.hooks.append(self.new_router.register_forward_pre_hook(lambda m,args:self._count('V1')))
        self.initialization_s=time.perf_counter()-start
    def _count(self,name):self.calls[name]+=1
    @torch.inference_mode()
    def visual_frame(self,frame):return self.encode_visual(self.model,self.processor,np.asarray(frame)[None],1).float()[0]
    @torch.inference_mode()
    def vision_latents(self,v):
        out=[]
        for x in v:out.append(self.new_router(x[None].to(self.device).float(),None)[0][0])
        return torch.stack(out)
