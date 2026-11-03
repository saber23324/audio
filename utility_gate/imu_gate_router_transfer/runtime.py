"""Existing frozen visual runtime and exactly the audited RA residual module."""
import copy
import torch
import torch.nn.functional as F
from utility_gate.imu_visual.runtime import VisionRuntime
from utility_gate.grounded_dual_image.runtime import insert_views
from utility_gate.gate_first_native.fusion import GroundedFusionRouter,load_fusion

SYSTEM=('You are answering a question about the observed video interval. RGB observations show the scene. A second view, when present, is an uncertain auxiliary observation: it may show a physical IMU time-series chart or repeat the RGB. IMU acceleration and angular velocity describe only the sensor location written on the chart, not every visible object. Preserve the original scene facts and use the auxiliary view only for supporting evidence. Use only the observed interval. Answer in the required format.')
def prompt(r,row,native=False,counts=None,only=False):
    from llava.conversation import conv_templates
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import tokenizer_image_token
    conv=copy.deepcopy(conv_templates['qwen_1_5']);conv.system='<|im_start|>system\n'+SYSTEM
    if native:
        lines=[]
        for t,n in zip([2,4,6,8],counts or [1]*4):
            lines.extend([f'At observed interval time {t} seconds:',('IMU view: ' if only else 'Original RGB observation: ')+DEFAULT_IMAGE_TOKEN])
            if n==2:lines.append('Second visual view: '+DEFAULT_IMAGE_TOKEN)
    else:lines=[DEFAULT_IMAGE_TOKEN,'The four arrived observations are at interval times2,4,6,8 seconds.']
    lines.extend([row['question'],*row['options'],'Respond only with the uppercase option letter A, B, C, or D.']);conv.append_message(conv.roles[0],'\n'.join(lines));conv.append_message(conv.roles[1],None)
    return tokenizer_image_token(conv.get_prompt(),r.tokenizer,return_tensors='pt')

class Runtime(VisionRuntime):
    @torch.inference_mode()
    def score(self,row,z):
        self.model.eval();return self.answer(prompt(self,row),[z.to(self.device)],row['answer_text'],mcqa=True)
    def first_loss(self,row,z):
        # Exact audited first-answer-token CE/alignment, adapted prompt only.
        from streamingtom.train_adbt_videomme import build_inputs
        embeds,labels,mask=build_inputs(self.model,self.tokenizer,prompt(self,row),row['answer_text'],z)
        h=self.model.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state[:,:-1];target=labels[:,1:];keep=target.ne(-100)
        logits=self.model.lm_head(h[keep]);return F.cross_entropy(logits[:1].float(),target[keep][:1])
def new_fusion(r):
    return GroundedFusionRouter(query_init=r.new_router.query_generator.query_tokens,key_init=r.new_router.visual_key.weight,key_dim=128,rank=32).to(r.device)

def fuse_sequence(router,rgb,auxiliary,base,mask=None):
    """Canonical online B=1 route, also used for cached full-QA Utility."""
    if mask is None:mask=[True]*len(rgb)
    return torch.stack([router(rgb[i:i+1],auxiliary[i:i+1],base[i:i+1])[0] if bool(mask[i]) else base[i] for i in range(len(rgb))])
