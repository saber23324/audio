"""Existing frozen runtime with new-run-only audit and no legacy writes."""
import copy,importlib.util,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18
from utility_gate.adaptive_grounded64.runtime import Runtime as PreviousRuntime,qa_prompt,gate_features,append_step_features
from utility_gate.grounded_dual_image.grounder import FrozenLocalizer,render,sham
from utility_gate.runtime import load_router
from .common import ROOT,REPO,NATIVE,V1,sha256,write_json

NATIVE_SYSTEM=('You will receive an original visual observation and a second visual view. The second view may be identical to the first or may contain a noisy visual cue derived from an auxiliary sensor. Treat the cue as uncertain. Preserve the original scene facts, use the extra view only when it provides clear supporting evidence, and answer the question using the visible observations. Do not assume a highlighted region is always correct.')

def native_prompt(runtime,row,timestamps,counts):
    from llava.conversation import conv_templates
    from llava.constants import DEFAULT_IMAGE_TOKEN,IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    lines=[]
    for t,count in zip(timestamps,counts):
        lines.append(f'At video time {float(t):.3f} seconds:')
        lines.append('Original visual observation: '+DEFAULT_IMAGE_TOKEN)
        if count==2:lines.append('Second visual view: '+DEFAULT_IMAGE_TOKEN)
    if row['dataset']=='music_avqa':lines.append('Answer the following question with only a short answer.\n'+row['question'])
    else:lines.append('Answer the following multiple-choice question. Respond only with the uppercase option letter A, B, C, or D.\n'+row['question']+'\n'+'\n'.join(row['options']))
    conv=copy.deepcopy(conv_templates['qwen_1_5']);conv.system='<|im_start|>system\n'+NATIVE_SYSTEM
    conv.append_message(conv.roles[0],'\n'.join(lines));conv.append_message(conv.roles[1],None)
    return tokenizer_image_token(conv.get_prompt(),runtime.tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt')

class Runtime(PreviousRuntime):
    def __init__(self):
        super().__init__();self.load_new_router(V1);self.audio_encoder.batch_size=1
        self.router_calls={'V0':0,'Stage3':0,'V1':0,'RF':0};self._hooks=[]
        for name,module in [('V0',self.vision_router),('Stage3',self.audio_router),('V1',self.new_router)]:
            self._hooks.append(module.register_forward_pre_hook(lambda m,args,n=name:self._count(n)))
    def _count(self,name):self.router_calls[name]+=1
    def attach_fusion(self,module):self._hooks.append(module.register_forward_pre_hook(lambda m,args:self._count('RF')))
    def first_token_loss_z(self,row,z):
        from streamingtom.train_adbt_videomme import build_inputs
        embeds,labels,mask=build_inputs(self.model,self.tokenizer,qa_prompt(self,row),row['answer_text'],z)
        h=self.model.model(inputs_embeds=embeds,attention_mask=mask,use_cache=False,return_dict=True).last_hidden_state[:,:-1]
        shifted=labels[:,1:];which=shifted.ne(-100)
        logits=self.model.lm_head(h[which]);return F.cross_entropy(logits[:1].float(),shifted[which][:1])
    @torch.inference_mode()
    def legacy_fixture(self):
        report={}
        for name,router in [('m0-vision-20261022',self.vision_router),('m0-gate-disabled-final-20261022',self.audio_router)]:
            f=torch.load(REPO/f'artifacts/relevance_gate/{name}/fixture.pt',map_location='cpu',weights_only=True)
            v=torch.cat([x['visual'] for x in f['chunks']]).to(self.device);a=f['audio_info']['embeddings'].to(self.device) if router.stage==3 else None
            z,p=router(v,a,f['timestamps'].to(self.device));g=self.generate(f['prompt_ids'],z)
            checks={'P':torch.equal(p.cpu(),torch.cat([x['attention'] for x in f['chunks']])),
                'Z':torch.equal(z.cpu(),torch.cat([x['latents'] for x in f['chunks']])),
                'generated_ids':g['generated_ids']==f['generated_ids'].reshape(-1).tolist()}
            if not all(checks.values()):raise AssertionError('legacy fixture changed')
            report[name]={**checks,'actual_generated_ids':g['generated_ids']}
        write_json(ROOT/'audit/legacy_regression.json',report);return report

class Localizer(FrozenLocalizer):
    def __init__(self,device):
        self.lock=json.loads((NATIVE/'config/localizer_lock.json').read_text());ckpt=Path(self.lock['checkpoint_path'])
        if sha256(ckpt)!=self.lock['checkpoint_sha256']:raise ValueError('EZ-VSL checkpoint drift')
        code=next(Path(p) for p in self.lock['source_hashes'] if p.endswith('/model.py'))
        if sha256(code)!=self.lock['source_hashes'][str(code)]:raise ValueError('EZ-VSL official source drift')
        spec=importlib.util.spec_from_file_location('gatefirst_official_ezvsl',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.resnet18=lambda *a,**kw:resnet18(weights=None)
        model=module.EZVSL(.03,512);payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        model.load_state_dict({k.removeprefix('module.'):v for k,v in payload['model'].items()},strict=True)
        self.model=model.to(device).eval().requires_grad_(False);self.device=device
        self.transform=transforms.Compose([transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        write_json(ROOT/'audit/localizer_loaded.json',{'checkpoint_sha256':self.lock['checkpoint_sha256'],'trainable_parameters':0,'GT_used':False,'batch_size':1,'crop_before_resample':True})

