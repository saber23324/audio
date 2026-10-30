"""Read-only frozen AudioRouter runtime, with audit writes scoped to this run."""
import importlib.util,json
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.models import resnet18
from utility_gate.gate_first_native.runtime import Runtime as HistoricalRuntime
from utility_gate.grounded_dual_image.grounder import FrozenLocalizer,render,sham
from utility_gate.adaptive_grounded64.runtime import qa_prompt,gate_features,append_step_features
from utility_gate.gate_first_native.fusion import load_fusion
from .common import ROOT,REPO,NATIVE,RA,sha256,write_json

class Runtime(HistoricalRuntime):
    def __init__(self):
        super().__init__();self.fusion,self.fusion_payload=load_fusion(RA,self.device);self.attach_fusion(self.fusion)
        self.visual_encoder_calls=0
        for m in (self.model,self.new_router,self.vision_router,self.audio_router,self.fusion):
            m.eval().requires_grad_(False)
        if sum(p.numel() for p in self.fusion.parameters())!=696320:raise ValueError('wrong R_A')
    @torch.inference_mode()
    def visual_frame(self,frame):
        self.visual_encoder_calls+=1
        return super().visual_frame(frame)
    @torch.inference_mode()
    def legacy_fixture(self):
        report={}
        for name,router in [('m0-vision-20261022',self.vision_router),('m0-gate-disabled-final-20261022',self.audio_router)]:
            f=torch.load(REPO/f'artifacts/relevance_gate/{name}/fixture.pt',map_location='cpu',weights_only=True)
            v=torch.cat([x['visual'] for x in f['chunks']]).to(self.device)
            a=f['audio_info']['embeddings'].to(self.device) if router.stage==3 else None
            z,p=router(v,a,f['timestamps'].to(self.device));g=self.generate(f['prompt_ids'],z)
            c={'P':torch.equal(p.cpu(),torch.cat([x['attention'] for x in f['chunks']])),
               'Z':torch.equal(z.cpu(),torch.cat([x['latents'] for x in f['chunks']])),
               'generated_ids':g['generated_ids']==f['generated_ids'].reshape(-1).tolist()}
            if not all(c.values()):raise AssertionError(c)
            report[name]={**c,'actual_generated_ids':g['generated_ids']}
        write_json(ROOT/'audit/legacy_regression.json',report);return report

class Localizer(FrozenLocalizer):
    def __init__(self,device):
        self.lock=json.loads((NATIVE/'config/localizer_lock.json').read_text());ckpt=Path(self.lock['checkpoint_path'])
        if sha256(ckpt)!=self.lock['checkpoint_sha256']:raise ValueError('EZ-VSL weights drift')
        code=next(Path(p) for p in self.lock['source_hashes'] if p.endswith('/model.py'))
        if sha256(code)!=self.lock['source_hashes'][str(code)]:raise ValueError('EZ-VSL source drift')
        spec=importlib.util.spec_from_file_location('fixed_router_frozen_ezvsl',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.resnet18=lambda *a,**kw:resnet18(weights=None)
        model=module.EZVSL(.03,512);payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        model.load_state_dict({k.removeprefix('module.'):v for k,v in payload['model'].items()},strict=True)
        self.model=model.to(device).eval().requires_grad_(False);self.device=device;self.forward_calls=0
        self.transform=transforms.Compose([transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def predict(self,frames,windows,valid):
        self.forward_calls+=1
        return super().predict(frames,windows,valid)

