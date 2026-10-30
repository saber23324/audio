"""Read-only official EZ-VSL adapter; audit files belong to this run only."""
import importlib.util,json
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.models import resnet18
from utility_gate.grounded_dual_image.grounder import FrozenLocalizer,spectrogram,render,sham
from .common import ROOT,PRIOR,sha256,write_json

class Localizer(FrozenLocalizer):
    def __init__(self,device):
        self.lock=json.loads((PRIOR/'config/localizer_lock.json').read_text())
        ckpt=Path(self.lock['checkpoint_path'])
        if sha256(ckpt)!=self.lock['checkpoint_sha256']:raise ValueError('frozen localizer drift')
        code=next(Path(p) for p in self.lock['source_hashes'] if p.endswith('/model.py'))
        if sha256(code)!=self.lock['source_hashes'][str(code)]:raise ValueError('localizer implementation drift')
        spec=importlib.util.spec_from_file_location('adaptive_frozen_ezvsl',code)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.resnet18=lambda *args,**kwargs:resnet18(weights=None)
        model=module.EZVSL(.03,512)
        payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        model.load_state_dict({k.removeprefix('module.'):v for k,v in payload['model'].items()},strict=True)
        self.model=model.to(device).eval().requires_grad_(False);self.device=device
        self.transform=transforms.Compose([transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.audit={'strict_load':True,'checkpoint_sha256':self.lock['checkpoint_sha256'],
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'GT_used':False,'native_audio_s':3,'causal_audio_s':2,'left_zero_padding_s':1,'batch_size':1}
        # Each worker has the same immutable audit; never call the legacy initializer.
        write_json(ROOT/'audit/localizer_loaded.json',self.audit)

