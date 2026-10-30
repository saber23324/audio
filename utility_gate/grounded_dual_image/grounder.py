"""Frozen official EZ-VSL AVL predictions from strict two-second past audio."""
from __future__ import annotations
import importlib.util,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import signal
import torch
import torch.nn.functional as F
from torchvision import transforms
from .common import ROOT,DATA,sha256,write_json,append_jsonl

def spectrogram(window):
    samples=np.asarray(window,dtype=np.float32)
    if samples.shape!=(32000,) or not np.isfinite(samples).all():raise ValueError('finite causal 2s/16k mono window required')
    # Original EZ-VSL expects 3s. Preserve ONLY the 2s history, left pad 1s;
    # do not use the upstream centered/full-track audio loader or repeat future.
    native=np.pad(np.clip(samples,-1,1),(16000,0))
    _,_,spec=signal.spectrogram(native,16000,nperseg=512,noverlap=274)
    return np.log(spec+1e-7).astype(np.float32)/12.

def unit_map(value):
    value=np.asarray(value,dtype=np.float32)
    if value.ndim!=2 or not np.isfinite(value).all():return np.zeros(value.shape[-2:],dtype=np.float32),False,'nonfinite_or_malformed_map'
    spread=float(value.max()-value.min())
    if spread<1e-8:return np.zeros_like(value),False,'constant_map'
    return (value-value.min())/spread,True,'ok'

def render(frame,heatmap,valid=True,alpha=.2,area=.2):
    frame=np.asarray(frame,dtype=np.uint8)
    h=np.asarray(heatmap,dtype=np.float32)
    if h.shape!=frame.shape[:2]:raise ValueError('map/RGB geometry mismatch')
    if not valid or not np.isfinite(h).all():return frame.copy()
    if not(0<=alpha<=1 and 0<area<=1):raise ValueError('invalid overlay parameters')
    # Exact fixed-area highlight; stable ties, no boxes/classes/QA information.
    k=max(1,int(np.ceil(h.size*area)));idx=np.argsort(h.reshape(-1),kind='stable')[-k:]
    mask=np.zeros(h.size,dtype=np.float32);mask[idx]=1;mask=mask.reshape(h.shape)
    weight=(alpha*h*mask)[...,None]
    return np.rint((1-weight)*frame.astype(np.float32)+weight*np.array([255,48,48],np.float32)).clip(0,255).astype(np.uint8)

def sham(heatmap,source_id,index):
    import hashlib
    h=np.asarray(heatmap)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'1234:{source_id}:{index}:sham'.encode()).digest()[:8],'big'))
    # A location-shift control, not a verified incorrect sound source label.
    dy=int(rng.integers(max(1,h.shape[0]//3),max(2,2*h.shape[0]//3)))
    dx=int(rng.integers(max(1,h.shape[1]//3),max(2,2*h.shape[1]//3)))
    return np.roll(h,(dy,dx),(0,1))

class FrozenLocalizer:
    def __init__(self,device):
        self.lock=json.loads((ROOT/'config/localizer_lock.json').read_text())
        ckpt=Path(self.lock['checkpoint_path'])
        if sha256(ckpt)!=self.lock['checkpoint_sha256']:raise ValueError('localizer checkpoint drift')
        code=DATA/'localizer/model.py'
        if sha256(code)!=self.lock['source_hashes'][str(code)]:raise ValueError('official localizer code drift')
        spec=importlib.util.spec_from_file_location('frozen_official_ezvsl',code);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from torchvision.models import resnet18
        # Every tensor is subsequently loaded strictly from the full official
        # AV checkpoint. Avoid an unused automatic ImageNet checkpoint download.
        module.resnet18=lambda *args,**kwargs:resnet18(weights=None)
        model=module.EZVSL(.03,512)
        checkpoint=torch.load(ckpt,map_location='cpu',weights_only=False)
        state={k.removeprefix('module.'):v for k,v in checkpoint['model'].items()}
        model.load_state_dict(state,strict=True)
        self.model=model.to(device).eval().requires_grad_(False);self.device=device
        self.transform=transforms.Compose([transforms.Resize((224,224),transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        write_json(ROOT/'audit/localizer_loaded.json',{'strict_checkpoint_load':True,'unfrozen_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':self.lock['checkpoint_sha256']})

    @torch.inference_mode()
    def predict(self,frames,windows,valid):
        from utility_gate.runtime import synchronize
        start=time.perf_counter()
        image=torch.stack([self.transform(Image.fromarray(f)) for f in frames]).to(self.device)
        audio=torch.from_numpy(np.stack([spectrogram(w) for w in windows]))[:,None].to(self.device)
        synchronize();encoded=time.perf_counter()
        logits=self.model(image,audio)[1]
        synchronize();model_s=time.perf_counter()-encoded
        maps=[];okay=[];reasons=[]
        for i,(frame,value) in enumerate(zip(frames,logits)):
            resized=F.interpolate(value[None,None],size=frame.shape[:2],mode='bilinear',align_corners=True)[0,0].float().cpu().numpy()
            h,good,reason=unit_map(resized)
            if not bool(valid[i]):h=np.zeros_like(h);good=False;reason='invalid_audio_rgb_fallback'
            maps.append(h);okay.append(good);reasons.append(reason)
        return {'maps':np.stack(maps),'native_logits':logits.float().cpu().numpy(),'valid':okay,'reasons':reasons,
            'timing':{'localizer_forward_s':model_s,'localizer_preprocess_and_total_s':time.perf_counter()-start}}

def save_maps(source,row,data,prediction,alpha=.2):
    directory=ROOT/'maps/predicted_rawcrop_v1';directory.mkdir(exist_ok=True);path=directory/f'{source}.npz'
    if not path.exists():np.savez_compressed(path,heatmaps=prediction['maps'],native_logits=prediction['native_logits'],valid=prediction['valid'],timestamps_s=data['timestamps'].numpy())
    for i,(frame,h,valid,t) in enumerate(zip(data['frames'],prediction['maps'],prediction['valid'],data['timestamps'].tolist())):
        output=ROOT/f'images/grounded_rawcrop_v1/{source}_{i}_a{alpha:.3f}.png';output.parent.mkdir(exist_ok=True)
        if not output.exists():Image.fromarray(render(frame,h,valid,alpha=alpha)).save(output)
        append_jsonl(ROOT/'map_provenance_rawcrop_v1.jsonl',{'source_id':row['source_id'],'source_video_id':source,'frame_index':i,
            'frame_time':t,'frame_pts_s':data['frame_pts_s'][i],'audio_start':max(0,t-2),'audio_end':t,
            'raw_audio_truncated_before_resampling':True,'raw_audio_trace':prediction['audio_traces'][i],'native_audio_padding_s':1.,'map_origin':'EZVSL_PRED',
            'component':'AVL only','oracle':False,'valid':valid,'quality_reason':prediction['reasons'][i],
            'alpha':alpha,'map_npz':str(path),'map_sha256':sha256(path),'render_png':str(output),'render_sha256':sha256(output),
            'localizer_lock_sha256':sha256(ROOT/'config/localizer_lock.json'),'question_or_answer_input':False})
