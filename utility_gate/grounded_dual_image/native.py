"""Actual LLaVA multi-image preparation, with one single-crop image per marker."""
from __future__ import annotations
import torch
from PIL import Image
from utility_gate.runtime import clocked

@torch.inference_mode()
def prepare(runtime,ids,frames):
    from llava.constants import IMAGE_TOKEN_INDEX
    if int((ids==IMAGE_TOKEN_INDEX).sum())!=len(frames):raise ValueError('actual image count differs from markers')
    def preprocess():
        images=[Image.fromarray(f) for f in frames]
        pixels=[runtime.processor.preprocess(im,return_tensors='pt')['pixel_values'].to(runtime.device,dtype=torch.float16) for im in images]
        return images,pixels
    (images,pixels),process_s=clocked(preprocess)
    def native_call():
        return runtime.model.prepare_inputs_labels_for_multimodal(ids[None].to(runtime.device),None,None,None,None,
            pixels,modalities=['image']*len(pixels),image_sizes=[im.size for im in images])[4]
    embeddings,vision_s=clocked(native_call)
    n=len(images);content_tokens=embeddings.shape[1]-(ids.numel()-n)
    if content_tokens!=730*n:raise AssertionError('native single-crop metadata contract changed')
    views=[]
    for i,p in enumerate(torch.where(ids==IMAGE_TOKEN_INDEX)[0].tolist()):
        start=p+i*(730-1);views.append(embeddings[0,start:start+730])
    # Prove the actual native preparation inserts the same distinct spans.
    from .runtime import insert_views
    if not torch.equal(insert_views(runtime.model,ids,views),embeddings):raise AssertionError('native marker layout mismatch')
    return embeddings,views,{'image_processor_s':process_s,'native_image_encoder_projector_and_prepare_s':vision_s,
        'actual_image_count':n,'image_markers':n,'native_tokens_per_image':730,'native_total_visual_tokens':content_tokens,
        'modalities':['image']*n,'same_time_view_pairs':n//2,'fake_video_frames':False}

def feature_mix(z_rgb,z_grounded,alpha):
    if z_rgb.shape!=z_grounded.shape or z_rgb.shape[-2]!=64:raise ValueError('same-bank 64-slot features required')
    if not 0<=alpha<=1:raise ValueError('invalid feature mixture alpha')
    if alpha==0:return z_rgb.clone()
    if alpha==1:return z_grounded.clone()
    return (1-alpha)*z_rgb+alpha*z_grounded
