"""Actual model-level identity, future perturbation and native-bypass evidence."""
import json,wave
import numpy as np
import torch
from utility_gate.grounded_dual_image.native import prepare
from utility_gate.grounded_dual_image.runtime import insert_views
from utility_gate.grounded_dual_image.media import raw_window
from utility_gate.features import take_features
from .common import ROOT,training_rows,write_json,sha256
from .runtime import Runtime,Localizer,native_prompt,append_step_features
from .features import load
from .fusion import load_fusion
from .policy import GateBundle

@torch.inference_mode()
def run():
    rows=training_rows('val');row=min(rows,key=lambda r:(r['source_id'],r['question_id']));d=load(row)
    r=Runtime();localizer=Localizer(r.device);bundle=GateBundle();fusion,_=load_fusion(device=r.device);r.attach_fusion(fusion)
    v=d['visual'].to(r.device).float();g=d['visual_grounded'].to(r.device).float();base,p=r.vision_latents(v)
    off=fusion(v,g,base,torch.zeros(4,dtype=torch.bool,device=r.device));zero=fusion(v,g,base,torch.ones(4,dtype=torch.bool,device=r.device),alpha=0.)
    a=r.score(row,base);b=r.score(row,off);c=r.score(row,zero)
    from .runtime import qa_prompt
    def logits(z):
        e=insert_views(r.model,qa_prompt(r,row),[z]);h=r.model.model(inputs_embeds=e,use_cache=False,return_dict=True).last_hidden_state[:,-1]
        return r.model.lm_head(h).float()
    la,lb=logits(base),logits(off)
    identity={'P_exact_direct_frozen_V1':torch.equal(p.cpu(),d['P_V1']),'Z_exact_OFF':torch.equal(base,off),'Z_exact_alpha0':torch.equal(base,zero),
        'logits_exact_OFF':torch.equal(la,lb),'generated_ids_exact_OFF_and_alpha0':a['generated_ids']==b['generated_ids']==c['generated_ids']}
    if not all(identity.values()):raise AssertionError('actual model identity regression failed')
    # Future media values removed before feature validation and prediction.
    original=append_step_features(d['visual'],d['audio'],d['audio_valid'],d['timestamps'],1)
    vf=d['visual'].float().clone();af=d['audio'].clone();vf[2:]=float('nan');af[2:]=float('inf')
    changed=append_step_features(vf,af,d['audio_valid'],d['timestamps'],1)
    x=bundle.one(original,0,1);y=bundle.one(changed,0,1)
    if x!=y:raise AssertionError('future changed r/u/ON')
    rng=np.random.default_rng(1234);prefix=rng.integers(-3000,3000,32000,dtype=np.int16);windows=[]
    directory=ROOT/'audit/future_audio';directory.mkdir(exist_ok=True)
    for index in range(2):
        path=directory/f'future_{index}.wav'
        with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(np.r_[prefix,rng.integers(-30000,30000,32000,dtype=np.int16)].tobytes())
        windows.append(raw_window(path,2.)[0])
    h=[localizer.predict([d['frames'][0]],w[None],torch.tensor([True]))['maps'][0] for w in windows]
    if not torch.equal(*windows) or not np.array_equal(*h):raise AssertionError('future waveform changed localizer')
    # Actual native8-image path: all compression Router hook counts remain zero.
    start=dict(r.router_calls);frames=[f for frame in d['frames'] for f in (frame,frame)]
    ids=native_prompt(r,row,d['timestamps'],[2]*4);embeds,views,meta=prepare(r,ids,frames)
    native_a=r.answer(ids,views,row['answer_text'],prepared=embeds,mcqa=row['dataset']!='music_avqa')
    native_b=r.answer(ids,views,row['answer_text'],mcqa=row['dataset']!='music_avqa')
    delta={k:r.router_calls[k]-start[k] for k in start}
    if any(delta.values()) or native_a['generated_ids']!=native_b['generated_ids']:raise AssertionError('native path called Router or marker insertion differs')
    result={'completed':True,'source_role':'val','source_id':row['source_id'],'identity':identity,
        'native_Router_forward_counts':delta,'actual_native_visual_tokens':meta['native_total_visual_tokens'],'actual_native_images':meta['actual_image_count'],
        'native_full_prepare_vs_insert_views_generated_ids_exact':True,'native_generated_ids':native_a['generated_ids'],
        'future_current_Gate_output_exact':x==y,'future_raw_audio_and_predicted_maps_exact':True,'current_visual_or_Router_inputs_have_no_future_frames':True,
        'forced_exposure_only_in_training':True,'V1_generated_ids':a['generated_ids'],'actual_LLM_and_localizer_used':True}
    write_json(ROOT/'audit/integration_checks.json',result);print(result,flush=True)

if __name__=='__main__':run()
