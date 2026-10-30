"""GPU identity and real conditional-call k0..4 acceptance, before confirmation."""
import json,wave
import numpy as np
import torch
from utility_gate.grounded_dual_image.runtime import insert_views
from utility_gate.grounded_dual_image.media import raw_window
from .common import ROOT,PREV,train_rows,write_json,read_jsonl
from .features import load
from .runtime import Runtime,Localizer,append_step_features,qa_prompt
from .online_inference import Controllers,execute
from .evaluate import variants,prepare_controllers
from .gate_models import history

@torch.inference_mode()
def run():
    runtime=Runtime();loc=Localizer(runtime.device);cs=Controllers();specs=variants();prepare_controllers(cs,specs)
    rows={r['source_id']:r for r in train_rows('val')};row=next(rows[s] for s in sorted(rows) if bool(load(rows[s])['audio_valid'].all()));d=load(row)
    base,p=runtime.vision_latents(d['visual'].to(runtime.device));v=d['visual'].float().to(runtime.device);g=d['visual_grounded'].float().to(runtime.device)
    off=runtime.fusion(v,g,base,torch.zeros(4,dtype=torch.bool,device=runtime.device));a=runtime.score(row,base);b=runtime.score(row,off)
    def logits(z):
        e=insert_views(runtime.model,qa_prompt(runtime,row),[z]);h=runtime.model.model(inputs_embeds=e,use_cache=False,return_dict=True).last_hidden_state[:,-1];return runtime.model.lm_head(h).float()
    checks={'V1_P_exact':torch.equal(p.cpu(),d['P_V1']),'OFF_Z_exact':torch.equal(base,off),'OFF_logits_exact':torch.equal(logits(base),logits(off)),
        'OFF_generated_IDs_exact':a['generated_ids']==b['generated_ids']}
    actual=[]
    for k in range(5):
        z,meta,acts=execute(runtime,loc,row,{'kind':'on'},cs,cap=k);score=runtime.score(row,z)
        if meta['k_extra_images']!=k or meta['encoded_image_count']!=4+k or meta['actual_localizer_calls']!=k:raise AssertionError('actual k/call interface failure')
        if k==0:checks.update(online_OFF_Z_exact=torch.equal(z,base),online_OFF_IDs_exact=score['generated_ids']==a['generated_ids'])
        if k==4:checks.update(online_ON_Z_exact=torch.equal(z.cpu(),d['R_A']))
        actual.append({'k':k,**meta,'generated_ids':score['generated_ids'],'gold_nll':score['gold_nll']})
    # Replay a genuine historical R_A generated-answer fixture using its prior
    # inputs. This old inspected validation source cannot enter fresh selection.
    historical=None
    for oldrow in sorted(read_jsonl(PREV/'manifests/val.jsonl'),key=lambda x:(x['source_id'],x['question_id'])):
        od=torch.load(PREV/f'features/development/{oldrow["source_id"]}.pt',map_location='cpu',weights_only=False)
        if not bool(od['audio_valid'].all()) or not all(od['map_valid']):continue
        oldscore=next(x for x in read_jsonl(PREV/f'results/val_AB/{oldrow["source_id"]}.jsonl') if x['strategy']=='B2_R_always' and x['question_id']==oldrow['question_id'])
        ov=od['visual'].float().to(runtime.device);og=od['visual_grounded'].float().to(runtime.device);oz,op=runtime.vision_latents(ov)
        z=torch.cat([runtime.fusion(ov[t:t+1],og[t:t+1],oz[t:t+1]) for t in range(4)])
        got=runtime.score(oldrow,z);checks['historical_R_A_generated_IDs_exact']=got['generated_ids']==oldscore['generated_ids'];historical={'source_id':oldrow['source_id'],'question_id':oldrow['question_id'],'generated_ids':got['generated_ids'],'historical_ids':oldscore['generated_ids']};break
    if historical is None:raise ValueError('historical trained R_A fixture missing')
    # Future corruption is sliced away before validation, pooling, or decisions.
    original=append_step_features(d['visual'],d['audio'],d['audio_valid'],d['timestamps'],1)
    vf=d['visual'].float().clone();af=d['audio'].clone();vf[2:]=float('nan');af[2:]=float('inf')
    changed=append_step_features(vf,af,d['audio_valid'],d['timestamps'],1)
    checks['future_features_exact']=all(torch.equal(getattr(original,k),getattr(changed,k)) for k in ('visual','audio','context','valid_audio'))
    masks={}
    for name in ('T3_accuracy','T4_cost','T7_accuracy_duplicate','T8_accuracy_sham','T9_accuracy_zero','T7_cost_duplicate','T8_cost_sham','T9_cost_zero'):
        spec=specs[name];z,meta,act=execute(runtime,loc,row,spec['spec'],cs,spec.get('view','real'));masks[name]=meta['on_mask'];s=runtime.score(row,z)
        actual.append({'policy':name,**meta,'generated_ids':s['generated_ids'],'gold_nll':s['gold_nll']})
    checks['same_mask_controls_exact']=all(masks[n]==masks[v['same_mask_as']] for n,v in specs.items() if v.get('same_mask_as'))
    random=np.random.default_rng(1234);prefix=random.integers(-3000,3000,32000,dtype=np.int16);ws=[]
    directory=ROOT/'audit/future_audio';directory.mkdir(exist_ok=True)
    for i in range(2):
        path=directory/f'{i}.wav'
        with wave.open(str(path),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(np.r_[prefix,random.integers(-30000,30000,32000,dtype=np.int16)].tobytes())
        ws.append(raw_window(path,2.)[0])
    maps=[loc.predict([d['frames'][0]],w[None],torch.tensor([True]))['maps'][0] for w in ws]
    checks['future_raw_PCM_and_localizer_map_exact']=torch.equal(*ws) and np.array_equal(*maps)
    checks['all_runtime_modules_frozen']=all(not p.requires_grad for module in (runtime.model,runtime.audio_encoder,runtime.vision_router,runtime.audio_router,runtime.new_router,runtime.fusion,loc.model) for p in module.parameters())
    result={'checks':checks,'completed':all(checks.values()),'fresh_val_source':row['source_id'],'historical_R_A_fixture':historical,'actual_GPU_generations':actual,'LLM_tokens_per_frame':64,'no_training_performed':True}
    write_json(ROOT/'audit/integration_checks.json',result)
    if not all(checks.values()):raise AssertionError(checks)
    print(result,flush=True)
if __name__=='__main__':run()
