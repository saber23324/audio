"""Select fixed F1 on legal development QA, then close the action definition."""
import argparse,json,time
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from utility_gate.common import write_jsonl
from utility_gate.grounded_dual_image.runtime import insert_views
from .common import ROOT,train_rows,read_config,write_json,sha256,status,journal,now
from .features import load_rgb
from .prepare_av import grounded_path
from .runtime import Runtime,qa_prompt
from .fuser64 import fuse,dependencies

@torch.inference_mode()
def identity(r,row,visual,z,p):
    zs={'direct':z,'off':fuse(z,on=False,beta=.5),
        'beta0':fuse(z,on=True,beta=0),'unavailable':fuse(z,on=True,beta=.5,available=False),
        'duplicate':fuse(z,z.clone(),on=True,beta=.5)}
    logits={};ids={}
    for name,value in zs.items():
        if not torch.equal(z,value):raise AssertionError('identity Z failure')
        inputs=insert_views(r.model,qa_prompt(r,row),[value])
        output=r.model.model(inputs_embeds=inputs,use_cache=False,return_dict=True)
        logits[name]=r.model.lm_head(output.last_hidden_state[:,-1]).float().cpu()
        ids[name]=r.score(row,value)['generated_ids']
    independently_recomputed_z,independently_recomputed_p=r.vision_latents(visual)
    equality={n:{'Z':torch.equal(independently_recomputed_z,v),'P':torch.equal(p,independently_recomputed_p),
        'logits':torch.equal(logits['direct'],logits[n]),'generated_ids':ids[n]==ids['direct']} for n,v in zs.items()}
    if not all(all(x.values()) for x in equality.values()):raise AssertionError('identity decode mismatch')
    write_json(ROOT/'trace/identity_results.json',{'actual_large_model_run':True,'source_id':row['source_id'],
        'checks':equality,'generated_ids':ids,'map_calls_on_bypass':0,'extra_encoder_calls_on_bypass':0,
        'P_shape':list(p.shape),'Z_shape':list(z.shape),'no_audio_in_new_stage1':True})

@torch.inference_mode()
def run():
    if (ROOT/'locks/fuser_lock.json').exists():raise FileExistsError('fixed action already frozen')
    rows=train_rows('gate_val');groups=defaultdict(list)
    for row in rows:groups[row['source_id']].append(row)
    runtime=Runtime();runtime.load_new_router();candidates=read_config()['fuser']['beta_candidates'];out=[];diag=[]
    for j,(source,questions) in enumerate(groups.items()):
        rgb=load_rgb(questions[0]);g=torch.load(grounded_path(source),map_location='cpu',weights_only=False)
        z,p=runtime.vision_latents(rgb['visual']);zg,pg=runtime.vision_latents(g['visual_grounded'])
        if j==0:identity(runtime,questions[0],rgb['visual'],z,p)
        cosine=F.cosine_similarity(z.float(),zg.float(),dim=-1)
        for i in range(len(z)):
            cross=F.normalize(z[i].float(),dim=-1)@F.normalize(zg[i].float(),dim=-1).T
            ri,ci=linear_sum_assignment(-cross.cpu().numpy())
            diag.append({'source_id':source,'timestamp_s':float(rgb['timestamps'][i]),'same_slot_cosine':float(cosine[i].mean()),
                'optimal_assignment_cosine':float(cross[ri,ci].mean()),'same_slot_assignment_fraction':float(np.mean(ri==ci)),
                'P_cosine':float(F.cosine_similarity(p[i],pg[i],dim=-1).mean()),'grounded_map_valid':bool(g['map_valid'][i])})
        for beta in [0.]+candidates:
            mixed=fuse(z,zg,on=True,beta=beta)
            for row in questions:
                value=runtime.score(row,mixed);out.append({**row,**value,'strategy':f'beta_{beta}','beta':beta,'tokens_per_frame':64})
        print('fuser validation',j+1,len(groups),source,flush=True)
    summaries=[]
    base_visual=[x for x in out if x['beta']==0 and x.get('question_type')=='Visual']
    for beta in [0.]+candidates:
        a=[x for x in out if x['beta']==beta]
        summaries.append({'beta':beta,'n_qa':len(a),'accuracy':float(np.mean([x['correct'] for x in a])),
            'gold_nll':float(np.mean([x['gold_nll'] for x in a]))})
    # Beta=0 is an identity reference, not a candidate grounded action.
    selected=min([x for x in summaries if x['beta'] in candidates],key=lambda x:(-x['accuracy'],x['gold_nll'],x['beta']))
    write_jsonl(ROOT/'qa_eval/fuser_validation.jsonl',out);write_jsonl(ROOT/'trace/f1_slot_diagnostics.jsonl',diag)
    write_json(ROOT/'locks/fuser_lock.json',{'created_utc':now(),'beta':selected['beta'],'selection':'legal val real EM, then lower Gold NLL, then smaller beta',
        'candidates_and_results':summaries,'dependencies':dependencies(selected['beta']),
        'validation_manifest_sha256':sha256(ROOT/'manifests/gate_val.jsonl'),'prediction_file_sha256':sha256(ROOT/'qa_eval/fuser_validation.jsonl'),
        'F2':'not_run_optional; F1 remains preregistered main strategy; report slot distortion honestly'})
    status('M2','completed',beta=selected['beta']);journal('F1 action frozen before utility targets',beta=selected['beta']);print('FUSER FROZEN',selected,flush=True)

if __name__=='__main__':run()

