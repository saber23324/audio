"""Fresh development sweeps, immutable pretest lock and one holdout run."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from utility_gate.common import REPO,append_jsonl,fingerprint,read_jsonl,seed_all,sha256,write_json
from utility_gate.grounded_image_pilot.proxy_eval import teacher_task
from utility_gate.grounded_image_pilot.report import pair
from utility_gate.grounded_image_pilot.routing_bias import resample
from utility_gate.runtime import score_answer
from utility_gate.strict_runtime import StrictStreamingRuntime
from .feature_diagnostics import encode_capture,evaluate_split,make_images
from .scaleup_manifest import CONFIG,ROOT,holdout_guard,now,stage
from .spatial_mapping import ImageTransform,pooled_occupancy,routing_metrics


def runtime_checks(runtime,root,label):
    disabled=runtime.regression_fixture(REPO/'artifacts/relevance_gate/m0-gate-disabled-final-20261022')
    vf=torch.load(REPO/'artifacts/relevance_gate/m0-vision-20261022/fixture.pt',map_location='cpu',weights_only=True)
    visual=torch.cat([c['visual'] for c in vf['chunks']]).to(runtime.device).float()
    z,p=runtime.vision_router(visual,None,vf['timestamps'].to(runtime.device))
    assert torch.equal(z.cpu(),torch.cat([c['latents'] for c in vf['chunks']]))
    generation=runtime.generate(vf['prompt_ids'],z)
    assert generation['generated_ids']==vf['generated_ids'].reshape(-1).tolist()
    write_json(root/'audit'/f'{label}_runtime_checks.json',{'Stage3_disabled':disabled,'Stage1_fixture_exact':True,
        'Stage1_generated_ids':generation['generated_ids'],'all_parameters_frozen':not any(p.requires_grad for m in
        (runtime.model,runtime.vision_router,runtime.audio_router,runtime.audio_encoder) for p in m.parameters())})


@torch.inference_mode()
def references(root,split,rows,runtime,pilot,resume=False):
    path=root/'per_source'/split/'stage3_references.jsonl'
    prior=read_jsonl(path) if path.exists() else []
    if prior and not resume:raise FileExistsError('stage3 reference file exists')
    done={r['clip_id'] for r in prior}
    silent=runtime.audio_encoder.encode_windows(torch.zeros(1,32000),runtime.device).to(runtime.device).float()
    for index,row in enumerate(rows):
        if row['clip_id'] in done:continue
        t=row['annotation_time_s'];windows,valid,quality,_=runtime.causal_windows(row.get('audio_path',row['media_path']),[t])
        result={'clip_id':row['clip_id'],'source_id':row['source_id'],'audio_valid':bool(valid[0]),'audio_quality':quality[0],
                'causal_window':[max(0,t-2),t],'single_frame_delta_t':0.,'methods':{}}
        if bool(valid[0]):
            rgb=np.asarray(Image.open(row['frame_path']).convert('RGB'));v=runtime.extractor.encode_visual(runtime.model,runtime.processor,rgb[None],1).float()
            audio=runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float();times=torch.tensor([t],device=runtime.device)
            transform=ImageTransform(rgb.shape[1],rgb.shape[0]);occupancy=pooled_occupancy(row['bbox'],transform)
            prompts={'teacher':runtime.prompt(teacher_task(pilot))}
            if row['proxy_category']['applicable']:prompts['category']=runtime.prompt(row['proxy_category'])
            for name,value in [('original_stage3',audio),('null_stage3',silent)]:
                z,p=runtime.audio_router(v,value,times);assert torch.equal(z,p@v)
                method={'stage':3,'gt_oracle':False,'visual_tokens':64,'spatial':routing_metrics(p[0].cpu().numpy(),occupancy,row['bbox'],transform)}
                for task,ids in prompts.items():
                    generation=runtime.generate(ids,z)
                    method[task]=score_answer(generation,row['proxy_category']['answer']) if task=='category' else generation
                result['methods'][name]=method
        else:result['status']='not_run_invalid_audio'
        append_jsonl(path,result)
        if (index+1)%40==0 or index==len(rows)-1:print({'stage3_reference_split':split,'completed':index+1,'total':len(rows)},flush=True)


@torch.inference_mode()
def sensitivity(root,rows,runtime,cfg,pilot,resume=False):
    path=root/'per_source/fresh_dev/sensitivity.jsonl';previous=read_jsonl(path) if path.exists() else []
    if previous and not resume:raise FileExistsError('sensitivity outputs exist')
    done={r['clip_id'] for r in previous}
    chosen=sorted(rows,key=lambda r:fingerprint([1234,'sensitivity',r['source_id']]))[:cfg['fresh_dev_sensitivity_max_sources']]
    write_json(root/'config/sensitivity_sources.json',{'source_ids':[r['source_id'] for r in chosen],'max_sources':cfg['fresh_dev_sensitivity_max_sources'],
        'alphas':cfg['sensitivity_alphas'],'lambdas':cfg['sensitivity_bias_lambdas'],'original_main_retained':True})
    for index,row in enumerate(chosen):
        if row['clip_id'] in done:continue
        rgb=np.asarray(Image.open(row['frame_path']).convert('RGB'));v=encode_capture(runtime,rgb)['visual_tokens_196']
        transform=ImageTransform(rgb.shape[1],rgb.shape[0]);occ=pooled_occupancy(row['bbox'],transform)
        gt_prior=torch.tensor(occ,device=runtime.device,dtype=torch.float32)
        _,_,_,meta=make_images(row,rgb,cfg)
        b=np.asarray(row['bbox'])+np.array([meta['dx_pixels']/rgb.shape[1],meta['dy_pixels']/rgb.shape[0]]*2)
        sham_prior=torch.tensor(pooled_occupancy(b,transform),device=runtime.device,dtype=torch.float32)
        rng=np.random.default_rng(int(fingerprint([1234,'permuted-prior',row['source_id']])[:16],16))
        shuffled_prior=gt_prior[torch.tensor(rng.permutation(196),device=runtime.device)]
        prompts={'teacher':runtime.prompt(teacher_task(pilot))}
        if row['proxy_category']['applicable']:prompts['category']=runtime.prompt(row['proxy_category'])
        variants={}
        for alpha in cfg['sensitivity_alphas']:
            images,_,_,_=make_images(row,rgb,cfg,alpha)
            for mode in ('gt_overlay','sham_overlay','random_color_overlay'):
                value=runtime.extractor.encode_visual(runtime.model,runtime.processor,images[mode][None],1).float()
                z,p=runtime.vision_router(value,None,None)
                name=f'alpha_{alpha:g}_{mode}';variants[name]=(z,p,{'alpha':alpha,'control':mode})
        for strength in cfg['sensitivity_bias_lambdas']:
            for name,prior in [('gt',gt_prior),('sham',sham_prior),('permuted',shuffled_prior)]:
                z,p=resample(runtime.vision_router,v,prior,strength)
                variants[f'lambda_{strength:g}_{name}']=(z,p,{'lambda':strength,'control':name})
        methods={}
        for name,(z,p,parameters) in variants.items():
            method={'parameters':parameters,'gt_oracle':parameters['control'] in ('gt_overlay','random_color_overlay','gt'),
                    'spatial':routing_metrics(p[0].cpu().numpy(),occ,row['bbox'],transform),'visual_tokens':64}
            for task,ids in prompts.items():
                generation=runtime.generate(ids,z)
                method[task]=score_answer(generation,row['proxy_category']['answer']) if task=='category' else generation
            methods[name]=method
        append_jsonl(path,{'clip_id':row['clip_id'],'source_id':row['source_id'],'split':'fresh_dev','methods':methods})
        if (index+1)%8==0 or index==len(chosen)-1:print({'dev_sensitivity_complete':index+1,'total':len(chosen)},flush=True)


def lock_pretest(root,cfg):
    path=root/'pretest_lock.json'
    if path.exists():raise FileExistsError('pretest lock is immutable')
    dev=read_jsonl(root/'per_source/fresh_dev/results.jsonl')
    dm=np.array([r['methods']['gt_overlay']['spatial']['mass']-r['methods']['rgb_clean']['spatial']['mass'] for r in dev])
    qa=[r for r in dev if 'category' in r['methods']['rgb_clean']]
    dq=np.array([int(r['methods']['gt_overlay']['category']['correct'])-int(r['methods']['rgb_clean']['category']['correct']) for r in qa])
    def n_required(a,delta):
        if len(a)<2 or float(a.var(ddof=1))<=0:return None
        return int(math.ceil((1.96+.84)**2*float(a.var(ddof=1))/delta**2))
    power={'design_only_fresh_dev_n':len(dev),'proxy_n':len(qa),'ROI_mass_minimum_important':cfg['minimum_important_roi_mass'],
           'proxy_accuracy_minimum_important':cfg['minimum_important_proxy_accuracy'],
           'ROI_N_approx_80pct':n_required(dm,cfg['minimum_important_roi_mass']),
           'proxy_accuracy_N_approx_80pct':n_required(dq,cfg['minimum_important_proxy_accuracy']),
           'formula':'normal approx (1.96+.84)^2 * fresh-dev paired sample variance / design delta^2; not power evidence from confirmation',
           'zero_discordance_policy':'null required N when observed paired variance is zero; cannot infer zero required samples',
           'no_holdout_topup_on_significance':True}
    write_json(root/'audit/power_design.json',power)
    files=[Path(__file__),Path(__file__).with_name('feature_diagnostics.py'),Path(__file__).with_name('feature_stats.py'),
           Path(__file__).with_name('scaleup_manifest.py'),Path(__file__).with_name('scaleup_data.py')]
    for name in ('heatmaps.py','image_fusion.py','routing_bias.py','proxy_eval.py'):
        files.append(REPO/'utility_gate/grounded_image_pilot'/name)
    files += [REPO/'utility_gate/runtime.py',REPO/'utility_gate/strict_runtime.py',REPO/'relevance_gate/media.py',
              REPO/'streamingtom/audio_bottleneck/adbt.py',REPO/'streamingtom/train_adbt_videomme.py']
    value={'created_utc':now(),'holdout_manifest_sha256':sha256(root/'manifests/fresh_holdout_LOCKED.jsonl'),
        'fresh_dev_manifest_sha256':sha256(root/'manifests/fresh_dev.jsonl'),
        'inference_source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in files},
        'config_sha256':sha256(root/'config/run.json'),'models':json.loads((root/'audit/M0_LOCK.json').read_text())['models'],
        'formal_alpha':cfg['alpha'],'formal_lambda':cfg['lambda_bias'],'formal_color':cfg['color_rgb'],
        'selection':'original configuration retained a priori, regardless of sweep; all sweeps development-only',
        'primary_comparisons':['B1-B0','B1-B2'],'secondary_comparisons':['B1-C1','B4-B0'],
        'bootstrap_replicates':2000,'seed':1234,'analysis_unit':'original source video; all strategy pairs have same source IDs',
        'minimum_important_mass':.01,'minimum_important_accuracy':.01,
        'GO_A':'B1-B0 and B1-B2 mass CI lower>0, both point estimates>=.01, and paired gold NLL or teacher KL CI upper<0 for both; labels remain proxy; no genuine QA claim',
        'GO_B':'B4-B0 proxy Gold-NLL CI upper<0 with n>=30, mechanical ROI shift and nonzero Z change, while B1 GO-A fails; only a lead for later GT-free predicted bias',
        'NO_GO':'B1 fails improvement and its spatial CI upper bounds < minimum important .01 vs B0 and Sham; not a general equivalence proof',
        'INCONCLUSIVE':'unavailable/new sources too few, proxy ceiling, or intervals broad; report with actual denominator',
        'stopping':'fixed acquired/QC-disjoint holdout, no optional stopping or result-driven sample additions',
        'holdout_execution':'two deterministic source-index shards on GPUs6/7, batch1; merge only after both complete; no extra source added',
        'B5_B6':'not_run: no predeclared verified independent localizer for fresh sources; existing legacy routing maps remain distinct',
        'holdout_predictions_at_lock':0,'power_design':power}
    write_json(path,value)
    for file in files:
        out=root/'audit/pretest_source_snapshot'/file.relative_to(REPO);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,out)
    stage(root,'M3','done',fresh_dev=len(dev),sensitivity_sources=cfg['fresh_dev_sensitivity_max_sources'],pretest_lock_sha256=sha256(path))
    print({'pretest_locked':True,'power_design':power},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--split',choices=['fresh_dev','fresh_holdout'],required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1)
    a=p.parse_args();root=a.output_dir;cfg=json.loads(a.config.read_text())
    if cfg!=json.loads((root/'config/run.json').read_text()):raise ValueError('run config differs from M0')
    manifest=root/'manifests'/('fresh_dev.jsonl' if a.split=='fresh_dev' else 'fresh_holdout_LOCKED.jsonl')
    if a.dry_run:print({'split':a.split,'manifest':str(manifest),'requires_pretest_lock':a.split=='fresh_holdout'});return
    if a.split=='fresh_holdout':holdout_guard(root,manifest)
    rows=read_jsonl(manifest)
    if not 0<=a.shard_index<a.num_shards:raise ValueError('invalid shard index')
    if a.split=='fresh_dev' and a.num_shards!=1:raise ValueError('fresh_dev uses one fixed execution')
    label=a.split if a.num_shards==1 else a.split+f'_shard{a.shard_index}'
    full_n=len(rows);rows=[r for i,r in enumerate(rows) if i%a.num_shards==a.shard_index]
    seed_all(cfg['seed']);stage(root,'M3' if a.split=='fresh_dev' else 'M4','running',sources=full_n,shards=a.num_shards)
    runtime=StrictStreamingRuntime();assert str(runtime.device)==a.device;runtime_checks(runtime,root,label)
    pilot=json.loads((root/'config/pilot_config.json').read_text())
    evaluate_split(root,label,rows,runtime,cfg,pilot,None,a.resume)
    references(root,label,rows,runtime,pilot,a.resume)
    if a.split=='fresh_dev':
        sensitivity(root,rows,runtime,cfg,pilot,a.resume)
        lock_pretest(root,cfg)
    elif a.num_shards==1:stage(root,'M4','done',sources=len(rows),formal_cohort_once=True)
    else:stage(root,'M4_shard_'+str(a.shard_index),'done',sources=len(rows),formal_cohort_once=True)


if __name__=='__main__':main()
