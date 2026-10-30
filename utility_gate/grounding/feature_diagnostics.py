"""Read-only hooks and real frozen VLM replay, isolated from the old pilot."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from utility_gate.common import REPO, append_jsonl, fingerprint, read_jsonl, seed_all, sha256, write_json
from utility_gate.grounded_image_pilot.eval import row_diagnostics,validate_geometry
from utility_gate.grounded_image_pilot.heatmaps import box_map,outline_map,rng_for,sham_map
from utility_gate.grounded_image_pilot.image_fusion import overlay
from utility_gate.grounded_image_pilot.proxy_eval import teacher_task
from utility_gate.grounded_image_pilot.routing_bias import resample
from utility_gate.runtime import score_answer
from utility_gate.strict_runtime import StrictStreamingRuntime
from .feature_stats import compression_decomposition,feature_response,pixel_response,routing_response
from .scaleup_manifest import CONFIG,ROOT,stage,now
from .spatial_mapping import ImageTransform,native_occupancy,pooled_occupancy,routing_metrics


class FeatureCapture:
    def __init__(self,runtime):self.runtime=runtime;self.values={};self.handles=[]
    def __enter__(self):
        def hook(name):
            def capture(module,args,output):
                value=output[0] if isinstance(output,(tuple,list)) else output
                self.values[name]=value.detach().float()
                # Returning None is essential: no replacement forward value.
            return capture
        self.handles=[self.runtime.model.get_vision_tower().register_forward_hook(hook('encoder_patch_features')),
            self.runtime.model.get_model().mm_projector.register_forward_hook(hook('visual_projector_features'))]
        return self
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()


@torch.inference_mode()
def encode_capture(runtime,rgb):
    with FeatureCapture(runtime) as capture:
        visual=runtime.extractor.encode_visual(runtime.model,runtime.processor,rgb[None],1).float()
    if capture.values['encoder_patch_features'].shape[1]!=729 or visual.shape[1]!=196:raise ValueError('unexpected feature grid')
    return {**capture.values,'visual_tokens_196':visual}


def make_images(row,rgb,cfg,alpha=None):
    alpha=cfg['alpha'] if alpha is None else alpha
    gt=box_map(row['bbox'],rgb.shape[0],rgb.shape[1]);sham,meta=sham_map(gt,row['source_id'],cfg['seed'])
    color=rng_for(row['source_id'],'random-color',cfg['seed']).integers(0,256,3).tolist()
    maps={'rgb_clean':np.zeros_like(gt),'gt_overlay':gt,'sham_overlay':sham,'uniform_overlay':np.full_like(gt,.5),
          'gt_routing_bias':np.zeros_like(gt),'random_color_overlay':gt,
          'bbox_outline_overlay':outline_map(row['bbox'],rgb.shape[0],rgb.shape[1])}
    images={name:overlay(rgb,h,alpha,color if name=='random_color_overlay' else cfg['color_rgb']) for name,h in maps.items()}
    return images,maps,gt,meta


def router_state(runtime,features,prior,mode,cfg):
    v=features['visual_tokens_196'];router=runtime.vision_router
    q=router.query_generator.static_queries(1)
    r=router.compute_attention_scores(q,router.visual_key(v))
    if mode=='gt_routing_bias':
        r=r+cfg['lambda_bias']/router.attention_temperature*(prior-prior.mean())[None,None]
        z,p=resample(router,v,prior,cfg['lambda_bias'])
    else:z,p=router(v,None,None)
    if not torch.equal(z,p@v):raise AssertionError('native visual values changed')
    return {**features,'router_queries':q,'router_logits_R':r,'routing_matrix_P':p,'compressed_tokens_Z':z}


def measurements(state,base,rgb,changed,gt,mask,row,occ729,occ196):
    metrics={'pixel':pixel_response(rgb,changed,gt,mask)};maps={}
    for name,w in [('encoder_patch_features',occ729),('visual_projector_features',occ729),('visual_tokens_196',occ196),('compressed_tokens_Z',None)]:
        metrics[name],values=feature_response(state[name],base[name],w)
        for key,v in values.items():maps[name+'_'+key]=v
    metrics['routing']=routing_response(state['router_logits_R'],state['routing_matrix_P'],base['router_logits_R'],base['routing_matrix_P'],occ196)
    metrics['compression_decomposition']=compression_decomposition(state['routing_matrix_P'],state['visual_tokens_196'],base['routing_matrix_P'],base['visual_tokens_196'])
    metrics['queries_exact_static_bank']=bool(torch.equal(state['router_queries'],base['router_queries']))
    return metrics,maps


def source_lock(root,split,rows):
    paths=[Path(__file__),Path(__file__).with_name('feature_stats.py'),
        Path(__file__).with_name('scaleup_manifest.py'),REPO/'utility_gate/grounded_image_pilot/heatmaps.py',
        REPO/'utility_gate/grounded_image_pilot/image_fusion.py',REPO/'utility_gate/grounded_image_pilot/routing_bias.py']
    if split!='legacy_dev':paths.append(Path(__file__).with_name('scaleup_eval.py'))
    value={'created_utc':now(),'split':split,'source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in paths if p.exists()},
           'manifest_fingerprint':fingerprint(rows),'config_sha256':sha256(root/'config/run.json')}
    target=root/'audit'/f'{split}_inference_lock.json'
    if target.exists():
        old=json.loads(target.read_text())
        for key in ('source_sha256','manifest_fingerprint','config_sha256'):
            if old[key]!=value[key]:raise ValueError('locked inference inputs changed')
    else:
        write_json(target,value)
        for p in paths:
            if p.exists():
                out=root/'source_snapshot'/p.relative_to(REPO);out.parent.mkdir(parents=True,exist_ok=True)
                if not out.exists():shutil.copyfile(p,out)


@torch.inference_mode()
def evaluate_split(root,split,rows,runtime,cfg,pilot,legacy_results=None,resume=False):
    out=root/'per_source'/split;out.mkdir(parents=True,exist_ok=True)
    tensor_dir=root/'features'/split;tensor_dir.mkdir(parents=True,exist_ok=True)
    previous=read_jsonl(out/'results.jsonl') if (out/'results.jsonl').exists() else []
    if previous and not resume:raise FileExistsError('predictions exist; explicit --resume required')
    done={r['clip_id'] for r in previous};source_lock(root,split,rows)
    saved=[]
    for index,row in enumerate(rows):
        if row['clip_id'] in done:continue
        begin=time.perf_counter();rgb=np.asarray(Image.open(row['frame_path']).convert('RGB')).copy()
        if sha256(row['frame_path'])!=row['frame_sha256']:raise ValueError('frame hash changed')
        images,masks,gt,sham=make_images(row,rgb,cfg)
        assert np.array_equal(overlay(rgb,gt,0,cfg['color_rgb']),rgb)
        transform=ImageTransform(rgb.shape[1],rgb.shape[0]);occ729=native_occupancy(row['bbox'],transform).flatten();occ196=pooled_occupancy(row['bbox'],transform)
        prior=torch.tensor(occ196,device=runtime.device,dtype=torch.float32)
        features=encode_capture(runtime,rgb);base=router_state(runtime,features,prior,'rgb_clean',cfg)
        if index==0:
            unhooked=runtime.extractor.encode_visual(runtime.model,runtime.processor,rgb[None],1).float()
            assert torch.equal(unhooked,features['visual_tokens_196'])
            alpha0=encode_capture(runtime,overlay(rgb,gt,0,cfg['color_rgb']))
            for key in alpha0:assert torch.equal(alpha0[key],features[key])
            zero=router_state(runtime,alpha0,prior,'rgb_clean',cfg)
            for key in ('routing_matrix_P','compressed_tokens_Z'):assert torch.equal(zero[key],base[key])
        prompts={'teacher':runtime.prompt(teacher_task(pilot))}
        if row['proxy_category']['applicable']:prompts['category']=runtime.prompt(row['proxy_category'])
        hashes={k:fingerprint(v.tolist()) for k,v in prompts.items()}
        teacher=runtime.generate(prompts['teacher'],runtime.full_vision(features['visual_tokens_196']))
        methods={};delta_maps={};raw={};replay_errors=[]
        for mode in cfg['modes']:
            current=features if mode in ('rgb_clean','gt_routing_bias') else encode_capture(runtime,images[mode])
            state=base if mode=='rgb_clean' else router_state(runtime,current,prior,mode,cfg)
            response,maps=measurements(state,base,rgb,images[mode],gt,masks[mode],row,occ729,occ196)
            delta_maps.update({mode+'__'+k:v for k,v in maps.items()})
            method={'stage':1,'gt_oracle':mode in ('gt_overlay','gt_routing_bias','random_color_overlay','bbox_outline_overlay'),
                    'GT_area_only':mode=='sham_overlay','visual_tokens':64,'features':response,
                    'spatial':routing_metrics(state['routing_matrix_P'][0].cpu().numpy(),occ196,row['bbox'],transform),
                    'routing_diagnostics':row_diagnostics(state['routing_matrix_P']),'prompt_fingerprints':hashes}
            for task,ids in prompts.items():
                generation=runtime.generate(ids,state['compressed_tokens_Z'])
                method[task]=score_answer(generation,row['proxy_category']['answer']) if task=='category' else generation
            methods[mode]=method
            raw[mode]={key:t.detach().cpu().to(torch.float16 if key in ('encoder_patch_features','visual_projector_features') else torch.float32)
                       for key,t in state.items()} if index<cfg['full_tensor_sources'] else {
                           key:state[key].detach().cpu() for key in ('router_logits_R','routing_matrix_P')}
            if legacy_results:
                old=legacy_results[row['clip_id']]['methods'][mode]
                for task in prompts:
                    if method[task]['generated_ids']!=old[task]['generated_ids']:replay_errors.append(mode+':'+task+':IDs')
                    if max(abs(method[task]['option_logits'][c]-old[task]['option_logits'][c]) for c in 'ABCD')>2e-5:
                        replay_errors.append(mode+':'+task+':logits')
                with np.load(REPO/cfg['legacy_root']/'spatial'/f"{row['clip_id']}_routing.npz") as prior_npz:
                    if not np.array_equal(prior_npz[mode],state['routing_matrix_P'][0].cpu().numpy()):replay_errors.append(mode+':P')
        # True extra forwards for exact zero controls, not cached answer reuse.
        zz,zp=resample(runtime.vision_router,features['visual_tokens_196'],prior,0.)
        assert torch.equal(zz,base['compressed_tokens_Z']) and torch.equal(zp,base['routing_matrix_P'])
        control=runtime.generate(prompts['teacher'],zz)
        assert control['generated_ids']==methods['rgb_clean']['teacher']['generated_ids']
        assert control['option_logits']==methods['rgb_clean']['teacher']['option_logits']
        if index==0:
            control_alpha=runtime.generate(prompts['teacher'],zero['compressed_tokens_Z'])
            assert control_alpha['generated_ids']==control['generated_ids'] and control_alpha['option_logits']==control['option_logits']
            write_json(root/'audit'/f'{split}_hook_check.json',{'real_GPU_hook_unchanged_features':True,
                'alpha0_all_feature_layers_P_Z_generated_ids_logits_exact':True,'lambda0_P_Z_generated_ids_logits_exact':True,
                'tensor_shapes':{k:list(v.shape) for k,v in base.items()},'tensor_dtypes':{k:str(v.dtype) for k,v in base.items()}})
        if replay_errors:raise AssertionError('old replay mismatch: '+str(replay_errors))
        record={key:row[key] for key in ('clip_id','source_id','split','class','bbox','frame_pts','annotation_time_s','media_sha256','proxy_category')}
        record.update(methods=methods,teacher_full196=teacher,roi_area_fraction=float(occ196.mean()),
            bbox_area=float(row.get('bbox_area',gt.mean())),audio_valid=row.get('historical_audio_valid'),
            audio_quality=row.get('historical_audio_quality','not_measured_in_image_core'),
            legacy_replay_exact=True if legacy_results else None,sham=sham,
            alpha0_pixels_exact=True,lambda0_exact=True,full_tensors_saved=index<cfg['full_tensor_sources'],
            inference_s=time.perf_counter()-begin)
        np.savez_compressed(tensor_dir/f"{row['clip_id']}_delta_maps.npz",**delta_maps,occupancy729=occ729,occupancy196=occ196)
        torch.save(raw,tensor_dir/f"{row['clip_id']}_tensors.pt")
        append_jsonl(out/'results.jsonl',record)
        if (index+1)%8==0 or index==len(rows)-1:print({'split':split,'completed':index+1,'total':len(rows),'legacy_exact':not replay_errors},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--split',default='legacy_dev',choices=['legacy_dev'])
    p.add_argument('--output-dir',type=Path,default=ROOT);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--device',default='cuda:0')
    a=p.parse_args();root=a.output_dir;cfg=json.loads(a.config.read_text());rows=read_jsonl(root/'manifests/legacy_dev.jsonl')
    if a.dry_run:print({'split':a.split,'sources':len(rows),'hooks':'read only','training':False});return
    stage(root,'M1','running',sources=len(rows));seed_all(cfg['seed']);runtime=StrictStreamingRuntime();assert str(runtime.device)==a.device
    write_json(root/'audit/runtime_metadata.json',runtime.metadata);write_json(root/'audit/geometry.json',validate_geometry(runtime))
    baseline=runtime.regression_fixture(REPO/'artifacts/relevance_gate/m0-gate-disabled-final-20261022')
    write_json(root/'audit/extension_disabled_Stage3_fixture.json',baseline)
    old={r['clip_id']:r for r in read_jsonl(REPO/cfg['legacy_root']/'results.jsonl')}
    pilot=json.loads((root/'config/pilot_config.json').read_text())
    evaluate_split(root,a.split,rows,runtime,cfg,pilot,old,a.resume)
    stage(root,'M1','done',sources=len(rows),all_old_replays_exact=True)


if __name__=='__main__':main()
