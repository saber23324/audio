"""Real frozen-model B0-B4 inference, followed by paired proxy evaluation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from utility_gate.common import REPO, append_jsonl, fingerprint, read_jsonl, seed_all, sha256, write_json
from utility_gate.grounding.spatial_mapping import ImageTransform, LAYOUT, pooled_occupancy, routing_metrics
from utility_gate.runtime import clocked, score_answer
from utility_gate.strict_runtime import StrictStreamingRuntime
from .audit import CONFIG, ROOT
from .heatmaps import box_map, outline_map, rng_for, sham_map
from .image_fusion import overlay
from .proxy_eval import teacher_task
from .routing_bias import resample

IDS={'rgb_clean':'B0','gt_overlay':'B1','sham_overlay':'B2','uniform_overlay':'B3','gt_routing_bias':'B4',
     'random_color_overlay':'C1','bbox_outline_overlay':'C2','original_stage3':'B7','null_stage3':'B8'}


def row_diagnostics(p):
    p=p[0].double(); q=p/p.norm(dim=1,keepdim=True).clamp_min(1e-15)
    similarity=q@q.T; mask=~torch.eye(64,device=p.device,dtype=torch.bool)
    return {'mean_row_cosine':float(similarity[mask].mean()),
            'row_pair_cosine_gt_099':float((similarity[mask]>.99).double().mean()),
            'unique_argmax_positions':int(p.argmax(1).unique().numel()),
            'mean_row_entropy':float(-(p*p.clamp_min(1e-30).log()).sum(1).mean()),
            'effective_spatial_positions':float((-(p.mean(0)*p.mean(0).clamp_min(1e-30).log()).sum()).exp())}


def validate_geometry(runtime):
    from streamingtom.tasks.vision import VisionTask
    tower=runtime.model.get_vision_tower()
    conv=tower.vision_tower.vision_model.embeddings.patch_embedding
    assert tuple(conv.kernel_size)==(14,14) and tuple(conv.stride)==(14,14) and conv.padding in ('valid',(0,0))
    grid=torch.arange(729,dtype=torch.float64).reshape(1,729,1)
    actual=VisionTask(None,None)._apply_2d_pool(grid)
    expected=torch.nn.functional.interpolate(grid.view(1,27,27,1).permute(0,3,1,2),size=(14,14),
                                               mode='bilinear',align_corners=False).flatten()
    assert torch.equal(actual.flatten(),expected)
    raw=np.zeros((1,80,160,3),dtype=np.uint8);raw[:,0,:,0]=255;raw[:,-1,:,2]=255
    pixels=runtime.processor.preprocess(raw,return_tensors='pt')['pixel_values']
    assert pixels.shape==(1,3,384,384)
    assert not any(p.requires_grad for module in (runtime.model,runtime.vision_router,runtime.audio_router,runtime.audio_encoder) for p in module.parameters())
    return {'actual_patch_kernel':list(conv.kernel_size),'actual_patch_stride':list(conv.stride),
            'actual_patch_padding':conv.padding,'pool_ramp_exact':True,'processor_shape':list(pixels.shape),
            'all_model_parameters_frozen':True,'layout':LAYOUT}


def show_example(path,row,images,maps,routings,occupancy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axs=plt.subplots(2,5,figsize=(16,7),constrained_layout=True)
    h,w=images['rgb_clean'].shape[:2]
    for ax,name in zip(axs[0],['rgb_clean','gt_overlay','sham_overlay','uniform_overlay','gt_routing_bias']):
        ax.imshow(images.get(name,images['rgb_clean']));ax.set_title(IDS[name]+' '+name,fontsize=9)
        for x0,y0,x1,y1 in row['bbox']:
            ax.add_patch(Rectangle((x0*w,y0*h),(x1-x0)*w,(y1-y0)*h,fill=False,color='lime',linewidth=1))
        ax.axis('off')
    for ax,name in zip(axs[1],['GT map','Sham map','Token ROI','B0 routing','B4 routing']):
        value={'GT map':maps['gt'],'Sham map':maps['sham'],'Token ROI':occupancy.reshape(14,14),
               'B0 routing':routings['rgb_clean'][0].mean(0).reshape(14,14),
               'B4 routing':routings['gt_routing_bias'][0].mean(0).reshape(14,14)}[name]
        ax.imshow(np.asarray(value),cmap='magma',interpolation='nearest');ax.set_title(name);ax.axis('off')
    fig.suptitle(f"{row['clip_id']} | Oracle diagnostic | PTS {row['frame_pts']:.5f} | {row['split']}",fontsize=10)
    fig.savefig(path,dpi=120);plt.close(fig)


def lock_code(output,config,rows):
    # Reporting is downstream of predictions; its independent provenance is
    # recorded by report.py without preventing inference-only resume.
    sources=[p for p in Path(__file__).parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='report.py']
    sources += [REPO/'tests/test_grounded_image_pilot.py']
    sources += [REPO/p for p in ('utility_gate/runtime.py','utility_gate/strict_runtime.py',
                'utility_gate/grounding/spatial_mapping.py','relevance_gate/media.py',
                'streamingtom/audio_bottleneck/adbt.py','streamingtom/train_adbt_videomme.py')]
    hashes={str(p.relative_to(REPO)):sha256(p) for p in sources if p.exists()}
    lockpath=output/'preprediction_lock.json'
    value={'source_sha256':hashes,'config_fingerprint':fingerprint(config),
           'manifest_sha256':sha256(output/'manifest.jsonl'),'sample_ids':[r['clip_id'] for r in rows],
           'created_utc':datetime.now(timezone.utc).isoformat(),
           'go1_rule':'Either B1 proxy accuracy point estimates beat B0 and Sham AND both paired proxy gold-NLL CI upper bounds <0; OR B1 ROI paired lower bounds >0 against both AND teacher-KL paired upper bounds <0 against both. All conditions exploratory, no multiple-test correction.',
           'go2_rule':'requires GT-free prediction and genuinely new source-disjoint confirmation; absent assets means not established',
           'parameters_selected_on_outcomes':False}
    replace_lock=not lockpath.exists()
    if lockpath.exists():
        old=json.loads(lockpath.read_text())
        changed=[k for k in ('source_sha256','config_fingerprint','manifest_sha256','sample_ids') if old[k]!=value[k]]
        if changed:
            if (output/'results.jsonl').exists() and (output/'results.jsonl').stat().st_size:
                raise ValueError('immutable inference code/config/manifest lock changed after predictions; cannot resume')
            if changed!=['source_sha256']:
                raise ValueError('data/config cannot change after M0')
            # Engineering fixes before the very first prediction are recorded,
            # with the failed lock and its source snapshot kept intact.
            tag=sha256(lockpath)[:12]
            shutil.copyfile(lockpath,output/f'preprediction_lock_failed_{tag}.json')
            if (output/'source_snapshot').exists():
                shutil.copytree(output/'source_snapshot',output/f'source_snapshot_failed_{tag}',dirs_exist_ok=True)
            value['preprediction_engineering_fix_replaces_lock_sha256']=sha256(lockpath)
            replace_lock=True
    if replace_lock:
        write_json(lockpath,value)
        for p in sources:
            if p.exists():
                target=output/'source_snapshot'/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)


@torch.inference_mode()
def run(args):
    output=Path(args.output);config=json.loads(Path(args.config).read_text())
    locked=json.loads((output/'configs_locked/config.json').read_text())
    if config!=locked:raise ValueError('config differs from M0 lock')
    rows=read_jsonl(args.sources_manifest or output/'manifest.jsonl')
    original=read_jsonl(output/'manifest.jsonl')
    if rows!=original:raise ValueError('only the M0-locked source manifest may be evaluated')
    if args.subset=='pilot':rows=[r for r in rows if r['pilot']]
    if args.max_sources:rows=rows[:args.max_sources]
    modes=args.modes.split(',') if args.modes else config['modes']+config['supplemental_modes']
    if not set(config['modes']).issubset(modes):raise ValueError('core B0-B4 must all be present')
    if args.dry_run:
        print({'sources':len(rows),'pilot':sum(r['pilot'] for r in rows),'modes':modes,
               'predicted':'not_run: no_predicted_map_asset','model_load':False},flush=True);return
    lock_code(output,config,original)
    previous=read_jsonl(output/'results.jsonl') if (output/'results.jsonl').exists() else []
    if previous and not args.resume:raise FileExistsError('existing predictions require --resume')
    done={r['clip_id'] for r in previous};seed_all(config['seed'])
    status=json.loads((output/'STATUS.json').read_text());status[args.subset]='running';write_json(output/'STATUS.json',status)
    print({'loading_frozen_VLM':True,'requested_device':args.device,'sources':len(rows)},flush=True)
    runtime=StrictStreamingRuntime()
    if str(runtime.device)!=args.device:raise ValueError(f'model loaded on {runtime.device}, requested {args.device}')
    write_json(output/'checks/runtime_metadata.json',runtime.metadata)
    geometry=validate_geometry(runtime)
    disabled=runtime.regression_fixture(REPO/'artifacts/relevance_gate/m0-gate-disabled-final-20261022')
    vf=torch.load(REPO/'artifacts/relevance_gate/m0-vision-20261022/fixture.pt',map_location='cpu',weights_only=True)
    v=torch.cat([c['visual'] for c in vf['chunks']]).to(runtime.device).float()
    z,p=runtime.vision_router(v,None,vf['timestamps'].to(runtime.device))
    assert torch.equal(z.cpu(),torch.cat([c['latents'] for c in vf['chunks']]))
    gen=runtime.generate(vf['prompt_ids'],z)
    assert gen['generated_ids']==vf['generated_ids'].reshape(-1).tolist()
    write_json(output/'checks/real_GPU_baselines.json',{'geometry':geometry,'extension_disabled_Stage3':disabled,
         'Stage1_P_Z_and_generated_ids_exact':True,'Stage1_generated_ids':gen['generated_ids'],
         'extension_disabled_is_Stage3_not_Stage1':True})
    silent=runtime.audio_encoder.encode_windows(torch.zeros(1,32000),runtime.device).to(runtime.device).float()
    for index,row in enumerate(rows):
        if row['clip_id'] in done:continue
        begin=time.perf_counter();clip=row['clip_id'];rgb=np.asarray(Image.open(row['frame_path']).convert('RGB')).copy()
        assert sha256(row['frame_path'])==row['frame_sha256']
        assert sha256(row['media_path'])==row['media_sha256']
        maps_start=time.perf_counter()
        gt=box_map(row['bbox'],row['height'],row['width'])
        sham,sham_info=sham_map(gt,row['source_id'],config['seed'])
        outline=outline_map(row['bbox'],row['height'],row['width'])
        uniform=np.full_like(gt,.5)
        maps={'gt':gt,'sham':sham,'uniform':uniform,'outline':outline}
        map_s=time.perf_counter()-maps_start
        np.savez_compressed(output/'maps'/f'{clip}.npz',**maps)
        assert np.array_equal(overlay(rgb,gt,0,config['color_rgb']),rgb)
        color=rng_for(row['source_id'],'random-color',config['seed']).integers(0,256,3).tolist()
        overlay_times={};images={'rgb_clean':rgb}
        for name,h,c in [('gt_overlay',gt,config['color_rgb']),('sham_overlay',sham,config['color_rgb']),
                         ('uniform_overlay',uniform,config['color_rgb']),('random_color_overlay',gt,color),
                         ('bbox_outline_overlay',outline,config['color_rgb'])]:
            start=time.perf_counter();images[name]=overlay(rgb,h,config['alpha'],c);overlay_times[name]=time.perf_counter()-start
        transform=ImageTransform(row['width'],row['height']);occupancy=pooled_occupancy(row['bbox'],transform)
        prior=torch.tensor(occupancy,device=runtime.device,dtype=torch.float32)
        visual,visual_s=clocked(lambda:runtime.extractor.encode_visual(runtime.model,runtime.processor,rgb[None],1).float())
        times=torch.tensor([row['annotation_time_s']],device=runtime.device)
        (base_z,base_p),base_route_s=clocked(lambda:runtime.vision_router(visual,None,times))
        zero_z,zero_p=resample(runtime.vision_router,visual,prior,0.)
        assert torch.equal(zero_p,base_p) and torch.equal(zero_z,base_z) and torch.equal(base_z,base_p@visual)
        if index==0:
            from relevance_gate.media import frame_at_or_before
            raw,pts=frame_at_or_before(row['media_path'],row['annotation_time_s'])
            assert pts==row['frame_pts'] and np.array_equal(raw,rgb)
            original_v=runtime.extractor.encode_visual(runtime.model,runtime.processor,raw[None],1).float()
            assert torch.equal(original_v,visual)
        prompts={'teacher':runtime.prompt(teacher_task(config))}
        if row['proxy_category']['applicable']:prompts['category']=runtime.prompt(row['proxy_category'])
        prompt_hash={k:fingerprint(v.tolist()) for k,v in prompts.items()}
        teacher=runtime.generate(prompts['teacher'],runtime.full_vision(visual))
        branch_inputs={};route_maps={};results={}
        for mode in modes:
            if mode=='rgb_clean':z,p=base_z,base_p;vs=visual_s;rs=base_route_s;values=visual
            elif mode=='gt_routing_bias':
                (z,p),rs=clocked(lambda:resample(runtime.vision_router,visual,prior,config['lambda_bias']));vs=visual_s;values=visual
            else:
                values,vs=clocked(lambda:runtime.extractor.encode_visual(runtime.model,runtime.processor,images[mode][None],1).float())
                (z,p),rs=clocked(lambda:runtime.vision_router(values,None,times))
            assert z.shape==(1,64,3584) and torch.equal(z,p@values)
            branch_inputs[mode]=z;route_maps[mode]=p.cpu().numpy()
            result={'id':IDS[mode],'map_origin':'GT_BOX_RENDERED' if mode in ('gt_overlay','gt_routing_bias','random_color_overlay','bbox_outline_overlay') else 'GT_AREA_SHAM' if mode=='sham_overlay' else 'CONSTANT' if mode=='uniform_overlay' else 'NONE',
                    'oracle':mode in ('gt_overlay','gt_routing_bias','random_color_overlay','bbox_outline_overlay'),
                    'GT_area_used':mode=='sham_overlay','stage':1,'visual_tokens':64,
                    'spatial':routing_metrics(p[0].cpu().numpy(),occupancy,row['bbox'],transform),
                    'routing_diagnostics':row_diagnostics(p),'timing':{'map_creation_shared_s':map_s if mode not in ('rgb_clean','uniform_overlay') else 0.,
                        'overlay_s':overlay_times.get(mode,0.),'visual_encoder_s':vs,'resampler_s':rs},'prompt_fingerprints':prompt_hash.copy()}
            for task,ids in prompts.items():
                generation=runtime.generate(ids,z)
                result[task]=score_answer(generation,row['proxy_category']['answer']) if task=='category' else generation
            results[mode]=result
        # Real separate forwards for zero-bias degeneration, every source.
        zero=runtime.generate(prompts['teacher'],zero_z)
        assert zero['generated_ids']==results['rgb_clean']['teacher']['generated_ids']
        assert zero['option_logits']==results['rgb_clean']['teacher']['option_logits']
        # References have the same single annotated frame, independent frozen
        # Stage3 checkpoint and delta_t=0; not a same-training ablation.
        (windows,valid,quality,rms),ad=clocked(lambda:runtime.causal_windows(row['media_path'],[row['annotation_time_s']]))
        audio_s=0.
        if bool(valid[0]):
            audio,audio_s=clocked(lambda:runtime.audio_encoder.encode_windows(windows,runtime.device).to(runtime.device).float())
            for mode,audio_value in [('original_stage3',audio),('null_stage3',silent)]:
                (z,p),rs=clocked(lambda:runtime.audio_router(visual,audio_value,times))
                assert torch.equal(z,p@visual)
                route_maps[mode]=p.cpu().numpy()
                result={'id':IDS[mode],'stage':3,'map_origin':'ORIGINAL_AUDIO_ROUTER','oracle':False,'visual_tokens':64,
                        'spatial':routing_metrics(p[0].cpu().numpy(),occupancy,row['bbox'],transform),
                        'routing_diagnostics':row_diagnostics(p),'timing':{'map_creation_shared_s':0.,'overlay_s':0.,
                            'visual_encoder_s':visual_s,'resampler_s':rs,'audio_decode_s':ad,'audio_encoder_s':audio_s},
                        'prompt_fingerprints':prompt_hash.copy(),'causal_audio_window':[max(0,row['annotation_time_s']-2),row['annotation_time_s']],
                        'time_conditioning':'single annotated frame, delta=0; distinct from old three-frame spatial diagnostic'}
                for task,ids in prompts.items():
                    generation=runtime.generate(ids,z)
                    result[task]=score_answer(generation,row['proxy_category']['answer']) if task=='category' else generation
                results[mode]=result
        record={k:row[k] for k in ('clip_id','source_id','split','pilot','class','bbox','width','height','frame_pts','annotation_time_s','media_sha256','roi_area_fraction','proxy_category')}
        record.update(methods=results,teacher_full196=teacher,audio_valid=bool(valid[0]),audio_quality=quality[0],
            sham=sham_info,random_color_rgb=color,control_checks={'alpha_zero_pixels_exact':True,
            'lambda_zero_P_Z_exact':True,'lambda_zero_generated_ids_and_logits_exact':True,
            'prompt_identical_across_variants':True,'native_values_exact':True},
            gate_applied=False,branch_selected='always_on_spatial_cue',extension_enabled=True,
            end_to_end_all_variants_s=time.perf_counter()-begin)
        np.savez_compressed(output/'spatial'/f'{clip}_routing.npz',**{k:v[0] for k,v in route_maps.items()},occupancy=occupancy)
        if index<4:
            torch.save({'visual':visual.cpu(),'base_P':base_p.cpu(),'base_Z':base_z.cpu(),'zero_P':zero_p.cpu(),
                        'biased_P':torch.from_numpy(route_maps['gt_routing_bias']),'teacher_prompt_ids':prompts['teacher'].cpu()},
                       output/'checks'/f'{clip}_fixture.pt')
        # Every source gets original/GT/sham/overlay/routing comparisons. These
        # audit figures never enter the model or serve as QA labels.
        show_example(output/'examples'/f'{clip}.png',row,images,maps,route_maps,occupancy)
        append_jsonl(output/'results.jsonl',record)
        append_jsonl(output/'map_provenance.jsonl',{'source_id':row['source_id'],'clip_id':clip,'map_origin':'GT_BOX_RENDERED',
             'frame_time':row['annotation_time_s'],'frame_pts':row['frame_pts'],'oracle':True,'uses_test_annotation_as_input':True,
             'annotation_version':row['annotation_version'],'audio_window':None,'npz':str(output/'maps'/f'{clip}.npz'),
             'sha256':sha256(output/'maps'/f'{clip}.npz'),'sham':sham_info})
        done.add(clip)
        print({'completed_sources':len(done),'planned':len(rows),'last':clip,'audio_valid':bool(valid[0]),
               'source_elapsed_s':round(time.perf_counter()-begin,2)},flush=True)
        if sum(r['pilot'] and r['clip_id'] in done for r in original)==config['pilot_sources']:
            status['pilot']='completed';status['pilot_native_and_generation_checks']='passed';write_json(output/'STATUS.json',status)
    status[args.subset]='completed'
    if args.subset=='all':status['expanded']='completed'
    write_json(output/'STATUS.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--modes');p.add_argument('--subset',choices=['pilot','all'],default='all');p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--sources-manifest',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-sources',type=int,default=0);args=p.parse_args()
    try:run(args)
    except Exception as e:
        status_path=args.output/'STATUS.json'
        if status_path.exists():
            status=json.loads(status_path.read_text());status[args.subset]='blocked'
            status['last_execution_error']=str(e);write_json(status_path,status)
        raise


if __name__=='__main__':main()
