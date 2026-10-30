"""M0 read-only asset audit, frozen selection and exact historical protection."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess

import av
import numpy as np
from PIL import Image

from relevance_gate.media import frame_at_or_before
from utility_gate.common import AUDIO_ROUTER, BEATS, DATA_ROOT, REPO, VISION_ROUTER, fingerprint, read_jsonl, sha256, write_json, write_jsonl
from utility_gate.grounding.spatial_mapping import ImageTransform, LAYOUT, pooled_occupancy
from .heatmaps import valid_boxes
from .proxy_eval import CLASS_MAP, category_task

ROOT = REPO/'artifacts/utility_gate/vggss_grounded_image_pilot_20261028'
OLD = REPO/'artifacts/utility_gate/gate2_grounding_20261027'
CONFIG = Path(__file__).parent/'configs/pilot.yaml'


def old_protection():
    paths = [p for p in (REPO/'artifacts').rglob('*') if p.is_file() and ROOT not in p.parents]
    for folder in ('streamingtom', 'relevance_gate', 'utility_gate', 'tests', 'scripts', 'docs', 'reports'):
        paths += [p for p in (REPO/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix in ('.py','.md','.sh','.yaml','.json') and 'grounded_image_pilot' not in p.parts]
    paths += [p for p in REPO.glob('*.md') if p.is_file()]
    paths += [AUDIO_ROUTER, VISION_ROUTER, BEATS]
    return {str(p): sha256(p) for p in sorted(set(paths))}


def collect_ids(value, target):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ('source_video_id','source_id','videoID','video_id','clip_id','file','canonical_clip_id','video_name') and isinstance(v,str):
                match = re.match(r'^([A-Za-z0-9_-]{11})(?:_|$)', v)
                if match: target.add(match[1])
            if k in ('train','val','test') and isinstance(v,list):
                for x in v:
                    if isinstance(x,str) and re.match(r'^[A-Za-z0-9_-]{11}$',x): target.add(x)
            collect_ids(v,target)
    elif isinstance(value,list):
        for x in value: collect_ids(x,target)


def historical_ids(data_root):
    ids=set(); files=[]
    for base in (REPO/'artifacts', data_root/'utility_gate', data_root/'dual_head_proxy_20261028_v1'):
        files += [p for p in base.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl')
                  and ROOT not in p.parents and 'source_snapshot' not in p.parts and p.stat().st_size < 35_000_000]
    files += [REPO/'results/adbt_videomme_split.json']
    for p in files:
        try:
            collect_ids(read_jsonl(p) if p.suffix=='.jsonl' else json.loads(p.read_text()),ids)
        except (ValueError, UnicodeError): pass
    return ids,files


def audit(output=ROOT, data_root=DATA_ROOT):
    output=Path(output); data_root=Path(data_root)
    if (output/'checkpoint_locks.json').exists(): raise FileExistsError('completed M0 is immutable; choose a fresh output')
    output.mkdir(parents=True,exist_ok=True)
    for folder in ('frames','maps','spatial','proxy_qa','examples','logs','configs_locked','source_snapshot','checks'):
        (output/folder).mkdir(exist_ok=True)
    config=json.loads(CONFIG.read_text())
    protected=old_protection()
    write_json(output/'historical_locks.json',protected)
    raw=data_root/'vggss/annotations/raw/vggss.json'
    official=json.loads(raw.read_text()); ann={r['file']:r for r in official}
    old=read_jsonl(OLD/'datasets/vggss_manifest.jsonl')
    old_spatial={r['clip_id']:r for r in read_jsonl(OLD/'spatial/per_sample.jsonl')}
    used,usedfiles=historical_ids(data_root)
    media={p.stem:p for base in (data_root/'avqa/videos',data_root/'llp_avvp/videos',data_root/'dual_head_proxy_20261028_v1')
           for p in base.rglob('*.mp4')}
    local=[r for r in official if r['file'] in media]
    extra=[r for r in local if r['file'][:11] not in used]
    selected=[{**r,'split':'development_previously_inspected'} for r in old]
    selected += [{**r,'clip_id':r['file'],'source_video_id':r['file'][:11], 'media_path':str(media[r['file']]),
                  'clip_offset_s':int(r['file'][12:]),'split':'confirmation_new_source'} for r in extra]
    selected.sort(key=lambda r:(r['split']!='development_previously_inspected',fingerprint([1234,'grounded-image',r['clip_id']])))
    rows=[]; failures=[]; provenance=[]; ids=set(); hashes=set(); aliases=set()
    for i,row in enumerate(selected):
        clip=row['clip_id']; source=row['source_video_id']; path=Path(row['media_path'])
        try:
            boxes=valid_boxes(ann[clip]['bbox']).tolist()
            if source in ids or str(path.resolve()) in aliases: raise ValueError('duplicate_source_or_alias')
            if not path.is_file(): raise ValueError('missing_media')
            digest=sha256(path)
            if digest in hashes: raise ValueError('duplicate_media_bytes')
            with av.open(str(path)) as c:
                stream=c.streams.video[0]; origin=(c.start_time or 0)/av.time_base
                duration=c.duration/av.time_base if c.duration is not None else float(stream.duration*stream.time_base)
                has_audio=bool(c.streams.audio)
                video_start=float(stream.start_time*stream.time_base)-origin if stream.start_time is not None else None
            # Official annotation is the midpoint of the nominal ten-second
            # VGGSound trim, NOT midpoint of container duration incl. mux delay.
            if not 9.5 <= duration <= 10.5: raise ValueError(f'unverified_nominal10s_trim_duration:{duration}')
            target=5.0
            frame,pts=frame_at_or_before(path,target)
            if not 0 <= target-pts <= .1: raise ValueError('annotation_center_PTS_mismatch')
            occ=pooled_occupancy(boxes,ImageTransform(frame.shape[1],frame.shape[0]))
            if occ.sum()<=0: raise ValueError('no_consumed_visual_bbox_support')
            framepath=output/'frames'/f'{clip}.png'; Image.fromarray(frame).save(framepath)
            prior=old_spatial.get(clip,{})
            task=category_task(ann[clip]['class'],source,config)
            result={'clip_id':clip,'source_id':source,'source_video_id':source,'video_id':clip,'media_path':str(path),
                'media_sha256':digest,'media_realpath':str(path.resolve()),'clip_offset_s':int(clip[12:]),
                'annotation_source_time_s':int(clip[12:])+target,'annotation_time_s':target,'frame_pts':pts,
                'frame_path':str(framepath),'frame_sha256':sha256(framepath),'duration_s':duration,
                'container_origin_s':origin,'video_start_relative_s':video_start,
                'actual_duration_midpoint_s':duration/2,'annotation_rule':'official nominal10s clip midpoint; verified source+offset identity; causal nearest PTS',
                'bbox':boxes,'class':ann[clip]['class'],'split':row['split'],'has_audio':has_audio,
                'historical_audio_valid':prior.get('audio_available'),'historical_audio_quality':prior.get('audio_quality'),
                'heatmap_source':'GT_BOX_RENDERED','annotation_version':sha256(raw),
                'proxy_category':task,'pilot':len(rows)<config['pilot_sources'] and row['split'].startswith('development'),
                'width':frame.shape[1],'height':frame.shape[0], 'roi_area_fraction':float(occ.mean())}
            rows.append(result); ids.add(source); hashes.add(digest);aliases.add(str(path.resolve()))
            provenance.append({'source_id':source,'clip_id':clip,'map_origin':'GT_BOX_RENDERED','oracle':True,
                'uses_test_annotation_as_input':True,'bbox_provenance':str(raw),'annotation_sha256':sha256(raw),
                'frame_time':target,'frame_pts':pts,'audio_window':None,'map_kind':config['map_kind'],
                'map_status':'planned_deterministic_render_after_M0','source_model':None})
        except Exception as e: failures.append({'clip_id':clip,'reason':str(e)})
        if (i+1)%32==0: print({'M0_attempted':i+1,'valid':len(rows),'failed':len(failures)},flush=True)
    # Old spatial report contains scalars and colored figures, no recoverable
    # numerical P or independent localizer output. Never invert JPEG heatmaps.
    assets=[{'map_origin':'LEGACY_ROUTING_PRED','path':str(OLD/'spatial/per_sample.jsonl'),
             'status':'visualization_or_metrics_only','usable_continuous_predicted_map_count':0,
             'reason':'per-sample metrics and rendered figures; no saved P matrix/raw heatmap'},
            {'map_origin':'UNKNOWN','status':'not_imported','reason':'no verified DenseAV/EZVSL numeric asset in current VGGSS assets'}]
    provenance += assets
    write_jsonl(output/'manifest.jsonl',rows);write_jsonl(output/'selection_manifest.jsonl',rows)
    write_jsonl(output/'asset_provenance.jsonl',provenance);write_jsonl(output/'audit_failures.jsonl',failures)
    write_json(output/'configs_locked/config.json',config);write_json(output/'configs_locked/category_mapping.json',CLASS_MAP)
    write_json(output/'historical_source_ids.json',{'source_ids':sorted(used),'files_scanned':len(usedfiles)})
    versions={}
    for k in ('torch','transformers','numpy','av','Pillow','pytest'):
        try: versions[k]=importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError: versions[k]='not_installed_in_original_environment'
    model_files=[]
    cache=Path('/home/yxd/.cache/huggingface/hub')
    for name in ('models--lmms-lab--llava-onevision-qwen2-7b-ov','models--google--siglip-so400m-patch14-384'):
        directory=cache/name
        for p in directory.glob('snapshots/*/*'):
            if p.is_file(): model_files.append({'path':str(p),'realpath':str(p.resolve()),'bytes':p.stat().st_size,
                'sha256':sha256(p)})
    lock={'created_utc':datetime.now(timezone.utc).isoformat(),'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'models':{name:{'path':str(p),'sha256':sha256(p)} for name,p in [('stage1',VISION_ROUTER),('stage3',AUDIO_ROUTER),('BEATs',BEATS)]},
        'frozen_backbone_cache_files':model_files,'annotation_sha256':sha256(raw),'environment':versions,
        'config_sha256':sha256(output/'configs_locked/config.json'),'manifest_sha256':sha256(output/'manifest.jsonl'),
        'mapping_sha256':sha256(output/'configs_locked/category_mapping.json'),'historical_files_protected':len(protected),
        'historical_ids_scanned':len(used),'spatial_layout':LAYOUT,
        'preexisting_predicted_maps_usable':0,'B5_B6':'not_run: no_predicted_map_asset',
        'independent_frame_aligned_QA':'not_found; AVQA whole-clip QA is not center-frame truth',
        'proxy_tasks':['VGGSS-derived source-category proxy','FullVision teacher agreement (not correctness)'],
        'n_official':len(official),'n_local_media':len(local),'n_selected':len(rows),'n_failures':len(failures),
        'n_development':sum(r['split'].startswith('development') for r in rows),
        'n_confirmation':sum(r['split'].startswith('confirmation') for r in rows),
        'n_audio_valid_historical':sum(r['historical_audio_valid'] is True for r in rows),
        'n_boxes':sum(len(r['bbox']) for r in rows),'n_unique_classes':len({r['class'] for r in rows}),
        'n_proxy_applicable':sum(r['proxy_category']['applicable'] for r in rows),
        'n_reusable_preexisting_images':0,'n_images_extracted_this_audit':len(rows),
        'internet_verification':'web tool returned HTTP429; reused pinned official annotation/source and prior documented paper section4.1'}
    write_json(output/'checkpoint_locks.json',lock)
    write_json(output/'STATUS.json',{'M0':'completed','pilot':'not_run','expanded':'not_run','report':'not_run',
               'B5':'not_run: no_predicted_map_asset','B6':'not_run: no_predicted_map_asset'})
    message=f'''# M0 VGG-SS grounded image 资产审计\n\n当前 revision `{lock['revision']}`。冻结 Stage1/Stage3/BEATs 和本地 backbone 权重已锁定 SHA256；保护 {len(protected)} 个旧文件。\n\n官方 {len(official)} 条标注，本地精确 clip+offset 媒体 {len(local)} 条。选择 {len(rows)} 个去重来源：已检视 development {lock['n_development']}，新 confirmation {lock['n_confirmation']}；历史有效音频 {lock['n_audio_valid_historical']}。框 {lock['n_boxes']} 个，类别 {lock['n_unique_classes']} 种，类别代理适用 {lock['n_proxy_applicable']}。失败 {len(failures)} 条，逐条见 audit_failures.jsonl。\n\n旧 spatial 只保存指标和渲染图，没有原始 P/连续预测图；DenseAV/EZVSL 没有可核验既存数值图。可用预测图=0，B5/B6 not_run。旧图是 LEGACY_ROUTING_PRED 可视化，绝不逆推数值。GT 数值图将在新目录确定性生成，标为 GT_BOX_RENDERED / Oracle。\n\n旧可复用独立原帧=0；本轮重解码保存 {len(rows)} 个原帧。逐个核对实际时长、容器起点、clip offset 与 PTS；标注为 nominal10s clip 的5s中心，非盲用实际 mux 时长一半。视频时长不支持 nominal10s 或 PTS 超差则拒绝。框为官方归一化 xyxy、多框连续并集；384 warp→valid14 27×27→bilinear14×14 的既有精确位置支持继续复用，运行时再用真实模型验证。\n\n没有独立中心帧 QA 标注。仅报告明确标记的声源类别代理（仅看类别名制定白名单，未知排除），和 GT-free 通用视觉问题的 FullVision196 回答/分布保持。Teacher不是ground truth。旧128来源均已看过，不称独立新测试。无训练、无调参。B0–B4统一64视觉tokens，GT提示仅Oracle，Stage3原方法另列。\n\n配置 alpha=.20，固定无类别编码颜色；bias λ=.05 作用于原温度前余弦分数（相当于当前τ=.05下logit加1倍coverage）。Sham 共同整数平移保证面积、形状、数量、重叠和强度完全不变；不筛选低GT重叠位置。Uniform为常数0.5，不接触GT面积。\n\n首次32来源先导，完成工程检查后扩大到全部可用选定来源；配置/类别映射/样本先锁。新增来源只有通过全部历史来源排除才算confirmation。本轮若无新来源，GO-2不能成立。参考外网核查遇HTTP429；本机已固定官方source与原审计保留。\n'''
    (output/'M0_asset_audit.md').write_text(message)
    (output/'M1_manifest_and_heatmap_provenance.md').write_text(message+'\n完整逐条来源见 selection_manifest.jsonl / asset_provenance.jsonl。\n')
    print({k:lock[k] for k in ('n_local_media','n_selected','n_development','n_confirmation','n_audio_valid_historical','n_proxy_applicable')},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--data-root',type=Path,default=DATA_ROOT)
    args=p.parse_args();audit(args.output,args.data_root)


if __name__=='__main__':main()
