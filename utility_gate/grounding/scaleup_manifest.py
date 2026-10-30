"""Immutable VGG-SS audit, source exclusions and public-media scale-up registry."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from utility_gate.common import AUDIO_ROUTER, BEATS, DATA_ROOT, REPO, VISION_ROUTER, fingerprint, read_jsonl, sha256, write_json, write_jsonl
from utility_gate.grounded_image_pilot.heatmaps import valid_boxes
from utility_gate.grounding.spatial_mapping import LAYOUT, union_area

ROOT=REPO/'artifacts/utility_gate/vggss_feature_scaleup_20261028_v1'
DATA=DATA_ROOT/'vggss_feature_scaleup_20261028_v1'
CONFIG=Path(__file__).with_name('scaleup_config.json')
NEW_NAMES={'feature_diagnostics.py','feature_stats.py','feature_viz.py','scaleup_manifest.py','scaleup_eval.py','scaleup_data.py','scaleup_config.json','test_vggss_feature_scaleup.py'}


def now():return datetime.now(timezone.utc).isoformat()


def stage(root,name,state,**details):
    path=Path(root)/'STATUS.json';value=json.loads(path.read_text()) if path.exists() else {}
    value[name]={'status':state,'updated_utc':now(),**details};write_json(path,value)


def protect(root):
    paths=[p for p in (REPO/'artifacts').rglob('*') if p.is_file() and Path(root) not in p.parents]
    for directory in ('utility_gate','relevance_gate','streamingtom','tests','scripts','docs','reports'):
        paths += [p for p in (REPO/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix in ('.py','.json','.yaml','.md','.sh') and p.name not in NEW_NAMES]
    paths += list(REPO.glob('*.md'))+[AUDIO_ROUTER,VISION_ROUTER,BEATS]
    return {str(p):sha256(p) for p in sorted(set(paths))}


def identity(value):
    if not isinstance(value,str):return None
    text=value.removeprefix('youtube:')
    match=re.fullmatch(r'([A-Za-z0-9_-]{11})(?:_(\d{6}))?',text)
    return (match[1],int(match[2]) if match[2] else None) if match else None


def collect_evidence(value,path,seen,hashes):
    if isinstance(value,dict):
        local=[]
        for key in ('source_id','source_video_id','videoID','video_id','clip_id','canonical_clip_id','video_name','file'):
            item=identity(value.get(key))
            if item:seen[item[0]].add(str(path));local.append(item)
        digest=value.get('media_sha256')
        if digest:hashes[digest].update(x[0] for x in local)
        for key,v in value.items():
            if key in ('source_ids','seen_sources','excluded_sources','train','val','test') and isinstance(v,list):
                for s in v:
                    item=identity(s)
                    if item:seen[item[0]].add(str(path))
            collect_evidence(v,path,seen,hashes)
    elif isinstance(value,list):
        for v in value:collect_evidence(v,path,seen,hashes)


def history(root):
    seen=defaultdict(set);hashes=defaultdict(set);scanned=[]
    for base in (REPO/'artifacts',DATA_ROOT/'utility_gate',DATA_ROOT/'relevance_gate',DATA_ROOT/'dual_head_proxy_20261028_v1'):
        for p in base.rglob('*'):
            if not p.is_file() or Path(root) in p.parents or 'source_snapshot' in str(p) or p.suffix not in ('.json','.jsonl') or p.stat().st_size>40_000_000:continue
            try:
                value=read_jsonl(p) if p.suffix=='.jsonl' else json.loads(p.read_text())
                collect_evidence(value,p,seen,hashes);scanned.append(str(p))
            except (ValueError,UnicodeError):continue
    p=REPO/'results/adbt_videomme_split.json';collect_evidence(json.loads(p.read_text()),p,seen,hashes)
    return seen,hashes,scanned


def validate_pools(pools,seen_ids,seen_hashes):
    assigned={};sources=set()
    for pool,rows in pools.items():
        for r in rows:
            source=r['source_id'];digest=r['media_sha256'];alias=str(Path(r['media_path']).resolve())
            if pool.startswith('fresh') and (source in seen_ids or digest in seen_hashes):
                raise ValueError('previously inspected source/byte alias entered fresh pool')
            if source in sources:raise ValueError('duplicate original source across or within pools')
            sources.add(source)
            for key in [('source',source),('hash',digest),('path',alias)]:
                if key in assigned and assigned[key]!=pool:raise ValueError('identity alias crosses pools')
                assigned[key]=pool
    return {'sources':len(sources),'identity_alias_checks':len(assigned),'cross_pool_leakage':False}


def holdout_guard(root,manifest_path):
    path=Path(root)/'pretest_lock.json'
    if not path.exists():raise ValueError('fresh_holdout cannot run before pretest_lock')
    lock=json.loads(path.read_text())
    if sha256(manifest_path)!=lock['holdout_manifest_sha256']:raise ValueError('holdout manifest changed after lock')
    for p,digest in lock['inference_source_sha256'].items():
        if sha256(REPO/p)!=digest:raise ValueError('inference source changed after pretest lock')
    return lock


def audit(root=ROOT,config=CONFIG):
    root=Path(root);cfg=json.loads(Path(config).read_text())
    if (root/'audit/M0_LOCK.json').exists():raise FileExistsError('M0 already locked; no overwrite')
    for name in ('audit','manifests','features','per_source','figures','reports','logs','config','source_snapshot'):
        (root/name).mkdir(parents=True,exist_ok=True)
    DATA.mkdir(parents=True,exist_ok=True)
    stage(root,'M0','running')
    protected=protect(root);write_json(root/'audit/protected_assets.json',protected)
    old=REPO/cfg['legacy_root'];oldlock=json.loads((old/'checkpoint_locks.json').read_text())
    model_bad=[r['path'] for r in oldlock['frozen_backbone_cache_files'] if sha256(r['path'])!=r['sha256']]
    if model_bad:raise ValueError('backbone changed since pilot')
    seen,hashes,scanned=history(root)
    write_jsonl(root/'seen_sources.jsonl',[{'source_id':source,'evidence_paths':sorted(paths)} for source,paths in sorted(seen.items())])
    write_json(root/'audit/seen_media_hashes.json',{k:sorted(v) for k,v in hashes.items()})
    legacy=read_jsonl(old/'manifest.jsonl');write_jsonl(root/'manifests/legacy_dev.jsonl',[{**r,'split':'legacy_dev'} for r in legacy])
    annpath=DATA_ROOT/'vggss/annotations/raw/vggss.json';annotations=json.loads(annpath.read_text())
    media={p.stem:p for base in (DATA_ROOT/'avqa/videos',DATA_ROOT/'llp_avvp/videos',DATA_ROOT/'dual_head_proxy_20261028_v1') for p in base.rglob('*.mp4')}
    mirror=json.loads((DATA_ROOT/'avqa/acquisition_20261027/media_repository.json').read_text())
    remote={Path(r['rfilename']).stem:r['rfilename'] for r in mirror['siblings'] if r['rfilename'].endswith('.mp4')}
    candidates=[];clips=set();source_clips=defaultdict(list)
    for r in annotations:
        item=identity(r['file'])
        if item is None:raise ValueError('official invalid clip name')
        source,offset=item
        status='existing_verified' if r['file'] in media else 'downloadable' if r['file'] in remote else 'missing'
        try:bbox=valid_boxes(r['bbox']).tolist();area=union_area(bbox)
        except ValueError:bbox=r['bbox'];area=None;status='invalid_bbox'
        if r['file'] in clips:status='duplicate'
        clips.add(r['file']);source_clips[source].append(r['file'])
        path=media.get(r['file']);digest=sha256(path) if path else None
        candidates.append({'clip_id':r['file'],'source_id':source,'clip_offset_s':offset,'class':r['class'],'bbox':bbox,
             'bbox_area':area,'bbox_count':len(bbox),'media_status':status,'seen_source':source in seen,
             'media_path':str(path) if path else None,'media_sha256':digest,
             'known_byte_alias_seen':digest in hashes if digest else False,
             'remote_repository':'juyil/AVQA-videos' if r['file'] in remote else None,
             'remote_revision':mirror['sha'] if r['file'] in remote else None,'remote_path':remote.get(r['file']),
             'official_annotation_sha256':sha256(annpath),'annotation_time_s':5.,
             'official_split':'VGGSound_test; official source CSV will be cross-checked during acquisition'})
    write_jsonl(root/'manifests/all_candidates.jsonl',candidates)
    with (root/'media_acquisition.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['clip_id','source_id','media_status','seen_source','media_path']);writer.writeheader()
        for r in candidates:writer.writerow({k:r[k] for k in writer.fieldnames})
    write_json(root/'tensor_layout.json',{**LAYOUT,'encoder_patch_features':[1,729,1152],
        'visual_projector_features':[1,729,3584],'visual_tokens_196':[1,196,3584],
        'router_queries':[1,64,256],'router_logits_R':'FP32 cosine/tau, before softmax; B4 adds lambda/tau*centered_prior',
        'routing_matrix_P':[1,64,196],'compressed_tokens_Z':[1,64,3584],
        'original_real_model_validation':json.loads((old/'checks/real_GPU_baselines.json').read_text())})
    write_json(root/'config/run.json',cfg)
    write_json(root/'config/pilot_config.json',json.loads((old/'configs_locked/config.json').read_text()))
    write_json(root/'config/category_mapping.json',json.loads((old/'configs_locked/category_mapping.json').read_text()))
    lock={'created_utc':now(),'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'old_pilot':str(old),'legacy_manifest_sha256':sha256(root/'manifests/legacy_dev.jsonl'),
          'official_annotations_sha256':sha256(annpath),'config_sha256':sha256(root/'config/run.json'),
          'protected_files':len(protected),'backbone_files':oldlock['frozen_backbone_cache_files'],
          'models':oldlock['models'],'historical_sources':len(seen),'history_files_scanned':len(scanned),
          'legacy_sources':len(legacy),'legacy_audio_valid':sum(r['historical_audio_valid'] is True for r in legacy),
          'legacy_proxy_n':sum(r['proxy_category']['applicable'] for r in legacy),
          'candidates':len(candidates),'candidate_sources':len(source_clips),
          'media_states':dict(Counter(r['media_status'] for r in candidates)),
          'existing_fresh':sum(r['media_status']=='existing_verified' and not r['seen_source'] for r in candidates),
          'AVQA_remote_fresh':sum(r['remote_path'] is not None and not r['seen_source'] for r in candidates),
          'existing_predicted_assets':{'legacy_Stage3_P':109,'kind':'LEGACY_ROUTING_PRED; not independent localizer',
              'trace':'prior raw routing NPZ + results causal window metadata + frozen Stage3 SHA',
              'independent_DenseAV_EZVSL':0,'fresh_source_predicted_maps':0},
          'authority':{'vggss':'https://www.robots.ox.ac.uk/~vgg/research/lvs/',
              'vggsound':'https://www.robots.ox.ac.uk/~vgg/data/vggsound/',
              'observed':'official VGGSound website says media download links no longer available; public mirrors must be separate provenance'}}
    write_json(root/'audit/M0_LOCK.json',lock)
    (root/'reports/M0_audit.md').write_text(f'''# M0 — 特征层诊断与扩展资产审计\n\n保护{len(protected)}个历史文件，含前一轮pilot所有B0–B8工件、源码、Stage1/Stage3/BEATs；17项backbone cache逐SHA核对一致。旧125来源（音频109、类别代理108）单列legacy_dev，绝不重新称独立确认。\n\n全量官方{len(candidates)}条/ {len(source_clips)}个源，标注SHA={sha256(annpath)}。当前状态{lock['media_states']}；已检视排除源{len(seen)}个，旧125/128、Gate2/AVQA/LLP/Dual-Head均纳入；目前AVQA匿名媒体镜像有{lock['AVQA_remote_fresh']}个未检视候选，实际可下载及质量还需M2检验。all_candidates.jsonl逐条保留bbox schema、clip offset、媒体状态/hash、来源别名；全量框的M0合法性/分层索引不用于holdout参数选择。\n\n路径：同一RGB处理器warp384→SigLIP实际返回729×1152→原projector729×3584→原bilinear27→14（align_corners=False，row-major）→196×3584→Stage1的64×256静态queries→FP32 cosine/tau logits R→P64×196→Z=P@V64×3584→纯视觉VLM。patch支持仅valid14覆盖378×378，非Transformer全局感受野。Hook只读输出，不返回替换tensor，真实GPU将在M1验证。B4始终使用clean RGB V，仅R/P改变。\n\n复用原alpha=.20、固定颜色[255,64,32]、lambda=.05及完整map/sham/color实现。新模块独立捕获各层，不改旧forward。原类别映射/问题模板锁定；类别代理与Teacher保真始终分栏。\n\n注意：上一轮B7已生成109个可追溯数值Stage3 P，它是原Router预测路由，不能继续说所有数值预测图为0；不是独立声源localizer。本轮主协议仍为B0–B4/C1/C2，B5/B6不引入新localizer，正式新来源没有预测图资产。Stage3 real/null保留独立参照。\n\n官方依据：[VGG-SS](https://www.robots.ox.ac.uk/~vgg/research/lvs/)说明bbox字段；[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)当前不再提供视频下载链接。第三方匿名镜像仅在来源、修订、hash及时间裁剪可核验后使用；不绕过鉴权。\n''')
    stage(root,'M0','done',sources=len(candidates),legacy=125,protected_files=len(protected))
    for name in ('M1','M2','M3','M4','M5'):stage(root,name,'pending')
    print(lock,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--split',default='audit');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:print({'operation':'M0 audit','output':str(a.output_dir),'read_only_history':True});return
    audit(a.output_dir,a.config)


if __name__=='__main__':main()
