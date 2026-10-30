"""Pinned public media acquisition and source-disjoint development/holdout pools."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
from pathlib import Path
import threading
import time

import av
import numpy as np
from PIL import Image
import requests

from relevance_gate.media import causal_window,decode_audio,frame_at_or_before
from utility_gate.common import REPO,append_jsonl,fingerprint,read_jsonl,sha256,write_json,write_jsonl
from utility_gate.grounded_image_pilot.proxy_eval import category_task
from .scaleup_manifest import CONFIG,DATA,ROOT,identity,now,stage,validate_pools
from .spatial_mapping import ImageTransform,pooled_occupancy

MIRROR='11hu83/vggsound'
REVISION='dc7815aa65132c48f070686e0004180f03b5abcf'


def download(url,path,limit=64*1024*1024):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return {'sha256':sha256(path),'bytes':path.stat().st_size,'reused':True}
    temporary=path.with_suffix(path.suffix+'.partial')
    response=requests.get(url,stream=True,timeout=(15,45))
    if response.status_code!=200:
        response.close();raise ValueError('HTTP_'+str(response.status_code))
    size=0
    with temporary.open('wb') as f:
        for chunk in response.iter_content(1024*512):
            size+=len(chunk)
            if size>limit:response.close();raise ValueError('media_size_limit')
            f.write(chunk)
    response.close();temporary.replace(path)
    info={'sha256':sha256(path),'bytes':size,'url':url,'acquired_utc':now(),'access':'anonymous public resolve endpoint',
          'repository':MIRROR,'revision':REVISION,'license_claim':'mirror card cc-by-4.0; original video copyright retained'}
    write_json(path.with_suffix('.source.json'),info)
    return info


def family(label):
    text=label.lower()
    if any(w in text for w in ['bird','cuckoo','canary','pheasant','warbler','owl','magpie','thrush','swallow','francolin','crow','woodpecker','chicken','turkey','penguin']):return 'birds'
    if any(w in text for w in ['cow','cattle','bull','dog','cat ','cat,','cat me','cat hi','lion','goat','snake','whale','donkey','otter','coyote','fox','chipmunk','crocodile','gibbon','horse','pig','sheep','elephant','frog','duck','goose','monkey']):return 'other_animals'
    if any(w in text for w in ['playing','singing','music','guitar','piano','drum','violin','instrument']):return 'music'
    if any(w in text for w in ['car ','train','airplane','helicopter','motor','vehicle','truck','bus','tractor','boat','snowmobile','subway']):return 'vehicles'
    if any(w in text for w in ['people','speech','human','baby','child','laugh','crying','eating','cough','sneeze']):return 'human'
    return 'objects_environment'


def stratum(row):
    area=row['bbox_area'];size='small' if area<=.1 else 'medium' if area<=.5 else 'large'
    return '|'.join((family(row['class']),size,'multi' if row['bbox_count']>1 else 'single'))


def proportional_sample(rows,n,tag):
    groups=defaultdict(list)
    for row in rows:groups[row['stratum']].append(row)
    n=min(n,len(rows));quota={k:int(n*len(v)/len(rows)) for k,v in groups.items()} if rows else {}
    remaining=n-sum(quota.values())
    order=sorted(groups,key=lambda k:(-(n*len(groups[k])/len(rows)-quota[k]),fingerprint([1234,tag,k])))
    for key in order[:remaining]:quota[key]+=1
    chosen=[]
    for key,group in groups.items():
        chosen.extend(sorted(group,key=lambda r:fingerprint([1234,tag,r['source_id']]))[:quota[key]])
    return sorted(chosen,key=lambda r:fingerprint([1234,tag,'order',r['source_id']]))


def plan(root=ROOT):
    root=Path(root);cfg=json.loads((root/'config/run.json').read_text())
    if (root/'manifests/acquisition_plan.jsonl').exists():raise FileExistsError('media plan already exists')
    source=DATA/'sources'/MIRROR.replace('/','__')
    metadata=requests.get('https://huggingface.co/api/datasets/'+MIRROR,timeout=35);metadata.raise_for_status();meta=metadata.json()
    if meta['gated'] or meta.get('private') or meta.get('disabled') or meta['sha']!=REVISION:raise ValueError('mirror access/revision mismatch')
    write_json(source.with_suffix('.json'),meta)
    readme=requests.get(f'https://huggingface.co/datasets/{MIRROR}/raw/{REVISION}/README.md',timeout=25);readme.raise_for_status()
    (DATA/'sources/11hu83_README.md').write_text(readme.text)
    paths={r['rfilename'] for r in meta['siblings']};rows=read_jsonl(root/'manifests/all_candidates.jsonl')
    for row in rows:
        clip=row['clip_id'];row['stratum']=stratum(row) if row['bbox_area'] is not None else 'invalid'
        row['scaleup_video_remote']=f'video/{clip}/video.mp4' if f'video/{clip}/video.mp4' in paths else None
        row['scaleup_audio_remote']=f'audio/{clip}/audio.wav' if f'audio/{clip}/audio.wav' in paths else None
        if row['scaleup_video_remote'] and row['media_status'] not in ('invalid_bbox','existing_verified'):row['media_status']='downloadable'
    write_jsonl(root/'manifests/all_candidates_indexed.jsonl',rows)
    groups=defaultdict(list)
    for row in rows:
        if row['seen_source'] or row['media_status']=='invalid_bbox' or not row['scaleup_video_remote']:continue
        groups[row['source_id']].append(row)
    unique=[min(v,key=lambda r:fingerprint([1234,'clip-within-source',r['clip_id']])) for v in groups.values()]
    # Availability/QC buffer is fixed before any new prediction. The final
    # target remains200+500; QC failures are retained, never replaced by scores.
    selected=proportional_sample(unique,cfg['fresh_dev_target']+cfg['fresh_holdout_target']+100,'media_acquisition')
    write_jsonl(root/'manifests/acquisition_plan.jsonl',selected)
    write_json(root/'audit/acquisition_plan_lock.json',{'created_utc':now(),'plan_sha256':sha256(root/'manifests/acquisition_plan.jsonl'),
        'mirror':MIRROR,'revision':REVISION,'eligible_unique_sources':len(unique),'planned':len(selected),
        'eligible_strata':dict(Counter(r['stratum'] for r in unique)),'sampling':'proportional family x bbox area x multi-box; seeded source hashes',
        'transfer_budget_bytes':cfg['data_transfer_budget_bytes'],'successful_target':700,'QC_buffer':100,'outcomes_used':False})
    print({'mirror':MIRROR,'candidate_matches':sum(r['scaleup_video_remote'] is not None for r in rows),
           'eligible_unique_sources':len(unique),'planned_downloads':len(selected)},flush=True)


def mirror_validation(root):
    root=Path(root);legacy=read_jsonl(root/'manifests/legacy_dev.jsonl')
    picked=sorted(legacy,key=lambda r:fingerprint([1234,'mirror-validation',r['source_id']]))[:3]
    checks=[]
    for row in picked:
        clip=row['clip_id'];url=f'https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/video/{clip}/video.mp4'
        target=DATA/'validation'/f'{clip}.mp4';info=download(url,target)
        frame,pts=frame_at_or_before(target,5.);original=np.asarray(Image.open(row['frame_path']).convert('RGB'))
        a=np.asarray(Image.fromarray(frame).resize((96,96)),np.float64)/255
        b=np.asarray(Image.fromarray(original).resize((96,96)),np.float64)/255
        cosine=float((a*b).sum()/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b)));mae=float(np.abs(a-b).mean())
        item={'clip_id':clip,'mirror_sha256':info['sha256'],'old_sha256':row['media_sha256'],
              'same_media_bytes':info['sha256']==row['media_sha256'],'frame_pts':pts,'old_frame_pts':row['frame_pts'],
              'frame_MAE':mae,'frame_RGB_cosine':cosine,'mirror_hw':list(frame.shape[:2]),'old_hw':list(original.shape[:2])}
        if not item['same_media_bytes'] and (cosine<.98 or mae>.05):raise ValueError('mirror center-frame geometry/content unverified: '+str(item))
        checks.append(item)
    write_json(root/'audit/mirror_validation.json',{'repository':MIRROR,'revision':REVISION,'checks':checks,
        'scope':'three predetermined old clips; ID/start naming and original CSV still checked per new clip; no claim of independent publisher identity'})
    print({'mirror_validation':'passed','byte_exact':sum(r['same_media_bytes'] for r in checks)},flush=True)


def acquire(root=ROOT,resume=False):
    root=Path(root);cfg=json.loads((root/'config/run.json').read_text());stage(root,'M2','running')
    if json.loads((root/'STATUS.json').read_text())['M1']['status']!='done':raise ValueError('M1 must finish before scale-up')
    if not (root/'manifests/acquisition_plan.jsonl').exists():plan(root)
    if not (root/'audit/mirror_validation.json').exists():mirror_validation(root)
    # Cross-check official VGGSound source/offset/split, never use new outputs.
    csvpath=DATA/'sources/vggsound_official.csv'
    if not csvpath.exists():
        commit=requests.get('https://api.github.com/repos/hche11/VGGSound/commits/master',timeout=25);commit.raise_for_status()
        revision=commit.json()['sha'];url=f'https://raw.githubusercontent.com/hche11/VGGSound/{revision}/data/vggsound.csv'
        response=requests.get(url,timeout=45);response.raise_for_status();csvpath.write_bytes(response.content)
        write_json(DATA/'sources/vggsound_official_source.json',{'revision':revision,'url':url,'sha256':sha256(csvpath)})
    lookup={}
    for item in csv.reader(csvpath.open()):
        if len(item)>=4:
            try:lookup[(item[0],int(item[1]))]={'class':item[2],'split':item[3]}
            except ValueError:pass
    write_json(root/'audit/official_metadata_source.json',{**json.loads((DATA/'sources/vggsound_official_source.json').read_text()),
        'sha256':sha256(csvpath),'rows':len(lookup),'local_path':str(csvpath)})
    rows=read_jsonl(root/'manifests/acquisition_plan.jsonl');log=root/'manifests/media_attempts.jsonl'
    previous=read_jsonl(log) if log.exists() else []
    if previous and not resume:raise FileExistsError('acquisition has prior attempts; use --resume')
    done={r['clip_id'] for r in previous};budget_lock=threading.Lock();transferred=[sum(r.get('downloaded_bytes',0) for r in previous)]
    frames=DATA/'frames';frames.mkdir(exist_ok=True)
    pilot=json.loads((root/'config/pilot_config.json').read_text());stop=threading.Event()
    def fetch(row):
        if stop.is_set():return {**row,'media_status':'failed_transient','reason':'public mirror rate-limited; pending requests stopped'}
        clip=row['clip_id'];source=row['source_id'];official=lookup.get((source,row['clip_offset_s']))
        if not official:raise ValueError('official VGGSound source+offset missing')
        if official['split']!='test':raise ValueError('VGGSS source is not official VGGSound test')
        target=DATA/'videos'/f'{clip}.mp4';url=f'https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/{row["scaleup_video_remote"]}'
        with budget_lock:
            if transferred[0]>cfg['data_transfer_budget_bytes']-cfg['maximum_media_bytes']:raise ValueError('predeclared transfer budget exhausted')
        info=download(url,target,cfg['maximum_media_bytes']);size=0 if info.get('reused') else info['bytes']
        with budget_lock:transferred[0]+=size
        with av.open(str(target)) as container:
            stream=container.streams.video[0];origin=(container.start_time or 0)/av.time_base
            duration=container.duration/av.time_base if container.duration is not None else float(stream.duration*stream.time_base)
            has_audio=bool(container.streams.audio)
        if not cfg['duration_range_s'][0]<=duration<=cfg['duration_range_s'][1]:
            return {**row,'media_status':'invalid_duration','reason':str(duration),'downloaded_bytes':size}
        frame,pts=frame_at_or_before(target,5.)
        if not 0<=5-pts<=cfg['frame_pts_tolerance_s']:raise ValueError('center frame PTS mismatch')
        occupancy=pooled_occupancy(row['bbox'],ImageTransform(frame.shape[1],frame.shape[0]))
        if occupancy.sum()<=0:raise ValueError('no consumed visual bbox support')
        fpath=frames/f'{clip}.png';Image.fromarray(frame).save(fpath)
        audio_path=target
        if not has_audio and row['scaleup_audio_remote']:
            audio_path=DATA/'audio'/f'{clip}.wav'
            ai=download(f'https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/{row["scaleup_audio_remote"]}',audio_path,8*1024*1024)
            audio_bytes=0 if ai.get('reused') else ai['bytes'];size+=audio_bytes
            with budget_lock:transferred[0]+=audio_bytes
        try:
            waveform,audio_origin,details=decode_audio(audio_path,end_s=5.)
            window=causal_window(waveform,5.,origin_s=audio_origin)
            valid,quality=window.valid,window.reason
            if details.get('pts_gaps'):valid,quality=False,'audio_pts_gap'
        except ValueError as error:
            valid,quality=False,'audio_pts_overlap' if 'overlapping' in str(error) else 'decode_error:'+str(error)
        return {**row,'media_status':'existing_verified','media_path':str(target),'media_sha256':info['sha256'],
            'frame_path':str(fpath),'frame_sha256':sha256(fpath),'frame_pts':pts,'annotation_time_s':5.,
            'width':frame.shape[1],'height':frame.shape[0],'duration_s':duration,'container_origin_s':origin,
            'has_audio_track':has_audio,'audio_path':str(audio_path),'audio_sha256':sha256(audio_path),
            'downloaded_bytes':size,'media_repository':MIRROR,'media_revision':REVISION,
            'original_csv_label':official['class'],'original_csv_split':official['split'],
            'proxy_category':category_task(row['class'],source,pilot),'roi_area_fraction':float(occupancy.mean()),
            'historical_audio_valid':bool(valid),'historical_audio_quality':quality,
            'quality_field_note':'compatibility field names: newly measured acquisition QC, not prior-model outcome'}
    pending=[r for r in rows if r['clip_id'] not in done]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures={executor.submit(fetch,row):row for row in pending}
        for i,future in enumerate(as_completed(futures),1):
            row=futures[future]
            try:result=future.result()
            except Exception as e:
                reason=str(e);state='restricted' if 'HTTP_403' in reason or 'HTTP_401' in reason else 'failed_transient'
                if 'HTTP_429' in reason:stop.set()
                result={**row,'media_status':state,'reason':reason}
            append_jsonl(log,result)
            if i%40==0 or i==len(pending):print({'media_attempts':len(done)+i,'planned':len(rows),'downloaded_bytes':transferred[0]},flush=True)
    allrows=read_jsonl(log);valid=[r for r in allrows if r['media_status']=='existing_verified']
    write_json(root/'audit/acquisition_summary.json',{'planned':len(rows),'attempts':len(allrows),'valid':len(valid),
        'states':dict(Counter(r['media_status'] for r in allrows)),'downloaded_bytes':transferred[0],
        'no_model_outcomes_used':True,'media_repository':MIRROR,'revision':REVISION})
    print({'acquired_valid':len(valid),'states':dict(Counter(r['media_status'] for r in allrows))},flush=True)


def split(root=ROOT):
    root=Path(root);cfg=json.loads((root/'config/run.json').read_text())
    if (root/'manifests/fresh_holdout_LOCKED.jsonl').exists():raise FileExistsError('pool manifests already locked')
    rows=[r for r in read_jsonl(root/'manifests/media_attempts.jsonl') if r['media_status']=='existing_verified']
    seen={r['source_id'] for r in read_jsonl(root/'seen_sources.jsonl')};hashes=json.loads((root/'audit/seen_media_hashes.json').read_text())
    filtered=[];used_hashes=set();fail=[]
    for row in sorted(rows,key=lambda r:fingerprint([1234,'dedup-byte',r['source_id']])):
        if row['source_id'] in seen or row['media_sha256'] in hashes or row['media_sha256'] in used_hashes:
            fail.append({'clip_id':row['clip_id'],'reason':'source_or_content_alias_to_seen_or_duplicate'});continue
        used_hashes.add(row['media_sha256']);filtered.append(row)
    # Keep the declared2:5 ratio if data are scarce, rather than spending the
    # entire small cohort on development and losing all independent evidence.
    total=min(len(filtered),cfg['fresh_dev_target']+cfg['fresh_holdout_target'])
    dev_n=min(cfg['fresh_dev_target'],round(total*2/7))
    dev=proportional_sample(filtered,dev_n,'fresh_dev');ids={r['source_id'] for r in dev}
    hold=proportional_sample([r for r in filtered if r['source_id'] not in ids],total-len(dev),'fresh_holdout')
    population=json.loads((root/'audit/acquisition_plan_lock.json').read_text())['eligible_strata']
    for name,pool in [('fresh_dev',dev),('fresh_holdout',hold)]:
        counts=Counter(r['stratum'] for r in pool)
        for row in pool:row['split']=name;row['poststratification_weight']=population[row['stratum']]/counts[row['stratum']]
    check=validate_pools({'fresh_dev':dev,'fresh_holdout':hold},seen,set(hashes))
    write_jsonl(root/'manifests/fresh_dev.jsonl',dev);write_jsonl(root/'manifests/fresh_holdout_LOCKED.jsonl',hold)
    write_jsonl(root/'manifests/dedup_rejections.jsonl',fail)
    write_json(root/'audit/pool_lock.json',{'created_utc':now(),'fresh_dev_sha256':sha256(root/'manifests/fresh_dev.jsonl'),
        'fresh_holdout_sha256':sha256(root/'manifests/fresh_holdout_LOCKED.jsonl'),'fresh_dev_n':len(dev),'fresh_holdout_n':len(hold),
        'source_alias_audit':check,'eligible_population_strata':population,'heldout_predictions_read':0,
        'weighted_estimand':'secondary standardization to mirror-eligible metadata strata, conditional on QC; not full-dataset unbiasedness'})
    (root/'reports/M2_data_scale.md').write_text(f'''# M2 — VGG-SS三池与实际规模\n\nlegacy_dev=125（以前已检视），fresh_dev={len(dev)}，fresh_holdout={len(hold)}。从公开镜像{MIRROR}@{REVISION}按源去重、类别大类×bbox面积×多框比例分层预选800个候选，按固定时长[9.5,10.5]、中心PTS误差≤.1s及坐标/媒体规则做QC，不读取任何新模型结果。\n\n官方全量5158条不等于媒体有效数；候选、实际下载、有效、去重与最终分池台账见manifests/all_candidates*.jsonl、media_attempts.jsonl、audit/acquisition_summary.json。byte/source/realpath去重已过：{check}。额外拒绝{len(fail)}个重复或已检视alias，逐条保留。源视频多clip只选一个。\n\n镜像README声明cc-by-4.0，仅第三方托管，未冒充官方视频分发。每文件URL/revision/hash/source sidecar保留在dataset专用新目录，旧媒体不动。三条预定旧clip的重叠对照见mirror_validation.json；逐clip与官方VGGSound CSV的YouTubeID/start/test split核对，CSV本轮已下载并锁SHA。bbox使用官方VGGSS原标签，仅M0合法性/分层会读全量框，holdout框不参与任何alpha/λ选择。\n\n如数量不足，仍保留原2:5开发/确认比例；没有根据确认表现追加来源。配置参数原样保留，主确认仅一次。分布与post-stratification辅助估计在最终统计中列出，不能宣称新旧同分布或全量VGGSS无偏。原模型预训练语料的逐视频重叠未能排查，本轮独立性仅针对项目训练/选模/既往结果检查。\n''')
    stage(root,'M2','done',fresh_dev=len(dev),fresh_holdout=len(hold),legacy_dev=125)
    print({'fresh_dev':len(dev),'fresh_holdout':len(hold),'isolation':check},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['plan','validate','acquire','split']);p.add_argument('--output-dir',type=Path,default=ROOT)
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--split',default='media');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.dry_run:print({'operation':a.operation,'network':a.operation!='split','no_model_predictions':True});return
    if a.operation=='plan':plan(a.output_dir)
    elif a.operation=='validate':mirror_validation(a.output_dir)
    elif a.operation=='acquire':acquire(a.output_dir,a.resume)
    else:split(a.output_dir)


if __name__=='__main__':main()
