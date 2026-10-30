"""Metadata-first secondary audit and a bounded fixed Valor confirmation cohort."""
from __future__ import annotations
import concurrent.futures,json,re,time
from collections import defaultdict
from pathlib import Path
import av,requests,pandas as pd
from .common import ROOT,DATA,META,sha256,write_json,write_jsonl,read_jsonl,append_jsonl,journal
from .data import order

def get_media(job):
    path,url,extra=job;path.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not path.exists():
            temporary=path.with_suffix('.part')
            with requests.get(url,stream=True,timeout=(20,180)) as r:
                r.raise_for_status()
                with temporary.open('wb') as out:
                    for b in r.iter_content(1<<20):out.write(b)
            temporary.rename(path)
        with av.open(str(path)) as c:
            duration=c.duration/av.time_base;v=c.streams.video[0];a=c.streams.audio[0] if c.streams.audio else None
            if a is None:raise ValueError('missing audio')
            info={'duration_s':duration,'width':v.width,'height':v.height,'audio_sample_rate':a.rate}
        return {**extra,**info,'status':'ok','video_path':str(path),'media_sha256':sha256(path),'bytes':path.stat().st_size,'url':url}
    except Exception as e:return {**extra,'status':'failed','error':str(e),'url':url}

def main():
    seen={r['source_id'] for r in read_jsonl(ROOT/'seen_sources.jsonl')}
    # Verify the index publisher's pinned official annotation content exactly.
    labels=json.loads((META/'avqa_hard_official_val_qa.json').read_text());lookup={str(r['id']):r for r in labels}
    hard=pd.read_parquet(META/'avqa_hard.parquet').to_dict('records')
    repository=json.loads(Path('/nvme_data/pkt/streaming/dataset/avqa/acquisition_20261027/media_repository.json').read_text())
    media={Path(r['rfilename']).stem:r['rfilename'] for r in repository['siblings'] if r['rfilename'].endswith('.mp4')}
    fresh=[{**lookup[r['id']],'expected_duration_s':float(r['video_duration'])} for r in hard if lookup[r['id']]['video_name'][:11] not in seen]
    possible=[r for r in fresh if r['video_name'] in media]
    samples=sorted(possible,key=lambda r:order(r['video_name'],'hard_media_probe'))[:8]
    jobs=[]
    for r in samples:
        name=r['video_name'];url=f"https://huggingface.co/datasets/juyil/AVQA-videos/resolve/e9538b7abd3dcf018f85c9e431d51f542728c13b/{media[name]}"
        jobs.append((DATA/'external_media_audit'/f'{name}.mp4',url,{'dataset':'avqa_hard','video_name':name,'question_id':str(r['id']),'expected_duration_s':r['expected_duration_s']}))
    probes=list(concurrent.futures.ThreadPoolExecutor(4).map(get_media,jobs))
    for r in probes:
        r['duration_aligned']=r['status']=='ok' and abs(r['duration_s']-r['expected_duration_s'])<=max(.25,.02*r['expected_duration_s'])
    write_json(ROOT/'audit/avqa_hard_media_alignment.json',{'index_n':len(hard),'joined_n':len(hard),'fresh_qa_n':len(fresh),'fresh_sources_n':len({r['video_name'][:11] for r in fresh}),'mirror_exact_name_available_n':len(possible),'official_annotation_sha256':sha256(META/'avqa_hard_official_val_qa.json'),'probe_rule':'8 SHA-ordered fresh exact-name clips, compare published probe duration; do not score QA on incomplete mismatched excerpts','probes':probes})
    # Bounded secondary cohort selected entirely from native fields, before any
    # VLM output. One AV spatial/action/temporal and one Visual question/source.
    rows=read_jsonl(META/'valor_test.jsonl');repo=json.loads((META/'inesriahi__valor32k-avqa-v2.json').read_text())
    available={Path(x['rfilename']).stem[:11]:x['rfilename'] for x in repo['siblings'] if x['rfilename'].endswith('.mp4')}
    groups=defaultdict(list)
    for r in rows:
        if r['video_id'] not in seen and r['video_id'] in available:groups[r['video_id']].append(r)
    eligible=[]
    for source,qs in groups.items():
        a=[r for r in qs if r['modality']=='audio-visual' and r['category'] in ('location','relative-position','action','temporal')]
        v=[r for r in qs if r['modality']=='visual']
        if a and v:eligible.append((source,sorted(a,key=lambda r:order(str(r['id']),'valor_q'))[0],sorted(v,key=lambda r:order(str(r['id']),'valor_q'))[0]))
    chosen=sorted(eligible,key=lambda r:order(r[0],'valor_confirm'))[:128]
    revision='1b916bd15c19302f319ee6dd1206c7555bb19d76';jobs=[]
    for source,*qs in chosen:
        url=f'https://huggingface.co/datasets/inesriahi/valor32k-avqa-v2/resolve/{revision}/{available[source]}'
        jobs.append((DATA/'valor_videos'/Path(available[source]).name,url,{'source_video_id':source,'revision':revision}))
    records=[]
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for r in pool.map(get_media,jobs):records.append(r);print('valor',r['source_video_id'],r['status'],flush=True)
    write_jsonl(ROOT/'audit/media_valor_confirm.jsonl',records);lookup={r['source_video_id']:r for r in records if r['status']=='ok'}
    manifest=[]
    for source,*qs in chosen:
        if source not in lookup:continue
        m=lookup[source];stem=Path(m['video_path']).stem;offset=float(stem.split('_')[-2])
        for q in qs:
            question=q['question']+'\n'+'\n'.join(f'{c}. {s}' for c,s in zip('ABCD',q['options']))+'\nRespond with only the letter (A, B, C, or D) of the correct option.'
            manifest.append({'dataset':'valor32k_avqa_v2','role':'valor_confirm','source_id':source,'source_video_id':source,'youtube_id':source,'clip_offset_s':offset,'source_split':'test','question_id':str(q['id']),
                'question':question,'answer_text':'ABCD'[q['correct_answer_idx']],'modality_type':'Audio-Visual' if q['modality']=='audio-visual' else 'Visual',
                'reasoning_type':{'location':'Location','relative-position':'Location','temporal':'Temporal','action':'Action','count':'Counting','description':'Description'}[q['category']],
                'native_modality':q['modality'],'native_category':q['category'],'quality_rating':q['quality_rating'],'video_path':m['video_path'],'media_sha256':m['media_sha256'],'duration_s':m['duration_s'],
                'media_has_audio':True,'allowed_video_prefix_end':None,'score_protocol':'strict generated ABCD, official options/correct_answer_idx; full-vocabulary first-logit and ABCD-normalized Gold NLL','oracle':False,'synthetic':False})
    write_jsonl(ROOT/'manifests/valor_confirm_available.jsonl',manifest)
    write_json(ROOT/'audit/valor_selection.json',{'native_test_qa':len(rows),'eligible_known_unseen_sources_with_both_modalities':len(eligible),'requested_sources':len(chosen),'actual_sources':len(lookup),'questions':len(manifest),'selection':'seed1234 SHA order; 128 public test sources with one AV location/relative-position/action/temporal and one Visual question; select questions by hash; no model outputs used','caption_and_source_tags_used':False,'independence_scope':'known official YouTube ID exclusions; MUSIC original IDs unavailable, all model pretraining overlap unknown','license':'uploader research use only; no media redistribution','revision':revision})
    journal('secondary metadata and media audit',avqa_hard_probes=probes,valor_sources=len(lookup),valor_qa=len(manifest))

if __name__=='__main__':main()
