"""Official-template MUSIC adapter and versioned, targeted public-media acquisition."""
from __future__ import annotations
import argparse, ast, concurrent.futures, hashlib, json, re, subprocess, time
from collections import Counter, defaultdict
from pathlib import Path
import requests
from .common import ROOT, DATA, META, OFFICIAL, MIRROR, now, initialize, journal, sha256, write_json, write_jsonl, read_jsonl, append_jsonl

def order(value, tag='roles'):
    return hashlib.sha256(f'1234:{tag}:{value}'.encode()).hexdigest()

def expand_question(row):
    # Match the official dataloader: strip final question punctuation, replace
    # whitespace-delimited placeholder tokens in order using templ_values.
    words=row['question_content'].rstrip().split(' ')
    if not words or not words[-1]: raise ValueError('empty question')
    words[-1]=words[-1][:-1]
    values=ast.literal_eval(row['templ_values']) if isinstance(row['templ_values'],str) else row['templ_values']
    n=0
    for i,w in enumerate(words):
        if '<' in w:
            if n>=len(values): raise ValueError('template underflow')
            words[i]=values[n]; n+=1
    if n!=len(values) or '<' in ' '.join(words):raise ValueError('template mismatch')
    return ' '.join(words)+'?'

def normalize(text, numerals=True):
    value=re.sub(r'\s+',' ',str(text).lower().strip().replace('_',' ')).strip(' .!?\n\t')
    if numerals:
        value={'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine','10':'ten','11':'eleven'}.get(value,value)
    return value

def validate_roles(pools,old_hashes=()):
    identities={};questions=set()
    for role,rows in pools.items():
        for row in rows:
            key=(row['dataset'],row['question_id'])
            if key in questions:raise ValueError('duplicate question across roles')
            questions.add(key)
            aliases=[('clip',row['source_id']),('path',str(Path(row['video_path']).resolve()))]
            if row.get('youtube_id'):aliases.append(('original',row['youtube_id']))
            if row.get('media_sha256'):
                if row['media_sha256'] in old_hashes:raise ValueError('historical media bytes in fresh cohort')
                aliases.append(('bytes',row['media_sha256']))
            for alias in aliases:
                if alias in identities and identities[alias]!=role:raise ValueError('source or media alias crosses dev/confirm')
                identities[alias]=role
    return {'question_count':len(questions),'identity_aliases':len(identities),'cross_role_overlap':False,
        'original_youtube_scope':'only IDs actually supplied; missing original identities are not verified'}

def metadata():
    initialize()
    if (ROOT/'manifests/music_dev.jsonl').exists():raise FileExistsError('existing role selection is immutable; use a new run ID')
    data={s:json.loads((META/f'official_updated_{s}.json').read_text()) for s in ('train','val','test')}
    checks={s:{'official_updated_equal_mirror':data[s]==json.loads((META/f'music_mirror_{s}.json').read_text()),'official_sha256':sha256(META/f'official_updated_{s}.json')} for s in data}
    # Fail closed if a changed mirror is mistaken for authoritative labels.
    if not all(v['official_updated_equal_mirror'] for v in checks.values()): raise ValueError(checks)
    real={s:{r['video_id'] for r in rows if re.fullmatch(r'\d{8}',r['video_id'])} for s,rows in data.items()}
    test=sorted(real['test'],key=lambda x:order(x,'confirm'))[:500]
    dev=sorted(real['train']-set(test),key=lambda x:order(x,'dev'))[:200]
    roles={'music_dev_pilot':dev[:32],'music_dev':dev,'music_confirm':test}
    vocabulary=list(dict.fromkeys(r['anser'] for r in data['train']))
    write_json(ROOT/'config/answer_vocabulary.json',{'labels':vocabulary,'source':f'official data/json_update/avqa-train.json at {OFFICIAL}','classification_protocol':'official fixed vocabulary; open generated answer EM is reported separately','normalization':'case, whitespace, underscore->space, terminal punctuation; numeral whole answers 0..11 -> words; raw normalized EM also retained'})
    inventory=[]
    for role,ids in roles.items():
        split='test' if role=='music_confirm' else 'train'
        rows=[]
        for a in data[split]:
            if a['video_id'] not in ids:continue
            modality,reasoning=ast.literal_eval(a['type'])
            rows.append({'dataset':'music_avqa','source_video_id':a['video_id'],'source_id':'music:'+a['video_id'],
                'question_id':str(a['question_id']),'source_split':split,'role':role,
                'video_path':str(DATA/'music_videos'/f"{a['video_id']}.mp4"),'question':expand_question(a),
                'answer_text':a['anser'],'modality_type':modality,'reasoning_type':reasoning,
                'allowed_video_prefix_end':None,'score_protocol':'official_updated_labels_generated_answer_normalized_EM',
                'oracle':False,'synthetic':False,'youtube_id':None,'clip_offset_s':None,
                'source_identity_scope':'official real-video ID; original YouTube mapping unavailable; global source independence unverified'})
        rows.sort(key=lambda r:(order(r['source_video_id'],'dev' if split=='train' else 'confirm'),int(r['question_id'])))
        write_jsonl(ROOT/f'manifests/{role}.jsonl',rows)
        inventory.append({'role':role,'n_clips':len(ids),'n_questions':len(rows),'modality':dict(Counter(r['modality_type'] for r in rows))})
    stats={s:{'qa':len(rows),'video_ids':len({r['video_id'] for r in rows}),'real_ids':len(real[s]),'types':dict(Counter(r['type'] for r in rows))} for s,rows in data.items()}
    stats['train_test_video_overlap']=len(real['train']&real['test'])
    write_json(ROOT/'audit/music_metadata.json',{'official_commit':OFFICIAL,'media_mirror_revision':MIRROR,'checks':checks,'native_splits':stats,'selected':inventory,'source_split_rule':'reserve 500 real official-test video IDs by SHA256(seed=1234,confirm,ID); choose 200 official-train real IDs excluding all reserved IDs; no QA model outcomes used','original_youtube_mapping':'not supplied by inspected official annotations; no claim of full source independence','synthetic_policy':'all synthetic IDs excluded to avoid unknown donor leakage','vocabulary_n':len(vocabulary)})
    journal('fixed MUSIC roles before QA inference',inventory=inventory)
    print(inventory,flush=True)

def acquire_one(identity):
    path=DATA/'music_videos'/f'{identity}.mp4';url=f'https://huggingface.co/datasets/UnFaZeD07/Music-AVQA/resolve/{MIRROR}/videos/{identity}.mp4'
    start=time.perf_counter()
    try:
        if not path.exists():
            tmp=path.with_suffix('.part')
            with requests.get(url,stream=True,timeout=(20,180)) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for part in r.iter_content(1<<20):f.write(part)
            tmp.rename(path)
        import av
        with av.open(str(path)) as container:
            video=next(s for s in container.streams if s.type=='video')
            audio=next((s for s in container.streams if s.type=='audio'),None)
            duration=float(container.duration/av.time_base)
            if duration<=2 or audio is None:raise ValueError('no audio or short video')
            info={'width':video.width,'height':video.height,'audio_sample_rate':audio.rate,'audio_channels':audio.channels,
                  'metadata':dict(container.metadata),'video_metadata':dict(video.metadata)}
        return {'source_video_id':identity,'path':str(path),'url':url,'revision':MIRROR,'sha256':sha256(path),'bytes':path.stat().st_size,'duration_s':duration,**info,'probe_backend':'PyAV','status':'ok','download_and_probe_s':time.perf_counter()-start,'utc':now()}
    except Exception as e:
        return {'source_video_id':identity,'url':url,'status':'failed','error':str(e),'utc':now()}

def acquire(role):
    if (ROOT/f'manifests/{role}_available.jsonl').exists():raise FileExistsError('existing acquisition manifest is preserved; use an explicit recorded transport-repair action')
    rows=read_jsonl(ROOT/f'manifests/{role}.jsonl');ids=list(dict.fromkeys(r['source_video_id'] for r in rows));records=[]
    for r in concurrent.futures.ThreadPoolExecutor(6).map(acquire_one,ids):
        records.append(r);append_jsonl(ROOT/('download_failures.jsonl' if r['status']!='ok' else f'audit/media_{role}.jsonl'),r)
        print(r['source_video_id'],r['status'],r.get('bytes',r.get('error')),flush=True)
    good={r['source_video_id']:r for r in records if r['status']=='ok'}
    final=[]
    for r in rows:
        if r['source_video_id'] in good:
            m=good[r['source_video_id']];final.append({**r,'media_sha256':m['sha256'],'duration_s':m['duration_s'],'media_has_audio':True})
    target=ROOT/f'manifests/{role}_available.jsonl'
    if target.exists():raise FileExistsError(target)
    write_jsonl(target,final);journal('acquire '+role,requested=len(ids),success=len(good),qa=len(final))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['metadata','acquire']);p.add_argument('--role',default='music_dev_pilot');a=p.parse_args()
    metadata() if a.action=='metadata' else acquire(a.role)
