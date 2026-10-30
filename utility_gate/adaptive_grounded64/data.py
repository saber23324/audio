"""Native-label roles fixed before fitting; old inspected sources stay excluded."""
from __future__ import annotations
import argparse,ast,concurrent.futures,hashlib,json,re,time
from collections import defaultdict,Counter
from pathlib import Path
import av,requests
from utility_gate.grounded_dual_image.data import expand_question,normalize
from .common import ROOT,DATA,MUSIC_META,MUSIC_REV,OFFICIAL,SEED,read_jsonl,write_jsonl,write_json,sha256,journal,status,read_config

def order(value,tag):return hashlib.sha256(f'{SEED}:adaptive_grounded64_v1:{tag}:{value}'.encode()).hexdigest()

def plan():
    if (ROOT/'config/selection_plan.json').exists():raise FileExistsError('source selection already fixed')
    seen=read_jsonl(ROOT/'manifests/seen_before_run.jsonl')
    old_music={r['source_id'].split(':',1)[1] for r in seen if r['source_id'].startswith('music:')}
    old_youtube={r['source_id'].removeprefix('youtube:')[:11] for r in seen if re.fullmatch(r'[A-Za-z0-9_-]{11}(?:_\d+)?',r['source_id'].removeprefix('youtube:'))}
    annotations={s:json.loads((MUSIC_META/f'official_updated_{s}.json').read_text()) for s in ('train','val','test')}
    real={s:{r['video_id'] for r in rows if re.fullmatch(r'\d{8}',r['video_id'])} for s,rows in annotations.items()}
    confirm=sorted(real['test']-old_music,key=lambda x:order(x,'confirm'))[:300]
    available=sorted(real['train']-old_music-set(confirm),key=lambda x:order(x,'development'))
    assignment={'stage1_train':available[:192],'stage1_val':available[192:240],'confirmation_music':confirm}
    assert len(old_music)==700 and len(set(sum(assignment.values(),[])))==540
    for role,ids in assignment.items():
        split='test' if role=='confirmation_music' else 'train';rows=[]
        for r in annotations[split]:
            if r['video_id'] not in ids:continue
            modality,reasoning=ast.literal_eval(r['type'])
            rows.append({'dataset':'music_avqa','role':role,'source_id':'music:'+r['video_id'],'source_video_id':r['video_id'],
                'youtube_id':None,'unknown_source_lineage':True,'source_split':split,'question_id':str(r['question_id']),
                'question':expand_question(r),'answer_text':r['anser'],'modality_type':modality,'reasoning_type':reasoning,
                'video_path':str(DATA/'music_videos'/f"{r['video_id']}.mp4"),'oracle':False,'synthetic':False})
        rank={s:i for i,s in enumerate(ids)};rows.sort(key=lambda r:(rank[r['source_video_id']],int(r['question_id'])))
        write_jsonl(ROOT/f'manifests/planned_{role}.jsonl',rows)
    # Full unchanged native test metadata is only used for role construction,
    # never by the train loader. One predeclared spatial/temporal AV and one
    # Visual QA per fresh physical YouTube source, no model outcomes involved.
    valor=read_jsonl(MUSIC_META/'valor_test.jsonl');repo=json.loads((MUSIC_META/'inesriahi__valor32k-avqa-v2.json').read_text())
    media={Path(r['rfilename']).stem[:11]:r['rfilename'] for r in repo['siblings'] if r['rfilename'].endswith('.mp4')}
    groups=defaultdict(list)
    for r in valor:
        if r['video_id'] not in old_youtube and r['video_id'] in media:groups[r['video_id']].append(r)
    eligible=[]
    for source,qs in groups.items():
        a=[q for q in qs if q['modality']=='audio-visual' and q['category'] in ('location','relative-position','temporal')]
        v=[q for q in qs if q['modality']=='visual']
        if a and v:eligible.append((source,sorted(a,key=lambda q:order(str(q['id']),'valor_qa'))[0],sorted(v,key=lambda q:order(str(q['id']),'valor_qa'))[0]))
    chosen=sorted(eligible,key=lambda x:order(x[0],'valor_confirm'))[:256];rows=[]
    for source,*qs in chosen:
        filename=media[source];stem=Path(filename).stem;offset=float(stem.split('_')[-2])
        for q in qs:
            rows.append({'dataset':'valor32k_avqa_v2','role':'confirmation_valor','source_id':source,'source_video_id':source,'youtube_id':source,
                'clip_offset_s':offset,'unknown_source_lineage':False,'source_split':'test','question_id':str(q['id']),
                'question':q['question'],'options':[f'{c}. {s}' for c,s in zip('ABCD',q['options'])],'answer_text':'ABCD'[q['correct_answer_idx']],
                'modality_type':'Audio-Visual' if q['modality']=='audio-visual' else 'Visual','reasoning_type':{'location':'Location','relative-position':'Location','temporal':'Temporal','action':'Action','count':'Counting','description':'Description'}[q['category']],
                'native_category':q['category'],'quality_rating':q['quality_rating'],'video_path':str(DATA/'valor_videos'/Path(filename).name),'remote_filename':filename,'oracle':False,'synthetic':False})
    write_jsonl(ROOT/'manifests/planned_confirmation_valor.jsonl',rows)
    selection={'seed':SEED,'Music_official_commit':OFFICIAL,'Music_media_revision':MUSIC_REV,'old_music_ids_excluded':len(old_music),
        'old_known_youtube_ids_excluded':len(old_youtube),'music_roles':assignment,'Valor_eligible_spatial_temporal_sources':len(eligible),
        'Valor_confirmation_sources':[x[0] for x in chosen],'source_selection':'SHA order before training or predictions; no result-dependent source filtering',
        'source_lineage_limit':'Music numerical real-video IDs cannot be joined to original YouTube; unknown_source_lineage recorded, no invented independence',
        'frame_policy':{'first_timestamp_s':2.,'period_s':2.,'max_frames':32,'schedule':'fixed past arrivals2,4,...,64; do not use final duration to respace earlier observations'},
        'bounded_run_size':'192 training source IDs,48 held-out development source IDs,300 fresh Music confirmation IDs and256 fresh known-YT Valor sources; failures retained, no outcome-based replacements'}
    write_json(ROOT/'config/selection_plan.json',selection)
    write_json(ROOT/'manifests/source_roles.json',{'Music':{s:role for role,ids in assignment.items() for s in ids},'Valor':{x[0]:'confirmation_valor' for x in chosen},'old_seen_ledger_sha256':sha256(ROOT/'manifests/seen_before_run.jsonl')})
    journal('M0 roles planned before any new model fitting',music_roles={k:len(v) for k,v in assignment.items()},valor_sources=len(chosen))
    print({r:len(read_jsonl(ROOT/f'manifests/planned_{r}.jsonl')) for r in [*assignment,'confirmation_valor']},flush=True)

def acquire_one(row):
    path=Path(row['video_path']);path.parent.mkdir(parents=True,exist_ok=True)
    revision=MUSIC_REV if row['dataset']=='music_avqa' else '1b916bd15c19302f319ee6dd1206c7555bb19d76'
    url=f'https://huggingface.co/datasets/UnFaZeD07/Music-AVQA/resolve/{revision}/videos/{path.name}' if row['dataset']=='music_avqa' else f"https://huggingface.co/datasets/inesriahi/valor32k-avqa-v2/resolve/{revision}/{row['remote_filename']}"
    errors=[]
    for attempt in range(2):
      try:
        if not path.exists():
            temp=path.with_suffix('.part')
            with requests.get(url,stream=True,timeout=(20,180)) as r:
                r.raise_for_status()
                with temp.open('wb') as f:
                    for block in r.iter_content(1<<20):f.write(block)
            temp.rename(path)
        with av.open(str(path)) as c:
            video=c.streams.video[0];audio=c.streams.audio[0] if c.streams.audio else None
            duration=float(c.duration/av.time_base)
            if duration<2:raise ValueError('less than first2s observation')
            details={'duration_s':duration,'width':video.width,'height':video.height,'media_has_audio':audio is not None,'native_sample_rate':audio.rate if audio else None}
        return {'source_id':row['source_id'],'source_video_id':row['source_video_id'],'video_path':str(path),'url':url,'revision':revision,'media_sha256':sha256(path),'bytes':path.stat().st_size,'status':'ok','transport_failures':errors,**details}
      except Exception as e:
        errors.append(str(e))
        if isinstance(e,requests.HTTPError) and e.response.status_code in (401,403):break
    return {'source_id':row['source_id'],'source_video_id':row['source_video_id'],'url':url,'status':'failed','errors':errors}

def acquire(role):
    if role not in ('stage1_train','stage1_val','confirmation_music','confirmation_valor'):raise ValueError(role)
    target=ROOT/f'manifests/{role}.jsonl'
    if target.exists():raise FileExistsError('preserve existing materialized role')
    rows=read_jsonl(ROOT/f'manifests/planned_{role}.jsonl');unique={}
    for r in rows:unique.setdefault(r['source_id'],r)
    records=[]
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for r in pool.map(acquire_one,unique.values()):records.append(r);print(role,r['source_id'],r['status'],flush=True)
    write_jsonl(ROOT/f'audit/media_{role}.jsonl',records);good={r['source_id']:r for r in records if r['status']=='ok'}
    final=[{**r,**{k:v for k,v in good[r['source_id']].items() if k not in ('source_id','source_video_id','url','status','errors')}} for r in rows if r['source_id'] in good]
    write_jsonl(target,final)
    if role in ('stage1_train','stage1_val'):
        other='gate_train' if role=='stage1_train' else 'gate_val'
        write_jsonl(ROOT/f'manifests/{other}.jsonl',[{**r,'role':other} for r in final])
    journal('M0 materialize role using metadata-only QC',role=role,planned_sources=len(unique),media_ok=len(good),qa=len(final),failures=len(records)-len(good))

def finalize_protocol():
    target=ROOT/'config/protocol.json'
    if target.exists():raise FileExistsError(target)
    roles=['stage1_train','stage1_val','gate_train','gate_val','confirmation_music','confirmation_valor']
    rows={r:read_jsonl(ROOT/f'manifests/{r}.jsonl') for r in roles}
    aliases={}
    for role in ['stage1_train','stage1_val','confirmation_music','confirmation_valor']:
        for r in rows[role]:
            for key in [('id',r['source_id']),('bytes',r['media_sha256']),('path',str(Path(r['video_path']).resolve()))]:
                if key in aliases and aliases[key]!=role:raise ValueError('role alias leakage')
                aliases[key]=role
    config={'seed':SEED,'task':'Gate-16 as authorized actual adaptive taskbook, M0-M7','model_training':{'stage1':{'mode':'warm_start_legacy_stage1_on_fresh_official_music_train','warm_start_checkpoint':str(__import__('utility_gate.common',fromlist=['VISION_ROUTER']).VISION_ROUTER),
        'epochs':3,'learning_rate':1e-5,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'max_grad_norm':1.,'gradient_accumulation_steps':1,'batch_size_qa':1,
        'objective':'exact old phase4 full-vocabulary cross entropy of FIRST answer token only; no Full Vision KD in audited legacy Stage1','gradient_checkpointing':True,'bottleneck_stage':1,'num_queries':64,'hidden':256,'heads':8,'temperature':.05,'value_mode':'native','latent_norm':'none',
        'only_active_parameters':['query_generator.query_tokens','visual_key.weight'],'selection':'lowest held-out stage1_val first-answer-token CE among actually trained epochs1,2,3; tie earlier epoch; epoch0 reference not selectable',
        'protocol_adaptations':['official MUSIC native open answer in same qwen_1_5 chat template; no fabricated multiple-choice alternatives','fixed chronological2s first32 arrivals replace old noncausal full-duration uniform frame selection','feature cache is frozen RGB only; no real/null audio in Stage1 fitting','fresh optimizer from old final weight checkpoint; warm-start adaptation, not from-scratch reproduction','local logs only, no W&B telemetry']},
        'utility_gate':{'epochs':30,'learning_rate':.001,'weight_decay':.01,'batch_size':64,'hidden':64,'dropout':.1,'loss':'source-weighted Huber on newly measured signed add-one-grounded sequence-NLL utility; target scale from train only','checkpoint_selection':'held-out gate_val MAE','models':['new_utility_only','new_utility_on_frozen_combined_corr_trunk']}}
        ,'frame_policy':{'first_timestamp_s':2.,'period_s':2.,'max_frames':32,'question_independent':True,'duration_not_used_to_respace':True},
        'scoring':{'MUSIC':'official updated native labels; normalized generated answer EM with prior frozen case/whitespace/underscore/terminal-punctuation/numeral0..11 normalization; canonical-string full-vocabulary sum NLL withoutEOS','Valor':'original legacy multiple-choice prompt with official4options; strict generated uppercaseABCD;4candidate-normalized first-logit NLL, full-vocab separate'},
        'fuser':{'kind':'F1 same new Stage1 slots, fixed beta selected on legal validation','beta_candidates':[.25,.5],'overlay_alpha':.2,'highlight_area':.2,'color_rgb':[255,48,48]},
        'utility_targets':{'scope':'actual add-one frame intervention with all other frames RGB new Stage1*','primary':'baseline complete-answer sequence NLL minus add-one sequence NLL','correctness_delta':'actual greedy EM difference','aggregation':'per-source/time median over native questions, no global-video label replication','uncertainty_band':.001},
        'policy_selection':{'u_threshold_candidates':['-inf',-.02,0.,.02,'inf'],'r_threshold_candidates':['-inf','old_corr',0.,'inf'],
            'objective':'maximize real complete-mixture validation EM, then minimize complete-mixture Gold NLL, Visual retention tolerance2pp, final tie larger threshold; no activation-rate objective',
            'random':'three Bernoulli seeds [1234,2234,3234], probability locked to actual main Gate validation ON rate; no confirmation-rate fitting','no_fixed_activation_rate':True},
        'development_manifest_sha256':{r:sha256(ROOT/f'manifests/{r}.jsonl') for r in roles if not r.startswith('confirmation')},
        'confirmation_sealed_manifests':{r:{'path':str(ROOT/f'manifests/{r}.jsonl'),'sha256':sha256(ROOT/f'manifests/{r}.jsonl'),'sources':len({x['source_id'] for x in rows[r]}),'qa':len(rows[r])} for r in roles if r.startswith('confirmation')},
        'source_scope':'Music IDs/bytes/paths disjoint, original YouTube lineage unknown; Valor knownYouTube disjoint against historical ledger, Music physical cross-lineage/pretraining not provable',
        'all_old_models_frozen':True,'large_model_optimizers':0,'train_test_alias_check':True}
    write_json(target,config);status('M0','completed',source_counts={k:len({x['source_id'] for x in v}) for k,v in rows.items()},scope='see source_lineage limitations')
    print('protocol locked',sha256(target),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','acquire','finalize']);p.add_argument('--role');a=p.parse_args()
    plan() if a.action=='plan' else acquire(a.role) if a.action=='acquire' else finalize_protocol()
