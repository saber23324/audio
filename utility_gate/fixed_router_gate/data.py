import argparse,ast,json,re,hashlib,concurrent.futures
from pathlib import Path
from collections import defaultdict
from utility_gate.adaptive_grounded64.data import acquire_one
from utility_gate.grounded_dual_image.data import expand_question
from .common import ROOT,PREV,DATA,DATA_ROOT,SEED,read_jsonl,write_jsonl,write_json,sha256,journal,status
META=DATA_ROOT/'audio_grounded_dual_image/metadata'
def order(x,tag):return hashlib.sha256(f'{SEED}:fixed_router_gate_v1:{tag}:{x}'.encode()).hexdigest()
def plan():
    if (ROOT/'data/selection.json').exists():raise FileExistsError('source plan already fixed')
    seen=read_jsonl(ROOT/'data/seen_before_run.jsonl');music={r['source_id'].split(':',1)[1] for r in seen if r['source_id'].startswith('music:')}
    youtube={r['source_id'].removeprefix('youtube:')[:11] for r in seen if re.fullmatch(r'[A-Za-z0-9_-]{11}(?:_\d+)?',r['source_id'].removeprefix('youtube:'))}
    rows=read_jsonl(PREV/'manifests/train.jsonl');write_jsonl(ROOT/'data/train.jsonl',[{**r,'role':'train','reuse_scope':'legal192 training sources used for fixed R_A; not new test'} for r in rows])
    ann={k:json.loads((META/f'official_updated_{k}.json').read_text()) for k in ('train','test')}
    test_ids=sorted({r['video_id'] for r in ann['test'] if re.fullmatch(r'\d{8}',r['video_id'])}-music,key=lambda x:order(x,'confirm_music'))[:200]
    val_ids=sorted({r['video_id'] for r in ann['train'] if re.fullmatch(r'\d{8}',r['video_id'])}-music-set(test_ids),key=lambda x:order(x,'fresh_validation'))[:96]
    for role,ids,split in [('val',val_ids,'train'),('confirmation_music',test_ids,'test')]:
        rows=[]
        for r in ann[split]:
            if r['video_id'] not in ids:continue
            modality,reasoning=ast.literal_eval(r['type']);rows.append({'dataset':'music_avqa','role':role,'source_id':'music:'+r['video_id'],'source_video_id':r['video_id'],
                'youtube_id':None,'unknown_source_lineage':True,'source_split':split,'question_id':str(r['question_id']),'question':expand_question(r),'answer_text':r['anser'],
                'modality_type':modality,'reasoning_type':reasoning,'video_path':str(DATA/'music_videos'/f"{r['video_id']}.mp4"),'oracle':False,'synthetic':False})
        ranks={s:i for i,s in enumerate(ids)};rows.sort(key=lambda r:(ranks[r['source_video_id']],r['question_id']));write_jsonl(ROOT/f'data/planned_{role}.jsonl',rows)
    metadata=json.loads((META/'inesriahi__valor32k-avqa-v2.json').read_text());media={Path(r['rfilename']).stem[:11]:r['rfilename'] for r in metadata['siblings'] if r['rfilename'].endswith('.mp4')};groups=defaultdict(list)
    for r in read_jsonl(META/'valor_test.jsonl'):
        if r['video_id'] not in youtube and r['video_id'] in media:groups[r['video_id']].append(r)
    eligible=[]
    for source,qs in groups.items():
        av=[q for q in qs if q['modality']=='audio-visual' and q['category'] in ('location','relative-position','temporal')];vs=[q for q in qs if q['modality']=='visual']
        if av and vs:eligible.append((source,min(av,key=lambda q:order(q['id'],'qa')),min(vs,key=lambda q:order(q['id'],'qa'))))
    selected=sorted(eligible,key=lambda x:order(x[0],'confirm_valor'))[:256];rows=[]
    for source,*qs in selected:
        name=media[source]
        for q in qs:rows.append({'dataset':'valor32k_avqa_v2','role':'confirmation_valor','source_id':source,'source_video_id':source,'youtube_id':source,'unknown_source_lineage':False,
            'source_split':'test','clip_offset_s':float(Path(name).stem.split('_')[-2]),'question_id':str(q['id']),'question':q['question'],
            'options':[f'{c}. {s}' for c,s in zip('ABCD',q['options'])],'answer_text':'ABCD'[q['correct_answer_idx']],
            'modality_type':'Audio-Visual' if q['modality']=='audio-visual' else 'Visual','reasoning_type':{'location':'Location','relative-position':'Location','temporal':'Temporal','action':'Action','count':'Counting','description':'Description'}[q['category']],
            'native_category':q['category'],'video_path':str(DATA/'valor_videos'/Path(name).name),'remote_filename':name,'oracle':False,'synthetic':False})
    write_jsonl(ROOT/'data/planned_confirmation_valor.jsonl',rows)
    write_json(ROOT/'data/selection.json',{'seed':SEED,'old_inspected_identities':len(seen),'old_music_ids_excluded':len(music),'old_known_youtube_ids_excluded':len(youtube),
        'new_validation_train_IDs':val_ids,'confirmation_music_IDs':test_ids,'confirmation_valor_YT_IDs':[x[0] for x in selected],
        'fresh_val_reason':'old48 repeatedly searched; new96 official-train IDs never inspected or used for R_A/V1/Gate fitting',
        'source_selection':'hash order before model scores; metadata-only QC >=8s; no answer-correctness selection',
        'residual_uncertainty':'MUSIC original YouTube lineage and cross-dataset/pretraining physical overlap unknown'})
    print('Planned192 train,96 fresh val,200 fresh MUSIC and256 fresh Valor sources',flush=True)
def acquire(role):
    out=ROOT/f'data/{role}.jsonl'
    if out.exists():raise FileExistsError(out)
    rows=read_jsonl(ROOT/f'data/planned_{role}.jsonl');unique={r['source_id']:r for r in rows};records=[]
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for r in pool.map(acquire_one,unique.values()):records.append(r);print(role,r['source_id'],r['status'],flush=True)
    write_jsonl(ROOT/f'audit/media_{role}.jsonl',records);good={r['source_id']:r for r in records if r['status']=='ok' and r['duration_s']>=8}
    output=[{**r,**{k:v for k,v in good[r['source_id']].items() if k not in ('source_id','source_video_id','url','status','errors')}} for r in rows if r['source_id'] in good]
    write_jsonl(out,output);journal('new role materialized before predictions',role=role,sources=len(good),qa=len(output),QC_excluded=len(records)-len(good))
def protocol():
    roles=('train','val','confirmation_music','confirmation_valor');allrows={r:read_jsonl(ROOT/f'data/{r}.jsonl') for r in roles};ids={}
    for role,rs in allrows.items():
        for r in rs:
            for identity in [('source',r['source_id']),('media',r['media_sha256']),('path',str(Path(r['video_path']).resolve()))]:
                if identity in ids and ids[identity]!=role:raise ValueError('source/bytes/path leakage')
                ids[identity]=role
    cfg={'seed':SEED,'timestamps_s':[2.,4.,6.,8.],'audio_window_s':2.,'tokens_per_frame':64,'only_train_new_Gate':True,'Router_optimizer_count':0,
        'train':{'families':['audio_r','audio_no_r','visual','acoustic','history'],'lambdas':[0.,.1,.5,1.],'temperatures':[.25,1.],
            'epochs':30,'learning_rate':.001,'weight_decay':.01,'batch_size_sources':16,'hidden':64,'dropout':.1,'seed':SEED,
            'target':'source-mean full canonical QA Gold NLL over all16 true masks, q=softmax(-(NLL+lambda*k)/temperature); invalid masks excluded',
            'objective':'sequence-level autoregressive all-mask cross entropy, no sum of window truth or independent marginal labels',
            'checkpoint_candidates':[1,10,20,30,'best_validation_distillation_loss'],'thresholds':[0.,.25,.5,.75,1.]},
        'selection':{'accuracy_first':'lambda0 candidates: maximize actual val full-mask EM, then lower Gold NLL, then smaller measured frontend cost; off/on included',
            'cost_aware':'within1pp absolute EM and+0.05 dataset-native Gold NLL of val Accuracy-first, minimize measured predicted frontend cost, then mean k, then maximize EM; no activation target',
            'accuracy_margin':.01,'NLL_margin':.05,'visual_harm_diagnostic_margin':.02,'cost_calibration':'first3 lexicographic all4-audio-valid fresh validation sources, causally firstK prefix and per-family prepaid encoder/gate cost',
            'decision_latency_median_or_mean':'mean across3 fixed sources; descriptive, model loaded, no feature cache, no Gold evaluation or playback wait',
            'random_seeds':[1234,2234,3234],'random_probability':'separately actual T3/T4 val decision ON rate, no test fitting',
            'online_caps':[0,1,2,3,4],'cap_rule':'counter of already encoded extra images; once exhausted skip further auxiliary processing and return V1; no future ranking'},
        'formal_native_reference':'N0 not_run_reference_only; user requires all formal paths64/frame',
        'map_failure':'raw invalid => forced OFF; ON with map invalid still encodes rendered RGB fallback once through R_A, counts as extra image; map validity reported',
        'scoring':'MUSIC normalized native open-answer EM and canonical-string sum NLL; Valor strict ABCD and4-candidate normalized Gold NLL. No mixed-dataset NLL',
        'development_manifest_sha256':{r:sha256(ROOT/f'data/{r}.jsonl') for r in ('train','val')},
        'sealed_confirmation':{r:{'path':str(ROOT/f'data/{r}.jsonl'),'sha256':sha256(ROOT/f'data/{r}.jsonl'),'sources':len({x['source_id'] for x in allrows[r]}),'qa':len(allrows[r])} for r in roles if r.startswith('confirmation')},
        'bootstrap':{'replicates':2000,'seed':SEED,'unit':'known Valor YouTube ID; MUSIC numeric ID only, original lineage unknown'},
        'judgment':{'GO_SELECT':'Accuracy-first actual nontrivial masks, lower95 paired Accuracy>0 and supported Gold NLL against predeclared best fixed endpoint and val-rate random mean on known-YT AV primary; not explained by always-off/on',
            'GO_EFFICIENCY':'Cost-aware vsR_A always within1pp Accuracy and+0.05 Gold NLL (source bootstrap CI), extra encoder count reduced and actual frontend+pipeline cost reduced beyond prepaid costs',
            'GO_AUDIO_SPECIFIC':'same exact mask real Grounded vsdup/Sham/zero each lower95 paired Accuracy>0 with supported Gold NLL; else at most generic second-path adaptation',
            'otherwise':'NO-GO current configuration if precise no increment/only duplicate benefit; INCONCLUSIVE for broad intervals or source/power limits; eachGO flag reported separately'},
        'source_scope':'8s observed-prefix QA, not official full-video score; MUSIC original lineage unknown; new known-YT Valor disjoint against known inspected identities only'}
    write_json(ROOT/'data/protocol.json',cfg);write_json(ROOT/'data/role_manifest.json',{'roles':{r:{'sources':len({x['source_id'] for x in rs}),'qa':len(rs),'sha256':sha256(ROOT/f'data/{r}.jsonl')} for r,rs in allrows.items()},'source_scope':cfg['source_scope']});print('Protocol data sealed',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','acquire','protocol']);p.add_argument('--role');a=p.parse_args();plan() if a.action=='plan' else acquire(a.role) if a.action=='acquire' else protocol()
