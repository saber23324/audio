import argparse,ast,json,re,hashlib,concurrent.futures
from collections import defaultdict
from pathlib import Path
from utility_gate.adaptive_grounded64.data import acquire_one
from utility_gate.grounded_dual_image.data import expand_question
from .common import ROOT,PREV,DATA,DATA_ROOT,read_jsonl,write_jsonl,write_json,sha256,journal,status,SEED
META=DATA_ROOT/'audio_grounded_dual_image/metadata'
def order(x,tag):return hashlib.sha256(f'{SEED}:gatefirst_v1:{tag}:{x}'.encode()).hexdigest()

def plan():
    if (ROOT/'config/source_selection.json').exists():raise FileExistsError('source selection immutable')
    seen=read_jsonl(ROOT/'manifests/seen_before_run.jsonl');music={r['source_id'].split(':',1)[1] for r in seen if r['source_id'].startswith('music:')}
    yt={r['source_id'].removeprefix('youtube:')[:11] for r in seen if re.fullmatch(r'[A-Za-z0-9_-]{11}(?:_\d+)?',r['source_id'].removeprefix('youtube:'))}
    for role,oldrole in [('train','stage1_train'),('val','stage1_val')]:
        rows=read_jsonl(PREV/f'manifests/{oldrole}.jsonl');write_jsonl(ROOT/f'manifests/{role}.jsonl',[{**r,'role':role,'legal_role_reuse':'official train-derived prior192/48; train/val membership preserved'} for r in rows])
    official=json.loads((META/'official_updated_test.json').read_text());ids={r['video_id'] for r in official if re.fullmatch(r'\d{8}',r['video_id'])}-music
    selected=sorted(ids,key=lambda s:order(s,'fresh_music'))[:200];rows=[]
    for r in official:
        if r['video_id'] not in selected:continue
        modality,reasoning=ast.literal_eval(r['type']);rows.append({'dataset':'music_avqa','role':'confirmation_music','source_id':'music:'+r['video_id'],'source_video_id':r['video_id'],
            'youtube_id':None,'unknown_source_lineage':True,'source_split':'test','question_id':str(r['question_id']),'question':expand_question(r),'answer_text':r['anser'],
            'modality_type':modality,'reasoning_type':reasoning,'video_path':str(DATA/'music_videos'/f"{r['video_id']}.mp4"),'oracle':False,'synthetic':False})
    rank={s:i for i,s in enumerate(selected)};rows.sort(key=lambda r:(rank[r['source_video_id']],r['question_id']));write_jsonl(ROOT/'manifests/planned_confirmation_music.jsonl',rows)
    meta=json.loads((META/'inesriahi__valor32k-avqa-v2.json').read_text());media={Path(r['rfilename']).stem[:11]:r['rfilename'] for r in meta['siblings'] if r['rfilename'].endswith('.mp4')};groups=defaultdict(list)
    for r in read_jsonl(META/'valor_test.jsonl'):
        if r['video_id'] not in yt and r['video_id'] in media:groups[r['video_id']].append(r)
    candidates=[]
    for source,qs in groups.items():
        av=[q for q in qs if q['modality']=='audio-visual' and q['category'] in ('location','relative-position','temporal')];visual=[q for q in qs if q['modality']=='visual']
        if av and visual:candidates.append((source,min(av,key=lambda q:order(q['id'],'qa')),min(visual,key=lambda q:order(q['id'],'qa'))))
    chosen=sorted(candidates,key=lambda x:order(x[0],'fresh_valor'))[:256];rows=[]
    for source,*qs in chosen:
        filename=media[source];offset=float(Path(filename).stem.split('_')[-2])
        for q in qs:rows.append({'dataset':'valor32k_avqa_v2','role':'confirmation_valor','source_id':source,'source_video_id':source,'youtube_id':source,'unknown_source_lineage':False,
            'source_split':'test','clip_offset_s':offset,'question_id':str(q['id']),'question':q['question'],'options':[f'{c}. {s}' for c,s in zip('ABCD',q['options'])],
            'answer_text':'ABCD'[q['correct_answer_idx']],'modality_type':'Audio-Visual' if q['modality']=='audio-visual' else 'Visual',
            'reasoning_type':{'location':'Location','relative-position':'Location','temporal':'Temporal','action':'Action','count':'Counting','description':'Description'}[q['category']],
            'native_category':q['category'],'video_path':str(DATA/'valor_videos'/Path(filename).name),'remote_filename':filename,'oracle':False,'synthetic':False})
    write_jsonl(ROOT/'manifests/planned_confirmation_valor.jsonl',rows)
    write_json(ROOT/'config/source_selection.json',{'seed':SEED,'prior_inspected_sources':len(seen),'prior_music_IDs_excluded':len(music),'prior_known_YT_excluded':len(yt),
        'new_music_IDs':selected,'new_valor_YT_IDs':[x[0] for x in chosen],'new_valor_eligible_sources':len(candidates),
        'source_selection':'hash order before any predictions; no correctness filtering','known_lineage_limits':'MUSIC original YT unknown; physical cross-dataset / pretraining overlap not verifiable',
        'training_reuse':'prior legal192/48 official MUSIC train-derived sources, role membership unchanged','frame_schedule_s':[2.,4.,6.,8.],
        'sampling_audit':'previous V1/F0 run used up to32 2s arrivals; historical native used4 duration-relative times. This task requires4: use first4 fixed causal arrivals. Native labels evaluated on8s observed prefix; not official full-video accuracy.'})
    print('roles planned',len(selected),len(chosen),flush=True)

def acquire(role):
    path=ROOT/f'manifests/{role}.jsonl'
    if path.exists():raise FileExistsError(path)
    planned=read_jsonl(ROOT/f'manifests/planned_{role}.jsonl');unique={r['source_id']:r for r in planned};details=[]
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for r in pool.map(acquire_one,unique.values()):details.append(r);print(role,r['source_id'],r['status'],flush=True)
    write_jsonl(ROOT/f'audit/media_{role}.jsonl',details);good={r['source_id']:r for r in details if r['status']=='ok'}
    result=[{**r,**{k:v for k,v in good[r['source_id']].items() if k not in ('source_id','source_video_id','url','status','errors')}} for r in planned if r['source_id'] in good]
    # Do not respace or select by outcomes; short clips cannot supply all4 arrivals.
    result=[r for r in result if r['duration_s']>=8.0]
    write_jsonl(path,result);journal('M0 new confirmation materialized',role=role,qa=len(result),sources=len({r['source_id'] for r in result}),failures=sum(r['status']!='ok' for r in details),short8s_excluded=len(good)-len({r['source_id'] for r in result}))

def protocol():
    roles=['train','val','confirmation_music','confirmation_valor'];rows={role:read_jsonl(ROOT/f'manifests/{role}.jsonl') for role in roles};identities={}
    for role,rs in rows.items():
        for r in rs:
            for ident in [('id',r['source_id']),('bytes',r['media_sha256']),('path',str(Path(r['video_path']).resolve()))]:
                if ident in identities and identities[ident]!=role:raise ValueError('cross-role source/media alias')
                identities[ident]=role
    cfg={'seed':SEED,'phase_order':['M0','A train Gate on fixed V1/F0','freeze Gate','B train new Router','C native bypass','new confirmation','statistics'],
        'timestamps_s':[2.,4.,6.,8.],'audio_window_s':2.,'visual_tokens_per_frame':64,'fixed_F0_beta':.25,'overlay_alpha':.2,'overlay_area':.2,
        'Gate_training':{'epochs':30,'lr':.001,'weight_decay':.01,'hidden':64,'dropout':.1,'batch_size_sources':16,'seed':SEED,
            'variants':['u_huber','u_rank','dual_rank','vision_rank','u_classification_if_complementary'],
            'policies':'all16 masks, actual full-sequence generation+Gold NLL for every native QA',
            'aggregate':'median native-QA policy NLL benefit per source; conditional marginal averages over enumerated future choices; no future input in Gate',
            'tie_band':.001,'rank_tie_band':.001,'checkpoint_candidates':[1,10,20,30,'best_heldout_loss'],
            'thresholds':['-inf',-.02,0.,.02,'inf'],'class_thresholds':['-inf',.25,.5,.75,'inf'],'r_thresholds':['-inf',0.,'old_corr','inf'],
            'selection':'real complete-policy val EM, then lower Gold NLL, Visual no worse than2pp, then larger thresholds; never activation rate',
            'minimum_sources_each_class':20},
        'Router_training':{'kind':'R-A zero-output low-rank cross-view64 residual','key_dim':128,'value_rank':32,'epochs':3,'lr':1e-5,'weight_decay':.01,'clip':1.,'seed':SEED,
            'loss':'verified legacy phase4 FIRST-answer-token CE only; no KD','batch_size_qa':1,'gradient_checkpointing':True,
            'exposure':'train Gate-selected ON plus explicit forced_on_exposure for other available windows; pure Gate trace retained',
            'selection':'best real Always-on validation EM then Gold NLL among trained epochs; frozen G_A path separately evaluated'},
        'random_seeds':[1234,2234,3234],'random_probability':'actual selected Gate legal val ON rate, frozen before confirmation',
        'native':'fixed2 images/time, original then secondary, frozen identical sensor-uncertainty system prompt; actual tokens metered; C2 zero-shot transfer from fixed-F0 Utility',
        'optional_not_run':['R-B joint resampler','Gate_B refit','native-specific Gate C8','second-first image order'],
        'scoring':'MUSIC prior normalized short-answer EM/canonical sequence NLL; Valor strict ABCD/four-candidate normalized NLL. No pooling NLL across datasets.',
        'development_manifest_sha256':{role:sha256(ROOT/f'manifests/{role}.jsonl') for role in ('train','val')},
        'sealed_confirmation':{role:{'path':str(ROOT/f'manifests/{role}.jsonl'),'sha256':sha256(ROOT/f'manifests/{role}.jsonl'),'sources':len({r['source_id'] for r in rows[role]}),'qa':len(rows[role])} for role in roles if role.startswith('confirmation')},
        'bootstrap':{'replicates':2000,'seed':SEED,'unit':'Valor original known YT; MUSIC numeric ID only, original lineage unknown'},
        'decision':'stage-wise GO requires paired primary known-YT AV lower95% accuracy delta>0 against appropriate component and same-mask/duplicate/Sham/random controls, supported NLL and Visual drop<=2pp; all-off is no Gate active-use evidence; router Always-on diagnosis mandatory regardless; otherwise INCONCLUSIVE unless both components have no reproducible increment then current-config NO-GO'}
    write_json(ROOT/'config/protocol.json',cfg);write_jsonl(ROOT/'results/lineage_and_splits.jsonl',[{'source_id':r['source_id'],'role':role,'media_sha256':r['media_sha256'],'youtube_id':r['youtube_id'],'unknown_source_lineage':r['unknown_source_lineage']} for role,rs in rows.items() for r in {x['source_id']:x for x in rs}.values()]);print('data protocol fixed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','acquire','protocol']);p.add_argument('--role');a=p.parse_args()
    plan() if a.action=='plan' else acquire(a.role) if a.action=='acquire' else protocol()
