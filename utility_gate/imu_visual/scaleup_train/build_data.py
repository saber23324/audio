"""Score-free source partition, official event QA, strict original windows."""
import argparse,json,hashlib,collections,concurrent.futures as cf
from fractions import Fraction
from pathlib import Path
import numpy as np
from .common import ROOT,DATA,OLD,read_jsonl,write_json,write_jsonl,sha256,now,parent_group
from .data import eligible,sensor_window,frames

def build():
    accepted=read_jsonl(ROOT/'M1_download_sync/ACCEPTED_RECORDINGS.jsonl')
    seen={x['source_id'] for x in read_jsonl(ROOT/'locks/historical_seen_sources.jsonl')}
    historical_groups={parent_group(s) for s in seen if s.startswith('holo:')}
    groups=collections.defaultdict(list)
    for record in accepted:groups[parent_group(record['session_id'])].append(record['session_id'])
    fresh=sorted([g for g in groups if g not in historical_groups],key=lambda s:hashlib.sha256(('1234:split:'+s).encode()).hexdigest())
    others=sorted([g for g in groups if g in historical_groups],key=lambda s:hashlib.sha256(('1234:split:'+s).encode()).hexdigest())
    nh=max(2,len(fresh)//5) if len(fresh)>=6 else 0
    hold_groups=fresh[-nh:] if nh else [];rest=fresh[:-nh] if nh else fresh
    pool=sorted(rest+others,key=lambda s:hashlib.sha256(('1234:remaining-split:'+s).encode()).hexdigest());nv=max(2,len(pool)//4) if len(pool)>=4 else 1
    val_groups=pool[-nv:];train_groups=pool[:-nv]
    train=[s for g in train_groups for s in sorted(groups[g])];val=[s for g in val_groups for s in sorted(groups[g])];hold=[s for g in hold_groups for s in sorted(groups[g])]
    if not train or not val:raise ValueError('no new usable train/validation recording groups')
    split={'utc':now(),'train_sessions':train,'val_sessions':val,'confirmation_sessions':hold,'holdout_sessions':hold,'participant_ids':None,'primary_isolation':'conservative recording family R/z numeric prefix, all task variants and Part segments grouped; not an asserted participant identity','train_groups':train_groups,'val_groups':val_groups,'holdout_groups':hold_groups,'old_seen_SHA':sha256(ROOT/'locks/historical_seen_sources.jsonl'),'no_new_predictions_before_lock':True,'rule':'new unseen recording-family hash reserve20percent min2 when>=6; remaining families jointly hashed,last25percent validation; score-free','confirmation_status':'sealed until after model and policy locks' if hold else 'holdout_not_available','annotation_boundary_given':True}
    write_json(ROOT/'locks/data_split_provisional.json',split)
    anns={x['video_name']:x for x in read_jsonl(ROOT/'data/official_annotation_complete_records.jsonl')};vocab=json.loads((OLD/'M2_frozen_imu_plots/holo_renderer_lock.json').read_text())['vocabulary'];by={x['session_id']:x for x in accepted};output=[];windows=[]
    relevant={'rotate','turn','move','slide','push','pull','open','close','lift','lower','insert','remove','shake','press','flip'}
    for source in train+val+hold:
        qc=by[source];capture=np.load(qc['camera_clock'])['times'];rate=float(Fraction(qc['video']['rate']))
        # First 8 eligible events, fixed before any model evaluation; overlapping
        # episodes explicitly retained, effective N is recording not question.
        for i,e in enumerate(eligible(anns[source],vocab)[:8]):
            end_index=int(np.ceil(e['end']*rate))
            if end_index>=len(capture):continue
            end=float(capture[end_index]);stamps=[end-6,end-4,end-2,end]
            if stamps[0]<2:continue
            qid=f'holo:{source}:scaleup:{i}';verb=e['attributes']['Verb'].strip();noun=e['attributes']['Noun'].strip();rng=np.random.default_rng(int.from_bytes(hashlib.sha256((qid+':ABCD').encode()).digest()[:8],'big'))
            opts=rng.choice([x for x in vocab if x!=verb],3,replace=False).tolist();gold=i%4;opts.insert(gold,verb)
            row={'dataset':'holoassist_derived_action_QA','role':'train' if source in train else 'val' if source in val else 'holdout','source_id':'holo:'+source,'session_id':source,'parent_recording_id':parent_group(source),'stat_group_id':parent_group(source),'participant_id':None,'qa_id':qid,'question':f'What action on the {noun.replace("_"," ")} is being completed at the end of this observed interval?','options':[f'{c}. {v.replace("_"," ")}' for c,v in zip('ABCD',opts)],'answer_text':'ABCD'[gold],'option_class_ids':opts,'class_id':verb,'class_name':verb,'imu_relevant_stratum':verb.lower() in relevant,'imu_relevant_definition':'preset physical motion verbs; never model-score selected','imu_path':qc['imu_path'],'imu_sha256':qc['imu_sha256'],'timestamps_s':stamps,'relative_timestamps_s':[2.,4.,6.,8.],'episode_start_s':end-8,'observed_end_s':end,'annotation_media_end_s':e['end'],'annotation_end_frame_index':end_index,'RGB_clock_path':qc['camera_clock'],'clock_grade':qc['clock_grade'],'annotation_boundary_given':True,'label_source':'official completed fine-grained Verb/Noun; no future correction labels','previously_inspected_source':('holo:'+source) in seen,'no_audio_used':True,'gold_provenance':{'official_annotation_record':source,'annotation_id':e.get('id'),'start_media_s':e['start'],'end_media_s':e['end'],'verb':verb,'noun':noun,'template':'completed-event-v1-balanced','label_available_at_capture_s':end,'future_annotation_fields_used':False},'episode_overlap_policy':'may overlap within recording; never split same recording; cluster bootstrap'}
            ww=[sensor_window(row,t) for t in stamps]
            if not all(w.valid for w in ww):continue
            output.append(row)
            for t,w in zip(stamps,ww):windows.append({'qa_id':qid,'source_id':row['source_id'],'role':row['role'],'t':t,'lower':max(0,t-2),'original_both_sensor_bounds_checked':True,'samples':len(w.times),'valid':w.valid})
    write_jsonl(ROOT/'data/qa_manifest.jsonl',output);write_jsonl(ROOT/'M2_qa_manifest_splits/imu_windows.jsonl',windows)
    # Fit normalization on training sources, never test/validation or per window.
    values=np.concatenate([np.load(by[s]['imu_path'])['values'] for s in train]);limits={'head:acc':float(np.quantile(np.abs(values[:,:3]),.995)*1.1),'head:gyro':float(np.quantile(np.abs(values[:,3:]),.995)*1.1)}
    write_json(ROOT/'locks/IMU_PLOT_V1.json',{'kind':'P1','y_limits':limits,'size_px':[768,768],'dpi':120,'font_size':11,'line_width':1.5,'scale_fit_role':'Router train sources only abs99.5percentile*1.1, locked before scoring'})
    proto=json.loads((ROOT.parent/'imu_gate_router_transfer_20261030_v3/locks/protocol.json').read_text());proto['Router']['init']='matched v3 variant warm-start, fresh AdamW state; canonical B1';proto['Router']['warm_start']=True;proto['Gate']['accuracy_training_target']='actual full-greedy correctness first (2*incorrect + NLL/(1+NLL)), bounded monotone GoldNLL tie-break; cost candidates GoldNLL+lambda*extra_images';proto['manifest_sha256']=sha256(ROOT/'data/qa_manifest.jsonl');proto['Router']['steps_planned_per_variant']=3*sum(q['role']=='train' for q in output);proto['independent_confirmation']='sealed new session-disjoint sources only';proto['source_availability_QC']=None
    write_json(ROOT/'locks/protocol.json',proto)
    write_json(ROOT/'M2_qa_manifest_splits/provisional_qa_counts.json',dict(collections.Counter(q['role'] for q in output)))
    print('PROVISIONAL MANIFEST',len(train),len(val),len(hold),len(output),flush=True)

def finish():
    # Missing source decoding is an acquisition failure, not a performance filter.
    split=json.loads((ROOT/'locks/data_split_provisional.json').read_text());allrows=read_jsonl(ROOT/'data/qa_manifest.jsonl');good={s for s in {r['session_id'] for r in allrows} if (ROOT/f'data/frames_{s}.json').exists()}
    old=ROOT/'data/qa_manifest_before_decode_QC.jsonl'
    if not old.exists():old.write_bytes((ROOT/'data/qa_manifest.jsonl').read_bytes())
    output=[r for r in allrows if r['session_id'] in good]
    for key in ['train_sessions','val_sessions','confirmation_sessions','holdout_sessions']:split[key]=[s for s in split[key] if s in good]
    assert split['train_sessions'] and split['val_sessions']
    for key,sourcekey in [('train_groups','train_sessions'),('val_groups','val_sessions'),('holdout_groups','holdout_sessions')]:split[key]=sorted({parent_group(s) for s in split[sourcekey]})
    assert set(split['train_groups']).isdisjoint(split['val_groups']) and set(split['holdout_groups']).isdisjoint(split['train_groups']+split['val_groups'])
    split['utc_final_lock']=now();split['acquisition_exclusions']=sorted({r['session_id'] for r in allrows}-good);split['confirmation_status']='sealed' if split['holdout_sessions'] else 'holdout_not_available'
    write_jsonl(ROOT/'data/qa_manifest.jsonl',output);write_json(ROOT/'locks/data_split.json',split);write_json(ROOT/'M1_download_sync/SPLIT_LOCK.json',split)
    plotfile=ROOT/'locks/IMU_PLOT_V1.json'
    (ROOT/'locks/IMU_PLOT_V1_before_decode_QC.json').write_bytes(plotfile.read_bytes())
    trainvalues=np.concatenate([np.load(next(q['imu_path'] for q in output if q['session_id']==source))['values'] for source in split['train_sessions']])
    plot=json.loads(plotfile.read_text());plot['y_limits']={'head:acc':float(np.quantile(np.abs(trainvalues[:,:3]),.995)*1.1),'head:gyro':float(np.quantile(np.abs(trainvalues[:,3:]),.995)*1.1)};plot['scale_fit_role']='final decoded Router train sources only,99.5percentile*1.1; frozen before features/training'
    write_json(plotfile,plot)
    proto=json.loads((ROOT/'locks/protocol.json').read_text());proto['Gate']['accuracy_training_target']='actual greedy correctness first:2*incorrect+NLL/(1+NLL); cost candidates GoldNLL+lambda*extra_images';proto['manifest_sha256']=sha256(ROOT/'data/qa_manifest.jsonl');proto['Router']['steps_planned_per_variant']=3*sum(r['role']=='train' for r in output);write_json(ROOT/'locks/protocol.json',proto)
    write_json(ROOT/'M2_qa_manifest_splits/train_val_holdout.json',split);write_jsonl(ROOT/'M2_qa_manifest_splits/qa_manifest.jsonl',output)
    lineage=[{'source_id':'holo:'+s,'parent_recording':parent_group(s),'participant_id':None,'role':next(r['role'] for r in output if r['session_id']==s),'historically_seen':next(r['previously_inspected_source'] for r in output if r['session_id']==s)} for s in sorted(good)]
    write_jsonl(ROOT/'M2_qa_manifest_splits/source_lineage.jsonl',lineage)
    stamps=[]
    for s in sorted(good):
        fr=json.loads((ROOT/f'data/frames_{s}.json').read_text())
        stamps.extend({'source_id':'holo:'+s,'decision_t':float(t),**a} for t,a in fr.items())
    write_jsonl(ROOT/'M2_qa_manifest_splits/frame_timestamps.jsonl',stamps)
    print('FINAL SOURCE LOCK',split,collections.Counter(r['role'] for r in output),flush=True)

def safe_frames(shard,shards):
    sources=sorted({r['session_id'] for r in read_jsonl(ROOT/'data/qa_manifest.jsonl')})
    for source in sources[shard::shards]:
        try:frames(only_source=source)
        except Exception as e:
            from .common import append_jsonl
            append_jsonl(ROOT/'M1_download_sync/FAILED_DOWNLOADS.jsonl',{'utc':now(),'phase':'actual_video_decode','source':source,'error':repr(e)})
            print('SOURCE VIDEO FAILURE',source,repr(e),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','finish','frames']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();safe_frames(a.shard,a.shards) if a.phase=='frames' else globals()[a.phase]()
