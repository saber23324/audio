"""Official HoloAssist paired original-clock intake; no writes to older runs."""
import argparse,json,tarfile,concurrent.futures,hashlib
from pathlib import Path
import numpy as np
import av
from PIL import Image
from utility_gate.imu_visual.intake import SOURCES
from utility_gate.imu_visual.holo_probe import load_sensor,combine
from utility_gate.imu_visual.causal_imu_window import causal_window
from .common import ROOT,DATA,OLD,OLD_DATA,SEED,write_json,write_jsonl,read_jsonl,sha256,journal,status,rows
from .range_io import get_range,RangeFile

def scan(url,limit,kind):
    pos=0;out=[];buffer=b'';start=-1
    for _ in range(limit*12):
        if not(start<=pos and pos+512<=start+len(buffer)):buffer=get_range(url,pos,4096);start=pos
        raw=buffer[pos-start:pos-start+512]
        if not any(raw):break
        h=tarfile.TarInfo.frombuf(raw,'utf-8','strict');body=pos+512
        if h.isfile() and h.name.endswith(kind):
            out.append({'name':h.name,'size':h.size,'offset':pos,'offset_data':body});print('INDEX',len(out),h.name,flush=True)
            if len(out)==limit:break
        pos=body+(h.size+511)//512*512
    return out
def indices():
    write_json(ROOT/'data/video_index.json',json.loads((OLD/'M1_data/holo_video_subset_index.json').read_text()))
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs={'camera':pool.submit(scan,SOURCES['holo_video'],80,'.txt'),'imu':pool.submit(scan,SOURCES['holo_imu'],96,'.txt')}
        for key,f in jobs.items():write_json(ROOT/f'data/{key}_index.json',f.result())
    journal('bounded official archive header expansion completed',video_index_sources=40,IMU_index_streams=96,camera_txts=80)

def eligible(ann,vocab):
    fine=[e for e in ann['events'] if e['label']=='Fine grained action'];out=[]
    for e in sorted(fine,key=lambda x:(x['end'],x['start'])):
        v=e['attributes'].get('Verb','').strip();n=e['attributes'].get('Noun','').strip()
        if not(8<=e['end']<=150 and e['end']-e['start']>=1 and v in vocab and n not in ('','none','other','others')):continue
        if any(q is not e and q['attributes'].get('Noun')==n and q['attributes'].get('Verb')!=v and q['start']<e['end'] and q['end']>e['end']-.15 for q in fine):continue
        out.append(e)
    return out

def probe(use_cached=False):
    anns={x['video_name']:x for x in read_jsonl(OLD_DATA/'holo/complete_annotation_records_from_partial.jsonl')};vocab=json.loads((OLD/'M2_frozen_imu_plots/holo_renderer_lock.json').read_text())['vocabulary'];v={x['name'].split('/')[0]:x for x in json.loads((ROOT/'data/video_index.json').read_text())};camera={};imu={}
    for key,target in [('camera',camera),('imu',imu)]:
        for e in json.loads((ROOT/f'data/{key}_index.json').read_text()):target.setdefault(e['name'].split('/')[0],{})[Path(e['name']).name]=e
    candidates=[s for s in v if s in anns and s in camera and s in imu and len(eligible(anns[s],vocab))>=8 and s not in ('R016-7July-Navvis','R028-12July-Switch') and {'Pose_sync.txt','VideoMp4Timing.txt'}<=camera[s].keys() and {'Accelerometer_sync.txt','Gyroscope_sync.txt'}<=imu[s].keys()]
    candidate_lock=ROOT/'data/candidate_lock.json'
    if candidate_lock.exists():assert json.loads(candidate_lock.read_text())['sources']==candidates
    else:write_json(candidate_lock,{'sources':candidates,'selection':'archive prefix availability and >=8 unambiguous official completed events, before current predictions; no accuracy selection','desired_sources':12,'minimum_sources_for_training':4,'source_scope':'previous annotation metadata inspected; candidates are train/dev only unless previously unseen provenance is separately established'})
    def one(source):
        path=ROOT/f'data/probe_{source}.json'
        if path.exists():return json.loads(path.read_text())
        if use_cached:return {'session_id':source,'valid':False,'status':'not_acquired','reason':'bounded expansion transport repeatedly failed; minimum verified-source training proceeds; clock not declared invalid'}
        try:
            e=camera[source]['VideoMp4Timing.txt'];origin=int(get_range(SOURCES['holo_video'],e['offset_data'],e['size']).decode().split()[0]);e=camera[source]['Pose_sync.txt'];raw=get_range(SOURCES['holo_video'],e['offset_data'],e['size']);ticks=np.array([int(x.split()[1]) for x in raw.decode().splitlines() if x.strip()],np.int64)
            if (np.diff(ticks)<0).any():raise ValueError('nonmonotonic camera clock')
            ticks=np.unique(ticks);clock=(ticks-origin).astype(np.float64)/1e7;cp=DATA/f'clock/{source}.npz';np.savez_compressed(cp,ticks=ticks,times=clock)
            e=v[source]
            with av.open(RangeFile(SOURCES['holo_video'],e['size'],e['offset_data'],block_size=262144)) as container:
                stream=container.streams.video[0];metadata={'frame_count':stream.frames,'rate':str(stream.average_rate),'width':stream.width,'height':stream.height,'duration_s':float(stream.duration*stream.time_base)}
            result={'session_id':source,'participant_id':None,'origin_tick':origin,'camera_clock':str(cp),'camera_clock_sha256':sha256(cp),'unique_camera_ticks':len(ticks),'video':metadata,'valid':len(ticks)==metadata['frame_count'],'reason':'exact monotonic original camera tick/frame count matching' if len(ticks)==metadata['frame_count'] else 'frame count mismatch; no guessed offset','eligible_annotated_events':len(eligible(anns[source],vocab))}
        except Exception as e:result={'session_id':source,'valid':False,'reason':type(e).__name__+': '+str(e)}
        write_json(path,result);print('CLOCK QC',source,result.get('valid'),result.get('reason'),flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(6) as pool:records=list(pool.map(one,candidates))
    write_jsonl(ROOT/'data/candidate_audit.jsonl',records)
    good=sorted([r['session_id'] for r in records if r['valid']],key=lambda s:hashlib.sha256(f'{SEED}:direct-transfer-source:{s}'.encode()).hexdigest())[:12]
    if len(good)<4:raise ValueError('fewer than four verifiable sessions; cannot legally split source-disjoint train/dev')
    nval=max(2,len(good)//3);val=good[-nval:];train=good[:-nval]
    write_json(ROOT/'locks/data_split.json',{'train_sessions':train,'val_sessions':val,'confirmation_sessions':[],'confirmation_status':'not_run: all bounded available annotation candidates were metadata-inspected in previous run; none may be called newly independent confirmation','participant_ids':None,'primary_isolation':'raw recording/session; participant independence unavailable','no_model_scores_used_to_split':True,'previous_IMU_seen_ledger_SHA':sha256(OLD/'all_seen_sources.jsonl'),'native_label_type':'derived four-choice completed-event action QA, not official QA','timestamps_relative_to_each_audited_8s_episode':[2.,4.,6.,8.],'event_boundary_given_for_task_construction':True})
    # Source roles are fixed before downloading signals / computing features.
    def materialize(source):
        qc=json.loads((ROOT/f'data/probe_{source}.json').read_text());streams={}
        completed=DATA/f'sensors/{source}.npz'
        if use_cached and completed.exists():return source,str(completed),sha256(completed)
        for name in ('Accelerometer_sync.txt','Gyroscope_sync.txt'):
            e=imu[source][name];p=DATA/f'sensors/{source}_{name}';old=OLD_DATA/f'holo/sync_probe/{source}_{name}'
            if p.exists() and p.stat().st_size==e['size']:raw=p.read_bytes()
            elif old.exists():raw=old.read_bytes()
            elif use_cached:return source,None,'not_acquired: complete signal transport failed; retained partial cache'
            else:raw=get_range(SOURCES['holo_imu'],e['offset_data'],e['size'])
            if len(raw)!=e['size']:raise ValueError('reused official stream byte size mismatch')
            p.write_bytes(raw);streams[name]=load_sensor(p,qc['origin_tick'])
        tt,values,original_times=combine(streams['Accelerometer_sync.txt'],streams['Gyroscope_sync.txt']);p=DATA/f'sensors/{source}.npz';np.savez_compressed(p,times=tt,values=values,original_times=original_times)
        return source,str(p),sha256(p)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:signal_records=list(pool.map(materialize,good))
    signals={s:(p,h) for s,p,h in signal_records if p is not None};write_json(ROOT/'data/complete_signal_QC.json',{'records':[{'source':s,'path':p,'SHA_or_failure':h} for s,p,h in signal_records],'QC_before_any_current_model_score':True})
    if set(signals)!=set(good):
        original=ROOT/'locks/data_split_provisional.json'
        if not original.exists():original.write_bytes((ROOT/'locks/data_split.json').read_bytes())
        good=[s for s in good if s in signals];train=[s for s in train if s in signals];val=[s for s in val if s in signals]
        if len(train)<2 or len(val)<2:raise ValueError('minimum source-disjoint actual signals not available')
        split=json.loads(original.read_text());split.update(train_sessions=train,val_sessions=val,availability_QC_pruned_only=True,provisional_split_SHA=sha256(original));write_json(ROOT/'locks/data_split.json',split)
    # Fit physical scales on the actual Router training sessions only.
    training=np.concatenate([np.load(signals[s][0])['values'] for s in train]);limits={'head:acc':float(np.quantile(np.abs(training[:,:3]),.995)*1.1),'head:gyro':float(np.quantile(np.abs(training[:,3:]),.995)*1.1)}
    render={'kind':'P1','y_limits':limits,'size_px':[768,768],'dpi':120,'font_size':11,'line_width':1.5,'scale_fit_role':'actual Router train sessions only; abs99.5percentile*1.1, never per-window'};write_json(ROOT/'locks/IMU_PLOT_V1.json',render)
    output=[]
    for source in good:
        qc=json.loads((ROOT/f'data/probe_{source}.json').read_text());capture=np.load(qc['camera_clock'])['times'];num,den=map(int,qc['video']['rate'].split('/'));rate=num/den
        for i,e in enumerate(eligible(anns[source],vocab)[:16]):
            end_index=int(np.ceil(e['end']*rate));end=float(capture[end_index]);timestamps=[end-6,end-4,end-2,end]
            if timestamps[0]<2:continue
            qid=f'holo:{source}:direct:{i}';verb=e['attributes']['Verb'].strip();noun=e['attributes']['Noun'].strip();rng=np.random.default_rng(int.from_bytes(hashlib.sha256((qid+':ABCD').encode()).digest()[:8],'big'));opts=[verb]+rng.choice([x for x in vocab if x!=verb],3,replace=False).tolist();rng.shuffle(opts)
            row={'dataset':'holoassist_derived_action_QA','role':'train' if source in train else 'val','source_id':'holo:'+source,'session_id':source,'participant_id':None,'qa_id':qid,'question':f'What action on the {noun.replace("_"," ")} is being completed at the end of this observed interval?','options':[f'{c}. {v.replace("_"," ")}' for c,v in zip('ABCD',opts)],'answer_text':'ABCD'[opts.index(verb)],'option_class_ids':opts,'class_id':verb,'class_name':verb,'imu_path':signals[source][0],'imu_sha256':signals[source][1],'timestamps_s':timestamps,'relative_timestamps_s':[2.,4.,6.,8.],'episode_start_s':end-8,'observed_end_s':end,'annotation_media_end_s':e['end'],'annotation_end_frame_index':end_index,'RGB_clock_path':qc['camera_clock'],'annotation_boundary_given':True,'label_source':'official completed fine-grained Verb/Noun event; no future correction labels','no_audio_used':True,'previously_inspected_source':True}
            if all(sensor_window(row,t).valid for t in timestamps):output.append(row)
    for role in ('train','val'):
        if len([r for r in output if r['role']==role])<16:raise ValueError('insufficient legitimate QA after causal QC')
    write_jsonl(ROOT/'data/qa_manifest.jsonl',output)
    write_json(ROOT/'locks/protocol.json',{'seed':SEED,'Router':{'architecture':'unmodified utility_gate.gate_first_native.fusion.GroundedFusionRouter','key_dim':128,'rank':32,'params':696320,'epochs':3,'learning_rate':1e-5,'weight_decay':.01,'clip':1.,'loss':'audited first-answer-token full-vocabulary CE','init':'V1 query/key slices; exactly zero output residual','variants':['R_REAL','R_DUP','R_ZERO'],'always_on':True},'Gate':{'epochs':30,'learning_rate':.001,'weight_decay':.01,'hidden':64,'dropout':.1,'batch_size_episodes':8,'candidates':[{'name':'ranking','loss':'pairwise','lambda':0.},{'name':'classification','loss':'hard_trajectory','lambda':0.},{'name':'cost_005','loss':'soft_trajectory','lambda':.05},{'name':'cost_020','loss':'soft_trajectory','lambda':.2}],'checkpoint_epochs':[1,10,20,30],'thresholds':[0.,.25,.5,.75,1.],'temperature':.25,'no_qa_or_source_in_features':True},'selection':{'accuracy_margin':.01,'NLL_margin':.05,'random_seeds':[1234,2234,3234],'caps':[0,1,2,3,4],'noninferiority':'1pp, source-cluster CI; insufficient-power if too few sessions','accuracy_first':'actual val generated Accuracy, then Gold NLL, then k','cost_aware':'among val policies within1pp Accuracy and+0.05NLL of selected accuracy policy minimize k, then measured overhead; endpoints included','no_ON_target':True},'manifest_sha256':sha256(ROOT/'data/qa_manifest.jsonl'),'scorer':'strict greedy16 generated ABCD, first-answer-token ABCD-normalized NLL, full vocabulary NLL also retained','native':'4 RGB or4 RGB+4 auxiliary images, native token count measured','token_contract':'all compressed strategies4x64=256','independent_confirmation':'not_run if no verified newly uninspected source available; prior metadata-inspected sessions are train/dev only'})
    journal('new legal source-disjoint train/dev manifest and minimum training protocol locked',train_sessions=len(train),val_sessions=len(val),qa=len(output));print('MANIFEST',len(train),len(val),len(output),flush=True);status('M1','data_locked',train_sessions=len(train),val_sessions=len(val),qa=len(output))

def sensor_window(row,t):
    d=np.load(row['imu_path']);lower=max(0.,float(t)-2.)
    # A held value can originate just before the displayed union time. BOTH
    # original sensor samples must satisfy the strict window, including its
    # lower boundary; filtering union times alone is insufficient.
    original=d['original_times'];allowed=(original>=lower).all(-1)&(original<=float(t)).all(-1)&(d['times']>=lower)&(d['times']<=float(t))
    w=causal_window(d['times'][allowed],d['values'][allowed],float(t),['head:'+s+'_'+a for s in ('acc','gyro') for a in 'xyz'],['m/s^2']*3+['rad/s']*3,['head'],2.,30.,'strict original acc+gyro ticks both in[t-2,t]; causal last-observation hold')
    if w.valid:
        assert (original[allowed]>=lower).all() and (original[allowed]<=t).all() and (original[allowed]<=d['times'][allowed,None]).all()
    return w
def frames(shard=0,shards=1,only_source=None):
    allrows=rows();sources=sorted({r['session_id'] for r in allrows});entries={e['name'].split('/')[0]:e for e in json.loads((ROOT/'data/video_index.json').read_text())}
    for source in ([only_source] if only_source is not None else sources[shard::shards]):
        out=ROOT/f'data/frames_{source}.json'
        if out.exists():continue
        selected=[r for r in allrows if r['session_id']==source];stamps=sorted({t for r in selected for t in r['timestamps_s']});capture=np.load(selected[0]['RGB_clock_path'])['times'];e=entries[source];idx=0;previous=None;meta={}
        old=OLD/f'M1_data/holo_frames_{source}.json'
        if old.exists():
            oldmeta=json.loads(old.read_text())
            if all(str(t) in oldmeta for t in stamps):
                for t in stamps:
                    a=oldmeta[str(t)];assert a['capture_time_s']<=t;meta[str(t)]={'path':a['path'],'capture_s':a['capture_time_s'],'media_pts_s':a['media_pts_s'],'frame_index':a['frame_index'],'sha256':a['sha256'],'reused_old_input_only':True}
                write_json(out,meta);print('REUSED VERIFIED CAUSAL RGB INPUTS',source,len(meta),flush=True);continue
        with av.open(RangeFile(SOURCES['holo_video'],e['size'],e['offset_data'])) as c:
            rate=float(c.streams.video[0].average_rate)
            for n,f in enumerate(c.decode(video=0)):
                media=float(f.pts*f.time_base)
                if abs(media-n/rate)>1e-4:raise ValueError('annotation CFR conversion failed')
                current=float(capture[n])
                while idx<len(stamps) and current>stamps[idx]:
                    if previous is None:raise ValueError('no past RGB')
                    t=stamps[idx];p=DATA/f'frames/{source}_{t:.7f}.png';Image.fromarray(previous).save(p);meta[str(t)]={'path':str(p),'capture_s':previous_time,'media_pts_s':previous_media,'frame_index':n-1,'sha256':sha256(p)};idx+=1
                if idx==len(stamps):break
                previous=f.to_ndarray(format='rgb24');previous_time=current;previous_media=media
        if idx!=len(stamps):raise ValueError('incomplete video decode')
        write_json(out,meta);print('CAUSAL RGB FRAMES',source,len(meta),flush=True)
    journal('new source-disjoint causal frames decoded',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['indices','probe','minimal','frames']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();frames(a.shard,a.shards) if a.phase=='frames' else probe(True) if a.phase=='minimal' else globals()[a.phase]()
