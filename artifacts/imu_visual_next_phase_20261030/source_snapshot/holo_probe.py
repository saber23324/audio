"""Original-clock, session-disjoint HoloAssist development diagnostics."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
import av
from PIL import Image
from utility_gate.grounded_dual_image.native import prepare
from .common import ROOT,DATA,read_jsonl,write_json,write_jsonl,sha256,journal
from .range_io import get_range,RangeFile
from .intake import SOURCES
from .causal_imu_window import times_from_dotnet_ticks,causal_window
from .native_imu_probe import prompt
from .render_imu_plot import render_imu_plot
from .runtime import VisionRuntime

def load_sensor(path,origin):
    lines=[x.split() for x in Path(path).read_text().splitlines() if x.strip()]
    ticks=np.array([int(x[1]) for x in lines],np.int64);v=np.array([[float(y) for y in x[2:5]] for x in lines]);ts=times_from_dotnet_ticks(ticks,origin)
    # Released synchronized streams can repeat an original sensor sample.
    keep=np.r_[True,np.diff(ticks)>0]
    for i in np.flatnonzero(np.diff(ticks)==0):
        if not np.array_equal(v[i],v[i+1]):raise ValueError('same sensor tick has different values')
    if np.any(np.diff(ticks)<0):raise ValueError('nonmonotonic original sensor clock')
    return ts[keep],v[keep]

def combine(acc,gyro):
    # Causal zero-order hold at the union of original sample times, not
    # linear interpolation with a later sample. Both originals are retained.
    ta,va=acc;tg,vg=gyro;tt=np.union1d(ta,tg);ia=np.searchsorted(ta,tt,side='right')-1;ig=np.searchsorted(tg,tt,side='right')-1
    ok=(ia>=0)&(ig>=0)&(tt>=0);tt=tt[ok];ia=ia[ok];ig=ig[ok]
    origins=np.column_stack((ta[ia],tg[ig]));assert (origins<=tt[:,None]).all()
    return tt,np.column_stack((va[ia],vg[ig])),origins

def window(row,t,W=2.,shift=0.):
    d=np.load(row['imu_path']);end=t-shift
    w=causal_window(d['times'],d['values'],end,['head:'+s+'_'+a for s in ('acc','gyro') for a in 'xyz'],['m/s^2']*3+['rad/s']*3,['head'],W,30.,'original .NET sensor ticks relative to MPEG origin; causal last-observation hold')
    if not w.valid:raise ValueError(row['source_id']+': '+w.reason)
    selected=(d['times']>=max(0,end-W))&(d['times']<=end)
    assert np.all(d['original_times'][selected]<=d['times'][selected,None]) and d['original_times'][selected].max()<=end
    return w

def build():
    if (ROOT/'M1_data/holo_probe_manifest.jsonl').exists():raise FileExistsError('manifest already locked')
    lock=json.loads((ROOT/'M1_data/holo_sync_probe_source_lock.json').read_text());anns={x['video_name']:x for x in read_jsonl(DATA/'holo/complete_annotation_records_from_partial.jsonl')};entries=json.loads((ROOT/'M1_data/holo_imu_subset_index.json').read_text())
    # Only two completely separate calibration recordings set physical scales.
    scale_sources=['R016-7July-Navvis','R028-12July-Switch'];scale={s:[] for s in ('acc','gyro')}
    for e in entries:
        source=e['name'].split('/')[0];sensor='acc' if 'Accelerometer' in e['name'] else 'gyro' if 'Gyroscope' in e['name'] else None
        if source not in scale_sources or sensor is None:continue
        raw=get_range(SOURCES['holo_imu'],e['offset_data'],e['size']);path=DATA/'holo'/('calibration_'+source+'_'+sensor+'.txt');path.write_bytes(raw)
        values=np.array([[float(v) for v in line.split()[2:5]] for line in raw.decode().splitlines() if line.strip()]);scale[sensor].append(values)
    limits={s:float(np.quantile(np.abs(np.concatenate(v)),.995)*1.1) for s,v in scale.items()}
    channels={f'head:{s}':v for s,v in limits.items()};config={'kind':'P1','y_limits':channels,'size_px':[768,768],'dpi':120,'font_size':11,'line_width':1.5,'scale_fit_role':'two separate calibration-only sessions, fixed abs99.5percentile*1.1'}
    manifest=[];qa_dir=DATA/'holo/qa_frames';qa_dir.mkdir(exist_ok=True)
    # Label vocabulary is fixed from other released annotation sessions only.
    vocab=sorted({e['attributes'].get('Verb','').strip() for s,a in anns.items() if s not in lock['sources'] for e in a['events'] if e['label']=='Fine grained action'}-{'','none','other','others'})
    for source in lock['sources']:
        if source not in anns:continue
        sync=json.loads((ROOT/f'M1_data/holo_sync_{source}.json').read_text());clock=json.loads((ROOT/f'M1_data/holo_rgb_clock_{source}.json').read_text())
        if not clock['one_to_one_count_equal']:continue
        capture=np.load(clock['capture_clock_path'])['times'];num,den=map(int,clock['MP4_rate'].split('/'));rate=num/den
        origin=sync['video_start_tick'];acc=load_sensor(sync['sensor_streams']['Accelerometer_sync.txt']['path'],origin);gyro=load_sensor(sync['sensor_streams']['Gyroscope_sync.txt']['path'],origin);tt,v,orig=combine(acc,gyro);path=DATA/'holo'/('causal_'+source+'.npz');np.savez_compressed(path,times=tt,values=v,original_times=orig)
        fine=[e for e in anns[source]['events'] if e['label']=='Fine grained action'];eligible=[]
        for e in sorted(fine,key=lambda x:(x['end'],x['start'])):
            verb=e['attributes'].get('Verb','').strip();noun=e['attributes'].get('Noun','').strip()
            if not(8<=e['end']<=150 and e['end']-e['start']>=1 and verb in vocab and noun not in ('','none','other','others')):continue
            ambiguous=any(q is not e and q['attributes'].get('Noun')==noun and q['attributes'].get('Verb')!=verb and q['start']<e['end'] and q['end']>e['end']-.15 for q in fine)
            if ambiguous:continue
            eligible.append(e)
            if len(eligible)==16:break
        for i,e in enumerate(eligible):
            sid='holo:'+source;verb=e['attributes']['Verb'].strip();noun=e['attributes']['Noun'].strip();qid=f'{sid}:completed:{i}';rng=np.random.default_rng(int.from_bytes(hashlib.sha256(qid.encode()).digest()[:8],'big'));choices=[verb]+rng.choice([x for x in vocab if x!=verb],3,replace=False).tolist();rng.shuffle(choices)
            end_frame=int(np.ceil(float(e['end'])*rate));end=float(capture[end_frame]);times=[end-6,end-4,end-2,end]
            if times[0]<2:continue
            row={'dataset':'holoassist_completed_action_derived_QA','role':'development_only','source_id':sid,'session_id':source,'participant_id':None,'qa_id':qid,'imu_path':str(path),'imu_sha256':sha256(path),'timestamps_s':times,'episode_start_s':float(e['end'])-8,'observed_end_s':float(e['end']),'target_event_start':e['start'],'target_event_end':e['end'],'class_id':verb,'class_name':verb,'noun':noun,'question':f'What action on the {noun.replace("_"," ")} is being completed at the end of this observed interval?','options':[f'{c}. {v.replace("_"," ")}' for c,v in zip('ABCD',choices)],'option_class_ids':choices,'answer_text':'ABCD'[choices.index(verb)],'answer_source':'official fine-grained action interval and Verb/Noun; derived4-choice QA, not official HoloAssist QA; completed event only','future_mistake_correction_labels_used':False,'subject_disjoint_claim':False,'session_disjoint_from_scale_fit':True,'original_tick_filter':True,'no_audio_used':True}
            row.update(observed_end_s=end,episode_start_s=end-8,annotation_media_end_s=e['end'],annotation_end_frame_index=end_frame,annotation_boundary_given_for_task_construction=True,RGB_clock_path=clock['capture_clock_path'])
            for t in times:window(row,t)
            manifest.append(row)
    if not manifest:raise ValueError('no legitimate completed-action QA')
    write_jsonl(ROOT/'M1_data/holo_probe_manifest.jsonl',manifest)
    write_json(ROOT/'M2_frozen_imu_plots/holo_renderer_lock.json',{'config':config,'calibration_sessions':scale_sources,'development_sessions':sorted({r['session_id'] for r in manifest}),'vocabulary':vocab,'manifest_sha256':sha256(ROOT/'M1_data/holo_probe_manifest.jsonl'),'no_model_scores_used':True,'W_primary':2,'representations':['P1','P2'],'prompts':['sensor_instructions','neutral'],'plan':'I0-I7 plus preregistered P2 sensor and P1 neutral (only/real/zero/wrong). No additional search. No independent confirmation claim.'})
    journal('Holo development manifest and renderer locked before model scoring',qa=len(manifest),sessions=len({r['session_id'] for r in manifest}))
    print('HOLO MANIFEST',len(manifest),flush=True)

def frames(shard=0,shards=1):
    rows=read_jsonl(ROOT/'M1_data/holo_probe_manifest.jsonl');sources=sorted({r['session_id'] for r in rows});entries={e['name'].split('/')[0]:e for e in json.loads((ROOT/'M1_data/holo_video_subset_index.json').read_text())};directory=DATA/'holo/qa_frames'
    for source in sources[shard::shards]:
        target=ROOT/f'M1_data/holo_frames_{source}.json'
        if target.exists():continue
        stamps=sorted({t for r in rows if r['session_id']==source for t in r['timestamps_s']});out={};e=entries[source];idx=0;previous=None;prevtime=None;capture=np.load(DATA/'holo'/('rgb_clock_'+source+'.npz'))['times'];fps=None
        with av.open(RangeFile(SOURCES['holo_video'],e['size'],e['offset_data'])) as c:
            fps=float(c.streams.video[0].average_rate)
            for frame_index,f in enumerate(c.decode(video=0)):
                media_pts=float(f.pts*f.time_base)
                if abs(media_pts-frame_index/fps)>1e-4:raise ValueError('CFR assumption for annotation conversion failed')
                current=float(capture[frame_index])
                while idx<len(stamps) and current>stamps[idx]:
                    if previous is None:raise ValueError('no causal RGB')
                    t=stamps[idx];p=directory/f'{source}_{t:.6f}.png';Image.fromarray(previous).save(p);out[str(t)]={'path':str(p),'pts_s':prevtime,'capture_time_s':prevtime,'requested_s':t,'frame_index':previous_index,'media_pts_s':previous_media,'sha256':sha256(p)};idx+=1
                if idx==len(stamps):break
                previous=f.to_ndarray(format='rgb24');prevtime=current;previous_index=frame_index;previous_media=media_pts
        if idx!=len(stamps):raise ValueError('short RGB stream')
        write_json(target,out);print('CAUSAL RGB',source,len(out),flush=True)

def variants():
    out=[('H0_RGB','rgb','P1','sensor_instructions'),('H0_V1','v1','P1','sensor_instructions'),('H1_plot','only','P1','sensor_instructions'),('H2_real','real','P1','sensor_instructions'),('H3_duplicate','duplicate','P1','sensor_instructions'),('H4_zero','zero','P1','sensor_instructions'),('H5_wrong','wrong','P1','sensor_instructions'),('H6_past_shift','shift','P1','sensor_instructions'),('H7_numeric_text','numeric','P1','neutral')]
    out.extend([('H1_zero_plot','zero_only','P1','sensor_instructions'),('H1_wrong_plot','wrong_only','P1','sensor_instructions'),('H1_prior_no_image','prior','P1','sensor_instructions')])
    for kind,style in [('P2','sensor_instructions'),('P1','neutral')]:
        out.extend([(f'Hs_{kind}_{style}_{view}',view,kind,style) for view in ('only','real','zero','wrong')])
    return out

@torch.inference_mode()
def run(shard=0,shards=1):
    rows=read_jsonl(ROOT/'M1_data/holo_probe_manifest.jsonl');cfg=json.loads((ROOT/'M2_frozen_imu_plots/holo_renderer_lock.json').read_text())['config'];r=VisionRuntime();outdir=ROOT/'M2_frozen_imu_plots/holo';outdir.mkdir(exist_ok=True)
    for row in rows[shard::shards]:
        target=outdir/(row['qa_id'].replace(':','_')+'.jsonl')
        if target.exists():continue
        fr=json.loads((ROOT/f'M1_data/holo_frames_{row["session_id"]}.json').read_text());ims=[np.array(Image.open(fr[str(t)]['path']).convert('RGB')) for t in row['timestamps_s']];candidates=[x for x in rows if x['session_id']!=row['session_id'] and x['class_id']!=row['class_id']];donor=min(candidates,key=lambda x:hashlib.sha256((row['qa_id']+x['qa_id']).encode()).hexdigest());results=[];cache={}
        for name,view,kind,style in variants():
            only=view in ('only','zero_only','wrong_only','prior');base_view=view.removesuffix('_only')
            before=dict(r.calls);start=time.perf_counter();images=list(ims);meta=None;w=window(donor if base_view=='wrong' else row,(donor if base_view=='wrong' else row)['observed_end_s'],shift=.5 if view=='shift' else 0.)
            if view not in ('rgb','v1','duplicate','numeric','prior'):
                key=(view,kind)
                if key not in cache:cache[key]=render_imu_plot(w,{**cfg,'kind':kind},zero=base_view=='zero')
                image,meta=cache[key];images=[np.array(image)] if only else images+[np.array(image)]
            elif view=='duplicate':images.append(ims[-1])
            elif view in ('numeric','prior'):images=[]
            ids=prompt(r,row,len(images),style,only=only or view=='numeric')
            if view=='numeric':
                # Separate alternative, with correct HoloAssist gyro units.
                text='IMU times relative to observation (s), head acceleration xyz (m/s^2), head angular velocity xyz (rad/s):\n'+'\n'.join(' '.join([f'{t-w.decision_time:.3f}']+[f'{x:.3f}' for x in v]) for t,v in zip(w.times,w.values));alternative={**row,'question':text+'\n'+row['question']};ids=prompt(r,alternative,0,'neutral',only=True);score=r.answer(ids,[],row['answer_text'],mcqa=True)
            elif view=='prior':score=r.answer(ids,[],row['answer_text'],mcqa=True)
            elif view=='v1':
                vv=torch.stack([r.visual_frame(f) for f in ims]);z=r.vision_latents(vv);ids=prompt(r,row,1,style);score=r.answer(ids,[z],row['answer_text'],mcqa=True)
            else:
                embeds,views,native=prepare(r,ids,images);score=r.answer(ids,views,row['answer_text'],prepared=embeds,mcqa=True)
            results.append({**row,**score,'variant':name,'plot_kind':kind,'prompt_style':style,'plot_metadata':meta,'donor_qa_id':donor['qa_id'] if base_view=='wrong' else None,'RGB_pts_s':[fr[str(t)]['pts_s'] for t in row['timestamps_s']],'actual_forward_hooks':{k:r.calls[k]-before[k] for k in before},'measured_eval_s':time.perf_counter()-start,'cost_not_isolated_pipeline':True,'no_training':True,'plot_image_count':int(view not in ('rgb','v1','duplicate','numeric','prior')),'predicted_class_id':row['option_class_ids']['ABCD'.index(score['text'])] if score['text'] in ('A','B','C','D') else None})
        write_jsonl(target,results);print('HOLO FROZEN',row['qa_id'],flush=True)
    journal('actual frozen HoloAssist development probe complete',shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','frames','run']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();build() if a.phase=='build' else globals()[a.phase](a.shard,a.shards)
