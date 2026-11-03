"""Authentic UTD recording-level smoke; no invented sensor timestamps/splits."""
import argparse,json,hashlib,io,re,zipfile,concurrent.futures
import numpy as np
from scipy.io import loadmat
from PIL import Image
from .common import ROOT,DATA,sha256,write_json,write_jsonl,read_jsonl,journal
from .causal_imu_window import causal_window
from .render_imu_plot import render_imu_plot

LABELS=['right arm swipe left','right arm swipe right','right hand wave','two hand front clap','right arm throw','cross arms in chest','basketball shoot','right hand draw X','right hand draw clockwise circle','right hand draw counterclockwise circle','right hand draw triangle','right hand bowling','front boxing','baseball swing from right','right hand tennis forehand swing','two arm curl','tennis serve','two hand push','right hand knock on door','right hand catch object','right hand pick up and throw','jogging in place','walking in place','sit to stand','stand to sit','left foot forward lunge','squat with arms stretched']
def identity(name):
    m=re.search(r'a(\d+)_s(\d+)_t(\d+)',name)
    return tuple(map(int,m.groups()))
def prepare():
    z=zipfile.ZipFile(DATA/'utd/Inertial.zip');archive=[];arrays={}
    for name in z.namelist():
        if not name.endswith('.mat'):continue
        action,subject,trial=identity(name);p=loadmat(io.BytesIO(z.read(name)));v=p['d_iner'];arrays[action,subject,trial]=v;archive.append({'source_id':f'utd:a{action}_s{subject}_t{trial}','participant_id':f'utd:s{subject}','session_id':f'utd:a{action}_s{subject}_t{trial}',
            'rows':len(v),'channels':v.shape[1],'exported_keys':[k for k in p if not k.startswith('__')],'nominal_duration_s':len(v)/50,'timestamp_exported':False,'original_archive_entry':name})
    write_jsonl(ROOT/'M1_data/utd_all_inertial_audit.jsonl',archive)
    summary={'recordings':len(archive),'participants':len({r['participant_id'] for r in archive}),'min_nominal_duration_s':min(r['nominal_duration_s'] for r in archive),'max_nominal_duration_s':max(r['nominal_duration_s'] for r in archive),
        'nominal_duration_ge8':sum(r['nominal_duration_s']>=8 for r in archive),'explicit_sensor_timestamp_fields':False,
        'clock_limit':'author reports timestamps at acquisition; distributed MAT files inspected here only expose d_iner[N,6]. Nominal50Hz index is not verified original video-clock timing. No strict8s streaming training is justified from this alone.',
        'class_and_units_source':'official UTD-MHAD page and Sample_Code/show_inertial.m: columns1-3 acceleration g,4-6 gyro deg/s; wrist actions1-21,thigh22-27'}
    write_json(ROOT/'M1_data/utd_timing_feasibility.json',summary)
    # Only independent earlier training takes fit fixed physical scaling.
    train=[v for (a,s,t),v in arrays.items() if s in (1,3,5) and t in (2,3,4)];physical=np.concatenate(train);physical[:,:3]*=9.80665
    limits={'acc':float(np.quantile(np.abs(physical[:,:3]),.995)*1.1),'gyro':float(np.quantile(np.abs(physical[:,3:]),.995)*1.1)}
    rows=[];directory=DATA/'utd/probe';directory.mkdir(parents=True,exist_ok=True)
    for (action,subject,trial),v0 in sorted(arrays.items()):
        if subject not in (1,3,5,7) or trial!=1:continue
        source=f'utd:a{action}_s{subject}_t{trial}';video=DATA/f'utd/RGB/a{action}_s{subject}_t{trial}_color.avi'
        if not video.exists():continue
        position='right_wrist' if action<=21 else 'right_thigh';values=v0.astype(np.float64).copy();values[:,:3]*=9.80665;times=np.arange(len(values),dtype=np.float64)/50
        pool=list(range(1,22)) if action<=21 else list(range(22,28));other=[a for a in pool if a!=action];rng=np.random.default_rng(int.from_bytes(hashlib.sha256((source+':options').encode()).digest()[:8],'big'));choices=[action]+rng.choice(other,3,replace=False).tolist();rng.shuffle(choices)
        out=directory/(source.replace(':','_')+'.npz');np.savez_compressed(out,times=times,values=values)
        row={'dataset':'utd_mhad_recording_smoke','role':'development_smoke_only','source_id':source,'participant_id':f'utd:s{subject}','session_id':source,'subject':subject,'trial':trial,'class_id':action,'class_name':LABELS[action-1],
            'video_path':str(video),'video_sha256':sha256(video),'imu_path':str(out),'imu_sha256':sha256(out),'imu_rate_hz':50.,'imu_channels':[position+':'+s+'_'+a for s in ('acc','gyro') for a in 'xyz'],
            'imu_unit_map':['m/s^2']*3+['deg/s']*3,'imu_sensor_positions':[position],'sensor_clock':'nominal50Hz sample index; explicit capture timestamps absent','strict_RGB_IMU_sync_verified':False,
            'timestamps_s':times[-1].item(),'question':'Which action was performed in this completed recording?','options':[f'{c}. {LABELS[a-1]}' for c,a in zip('ABCD',choices)],'option_class_ids':choices,'answer_text':'ABCD'[choices.index(action)],
            'answer_source':'official full-recording action label; derived4choice QA, distractors within same sensor-placement group','scope':'offline completed-recording sanity; no8s streaming/advance-prediction claim','oracle':False}
        rows.append(row)
    write_jsonl(ROOT/'M1_data/utd_probe_manifest.jsonl',rows)
    config={'kind':'P1','y_limits':{pos+':'+s:limits[s] for pos in ('right_wrist','right_thigh') for s in ('acc','gyro')},'size_px':[768,768],'dpi':120,'font_size':11,'line_width':1.5,
        'scale_fit_role':'subjects1,3,5 trials2,3,4 only, no probe trial1 or subject7;99.5percentile absolute times1.1','scale_source_sha256':sha256(DATA/'utd/Inertial.zip')}
    write_json(ROOT/'M2_frozen_imu_plots/utd_renderer_lock.json',{'config':config,'W_primary':2.,'W_sensitivity':4.,'representations':['P1','P2'],'prompt_variants':['sensor_instructions','neutral'],
        'smoke_sources':len(rows),'source_sha256':sha256(ROOT/'M1_data/utd_probe_manifest.jsonl'),'not_a_strict_sync_bridge_claim':True,'no_trainable_model':True})
    print(summary,'probe_recordings',len(rows),flush=True);journal('real UTD sensor/raw RGB smoke prepared with explicit clock limitation',sources=len(rows),timing=summary)
if __name__=='__main__':prepare()
