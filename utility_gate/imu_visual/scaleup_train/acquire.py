"""Incremental official acquisition, independent of model predictions."""
import argparse, concurrent.futures as cf, json, tarfile, hashlib
from pathlib import Path
import numpy as np
import av
from .common import ROOT,DATA,OLD_DATA,write_json,write_jsonl,sha256,append_jsonl,now
from .range_io import get_range,RangeFile
from .data import eligible
from utility_gate.imu_visual.intake import SOURCES
from utility_gate.imu_visual.holo_probe import load_sensor,combine

V3=ROOT.parent/'imu_gate_router_transfer_20261030_v3'
V1=ROOT.parent/'imu_gate_router_transfer_20261030_v1'

def indices(kind):
    prior=json.loads((V1/f'data/{kind}_index.json').read_text())
    out=ROOT/f'data/{kind}_index.json'
    records=json.loads(out.read_text()) if out.exists() else prior
    last=records[-1];pos=last['offset_data']+(last['size']+511)//512*512
    limit={'video':400,'camera':300,'imu':450}[kind]
    url=SOURCES['holo_imu' if kind=='imu' else 'holo_video'];suffix='.mp4' if kind=='video' else '.txt'
    write_json(out,records)
    while len(records)<limit:
        raw=get_range(url,pos,512)
        if not any(raw):break
        h=tarfile.TarInfo.frombuf(raw,'utf-8','strict');body=pos+512
        if h.isfile() and h.name.endswith(suffix):
            records.append({'name':h.name,'size':h.size,'offset':pos,'offset_data':body})
            write_json(out,records)
            print('INDEX',kind,len(records),h.name,flush=True)
        pos=body+(h.size+511)//512*512

def annotations():
    # Extend the verified original prefix. Only complete JSON objects accepted.
    old=OLD_DATA/'holo/annotations.json.part';begin=old.stat().st_size;end=36*1024**2
    chunks=[(p,min(262144,end-p)) for p in range(begin,end,262144)]
    def one(pair):return get_range(SOURCES['holo_annotations'],*pair)
    with cf.ThreadPoolExecutor(12) as pool:parts=list(pool.map(one,chunks))
    raw=old.read_bytes()+b''.join(parts);p=DATA/'raw/annotations_36MiB_prefix.json.part';p.write_bytes(raw)
    decoder=json.JSONDecoder();text=raw.decode();i=text.index('[')+1;records=[]
    while i<len(text):
        while i<len(text) and text[i] in ' \r\n\t,':i+=1
        try:record,j=decoder.raw_decode(text,i)
        except json.JSONDecodeError:break
        if not isinstance(record,dict):break
        records.append(record);i=j
    write_jsonl(ROOT/'data/official_annotation_complete_records.jsonl',records)
    write_json(ROOT/'M1_download_sync/annotation_acquisition.json',{'url':SOURCES['holo_annotations'],'bytes':len(raw),'sha256':sha256(p),'complete_records':len(records),'full_file':False,'truncated_tail_excluded':True})
    print('ANNOTATIONS',len(records),flush=True)

def pair(source,video,camera,imu):
    target=ROOT/f'data/probe_{source}.json'
    if target.exists():return json.loads(target.read_text())
    provenance=[]
    def read(kind,name):
        e=(camera if kind=='video' else imu)[source][name];url=SOURCES['holo_'+kind]
        raw=get_range(url,e['offset_data'],e['size']);p=DATA/'raw'/source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        provenance.append({'path':str(p),'URL':url,'entry':e['name'],'bytes':len(raw),'sha256':sha256(p),'range_start':e['offset_data']})
        return p
    try:
        origin=int(read('video','VideoMp4Timing.txt').read_text().split()[0]);pose=read('video','Pose_sync.txt')
        ticks=np.array([int(x.split()[1]) for x in pose.read_text().splitlines() if x.strip()],dtype=np.int64)
        if (np.diff(ticks)<0).any():raise ValueError('nonmonotonic camera')
        ticks=np.unique(ticks);clock=(ticks-origin).astype(np.float64)/1e7;cp=DATA/f'clock/{source}.npz';np.savez_compressed(cp,ticks=ticks,times=clock)
        e=video[source]
        with av.open(RangeFile(SOURCES['holo_video'],e['size'],e['offset_data'],block_size=65536)) as c:
            st=c.streams.video[0];meta={'frame_count':st.frames,'rate':str(st.average_rate),'width':st.width,'height':st.height,'duration_s':float(st.duration*st.time_base)}
        if len(ticks)!=meta['frame_count']:raise ValueError('camera ticks/frame count mismatch; no guessed offset')
        acc=load_sensor(read('imu','Accelerometer_sync.txt'),origin);gyro=load_sensor(read('imu','Gyroscope_sync.txt'),origin)
        tt,values,orig=combine(acc,gyro)
        if not np.isfinite(values).all():raise ValueError('nonfinite sensors')
        sp=DATA/f'sensors/{source}.npz';np.savez_compressed(sp,times=tt,values=values,original_times=orig)
        result={'session_id':source,'participant_id':None,'valid':True,'clock_grade':'B','reason':'official synchronized streams retain original sample ticks; exact unique RGB tick/frame mapping; original acquisition precision uncertified','origin_tick':origin,'camera_clock':str(cp),'camera_clock_sha256':sha256(cp),'unique_camera_ticks':len(ticks),'video':meta,'imu_path':str(sp),'imu_sha256':sha256(sp),'sensor_rows':len(tt),'span_s':float(tt[-1]-tt[0]),'max_gap_s':float(np.diff(tt).max()),'raw_provenance':provenance}
    except Exception as e:
        result={'session_id':source,'valid':False,'reason':type(e).__name__+': '+str(e),'raw_provenance':provenance,'clock_grade':'Rejected' if 'mismatch' in str(e) or 'nonmonotonic' in str(e) else 'not_acquired'}
        append_jsonl(ROOT/'M1_download_sync/FAILED_DOWNLOADS.jsonl',{'utc':now(),**result})
    write_json(target,result);print('PAIR QC',source,result['valid'],result['reason'],flush=True);return result

def pairs(fresh_only=False):
    from .common import read_jsonl
    anns={r['video_name']:r for r in read_jsonl(ROOT/'data/official_annotation_complete_records.jsonl')}
    from .common import OLD
    vocab=json.loads((OLD/'M2_frozen_imu_plots/holo_renderer_lock.json').read_text())['vocabulary']
    video={r['name'].split('/')[0]:r for r in json.loads((ROOT/'data/video_index.json').read_text())}
    maps=[]
    for kind in ['camera','imu']:
        d={}
        combined=json.loads((ROOT/f'data/{kind}_index.json').read_text())
        for partial in sorted((ROOT/'data').glob(f'{kind}_sparse_*.json')):combined+=json.loads(partial.read_text())
        for r in combined:d.setdefault(r['name'].split('/')[0],{})[Path(r['name']).name]=r
        maps.append(d)
    camera,imu=maps;seen={r['source_id'] for r in read_jsonl(ROOT/'locks/historical_seen_sources.jsonl')}
    candidates=[s for s in video if s in anns and s in camera and s in imu and len(eligible(anns[s],vocab))>=4 and {'Pose_sync.txt','VideoMp4Timing.txt'}<=camera[s].keys() and {'Accelerometer_sync.txt','Gyroscope_sync.txt'}<=imu[s].keys()]
    # Newly acquired sources prioritized, deterministic and score-free.
    prior_main={'R005-7July-GoPro','R007-7July-DSLR','R012-7July-Nespresso','R0027-12-GoPro'}
    if fresh_only:candidates=[s for s in candidates if 'holo:'+s not in seen]
    candidates=sorted(set(candidates)-prior_main,key=lambda s:(('holo:'+s) in seen,hashlib.sha256(('1234:scaleup:'+s).encode()).hexdigest()))[:70]
    append_jsonl(ROOT/'M1_download_sync/intake_candidate_versions.jsonl',{'utc':now(),'sources':candidates})
    write_json(ROOT/'M1_download_sync/intake_candidate_lock.json',{'utc':now(),'sources':candidates,'no_model_outputs':True,'target_qualified_sessions':50,'rule':'>=4 eligible official events; archive availability; new-source-first deterministic hash'})
    with cf.ThreadPoolExecutor(5) as pool:results=list(pool.map(lambda s:pair(s,video,camera,imu),candidates))
    by={r['session_id']:r for r in results}
    for previous in (ROOT/'data').glob('probe_*.json'):
        q=json.loads(previous.read_text());by.setdefault(q['session_id'],q)
    results=list(by.values())
    write_jsonl(ROOT/'M1_download_sync/SYNC_AUDIT.jsonl',results)
    write_jsonl(ROOT/'M1_download_sync/ACCEPTED_RECORDINGS.jsonl',[r for r in results if r['valid']])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['video','camera','imu','annotations','pairs','pairs_fresh']);a=p.parse_args()
    indices(a.phase) if a.phase in ['video','camera','imu'] else pairs(True) if a.phase=='pairs_fresh' else globals()[a.phase]()
