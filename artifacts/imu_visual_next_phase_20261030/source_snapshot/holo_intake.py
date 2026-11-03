"""Selective official tar indexing and original-clock HoloAssist diagnostics."""
import argparse,json,tarfile,concurrent.futures
from pathlib import Path
import numpy as np
import av
from PIL import Image
from .common import ROOT,DATA,write_json,write_jsonl,sha256,journal
from .intake import SOURCES
from .range_io import get_range,RangeFile
from .causal_imu_window import times_from_dotnet_ticks

def scan(url,start=0,limit=100,only_kind=None):
    pos=start;out=[];buffer=b'';buffer_start=-1
    for step in range(limit*10):
        if not (buffer_start<=pos and pos+512<=buffer_start+len(buffer)):
            buffer=get_range(url,pos,4096);buffer_start=pos
        block=buffer[pos-buffer_start:pos-buffer_start+512]
        if not any(block):break
        info=tarfile.TarInfo.frombuf(block,encoding='utf-8',errors='strict');body=pos+512
        if info.isfile() and (only_kind is None or info.name.endswith(only_kind)):
            out.append({'name':info.name,'size':info.size,'offset':pos,'offset_data':body});print('INDEX',len(out),info.name,info.size,flush=True)
            if len(out)>=limit:break
        pos=body+((info.size+511)//512)*512
    return out
def indices():
    boundary=json.loads((ROOT/'M1_data/holo_video_binary_search.json').read_text())['validated_mp4_headers'][0]
    videos=scan(SOURCES['holo_video'],boundary['offset'],40,'.mp4');write_json(ROOT/'M1_data/holo_video_subset_index.json',videos)
    # Complete JSON objects only, explicitly not claiming the partial file is complete.
    text=(DATA/'holo/annotations.json.part').read_text();decoder=json.JSONDecoder();cursor=text.find('[')+1;records=[]
    while cursor<len(text):
        while cursor<len(text) and text[cursor] in ' \r\n\t,':cursor+=1
        try:obj,end=decoder.raw_decode(text,cursor)
        except json.JSONDecodeError:break
        records.append(obj);cursor=end
    write_jsonl(DATA/'holo/complete_annotation_records_from_partial.jsonl',records)
    write_json(ROOT/'M1_data/holo_annotation_subset_provenance.json',{'original_url':SOURCES['holo_annotations'],'partial_download_bytes':(DATA/'holo/annotations.json.part').stat().st_size,'complete_objects_recovered':len(records),'recovered_subset_sha256':sha256(DATA/'holo/complete_annotation_records_from_partial.jsonl'),'full_original_annotation_file_downloaded':False,'no_model_score_used_to_choose_records':True})
    times=scan(SOURCES['holo_video'],0,12,'VideoMp4Timing.txt');write_json(ROOT/'M1_data/holo_timing_subset_index.json',times)
    imus=scan(SOURCES['holo_imu'],0,36,'.txt');write_json(ROOT/'M1_data/holo_imu_subset_index.json',imus)
def paired_probe(shard=0,shards=1):
    videos=json.loads((ROOT/'M1_data/holo_video_subset_index.json').read_text());times=json.loads((ROOT/'M1_data/holo_timing_subset_index_v2.json').read_text());imus=json.loads((ROOT/'M1_data/holo_imu_subset_index.json').read_text());v={r['name'].split('/')[0]:r for r in videos};tt={r['name'].split('/')[0]:r for r in times};groups={}
    for r in imus:groups.setdefault(r['name'].split('/')[0],{})[Path(r['name']).name]=r
    common=[s for s in v if s in tt and {'Accelerometer_sync.txt','Gyroscope_sync.txt'}<=groups.get(s,{}).keys()][:10]
    assert len(common)>=10, 'At least ten matched sessions required; do not silently lower the intake target'
    lock=ROOT/'M1_data/holo_sync_probe_source_lock.json'
    if lock.exists():assert json.loads(lock.read_text())['sources']==common
    else:write_json(lock,{'sources':common,'selection':'first10 authentic matched official archive sessions before any model prediction; availability-only','participant_ID':'not inferred from recording IDs'})
    outdir=DATA/'holo/sync_probe';outdir.mkdir(exist_ok=True)
    for source in common[shard::shards]:
        if (ROOT/f'M1_data/holo_sync_{source}.json').exists():continue
        td=tt[source];ticks=[int(x) for x in get_range(SOURCES['holo_video'],td['offset_data'],td['size']).decode().split()];record={'source_id':'holo:'+source,'participant_id':None,'session_id':source,'video_start_tick':ticks[0],'video_end_tick':ticks[1],'physical_duration_s':(ticks[1]-ticks[0])/1e7,'sensor_streams':{},'subject_disjoint_claim':False}
        for name in ('Accelerometer_sync.txt','Gyroscope_sync.txt'):
            entry=groups[source][name];raw=get_range(SOURCES['holo_imu'],entry['offset_data'],entry['size']);p=outdir/(source+'_'+name);p.write_bytes(raw)
            lines=[x.split() for x in raw.decode().splitlines() if x.strip()];sensor_tick=np.array([int(x[1]) for x in lines],np.int64);relative=times_from_dotnet_ticks(sensor_tick,ticks[0]);rgb=np.array([float(x[0]) for x in lines]);values=np.array([[float(y) for y in x[2:5]] for x in lines]);diff=relative-rgb
            record['sensor_streams'][name]={'path':str(p),'sha256':sha256(p),'rows':len(lines),'explicit_original_sensor_ticks':True,'first_relative_s':float(relative[0]),'last_relative_s':float(relative[-1]),'median_source_interval_s':float(np.median(np.diff(relative))),
                'sync_row_sensor_later_than_RGB_nominal_count':int((diff>1e-6).sum()),'sensor_minus_RGB_nominal_range_s':[float(diff.min()),float(diff.max())],'finite':bool(np.isfinite(values).all()),'norm_median':float(np.median(np.linalg.norm(values,axis=1)))}
        entry=v[source];stream=RangeFile(SOURCES['holo_video'],entry['size'],entry['offset_data']);frames=[]
        with av.open(stream,mode='r') as c:
            vs=c.streams.video[0];record['video_metadata']={'width':vs.width,'height':vs.height,'average_rate':str(vs.average_rate),'time_base':str(vs.time_base),'duration_ticks':vs.duration,'frames':vs.frames}
            for i,f in enumerate(c.decode(video=0)):
                pts=float(f.pts*f.time_base)
                if i in (0,30,60,120,180,240):
                    p=outdir/f'{source}_{i}.png';Image.fromarray(f.to_ndarray(format='rgb24')).save(p);frames.append({'pts_s':pts,'frame_index':i,'path':str(p),'sha256':sha256(p)})
                if pts>=8:break
        record['sampled_RGB']=frames;write_json(ROOT/f'M1_data/holo_sync_{source}.json',record);print('PAIRED REAL HOLO',source,record,flush=True)
    journal('actual HoloAssist original-tick and RGB paired probes completed',sources=len(common[shard::shards]),shard=shard,shards=shards)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['indices','paired_probe']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();globals()[a.phase](a.shard,a.shards) if a.phase=='paired_probe' else indices()
