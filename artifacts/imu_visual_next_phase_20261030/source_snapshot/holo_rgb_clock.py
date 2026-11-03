"""Recover RGB capture time from retained camera pose ticks, not MP4 PTS."""
import json
import numpy as np
from .common import ROOT,DATA,write_json,sha256,journal
from .holo_intake import scan
from .intake import SOURCES
from .range_io import get_range
from .causal_imu_window import times_from_dotnet_ticks

def audit():
    path=ROOT/'M1_data/holo_pose_subset_index.json'
    if path.exists():entries=json.loads(path.read_text())
    else:entries=scan(SOURCES['holo_video'],0,12,'Pose_sync.txt');write_json(path,entries)
    by_source={e['name'].split('/')[0]:e for e in entries};sources=json.loads((ROOT/'M1_data/holo_sync_probe_source_lock.json').read_text())['sources'];summary=[]
    for source in sources:
        p=ROOT/f'M1_data/holo_sync_{source}.json'
        if not p.exists():continue
        sync=json.loads(p.read_text());e=by_source[source];raw=get_range(SOURCES['holo_video'],e['offset_data'],e['size']);out=DATA/'holo'/('rgb_pose_'+source+'.txt');out.write_bytes(raw)
        ticks=np.array([int(line.split()[1]) for line in raw.decode().splitlines() if line.strip()],np.int64)
        if np.any(np.diff(ticks)<0):raise ValueError('camera capture clock not monotonic')
        ticks=np.unique(ticks);frames=sync['video_metadata']['frames'];count_equal=len(ticks)==frames
        relative=times_from_dotnet_ticks(ticks,sync['video_start_tick']);clock=DATA/'holo'/('rgb_clock_'+source+'.npz');np.savez_compressed(clock,ticks=ticks,times=relative)
        row={'session_id':source,'pose_sync_sha256':sha256(out),'retained_unique_original_camera_ticks':len(ticks),'MP4_frame_count':frames,'one_to_one_count_equal':count_equal,'capture_clock_path':str(clock),'capture_clock_sha256':sha256(clock),'origin_tick':sync['video_start_tick'],'MP4_rate':sync['video_metadata']['average_rate'],'first_capture_s':float(relative[0]),'last_capture_s':float(relative[-1]),'camera_interval_range_s':[float(np.diff(relative).min()),float(np.diff(relative).max())],'important':'MP4 is constant-rate packed frames; PTS is not original acquisition time. Use unique monotonic camera ticks in frame order only when exact count matches. Annotation-media-time is converted to frame index then capture time. No camera index/30 assumption.'}
        write_json(ROOT/f'M1_data/holo_rgb_clock_{source}.json',row);summary.append(row);print('CAPTURE CLOCK',source,frames,len(ticks),count_equal,flush=True)
    write_json(ROOT/'M1_data/holo_rgb_clock_audit.json',summary);journal('Holo camera original clock audited against actual MP4 frame counts',sessions=len(summary),passed=sum(r['one_to_one_count_equal'] for r in summary))
if __name__=='__main__':audit()
