"""Full-timeline sensor audit and RGB thumbnails, separate from VLM evidence."""
import json,concurrent.futures
import numpy as np
import av
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .common import ROOT,DATA,write_json,journal
from .holo_probe import load_sensor
from .range_io import RangeFile
from .intake import SOURCES

def one(source):
    report=ROOT/f'M1_data/timeline_{source}.json'
    if report.exists():return
    sync=json.loads((ROOT/f'M1_data/holo_sync_{source}.json').read_text());clock=json.loads((ROOT/f'M1_data/holo_rgb_clock_{source}.json').read_text());entries=json.loads((ROOT/'M1_data/holo_video_subset_index.json').read_text());entry=next(x for x in entries if x['name'].split('/')[0]==source);acc=load_sensor(sync['sensor_streams']['Accelerometer_sync.txt']['path'],sync['video_start_tick']);gyro=load_sensor(sync['sensor_streams']['Gyroscope_sync.txt']['path'],sync['video_start_tick']);capture=np.load(clock['capture_clock_path'])['times'];frames=[]
    with av.open(RangeFile(SOURCES['holo_video'],entry['size'],entry['offset_data'])) as c:
        stream=c.streams.video[0];duration=float(stream.duration*stream.time_base);rate=float(stream.average_rate)
        for fraction in (.05,.5,.95):
            target=duration*fraction;c.seek(int(target/stream.time_base),stream=stream,backward=True);selected=None
            for f in c.decode(video=0):
                pts=float(f.pts*f.time_base)
                if pts>=target:selected=(f.to_ndarray(format='rgb24'),pts);break
            if selected is None:raise ValueError('missing audit frame')
            array,pts=selected;index=int(round(pts*rate));original=float(capture[index]) if clock['one_to_one_count_equal'] else None;frames.append((array,pts,original))
    fig=Figure(figsize=(12,8),dpi=120);canvas=FigureCanvasAgg(fig)
    for i,(array,pts,original) in enumerate(frames):
        ax=fig.add_subplot(3,3,i+1);ax.imshow(array);ax.axis('off');ax.set_title(f'Playback {pts:.2f}s / capture '+(f'{original:.2f}s' if original is not None else 'UNRESOLVED'),fontsize=9)
    for n,(sensor,(tt,v),unit) in enumerate([('acc',acc,'m/s^2'),('gyro',gyro,'rad/s')]):
        ax=fig.add_subplot(3,1,n+2);ax.plot(tt,np.linalg.norm(v,axis=1),linewidth=.5);ax.set_ylabel(sensor+' norm '+unit);ax.set_xlabel('Original capture time relative to video origin (s)');ax.grid(alpha=.2)
        for _,pts,original in frames:
            if original is not None:ax.axvline(original,color='red',alpha=.5)
    fig.suptitle(source+' — offline full-timeline audit, NEVER a model input',fontsize=12);fig.tight_layout(rect=[0,0,1,.96]);canvas.draw();p=ROOT/f'M1_data/timeline_{source}.png';Image.fromarray(np.asarray(canvas.buffer_rgba())[...,:3].copy()).save(p)
    write_json(report,{'source_id':'holo:'+source,'sensor_full_streams_read':True,'video_full_duration_s':duration,'video_thumbnails_across_full_timeline':len(frames),'whole_video_downloaded_or_all_frames_decoded':False,'clock_count_match':clock['one_to_one_count_equal'],'frames':[{'playback_time_s':x[1],'capture_time_s':x[2]} for x in frames],'manual_physical_alignment':'qualitative gallery only; not a calibrated residual sync error measurement','plot_not_used_in_model':True,'gallery':str(p)})
    print('TIMELINE AUDIT',source,flush=True)

def run():
    sources=json.loads((ROOT/'M1_data/holo_sync_probe_source_lock.json').read_text())['sources']
    with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(one,sources))
    journal('full sensor streams and full video timeline sparse paired audit completed',sessions=len(sources),whole_videos_downloaded=False)
if __name__=='__main__':run()
