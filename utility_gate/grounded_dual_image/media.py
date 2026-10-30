"""Clip BOTH raw native-rate audio boundaries before any resampling for maps."""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.signal import resample_poly

def raw_window(path,t):
    import av
    if not math.isfinite(t) or t<=0: return torch.zeros(32000),False,'invalid_timestamp',{}
    start=max(0.,t-2.);chunks=[];rate=None
    with av.open(str(path)) as c:
        if not c.streams.audio:return torch.zeros(32000),False,'missing_audio',{}
        origin=(c.start_time or 0)/av.time_base;stream=c.streams.audio[0]
        for f in c.decode(stream):
            if f.pts is None:return torch.zeros(32000),False,'missing_audio_pts',{}
            begin=float(f.pts*f.time_base)-origin;sr=f.sample_rate
            if begin>=t:break
            if begin+f.samples/sr<=start:continue
            first=max(0,math.ceil((start-begin)*sr-1e-8));last=min(f.samples,math.floor((t-begin)*sr+1e-8))
            if last<=first:continue
            values=f.to_ndarray()
            if not f.format.is_planar:values=values.reshape(-1,len(f.layout.channels)).T
            values=values[:,first:last]  # BOTH native PCM boundaries, before mono or resampling.
            if values.dtype.kind in 'iu':values=values.astype(np.float32)/float(max(abs(np.iinfo(values.dtype).min),np.iinfo(values.dtype).max))
            else:values=values.astype(np.float32)
            if rate is not None and rate!=sr:return torch.zeros(32000),False,'changing_native_sample_rate',{}
            rate=sr;chunks.append((begin+first/sr,values.mean(0)))
    if not chunks:return torch.zeros(32000),False,'empty_audio_window',{}
    first_time=chunks[0][0];end_time=max(s+len(a)/rate for s,a in chunks)
    n=int(round((end_time-first_time)*rate));native=np.zeros(n,np.float32);mask=np.zeros(n,bool)
    for stamp,values in chunks:
        offset=int(round((stamp-first_time)*rate));end=offset+len(values)
        if mask[offset:end].any():return torch.zeros(32000),False,'audio_pts_overlap',{}
        native[offset:end]=values;mask[offset:end]=True
    if not np.isfinite(native).all():return torch.zeros(32000),False,'nonfinite_audio',{}
    divisor=math.gcd(rate,16000)
    resampled=resample_poly(native,16000//divisor,rate//divisor,padtype='constant').astype(np.float32)
    samples=np.zeros(32000,np.float32)
    if len(resampled)>32000:raise ValueError('raw crop exceeded two-second model input')
    samples[-len(resampled):]=resampled
    valid=bool(mask.all() and first_time<=start+1/rate+1e-7 and end_time>=t-1/rate-1e-7 and np.sqrt(np.mean(native**2))>1e-5)
    reason='ok' if valid else 'audio_pts_gap_or_short_window' if not mask.all() or end_time<t-1/rate-1e-7 else 'silence'
    trace={'requested_raw_start_s':start,'requested_raw_end_s':t,'first_raw_sample_s':first_time,'last_raw_sample_interval_end_s':end_time,
        'native_sample_rate':rate,'native_sample_count':len(native),'resampled_16k_sample_count':len(resampled),
        'two_second_window_zero_left_padding_s':(32000-len(resampled))/16000.,
        'raw_both_boundaries_cropped_before_mono_and_resampling':True,'resampler':'scipy.signal.resample_poly, constant zero boundary; no outside-window filter state',
        'map_audio_preprocessing_version':'rawcrop_v1','model_zero_left_padding_s':1.}
    if first_time<start-1e-7 or end_time>t+1e-7:raise AssertionError('raw window escaped allowed bounds')
    return torch.from_numpy(samples),valid,reason,trace

def raw_windows(path,timestamps):
    import av
    values=[]
    for t in timestamps:
        try:values.append(raw_window(path,float(t)))
        except (av.error.FFmpegError,ValueError,OSError,TypeError) as error:
            values.append((torch.zeros(32000),False,'audio_decode_exception',{'error':str(error),'requested_raw_end_s':float(t)}))
    return torch.stack([v[0] for v in values]),torch.tensor([v[1] for v in values],dtype=torch.bool),[v[2] for v in values],[v[3] for v in values]
