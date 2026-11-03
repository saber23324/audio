from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class IMUWindow:
    times:np.ndarray
    values:np.ndarray
    decision_time:float
    window_s:float
    channels:tuple
    units:tuple
    positions:tuple
    valid:bool
    reason:str
    clock_provenance:str

def times_from_dotnet_ticks(ticks,video_origin_tick):
    # Absolute .NET ticks exceed float64's integer precision; subtract first.
    value=np.asarray(ticks,dtype=np.int64)
    return (value-np.int64(video_origin_tick)).astype(np.float64)/10000000.

def causal_window(times,values,t,channels,units,positions,window_s=2.,nominal_rate_hz=None,clock_provenance='explicit_sensor_timestamp'):
    if not math.isfinite(t) or t<0 or not math.isfinite(window_s) or window_s<=0:raise ValueError('finite nonnegative time and positive window required')
    times=np.asarray(times,dtype=np.float64);values=np.asarray(values,dtype=np.float64)
    if times.ndim!=1 or values.ndim!=2 or len(times)!=len(values) or values.shape[1]!=len(channels) or len(units)!=len(channels):raise ValueError('explicit time/channel shape required')
    # Exclude future/out-of-window samples before checking values or pooling.
    mask=np.isfinite(times)&(times>=max(0.,t-window_s))&(times<=t)
    tt=times[mask].copy();vv=values[mask].copy();reason='ok'
    if len(tt)<2:reason='insufficient_samples'
    elif not np.isfinite(vv).all():reason='nonfinite_current_sensor'
    elif (np.diff(tt)<=0).any():reason='nonmonotonic_or_duplicate_current_timestamp'
    elif nominal_rate_hz and np.diff(tt).max()>2.5/nominal_rate_hz:reason='sample_gap'
    elif nominal_rate_hz and (tt[0]>max(0.,t-window_s)+1.5/nominal_rate_hz or tt[-1]<t-1.5/nominal_rate_hz):reason='short_window'
    return IMUWindow(tt,vv,float(t),float(window_s),tuple(channels),tuple(units),tuple(positions),reason=='ok',reason,clock_provenance)

def cheap_stats(window):
    if not window.valid:return np.zeros(len(window.channels)*4+3,np.float32)
    v=window.values;dt=np.diff(window.times);jerk=np.diff(v,axis=0)/dt[:,None]
    return np.r_[v.mean(0),v.std(0),np.sqrt(np.mean(v*v,0)),np.sqrt(np.mean(jerk*jerk,0)),len(v),window.times[-1]-window.times[0],1.].astype(np.float32)

