import numpy as np
import pytest
from utility_gate.imu_visual.causal_imu_window import causal_window,cheap_stats,times_from_dotnet_ticks
from utility_gate.imu_visual.render_imu_plot import render_imu_plot

CHANNELS=tuple('right_wrist:'+s+'_'+a for s in ('acc','gyro') for a in 'xyz')
UNITS=('m/s^2',)*3+('deg/s',)*3
def make(values=None,t=2.):
    times=np.arange(201)/50.;values=np.column_stack([np.sin(times*(i+1)) for i in range(6)]) if values is None else values
    return causal_window(times,values,t,CHANNELS,UNITS,('right_wrist',),2.,50.)
def cfg():return {'kind':'P1','y_limits':{'right_wrist:acc':2.,'right_wrist:gyro':2.},'size_px':[384,384],'scale_fit_role':'train'}
def test_future_nan_noise_impulse_does_not_change_current_sensor_or_plot():
    w=make();values=np.column_stack([np.sin(np.arange(201)/50.*(i+1)) for i in range(6)]);values[101:]=np.nan;q=make(values)
    assert np.array_equal(w.times,q.times) and np.array_equal(w.values,q.values)
    assert np.array_equal(cheap_stats(w),cheap_stats(q))
    assert render_imu_plot(w,cfg())[1]['pixel_sha256']==render_imu_plot(q,cfg())[1]['pixel_sha256']
    assert w.times.min()>=0 and w.times.max()<=2
def test_physical_units_and_no_per_window_minmax():
    w=make();from dataclasses import replace
    q=replace(w,values=w.values*4)
    a,ma=render_imu_plot(w,cfg());b,mb=render_imu_plot(q,cfg())
    assert ma['pixel_sha256']!=mb['pixel_sha256'] and mb['clipped_fraction']>ma['clipped_fraction']
    assert all(x in ('right wrist / acceleration','right wrist / angular velocity','m/s^2','deg/s') for x in ma['physical_labels'])
def test_renderer_rejects_class_question_source_metadata():
    for key in ('question','answer','class_name','source_id','title'):
        c={**cfg(),key:'jogging'}
        with pytest.raises(ValueError):render_imu_plot(make(),c)
    from dataclasses import replace
    with pytest.raises(ValueError):render_imu_plot(replace(make(),positions=('jogging',)),cfg())
    with pytest.raises(ValueError):render_imu_plot(replace(make(),units=('the answer is A',)*6),cfg())
def test_clock_gap_missing_channel_and_polarity():
    v=np.ones((201,6));v[50]=np.nan;assert not make(v).valid
    with pytest.raises(ValueError):causal_window(np.arange(20),np.zeros((20,3)),2,CHANNELS,UNITS,('right_wrist',))
    t=np.r_[np.arange(20)/50.,1.,1.02];w=causal_window(t,np.zeros((len(t),6)),1.02,CHANNELS,UNITS,('right_wrist',),2,50)
    assert not w.valid and w.reason=='sample_gap'
    w=make();from dataclasses import replace
    q=replace(w,values=-w.values)
    assert render_imu_plot(w,cfg())[1]['pixel_sha256']!=render_imu_plot(q,cfg())[1]['pixel_sha256']
def test_absolute_tick_precision_and_future_boundary():
    origin=637932482246358268;ts=times_from_dotnet_ticks([origin,origin+1,origin+20000000,origin+20000001],origin)
    assert ts[1]==1e-7 and ts[2]==2 and ts[3]>2
    w=causal_window(ts,np.ones((4,6)),2,CHANNELS,UNITS,('right_wrist',))
    assert len(w.times)==3
def test_actual_wear_acceleration_only_panels_have_no_fake_gyro():
    channels=tuple(p+':acc_'+a for p in ('right_wrist','right_ankle','left_ankle','left_wrist') for a in 'xyz');ts=np.arange(101)/50
    w=causal_window(ts,np.zeros((101,12)),2,channels,('g',)*12,('right_wrist','right_ankle','left_ankle','left_wrist'),nominal_rate_hz=50)
    _,m=render_imu_plot(w,{'kind':'P1','y_limits':{p+':acc':8 for p in w.positions},'size_px':[384,768]})
    assert len(m['physical_labels'])==8 and not any('angular' in x or 'gyro' in x for x in m['physical_labels'])

def test_async_holo_last_observation_hold_never_uses_later_sample():
    from utility_gate.imu_visual.holo_probe import combine
    ta=np.array([0.,.03,.07,.10,1.1]);tg=np.array([.01,.04,.08,.11,1.2])
    va=np.arange(15).reshape(5,3).astype(float);vg=va+100
    tt,vv,orig=combine((ta,va),(tg,vg));assert (orig<=tt[:,None]).all()
    at=np.where(tt==.07)[0][0];assert np.array_equal(vv[at,3:],vg[1])
    va[-1]=np.nan;vg[-1]=np.nan
    t2,v2,o2=combine((ta,va),(tg,vg));mask=tt<=.1
    assert np.array_equal(vv[mask],v2[mask])

def test_packed_media_pts_are_not_camera_capture_time():
    # A dropped capture frame makes CFR presentation time an unsafe clock.
    capture=np.array([0.,.0333,.1000,.1333]);pts=np.arange(4)/24.
    t=.09;index=np.searchsorted(capture,t,side='right')-1
    assert capture[index]<=t and capture[index+1]>t
    assert int(np.searchsorted(pts,t,side='right')-1)!=index
