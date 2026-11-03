"""Deterministic physical-axis plots; no task, source or class text accepted."""
import io,hashlib
import numpy as np
from PIL import Image
from .causal_imu_window import IMUWindow
ALLOWED={'acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z'}
POSITIONS={'head','right_wrist','left_wrist','right_ankle','left_ankle','right_thigh'}
UNITS={'m/s^2','g','rad/s','deg/s'}
def render_imu_plot(window,config,zero=False):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if not isinstance(window,IMUWindow):raise TypeError('typed sensor-only window required')
    if set(config)-{'kind','y_limits','size_px','dpi','line_width','font_size','scale_fit_role','scale_source_sha256'}:raise ValueError('unrecognized renderer config; labels/answers forbidden')
    if not window.valid:raise ValueError('invalid sensor window: '+window.reason)
    if set(window.positions)-POSITIONS or set(window.units)-UNITS:raise ValueError('physical sensor identity and units only')
    if any(c.split(':')[0] not in window.positions for c in window.channels):raise ValueError('unknown sensor placement')
    if any(c.split(':')[-1] not in ALLOWED for c in window.channels):raise ValueError('canonical physical channel names required')
    kind=config['kind']
    if kind not in ('P1','P2'):raise ValueError('only preregistered waveform/norm renderers')
    width,height=config.get('size_px',[768,768]);dpi=config.get('dpi',120);fig=Figure(figsize=(width/dpi,height/dpi),dpi=dpi,facecolor='white');canvas=FigureCanvasAgg(fig)
    groups=[]
    for position in window.positions:
        for sensor in ('acc','gyro'):
            idx=[i for i,c in enumerate(window.channels) if c.startswith(position+':'+sensor+'_')]
            if idx:groups.append((position,sensor,idx))
    if not groups:raise ValueError('no actual channels')
    values=np.zeros_like(window.values) if zero else window.values;tt=window.times-window.decision_time;clipped=total=0;labels=[]
    for n,(position,sensor,idx) in enumerate(groups):
        ax=fig.add_subplot(len(groups),1,n+1);ylim=float(config['y_limits'][position+':'+sensor]);data=values[:,idx];clipped+=int((np.abs(data)>ylim).sum());total+=data.size
        for j,i in enumerate(idx):ax.plot(tt,values[:,i],color=['#d62728','#2ca02c','#1f77b4'][j%3],linewidth=config.get('line_width',1.5),label=window.channels[i].split('_')[-1],antialiased=True)
        if kind=='P2':
            norm=np.linalg.norm(data,axis=1);clipped+=int((norm>ylim).sum());total+=len(norm);ax.plot(tt,norm,color='black',linewidth=1.8,label='norm')
        title=position.replace('_',' ')+' / '+('acceleration' if sensor=='acc' else 'angular velocity');unit=window.units[idx[0]];labels.extend([title,unit])
        ax.set_title(title,fontsize=config.get('font_size',11));ax.set_ylabel(unit,fontsize=10);ax.set_xlim(-window.window_s,0);ax.set_ylim(-ylim,ylim);ax.grid(alpha=.25);ax.legend(loc='upper right',ncol=4,fontsize=9);ax.tick_params(labelsize=9)
        if n==len(groups)-1:ax.set_xlabel('Time relative to observation (s)',fontsize=10)
    fig.tight_layout(pad=1.4);canvas.draw();rgb=np.asarray(canvas.buffer_rgba())[...,:3].copy();image=Image.fromarray(rgb)
    return image,{'kind':kind,'size_px':[width,height],'physical_labels':labels,'clipped_fraction':clipped/max(1,total),'first_original_sensor_time':float(window.times[0]),'last_original_sensor_time':float(window.times[-1]),
        'decision_time':window.decision_time,'clock_provenance':window.clock_provenance,'numeric_signal_resampling':False,'pixel_line_antialiasing':True,'zero_control':zero,'pixel_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'no_QA_class_or_source_text':True}
