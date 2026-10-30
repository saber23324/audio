"""Pretest development-only actual front-end calibration, including prepaid cost."""
import json,time
from .common import ROOT,train_rows,write_json,journal
from .features import load
from .runtime import Runtime,Localizer
from .online_inference import Controllers,execute
def run():
    catalog=json.loads((ROOT/'train/catalog.json').read_text());families={x['family']:x for x in catalog if x['lambda']==0 and x['temperature']==.25}
    rows={r['source_id']:r for r in train_rows('val')};eligible=[rows[s] for s in sorted(rows) if bool(load(rows[s])['audio_valid'].all())][:3]
    if len(eligible)!=3:raise ValueError('need3 predeclared all-valid cost calibration sources')
    runtime=Runtime();loc=Localizer(runtime.device);controllers=Controllers();specs={'off':{'kind':'off'},'on':{'kind':'on'}}
    for family,entry in families.items():
        cp=next(x['path'] for x in entry['candidate_checkpoints'] if x['epoch']==30);controllers.model(cp)
        for k in range(5):specs[f'{family}_k{k}']={'kind':'prefix','family':family,'checkpoint':cp,'threshold':.5,'K':k}
    # One non-scored warm-up source; excludes model initialization and CUDA warm-up.
    execute(runtime,loc,eligible[0],specs['on'],controllers)
    records=[]
    for row in eligible:
        for name,spec in specs.items():
            z,meta,acts=execute(runtime,loc,row,spec,controllers)
            if spec['kind']=='prefix' and meta['k_extra_images']!=spec['K']:raise AssertionError('wrong real encode count')
            records.append({'source_id':row['source_id'],'policy':name,**meta});print('COST ACTUAL',row['source_id'],name,meta['frontend_wall_s'],meta['k_extra_images'],flush=True)
    mean={name:sum(x['frontend_wall_s'] for x in records if x['policy']==name)/3 for name in specs}
    write_json(ROOT/'timing/development_calibration.json',{'sources':[x['source_id'] for x in eligible],'source_role':'fresh val','replicates_sources':3,'warmup_excluded':True,
        'recorded_before_confirmation':True,'estimator':'mean of actually uncached front ends; per-k lookup used for validation only; final cost remeasured independently',
        'mean_frontend_s':mean,'measurements':records});journal('pretest real cost calibration complete',sources=3,frontends=len(records))
if __name__=='__main__':run()

