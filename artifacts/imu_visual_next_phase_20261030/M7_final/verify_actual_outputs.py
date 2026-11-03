import json,math
from collections import Counter
import numpy as np
from utility_gate.imu_visual.common import ROOT,read_jsonl,write_json,write_jsonl,journal
from utility_gate.imu_visual.holo_probe import window
expected={'utd':(108,25),'utd_post_primary_controls':(108,5),'holo':(64,20)}
counts={};hooks={};failures=[]
for name,(sources,variants) in expected.items():
 files=[p for p in (ROOT/f'M2_frozen_imu_plots/{name}').glob('*.jsonl') if not p.name.endswith('.plots.jsonl')]
 assert len(files)==sources,(name,len(files));count=0;calls=Counter();real_pixels=[];per_qa={}
 for p in files:
  rows=read_jsonl(p);assert len(rows)==variants and len({x['variant'] for x in rows})==variants
  for r in rows:
   assert math.isfinite(r['gold_nll']) and r['no_training'] and r['no_audio_used']
   calls.update(r['actual_forward_hooks']);count+=1
  d={x['variant']:x for x in rows}
  if name=='holo':
   real=d['H2_real'];zero=d['H4_zero'];wrong=d['H5_wrong'];v1=d['H0_V1'];rgb=d['H0_RGB']
   assert real['plot_metadata']['pixel_sha256']!=zero['plot_metadata']['pixel_sha256']
   assert real['plot_metadata']['pixel_sha256']!=wrong['plot_metadata']['pixel_sha256']
   assert real['plot_metadata']['last_original_sensor_time']<=real['observed_end_s']
   assert real['plot_metadata']['first_original_sensor_time']>=real['observed_end_s']-2
   assert all(pt<=t for pt,t in zip(real['RGB_pts_s'],real['timestamps_s']))
   for label,images,tokens in [('H2_real',5,3650),('H3_duplicate',5,3650),('H0_RGB',4,2920),('H0_V1',4,256),('H1_plot',1,730),('H1_prior_no_image',0,0),('H7_numeric_text',0,0)]:
    assert d[label]['actual_forward_hooks']['SigLIP_images']==images,(p,label)
    assert d[label]['visual_tokens']==tokens
   assert v1['actual_forward_hooks']['V1']==4
   if real['correct']!=rgb['correct']:failures.append({'dataset':'holo','qa_id':real['qa_id'],'kind':'rescue' if real['correct'] else 'harm','RGB_prediction':rgb['text'],'REAL_prediction':real['text'],'gold':real['answer_text'],'zero_prediction':zero['text'],'wrong_prediction':wrong['text'],'source_id':real['source_id']})
  if name=='utd':
   real=d['I2_real'];rgb=d['I0_native_RGB']
   if real['correct']!=rgb['correct']:failures.append({'dataset':'utd','qa_id':real['source_id'],'kind':'rescue' if real['correct'] else 'harm','RGB_prediction':rgb['text'],'REAL_prediction':real['text'],'gold':real['answer_text'],'source_id':real['source_id']})
 counts[name]={'files':len(files),'conditions_per_qa':variants,'actual_generations':count};hooks[name]=dict(calls)
for r in read_jsonl(ROOT/'M1_data/holo_probe_manifest.jsonl'):
 for t in r['timestamps_s']:assert window(r,t).valid
cost=read_jsonl(ROOT/'M6_pipeline_cost/resident_native_cost.jsonl');assert len(cost)==72
for r in cost:
 assert r['actual_forward_hooks']['SigLIP_images']==4+r['k']
 assert r['plot_render_count']==r['k']
 if r['k']==0:assert r['plot_render_s']==0
assert sum(x['actual_generations'] for x in counts.values())==4520
write_json(ROOT/'M7_final/actual_output_contract_audit.json',{'all_passed':True,'counts':counts,'actual_hooks':hooks,'Holo_real_pixel_distinct_from_zero_and_wrong':True,'Holo_all_256_sensor_windows_original_time_causal':True,'Holo_all_RGB_capture_times_causal':True,'resident_cost_measurements':len(cost),'audio_used':False,'new_training_steps':0})
write_jsonl(ROOT/'M7_final/development_rescue_harm_cases.jsonl',failures)
journal('raw-output integration audit passed: every prediction, causal sensor/RGB bound, physical distinctness, visual token count and actual image forward count checked',QA_generations=4520,cost_measurements=72)
print('ALL ACTUAL OUTPUT CONTRACTS PASSED',counts,hooks)
