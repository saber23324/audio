"""Read-only experiment audit. Writes ONLY under this new report directory."""
import json,csv,hashlib,collections,ast,math,sys,time
from pathlib import Path
import numpy as np
REPO=Path('/home/yxd/audio_pic_gate');OUT=Path(__file__).resolve().parents[1]
ROOTS={
 'frozen':REPO/'artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1',
 'adaptive':REPO/'artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1',
 'gatefirst':REPO/'artifacts/utility_gate/gate_first_router_second_native_ablation_v1',
 'fixed':REPO/'artifacts/utility_gate/fixed_router_gate_v1',
 'imu':REPO/'artifacts/utility_gate/imu_visual_scaleup_train_v1',
 'imu_small':REPO/'artifacts/utility_gate/imu_gate_router_transfer_20261030_v3'}
GUARD={};EVIDENCE=[];CHECKS=[];ISSUES=[]
def rel(p):
 p=Path(p).resolve()
 try:return str(p.relative_to(REPO))
 except ValueError:return str(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def track(p):
 p=Path(p).resolve();key=rel(p)
 if key not in GUARD:GUARD[key]={'sha256':sha(p),'bytes':p.stat().st_size}
 return p
def js(p):return json.loads(track(p).read_text())
def lines(p):return track(p).read_text()
def jl(p):
 p=track(p)
 with p.open() as f:
  for n,line in enumerate(f,1):
   if line.strip():yield json.loads(line),f'{rel(p)}:{n}'
def write(p,value):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
 if not rows:return
 with (OUT/p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def check(kind,ok,**kw):CHECKS.append({'kind':kind,'passed':bool(ok),**kw})
def ev(stage,role,sid,metric,value,path,key,code='',status='原始记录重算通过'):
 eid=f'E{len(EVIDENCE)+1:05d}';EVIDENCE.append({'evidence_id':eid,'stage':stage,'cohort':role,'strategy':sid,'metric':metric,'value':json.dumps(value,ensure_ascii=False),'source_file':rel(path),'result_key_or_selector':key,'code_location':code,'verification_status':status});return eid
KEEP=['dataset','role','source_id','source_video_id','youtube_id','question_id','qa_id','session_id','stat_group_id','parent_recording_id','answer_text','class_id','modality_type','reasoning_type','strategy','text','prediction','correct','strict_normalized_correct','gold_nll','gold_nll_full_vocab','gold_nll_per_token','gold_token_nll','nll_protocol','visual_tokens','visual_tokens_actual','tokens_per_frame','on_count','window_count','on_mask','mask','k_extra_images','k','ON_rate','actual_extra_encoder_calls','actual_second_encoder_calls','actual_localizer_calls','encoded_image_count','timestamps_s','duration_s','llm_total_s','llm_ttft_s','frontend_wall_s','pipeline_total_s','pipeline_ttft_s','pipeline_generation_s','frontend_components_s','source_paid_record','source_paid_record_SHA','prediction_origin','generated_ids','options','clock_grade','imu_relevant_stratum']
def raw(stage):
 r=ROOTS[stage]
 if stage=='frozen':ps=sorted([p for pat in ['qa_eval/baseline_*confirm_shard*.jsonl','qa_eval/dual_*confirm_shard*.jsonl'] for p in r.glob(pat)])
 elif stage=='adaptive':ps=[r/'qa_eval/per_source_results.jsonl']
 elif stage=='gatefirst':ps=[r/'results/per_qa.jsonl']
 elif stage=='fixed':ps=[r/'per_qa_results.jsonl']
 else:ps=sorted((r/'results/eval').glob('*.jsonl'))
 result=[]
 for p in ps:
  for a,ref in jl(p):result.append({**{k:a[k] for k in KEEP if k in a},'_ref':ref})
 return result

def aggregate(stage,a,sid,role,sourcepath,stored=None):
 unique={r['source_id']:r for r in a};n=len(a);correct=sum(float(r['correct']) for r in a);nll=float(np.mean([r['gold_nll'] for r in a]));tok=float(np.mean([r.get('visual_tokens',r.get('visual_tokens_actual')) for r in a]));r0=a[0]
 if stage.startswith('imu'):
  on=sum(sum(r['mask']) for r in a);den=4*n;k=on/n;unit='QA的4个窗口（同录制问题相关）';t=float(np.mean([r['pipeline_generation_s'] for r in a])) if all('pipeline_generation_s' in r for r in a) else None
 elif stage in ['fixed','adaptive','gatefirst']:
  on=sum(r.get('k_extra_images',r.get('on_count',0)) for r in unique.values());den=sum(r.get('window_count',4) for r in unique.values());k=on/len(unique);unit='去重视频来源的合格观察窗口';t=float(np.mean([r['pipeline_total_s'] for r in a])) if stage=='fixed' else None
 else:on=den=k=None;unit='无Gate，音频端点参照另定义';t=None
 vals={'n_qa':n,'n_sources':len(unique),'correct':correct,'accuracy':correct/n,'gold_nll':nll,'visual_tokens_mean':tok,'on_count':on,'window_count':den,'on_rate':on/den if den else None,'mean_k':k,'pipeline_s':t}
 refs={}
 for key,value in vals.items():refs[key]=ev(stage,role,sid,key,value,sourcepath,f"role={role}; strategy={sid}; recompute:{key}")
 if stored:
  for dst,aliases in [('n_qa',['n_qa','qa','n_questions']),('n_sources',['n_sources','sources']),('correct',['correct']),('accuracy',['accuracy']),('gold_nll',['gold_nll','Gold_NLL']),('on_rate',['on_rate','ON_rate']),('visual_tokens_mean',['visual_tokens_mean','visual_tokens','mean_visual_tokens']),('pipeline_s',['pipeline_s_per_QA'])]:
   for key in aliases:
    v=stored.get(key)
    if isinstance(v,(int,float)) and vals.get(dst) is not None:check('aggregate',math.isclose(v,vals[dst],rel_tol=1e-10,abs_tol=1e-9),stage=stage,role=role,strategy=sid,metric=dst,stored=v,recomputed=vals[dst]);break
 return {**vals,'stage':stage,'role':role,'dataset':r0['dataset'],'strategy':sid,'on_unit':unit,'nll_protocol':r0.get('nll_protocol','ABCD归一化首答案Token，源码另核'),'raw_files':sorted({x['_ref'].rsplit(':',1)[0] for x in a}),'refs':refs}

def predkey(r):return r['source_id'],str(r.get('question_id',r.get('qa_id')))
def pairing(stage,role,a,b,aid,bid,stored=None,stratum='全部QA'):
 bm={predkey(r):r for r in b};ok=set(map(predkey,a))==set(bm);check('paired_key_set',ok,stage=stage,role=role,a=aid,b=bid,stratum=stratum)
 if not ok:return None
 rescue=harm=0.;nll=[]
 for r in a:
  d=float(r['correct'])-float(bm[predkey(r)]['correct']);rescue+=max(d,0);harm+=max(-d,0);nll.append(bm[predkey(r)]['gold_nll']-r['gold_nll'])
 out={'stage':stage,'role':role,'stratum':stratum,'a':aid,'b':bid,'n':len(a),'rescue':rescue,'harm':harm,'delta':(rescue-harm)/len(a),'nll_improvement':float(np.mean(nll))}
 if stored:
  for key,ours in [('Rescue',rescue),('rescue',rescue),('rescue_vs_V1',rescue),('Harm',harm),('harm',harm),('accuracy_delta',out['delta']),('gold_nll_improvement',out['nll_improvement'])]:
   if isinstance(stored.get(key),(int,float)):check('paired_metric',math.isclose(float(stored[key]),ours,abs_tol=1e-9),stage=stage,role=role,a=aid,b=bid,metric=key,stored=stored[key],recomputed=ours)
 return out

def build():
 allrows={};tables=[];pairs=[];summaries={}
 paths={'frozen':'summary.json','adaptive':'qa_eval/paired_summary.json','gatefirst':'results/summary.json','fixed':'paired_summary.json','imu':'M6_confirmation/PAIRED_SUMMARY.json','imu_small':'results/paired_summary.json'}
 for stage in ROOTS:
  print('READ/AUDIT',stage,flush=True);r=ROOTS[stage];summaries[stage]=js(r/paths[stage]);rr=raw(stage);allrows[stage]=rr;groups=collections.defaultdict(list)
  for a in rr:groups[(a['role'],a['strategy'])].append(a)
  # Random means are averages of the same questions' three runs, not extra QAs.
  if stage in ['adaptive','gatefirst','fixed']:
   families={'adaptive':[('G3_seed','G3_three_seed_mean')],'gatefirst':[('A4_random','A4_random_mean'),('B7_randomR','B7_randomR_mean'),('C5_random','C5_random_mean')],'fixed':[('T5_random_accuracy_','T5_random_accuracy_mean'),('T5_random_cost_','T5_random_cost_mean')]}[stage]
   for role in {x['role'] for x in rr}:
    for prefix,name in families:
     gg=collections.defaultdict(list)
     for x in rr:
      if x['role']==role and x['strategy'].startswith(prefix):gg[predkey(x)].append(x)
     out=[]
     for kk,xx in gg.items():
      check('three_random_seeds',len(xx)==3,stage=stage,role=role,qa=kk)
      y=dict(xx[0]);y['strategy']=name
      for field in ['correct','gold_nll','on_count','k_extra_images','actual_localizer_calls','actual_extra_encoder_calls','actual_second_encoder_calls','frontend_wall_s','pipeline_total_s','llm_total_s','llm_ttft_s']:
       if field in y:y[field]=float(np.mean([z[field] for z in xx]))
      if 'on_mask' in y:y['on_mask']=np.mean([z['on_mask'] for z in xx],0).tolist()
      out.append(y)
     if out:groups[(role,name)]=out
  for (role,sid),a in sorted(groups.items()):
   stored=None;source=r/paths[stage]
   if stage in ['adaptive','gatefirst','fixed']:stored=summaries[stage].get(role,{}).get('strategies',{}).get(sid)
   elif stage=='frozen':
    dataset='MUSIC' if role=='music_confirm' else 'Valor';sd=summaries[stage][dataset]
    stored=sd.get('aggregates',{}).get(sid,{}).get('all');stored={**stored,'accuracy':stored['accuracy']['mean'],'gold_nll':stored['gold_nll']['mean']} if stored else None
   elif stage=='imu':stored=next((x for x in summaries[stage]['metrics'] if x['role']==role and x['strategy']==sid),None)
   tables.append(aggregate(stage,a,sid,role,source,stored))
   # Recorded correctness is also independently recomputed from frozen scoring code.
   for x in a:
    if not isinstance(x['correct'],bool):continue
    if x.get('prediction') is not None and x.get('answer_text') is not None:
     if x['dataset']=='music_avqa':
      from utility_gate.grounded_dual_image.data import normalize
      actual=normalize(x['text'])==normalize(x['answer_text'])
     else:actual=x.get('prediction',x.get('text','')).strip()==x['answer_text']
     check('score_rule',actual==x['correct'],stage=stage,ref=x['_ref'])
  if stage in ['adaptive','gatefirst','fixed']:
   for role,d in summaries[stage].items():
    for stratum,cc in d.get('paired_strata',{}).items():
     for contrast,ss in cc.items():
      if stage=='adaptive':aid,bid=contrast.split('-')
      else:aid,bid=contrast.split(' minus ')
      def filt(a):return a if stratum in ('ALL','all') else [x for x in a if x.get('modality_type')==stratum] if stratum in ('Visual','Audio','Audio-Visual') else [x for x in a if x.get('modality_type')=='Audio-Visual' and x.get('reasoning_type') in ('Location','Temporal')]
      aa=filt(groups.get((role,aid),[]));bb=filt(groups.get((role,bid),[]))
      if aa and bb:pairs.append({**pairing(stage,role,aa,bb,aid,bid,ss,stratum),'stored':ss,'source_file':rel(r/paths[stage]),'source_key':f'{role}.paired_strata.{stratum}.{contrast}'})
  elif stage=='imu':
   for pp in summaries[stage]['paired']:
    role=pp['role'];aa=groups[(role,pp['a'])];bb=groups[(role,pp['b'])];pairs.append({**pairing(stage,role,aa,bb,pp['a'],pp['b'],pp),'stored':pp,'source_file':rel(r/paths[stage]),'source_key':f"paired[a={pp['a']},b={pp['b']},role={role}]"})
  if stage=='frozen':
   for dataset,role in [('MUSIC','music_confirm'),('Valor','valor_confirm')]:
    for name,cc in summaries[stage][dataset]['paired'].items():
     aid,bid=name.split(' - ');aa=groups.get((role,aid),[]);bb=groups.get((role,bid),[])
     if aa and bb:pairs.append({**pairing(stage,role,aa,bb,aid,bid,cc['all']),'stored':cc['all'],'source_file':rel(r/paths[stage]),'source_key':f'{dataset}.paired.{name}.all'})
  write('audit_tools/'+stage+'_row_locations.json',{'rows':len(rr),'roles':dict(collections.Counter(x['role'] for x in rr)),'strategies':sorted({x['strategy'] for x in rr})})
 write('audited_results.json',{'tables':tables,'pairs':pairs,'summary_paths':{k:rel(ROOTS[k]/v) for k,v in paths.items()}})
 # Validation of source-level window traces is handled in the next pass.
 write('audit_checks.json',CHECKS);write('primary_audit_checks.json',CHECKS);write('audit_input_guard.json',GUARD);csvout('evidence_index.csv',EVIDENCE)
 print('AUDIT stage results',len(tables),'paired',len(pairs),'checks',len(CHECKS),'failed',sum(not x['passed'] for x in CHECKS),flush=True)
if __name__=='__main__':build()
