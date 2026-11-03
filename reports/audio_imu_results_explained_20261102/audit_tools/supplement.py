"""Second read-only pass: paid timing, training, splits, diagnostics, window traces."""
from audit import *
from sklearn.metrics import roc_auc_score, average_precision_score
GUARD.update(js(OUT/'audit_input_guard.json'))
CHECKS.extend(js(OUT/'primary_audit_checks.json'))
EVIDENCE.extend(list(csv.DictReader((OUT/'evidence_index.csv').open())))
S={};B=REPO/'artifacts/utility_gate'
def load(k,p):
 try:a=js(p)
 except json.JSONDecodeError:a=[x for x,_ in jl(p)];print('JSON filename with JSONL content',rel(p),flush=True)
 S[k]={'path':rel(p),'data':a};return a
def meta():
 for stage,patterns in {'frozen':['config/*.json','audit/music_metadata.json','pretest_lock.json','final_audit.json'], 'adaptive':['locks/stage1_retrained_lock.json','locks/policy_lock.json','locks/fuser_lock.json','locks/utility_label_lock.json','config/*.json','models/*/*.json','logs/stage1_history.json','models/*/*lock*.json'], 'gatefirst':['locks/gate_a_lock.json','locks/fusion_router_lock.json','config/*.json','commands.json','pretest_lock.json'], 'fixed':['train/catalog.json','locks/selection_lock.json','locks/policy_label_lock.json','data/selection.json','commands.json','pretest_lock.json'], 'imu':['locks/*.json','M3_router_train/*.json','M6_confirmation/*.json','M7_cost_final/*.json','FINAL_ASSET_SHA_AUDIT.json']}.items():
  for pat in patterns:
   for p in sorted(ROOTS[stage].glob(pat)):load(f'{stage}:{p.relative_to(ROOTS[stage])}',p)
 # Register live code and immutable experiment source snapshots actually used for explanation.
 for directory in ['utility_gate/grounded_dual_image','utility_gate/adaptive_grounded64','utility_gate/gate_first_native','utility_gate/fixed_router_gate','utility_gate/imu_visual/scaleup_train','utility_gate/imu_gate_router_transfer','utility_gate/grounding','utility_gate/dual_head/proxy','streamingtom/audio_bottleneck']:
  for p in (REPO/directory).glob('*.py'):track(p)
 for p in [REPO/'streamingtom/train_adbt_videomme.py',REPO/'CODEX_AUDIO_IMU_RESULTS_REPORT_INSTRUCTIONS.md',REPO/'AGENTS.md']:track(p)
 # A compact complete manifest ledger. Counts are computed, not copied from prose.
 manifests=[]
 for stage,folder in [('frozen','manifests'),('adaptive','manifests'),('gatefirst','manifests'),('fixed','data'),('imu','data')]:
  for p in sorted((ROOTS[stage]/folder).glob('*.jsonl')):
   if any(x in p.name for x in ['seen','candidate','attempt','history','planned','acquisition']):continue
   rr=[x for x,_ in jl(p)]
   if not rr or 'source_id' not in rr[0]:continue
   roles=collections.defaultdict(list)
   for x in rr:roles[x.get('role',p.stem)].append(x)
   for role,aa in roles.items():
    row={'stage':stage,'manifest':rel(p),'role':role,'qa':len(aa),'sources':len({x['source_id'] for x in aa}),'families':len({x.get('stat_group_id',x.get('parent_recording_id',x['source_id'])) for x in aa}),'qa_types':dict(collections.Counter(x.get('modality_type',x.get('class_id','unspecified')) for x in aa))}
    manifests.append(row)
 S['manifest_ledger']=manifests;csvout('dataset_split_ledger.csv',[{**x,'qa_types':json.dumps(x['qa_types'],ensure_ascii=False)} for x in manifests])
 # Training update logs, catalog metadata and selected checkpoint byte hashes.
 train=[]
 for p in sorted((ROOTS['imu']/'locks').glob('R_*.json')):
  a=js(p);lp=ROOTS['imu']/f'logs/{p.stem}_train_steps.jsonl';rr=[x for x,_ in jl(lp)]
  check('training_steps',len(rr)==a['optimizer_steps'] and max(x['step'] for x in rr)==a['optimizer_steps'],model=p.stem,source=rel(lp))
  check('frozen_gradient_log',all(x['frozen_gradient_count']==0 for x in rr),model=p.stem)
  cp=Path(a['checkpoint']);check('selected_checkpoint_sha',sha(track(cp))==a['sha256'],model=p.stem)
  train.append({'model':p.stem,'steps':len(rr),'qa':a['training_QA'],'groups':len(a['training_groups']),'sources':len(a['training_sources']),'epoch':a['selected_epoch'],'params':a['params'],'checkpoint':rel(cp),'sha256':a['sha256'],'selected_val':a['selected_val'],'loss':a['loss'],'init':a['training_init'],'warm_start':a.get('warm_start_checkpoint'),'source':rel(p)})
 S['imu_router_training']=train
 cat=js(ROOTS['imu']/'locks/Gate_training_catalog.json')
 for a in cat:
  p=ROOTS['imu']/f'logs/G_{a["name"]}_train.jsonl';rr=[x for x,_ in jl(p)]
  check('gate_update_count',len(rr)==a['epochs'] and rr[-1]['optimizer_steps']==a['steps'] and rr[-1]['nonzero_gradient_steps']==a['steps'],model=a['name'])
 load('original_training_metadata',REPO/'artifacts/imu_visual_next_phase_20261030/M0_snapshot/checkpoint_training_metadata.json')
 load('original_model_inventory',REPO/'artifacts/imu_visual_next_phase_20261030/M0_snapshot/model_inventory.json')
 load('original_split',REPO/'results/adbt_videomme_split.json')
 load('proxy_training',B/'dual_head_proxy_20261028_v1/model/training_configuration.json')
 S['training_totals']={'imu_router_models':len(train),'imu_router_steps':sum(x['steps'] for x in train),'imu_gate_models':len(cat),'imu_gate_steps':sum(x['steps'] for x in cat)}
 for k in ['accuracy','cost']:
  c=js(ROOTS['imu']/'locks/validation_selection.json')['selected'][k];check('selected_checkpoint_sha',sha(track(c['checkpoint']))==c['checkpoint_SHA'],model='G_IMU_'+k)
 for stage,file in [('gatefirst','locks/fusion_router_lock.json'),('fixed','locks/selection_lock.json')]:
  a=js(ROOTS[stage]/file)
  if stage=='gatefirst':check('selected_checkpoint_sha',sha(track(a['selected_checkpoint']))==a['selected_sha256'],model='R_A')
  else:
   for k in ['accuracy_first','cost_aware']:
    z=a['selected'][k]['spec'];check('selected_checkpoint_sha',sha(track(z['checkpoint']))==z['checkpoint_sha256'],model='AudioGate_'+k)

def paid():
 r=ROOTS['imu'];aa=[]
 for p in sorted((r/'results/cost').glob('*.json')):
  a=js(p);a['_ref']=rel(p);aa.append(a)
 groups=collections.defaultdict(list)
 for a in aa:groups[a['role'],a['strategy']].append(a)
 summary=js(r/'M6_confirmation/PAIRED_SUMMARY.json');out=[];pairs=[]
 # Native B1 results do not use a Router; their masks are bookkeeping, not a learned Gate.
 for (role,sid),rows in sorted(groups.items()):
  n=len(rows);correct=sum(x['correct'] for x in rows);k=float(np.mean([x['k'] for x in rows]));native=sid.startswith('N')
  z={'stage':'imu','role':role,'dataset':rows[0]['dataset'],'strategy':sid,'n_qa':n,'n_sources':len({x['source_id'] for x in rows}),'families':len({x['stat_group_id'] for x in rows}),'correct':correct,'accuracy':correct/n,'gold_nll':float(np.mean([x['gold_nll'] for x in rows])),'visual_tokens_mean':float(np.mean([x['visual_tokens'] for x in rows])),'on_count':sum(x['k'] for x in rows),'window_count':4*n,'on_rate':k/4,'mean_k':k,'pipeline_s':float(np.mean([x['pipeline_generation_s'] for x in rows])),'pipeline_median_s':float(np.median([x['pipeline_generation_s'] for x in rows])),'refs':{},'raw_files':[rel(r/'results/cost')+'/*.json']}
  base={x['qa_id']:x for x in groups[role,'T0_V1']}
  z['rescue']=sum(x['correct'] and not base[x['qa_id']]['correct'] for x in rows);z['harm']=sum(not x['correct'] and base[x['qa_id']]['correct'] for x in rows)
  for x in rows:
   check('paid_score_rule',x['correct']==(x['text']==x['answer_text']),ref=x['_ref'])
   if not native:
    check('paid_window_counts',x['k']==sum(x['mask']) and len(x['mask'])==4,ref=x['_ref'])
    check('paid_token_budget',x['visual_tokens']==256,ref=x['_ref'])
    if 'counts' in x:check('paid_extra_encodes',x['counts']['extra_encode']==x['k'],ref=x['_ref'])
  stored=next((x for x in summary['metrics'] if x['role']==role and x['strategy']==sid),None)
  if stored:
   for key,orig in [('correct','correct'),('accuracy','accuracy'),('gold_nll','gold_nll'),('on_rate','on_rate'),('pipeline_s','mean_pipeline_s'),('pipeline_median_s','median_pipeline_s')]:check('paid_summary',math.isclose(z[key],stored[orig],abs_tol=1e-9),role=role,strategy=sid,metric=key)
   z['ci95']=stored['accuracy_cluster_CI'];z['delta_ci95']=stored['delta_cluster_CI']
  for key,value in list(z.items()):
   if key in ['correct','accuracy','gold_nll','on_count','window_count','on_rate','mean_k','pipeline_s','pipeline_median_s','rescue','harm','visual_tokens_mean','ci95','delta_ci95']:z['refs'][key]=ev('imu',role,sid,key,value,r/'M6_confirmation/PAIRED_SUMMARY.json',f'metrics[role={role},strategy={sid}].{key}; raw=results/cost/*', 'utility_gate/imu_visual/scaleup_train/report.py')
  out.append(z)
 for role in ['val','holdout']:
  for sid in [x[1] for x in groups if x[0]==role]:
   for bid in ['T0_V1','T1_REAL_always','Fixed_k_2','T8_DUP_trained_always','T9_ZERO_trained_always','T2_accuracy','T2_accuracy_same_mask_DUP','T2_accuracy_same_mask_ZERO','T2_accuracy_same_mask_WRONG','T4_random_accuracy_1234','T4_random_accuracy_2234','T4_random_accuracy_3234']:
    pairs.append(pairing('imu',role,groups[role,sid],groups[role,bid],sid,bid))
 S['imu_paid']=out;S['imu_paid_pairs']=pairs;S['imu_paid_records']=len(aa)
 csvout('imu_paid_results.csv',[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in x.items()} for x in out])
 # Tail examples explain why accuracy and proper NLL can move oppositely.
 base={x['qa_id']:x for x in groups['holdout','T0_V1']};tails=[]
 for x in groups['holdout','T1_REAL_always']:
  y=base[x['qa_id']];tails.append({'qa_id':x['qa_id'],'family':x['stat_group_id'],'gold':x['answer_text'],'new_prediction':x['text'],'new_correct':x['correct'],'V1_prediction':y['text'],'V1_correct':y['correct'],'new_gold_nll':x['gold_nll'],'V1_gold_nll':y['gold_nll'],'nll_change':x['gold_nll']-y['gold_nll'],'raw_real':x['_ref'],'raw_V1':y['_ref']})
 S['imu_nll_tails']=sorted(tails,key=lambda x:-x['nll_change'])[:8];csvout('imu_gold_nll_tail_examples.csv',S['imu_nll_tails'])
 # Validate paired cost savings using the actual paid per-question records.
 for x in js(r/'M7_cost_final/PAIRED_PIPELINE_COST.json'):
  a=next(z for z in out if z['role']==x['role'] and z['strategy']==x['a']);b=next(z for z in out if z['role']==x['role'] and z['strategy']==x['b'])
  check('cost_saving_formula',math.isclose(1-a['pipeline_s']/b['pipeline_s'],x['saving_fraction'],abs_tol=1e-12),role=x['role'],a=x['a'],b=x['b'])

def diagnostics():
 # Spatial values recomputed from exported per-source results. No new model call.
 dg={}
 for rn,kind in [('vggss_grounded_image_pilot_20261028','pilot'),('vggss_feature_scaleup_20261028_v1','scaleup')]:
  r=B/rn;su=js(r/'summary.json');dg[kind]={'root':rel(r),'summary':su}
  if kind=='pilot':
   rr=[x for x,_ in jl(r/'results.jsonl')];dg[kind]['raw_schema']={'keys':list(rr[0]),'rows':len(rr)}
  else:
   p=track(r/'per_source_results.csv');rr=list(csv.DictReader(p.open()));dg[kind]['raw_schema']={'columns':list(rr[0]),'rows':len(rr)}
   for role in ['legacy_dev','fresh_dev','fresh_holdout']:
    for mode,st in su[role]['methods'].items():
     ar=[x for x in rr if x['split']==role and x['mode']==mode]
     for col,key in [('ROI_mass','mass'),('mass_excess','excess_mass'),('mass_lift','lift')]:check('VGG_spatial_mean',math.isclose(float(np.mean([float(x[col]) for x in ar])),st['spatial'][key]['mean'],abs_tol=1e-10),role=role,mode=mode,metric=key)
  for p in r.glob('features/*/summary.json'):dg[kind][str(p.parent.name)+'_features']=js(p)
 r=B/'dual_head_proxy_20261028_v1';su=js(r/'summary.json');rr=[x for x,_ in jl(r/'evaluation/LLP_new_confirmation_predictions.jsonl')];stats=[]
 for grid in su['correspondence']:
  if not isinstance(su['correspondence'][grid],dict) or 'methods' not in su['correspondence'][grid]:continue
  a=[x for x in rr if x['grid']==grid and x['available'] and x['proxy_target'] is not None]
  if not a:continue
  for method in su['correspondence'][grid]['methods']:
   if method not in a[0]['scores']:continue
   y=[x['proxy_target'] for x in a];v=[x['scores'][method]['u' if method=='utility_only' else 'r'] for x in a];au=roc_auc_score(y,v);ap=average_precision_score(y,v);stored=su['correspondence'][grid]['methods'][method]['primary_same_bin_vs_proxy_negative']
   check('LLP_AUROC',math.isclose(au,stored['AUROC'],abs_tol=1e-10),grid=grid,method=method)
   check('LLP_AUPRC',math.isclose(ap,stored['AUPRC'],abs_tol=1e-10),grid=grid,method=method)
   stats.append({'grid':grid,'method':method,'n':len(a),'sources':len({x['source_video_id'] for x in a}),'positives':sum(y),'AUROC':au,'AUPRC':ap,'AUROC_ci95':stored.get('cluster_ci95',{}).get('AUROC',{}).get('ci95'),'evidence':ev('LLP',grid,method,'AUROC/AUPRC',[au,ap],r/'evaluation/LLP_new_confirmation_predictions.jsonl',f'available,proxy_target known,grid={grid}; scores.{method}.r','utility_gate/dual_head/proxy/evaluation.py')})
 qq=[x for x,_ in jl(r/'downstream/new_QA_predictions.jsonl')];qa=[]
 for method,stored in su['downstream']['methods'].items():
  a=[x['branches'][method] for x in qq];c=sum(x['correct'] for x in a);nl=float(np.mean([x['gold_nll'] for x in a]));check('proxy_QA_accuracy',c==stored['correct'],method=method);check('proxy_QA_NLL',math.isclose(nl,stored['gold_NLL']['mean'],abs_tol=1e-9),method=method)
  qa.append({'method':method,'correct':c,'n':len(a),'gold_nll':nl,'on_rate':stored['mean_audio_activation'],'evidence':ev('proxy_QA','confirmation',method,'correct/NLL',[c,nl],r/'downstream/new_QA_predictions.jsonl',f'branches.{method}','utility_gate/dual_head/proxy/downstream.py')})
 dg['LLP']={'root':rel(r),'stats':stats,'qa':qa,'summary':su};S['diagnostics']=dg

def timing():
 timing={}
 for st,p in [('frozen','qa_eval/pipeline_cost_summary.json'),('adaptive','qa_eval/pipeline_cost_summary.json'),('gatefirst','results/pipeline_cost.json')]:timing[st]=js(ROOTS[st]/p)
 S['timing']=timing
 # Deduplicate actual per-window logs. ON is defined by the archived run, never inferred from accuracy.
 traces=[]
 for st,p in [('adaptive','qa_eval/per_window_actions.jsonl'),('gatefirst','results/per_window_gate_decisions.jsonl'),('fixed','per_window_gate.jsonl')]:
  counts=collections.defaultdict(lambda:collections.Counter());sample=None;names=None
  for a,ref in jl(ROOTS[st]/p):
   if sample is None:sample=a
   counts[(a.get('role','unknown'),a.get('strategy',a.get('method','unknown')))]['rows']+=1
   for k in ['ON','on','accepted','gate_on','actual_extra_encoder_calls','extra_encoder_calls','localizer_called','Localizer_calls']:
    if k in a and isinstance(a[k],(int,float,bool)):counts[(a.get('role','unknown'),a.get('strategy',a.get('method','unknown')))]['sum_'+k]+=a[k]
  traces.append({'stage':st,'path':rel(ROOTS[st]/p),'sample':sample,'counts':[{'role':k[0],'strategy':k[1],**v} for k,v in counts.items()]})
 S['window_traces']=traces

if __name__=='__main__':
 for f in [meta,paid,diagnostics,timing]:print('PASS',f.__name__,flush=True);f()
 write('supplement.json',S);write('audit_checks.json',CHECKS);write('audit_input_guard.json',GUARD);csvout('evidence_index.csv',EVIDENCE)
 print('SUPPLEMENT checks',len(CHECKS),'failed',sum(not x['passed'] for x in CHECKS),flush=True)
