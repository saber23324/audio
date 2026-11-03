"""Close the read-only audit, retain historical metadata discrepancies explicitly."""
from render_report import *
import ast,subprocess
CHECKS.extend(x for x in js(OUT/'audit_checks.json') if x['kind']!='all_local_links_exist')
def history():
 r=REPO/'artifacts/utility_gate/gate2_20261027';g=REPO/'artifacts/utility_gate/gate2_grounding_20261027';su=js(r/'summary.json');rows=[]
 lab=[x for x,_ in jl(r/'labels/evaluation_assets/test.jsonl')];qualified=[x for x,_ in jl(r/'qualified_labels/evaluation_assets/test.jsonl')];pred=[x for x,_ in jl(r/'integration/gated_results.jsonl')]
 for nm,aa,key,expect in [('固定视觉64',lab,'vision_correct',222),('原始音频Router64',lab,'audio_correct',211),('合法可用性音频Router64',qualified,'audio_correct',213),('旧视觉锚定前缀Gate',pred,'correct',222)]:
  c=sum(x[key] for x in aa);check('historical_gate2_correct',c==expect and len(aa)==256,method=nm);rows.append([nm,f'{c}/{len(aa)}',pct(c/len(aa)),'前缀可行性代理，最多10秒5次观察；非逐帧收益Gate',link(r/'integration/gated_results.jsonl' if key=='correct' else r/('qualified_labels/evaluation_assets/test.jsonl' if '合法' in nm else 'labels/evaluation_assets/test.jsonl'))])
 text='''# 更早的 Audio Gate 历史：单独队列，不能与 MUSIC／Valor 混排

## 2026-10-27 视觉锚定Gate2

该阶段来自原AVQA标签（不是MUSIC），训练512题／512原始YouTube源，验证128／128，确认256／256；View4、Sound10、Both242。每源一道四选一问题，严格生成ABCD匹配；首答案Token四选项归一化NLL。最多10秒观察、2秒因果窗口、每帧64视觉Token，五帧时320Token。前缀Gate在问题到来时决定整段是否用音频，逐窗口Gate另行训练，二者监督目标不同。
'''+table(['前缀策略中文名','正确／QA','Accuracy ↑','时间与决策协议','原始记录'],rows)
 text+='完整视觉196参考225/256来自原聚合summary.prefix.methods.full_vision，本次没有重验该端点生成文件，不把它当逐题重验通过。精确匹配启用数量的随机策略与Gate同222/256、能量阈值全关；该两端点来自原summary，仅聚合引用。Gate只开启3/256前缀，不能据此证明有益片段识别。'+link(r/'summary.json','prefix.methods')+'。\n'
 rr=[x for x,_ in jl(r/'temporal_evaluation/streaming_results.jsonl')];rows=[]
 for k,nm in [('vision_correct','固定视觉64'),('legacy_audio_correct','原始音频Router64'),('safe_audio_correct','可用性约束音频64'),('full_vision_correct','完整视觉196'),('correct','逐时刻因果Gate64'),('random_correct','验证预算随机Gate64')]:
  c=sum(x[k] for x in rr);rows.append([nm,f'{c}/{len(rr)}',pct(c/len(rr))]);check('historical_64_accuracy',len(rr)==64 and c/len(rr)==js(r/'temporal_evaluation/summary.json')[{'vision_correct':'vision_accuracy','legacy_audio_correct':'legacy_audio_accuracy','safe_audio_correct':'safe_audio_accuracy','full_vision_correct':'full_vision_accuracy','correct':'accuracy','random_correct':'random_accuracy'}[k]],method=nm)
 text+='\n逐时刻因果重放：另64题／64源，原AVQA四选项，2秒到达、总317观察窗，保持每时刻64Token；编码器缓存时间不能当完整Pipeline。\n'+table(['该64题策略','正确／QA','准确率 ↑'],rows)
 text+='训练：视觉+严格过去音频摘要、两层小MLP；只训练Gate，原Router／BEATs／VLM冻结。训练配置为AdamW lr.001、wd.01、batch32、最多60轮，种子1234／2333／3407；两布局、辅助分类／权重候选在验证选择，主seed预先固定。旧前缀、可用性修正、逐窗口三个拟合阶段不得合算独立数据。每步更新未在这次报告重建，精确轮数与配置保存于'+link(r/'qualified_model/train_config.json')+'、'+link(r/'temporal_model/train_config.json')+'；原源码及完整候选选择见'+link('reports/GATE2_EXPERIMENT_RESULTS_20261027.md')+'。\n'
 # New 197-source confirmation retains all branch-level raw QA outputs.
 aa=[x for x,_ in jl(g/'avqa/per_qa.jsonl')];ss=js(g/'avqa/summary.json');rows=[]
 nms={'vision_only_64':'固定视觉路由64','legacy_audio_64':'原始真实音频路由64','qualified_audio_64':'可用性约束音频路由64','full_vision_196':'完整视觉196参考','stage3_null_audio_64':'同Stage3零波形BEATs编码','gate2_prefix_64':'冻结前缀Gate','gate2_stream_64':'冻结逐窗口Gate','random_matched_prefix_64':'匹配前缀启用数随机','random_matched_stream_64':'匹配窗口预算随机'}
 for sid in aa[0]['branches']:
  b=[x['branches'][sid] for x in aa];c=sum(x['correct'] for x in b);nl=float(np.mean([x['gold_nll'] for x in b]));expected=ss['all']['methods'][sid];check('historical_197_accuracy',c==expected['correct'],method=sid);check('historical_197_NLL',math.isclose(nl,expected['gold_letter_nll']['mean'],abs_tol=1e-9),method=sid)
  base=[x['branches']['vision_only_64'] for x in aa];rescue=sum(x['correct'] and not y['correct'] for x,y in zip(b,base));harm=sum(not x['correct'] and y['correct'] for x,y in zip(b,base));eid=ev('Gate2-old','AVQA197',sid,'correct,NLL,Rescue,Harm',[c,nl,rescue,harm],g/'avqa/per_qa.jsonl',f'branches.{sid}','utility_gate/grounding/avqa_group_eval.py')
  rows.append([nms[sid],f'{c}/197',pct(c/197),num(nl),f'{rescue}/{harm}',ci(expected['accuracy']['ci95']),eid])
 text+='\n## 冻结Gate2的全新197源确认\n\n192 Both、1 View、4 Sound；独立197题／197源，四个2秒到达观察（8秒）、严格生成ABCD、四选项NLL。冻结Gate2／原Router，没有重新训练或重选阈值；与前面256题是不同队列。\n'+table(['完整策略名','正确／QA','准确率 ↑','Gold NLL ↓','相对视觉挽救／损害','既有源95%CI','证据'],rows)
 text+='原始文件'+link(g/'avqa/per_qa.jsonl')+'，汇总键`all.methods`；这轮仍未超过固定视觉。并行空间验证109合格VGG-SS源，Stage3真实减同权重零波形ROI质量+.030011，CI[.016501,.044311]；旧LLP126源／617合格2秒点AUROC .471382，CI[.390384,.553650]。这些是旧的诊断协议，不能与随后训练代理头的571点AUROC混比。仅聚合引用，未在本附录重算原空间矩阵与617点分数，保留限制。'+link(g/'combined/summary.json','main_table')+'。\n'
 md('historical_audio.md',text)
 p=OUT/'audio_results.md';p.write_text(p.read_text()+'\n更早的256／64／197题Gate2实际结果及其数据、训练定义另见 [历史Audio独立附录](historical_audio.md)，不与本页MUSIC／Valor计算统一提升。\n')

def training_catalog():
 rows=[]
 for a in S['fixed:train/catalog.json']['data']:
  rows.append({'stage':'固定Audio R_A只训练Gate','model':a['model'],'train_QA':656,'train_sources':a['train_sources'],'val_QA':335,'val_sources':a['val_sources'],'params':a['parameters'],'epochs':a['epochs'],'updates':a['optimizer_steps'],'initial_checkpoint':'fresh same seed; R_A frozen','loss':'完整16mask软轨迹CE','selection':'合法验证真实EM/NLL/成本','source':src('fixed','train/catalog.json',a['model'])})
 for a in S['imu_router_training']:rows.append({'stage':'IMU扩充Router','model':a['model'],'train_QA':a['qa'],'train_sources':a['sources'],'val_QA':166,'val_sources':22,'params':a['params'],'epochs':3,'updates':a['steps'],'initial_checkpoint':a['warm_start'] or '同V1初始化query/key，output零初始化','loss':a['loss'],'selection':f'验证生成EM再NLL，epoch{a["epoch"]}','source':a['source']})
 for a in S['imu:locks/Gate_training_catalog.json']['data']:rows.append({'stage':'IMU扩充Gate','model':a['name'],'train_QA':500,'train_sources':65,'val_QA':166,'val_sources':22,'params':a['parameters'],'epochs':a['epochs'],'updates':a['steps'],'initial_checkpoint':'fresh same seed1234','loss':str(a['objective']),'selection':'实际验证完整QA策略锁定','source':src('imu','locks/Gate_training_catalog.json',a['name'])})
 csvout('training_model_ledger.csv',rows)

def verify():
 # Deliberately keep the observed raw-field mismatch; explain its actual meaning.
 metadata=[]
 for x in CHECKS:
  if not x['passed'] and x['kind']=='trace_causality_and_conditional_calls' and x.get('stage')=='gatefirst' and x.get('violations')=={'extra_encode_OFF':1379}:
   x['classification']='historical_metadata_semantics_not_physical_extra_call';x['interpretation']='C_DEPLOY在1379个OFF窗只用1图，但旧actual_second_encoder_called=bool(native or ese>0)恒真；按native_image_count和源码识别。旧字段原样保留。';metadata.append(x)
 note='''\n## 本次审计发现的历史字段歧义

Gate-first原生分支的`actual_second_encoder_called`在源码中写成`bool(native or ese>0)`，所有原生调用都记真。动态单图C_DEPLOY有1379个OFF窗口同条记录`native_image_count=1`，因此不能把该布尔字段当物理“第二图调用”计数。源码按`ims=[frame] if C_DEPLOY and not on else [frame,second]`执行。保留旧日志，重算图数时按native_image_count，性能结论使用付费计时；不把它报告为Gate关闭仍偷偷编码第二图。自动核验保留这一字段不一致项并分类说明。

原始证据：[逐窗口记录](../../artifacts/utility_gate/gate_first_router_second_native_ablation_v1/results/per_window_gate_decisions.jsonl)，`strategy=C_DEPLOY,gate_on=false`；[代码](../../utility_gate/gate_first_native/evaluate.py:126)。另有固定R_A旧“amortized”成本字段的单位歧义，已在成本段按实际公式解释，未修改任何旧JSON。
'''
 for name in ['README.md','audio_results.md']:
  p=OUT/name
  if '## 本次审计发现的历史字段歧义' not in p.read_text():p.write_text(p.read_text()+note)
 # Add the audit-command continuation and ledger links before validating links.
 p=OUT/'README.md'
 if '附加可复现审计脚本' not in p.read_text():p.write_text(p.read_text()+'\n附加可复现审计脚本：`audit_tools/architecture.py`（CPU权重形状）、`audit_tools/enrich.py`（窗口／成本核验与附表）、`audit_tools/finalize.py`（历史附录与最终SHA）。使用上文相同CPU环境运行。全候选模型日志索引：[training_model_ledger.csv](training_model_ledger.csv)。\n')
 # Actual function line numbers are recorded for the code paths used by evidence.
 functions=[]
 for file in sorted({Path(v) for v in CODE.values()}|{REPO/'streamingtom/audio_bottleneck/adbt.py',REPO/'utility_gate/gate_first_native/fusion.py',REPO/'utility_gate/fixed_router_gate/gate_models.py',REPO/'utility_gate/imu_gate_router_transfer/gate.py'}):
  file=file if file.is_absolute() else REPO/file
  if not file.exists():continue
  tree=ast.parse(lines(file))
  for node in ast.walk(tree):
   if isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)):functions.append({'file':rel(file),'symbol':node.name,'line':node.lineno})
 csvout('code_symbol_index.csv',functions)
 for e in EVIDENCE:
  e['code_location']=e.get('code_location','').replace('utility_gate/dual_head/proxy/downstream.py','utility_gate/dual_head/proxy/evaluation.py')
  paths=re.findall(r'[a-zA-Z0-9_/.-]+\.py',e['code_location'])
  if paths:
   matches=[a for a in functions if a['file']==paths[0] and a['symbol'] in ['run','forward','answer','execute','summarise','paired','build']]
   if matches:e['code_location']+='; '+matches[0]['file']+':'+str(matches[0]['line'])+' '+matches[0]['symbol']
 csvout('evidence_index.csv',EVIDENCE)
 # Links are local; no web request is required to audit repository results.
 broken=[]
 for p in OUT.glob('*.md'):
  for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
   if target.startswith(('http:','https:','#')):continue
   clean=target.strip('<>').split('#')[0];clean=re.sub(r':\d+$','',clean)
   if not (p.parent/clean).resolve().exists():broken.append({'report':p.name,'target':target})
 # The verification file is the one intentionally created after the link check.
 broken=[x for x in broken if x['target']!='verification_report.json']
 check('all_local_links_exist',not broken,broken=broken)
 protected={k:v for k,v in GUARD.items() if not (REPO/k).resolve().is_relative_to(OUT)};changed=[];missing=[]
 for key,old in protected.items():
  p=REPO/key
  if not p.exists():missing.append(key)
  elif sha(p)!=old['sha256']:changed.append(key)
 check('source_input_sha_unchanged',not changed and not missing,files=len(protected),changed=changed,missing=missing)
 unexpected=[x for x in CHECKS if not x['passed'] and x not in metadata]
 required=['README.md','audio_results.md','imu_results.md','metric_dictionary.md','strategy_name_mapping.csv','evidence_index.csv','architecture_audio_imu.md']
 rows=list(csv.DictReader((OUT/'strategy_name_mapping.csv').open()))
 for x in rows:
  sid=x['raw_strategy_id']
  if x['stage']=='LLP共同事件代理与独立AVQA':
   x['plain_chinese_name']=proxy_name(sid)
   if sid=='same_stage3_null_audio':x['inference_sensor_input']='直接将已提取音频状态causal_audio置零，不是零波形经过BEATs；该代理阶段特有实现'
  if x['stage']==ST['imu']:
   cp=Path(x.get('checkpoint_path',''));model=cp.parent.name;lock=S.get('imu:locks/'+model+'.json',{}).get('data',{})
   if lock:x['init_checkpoint']=lock.get('warm_start_checkpoint') or 'V1查询与键初始化，低秩output全零；scratch同初始化种子'
   if sid.startswith('T2_') or sid.startswith('BUDGET_'):
    key='accuracy' if 'accuracy' in sid else 'cost';x['gate_checkpoint']=S['imu:locks/validation_selection.json']['data']['selected'][key]['checkpoint'];x['gate_init']='fresh seed1234；Router先冻结'
  if x['stage']==ST['fixed']:
   x['init_checkpoint']='当前R_A权重冻结；本阶段仅Gate从相同seed重新初始化'
   key='accuracy_first' if 'accuracy' in sid or sid=='T3_accuracy' else 'cost_aware' if 'cost' in sid or sid=='T4_cost' else None
   if key:x['gate_checkpoint']=S['fixed:locks/selection_lock.json']['data']['selected'][key]['spec']['checkpoint']
  if x['raw_strategy_id']=='V1':x['init_checkpoint']=S['original_model_inventory']['data']['components']['Stage1_original']['checkpoint']
  if x['stage']==ST['gatefirst'] and x['raw_strategy_id'].startswith('B') and x['raw_strategy_id'] not in ['B0_V1','B1_F0','B4_GA_F0']:x['init_checkpoint']='V1查询前128维与键前128行，output零初始化；V1 checkpoint='+S['original_model_inventory']['data']['components']['V1_retrained']['checkpoint']
 missing_names=[x['raw_strategy_id'] for x in rows if not x['plain_chinese_name']]
 # Empty diagnostic-only cells are explicitly labeled rather than silently guessed.
 for x in rows:
  for key,value in x.items():
   if value=='':x[key]='未核验／该诊断阶段不适用，详见source_file'
 csvout('strategy_name_mapping.csv',rows)
 result={'audit_date':'2026-11-02','status':'complete_with_disclosed_historical_metadata_issue' if not unexpected else 'needs_review','new_training_runs':0,'new_model_inference_runs':0,'checkpoint_mutations':0,'completeness':{k:(OUT/k).exists() for k in required},'primary_aggregate_groups':len(A['tables']),'primary_stored_paired_contrasts':len(A['pairs']),'imu_paid_records':S['imu_paid_records'],'imu_paid_policies_per_role':44,'strategy_mapping_rows':len(rows),'mapping_missing_chinese_names':missing_names,'checks':{'total':len(CHECKS),'passed':sum(x['passed'] for x in CHECKS),'failed_metadata_semantics':len(metadata),'unexpected_failed':len(unexpected),'unexpected_failures':unexpected,'all_kinds':dict(collections.Counter(x['kind'] for x in CHECKS))},'disclosed_historical_metadata_discrepancy':metadata,'source_asset_protection':{'scope':'本轮实际读取的原始预测、配置、日志、源码及选定小模型checkpoint；读取前与报告完成前逐文件SHA256比较；不等同重新核验全部7B权重分片或未读媒体','files':len(protected),'changed':changed,'missing':missing,'old_experiment_writes':0,'new_files_only_under':rel(OUT)},'statistical_scope':{'Audio':'按同一来源聚类，通常2000次bootstrap、seed1234；各队列独立，不合算排名','IMU':'55QA/7录制/4保守录制家族；10000次家族bootstrap；参与者未知；功效有限','random_means':'同QA三个seed的平均，非额外独立样本','CIs_not_created':'本轮只引用原有CI，未伪造未存区间'},'missing_or_not_reverified':['旧Stage3历史实际活跃训练参数与旧checkpoint最优选择未核验；元数据无optimizer状态','最早UTD/Holo零样本聚合仅引用，未重复跑模型／未全量重算逐题','旧Gate2 FullVision256/随机/能量及109空间/617点时序附录聚合明确未逐题重验','LLP六模型有30轮历史及选定轮，逐optimizer更新计数未保存，未补推','7B冻结主干参数量来自已有运行审计，本轮未加载主干重新核数／未重新hash全部大分片','原生、部分cap等策略主聚合未存每项CI；明确留空，不补造','LLP代理阶段same_stage3_null_audio实际直接置零causal_audio特征；与主MUSIC/Valor及VGG-SS零波形经BEATs的null不同，报告已分别解释','未测能耗，未包含网络、冷加载、物理采集等待；不可跨阶段排名延迟'],'comparison_integrity':{'separate_confirmation_cohorts':True,'sensor_specific_vs_task_adaptation_separated':True,'Oracle_and_proxy_separate_from_QA':True,'Audio_B8_inference_intervention_not_DUP_training':True,'IMU_DUP_ZERO_independently_trained':True,'same_IMU_Accuracy_Cost_Gate_named_explicitly':True},'broken_links':broken,'scripts':['audit_tools/audit.py','audit_tools/supplement.py','audit_tools/render_report.py','audit_tools/architecture.py','audit_tools/enrich.py','audit_tools/finalize.py']}
 write('verification_report.json',result);write('audit_checks.json',CHECKS);write('audit_input_guard.json',protected)
 print(json.dumps({'status':result['status'],'checks':result['checks']['total'],'unexpected_failed':len(unexpected),'metadata_issue':len(metadata),'protected_files':len(protected),'changed':len(changed),'broken_links':broken},ensure_ascii=False),flush=True)

if __name__=='__main__':history();training_catalog();verify()
