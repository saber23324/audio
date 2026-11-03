"""Additional trace verification and reader-facing training/data details."""
from render_report import *
CHECKS.extend(js(OUT/'audit_checks.json'))

def trace_checks():
 details=[]
 for z in S['window_traces']:
  stage=z['stage'];actual=collections.Counter();ons=collections.Counter();extras=collections.Counter();violations=collections.Counter()
  for x,ref in jl(REPO/z['path']):
   sid=x['strategy'];actual[sid]+=1;ons[sid]+=x['gate_on'];extras[sid]+=x.get('actual_extra_encoder_called',x.get('actual_second_encoder_called',False));t=x['timestamp_s']
   if x.get('frame_pts_s',t)>t+1e-6:violations['future_RGB']+=1
   tr=x.get('raw_audio_trace')
   if tr and tr.get('raw_both_boundaries_cropped_before_mono_and_resampling') is not True:violations['raw_crop_order']+=1
   if tr and (tr['first_raw_sample_s']<max(0,t-2)-1e-6 or tr['last_raw_sample_interval_end_s']>t+1e-6):violations['audio_out_of_window']+=1
   if x.get('actual_extra_encoder_called',x.get('actual_second_encoder_called',False)) and not x['gate_on']:
    # Native fixed duplicate pairs intentionally encode a second image even OFF.
    if not (stage=='gatefirst' and sid.startswith('C') and sid!='C_DEPLOY'):violations['extra_encode_OFF']+=1
  for sid,count in actual.items():
   tt=[a for a in A['tables'] if a['stage']==stage and a['strategy']==sid];expected=sum(a['window_count'] for a in tt);on=sum(a['on_count'] for a in tt)
   check('trace_window_denominator',expected==count,stage=stage,strategy=sid,actual=count,expected=expected)
   check('trace_ON_numerator',math.isclose(on,ons[sid],abs_tol=1e-8),stage=stage,strategy=sid,actual=ons[sid],expected=on)
   details.append({'stage':stage,'strategy':sid,'actual_windows':count,'actual_ON':ons[sid],'second_encoder_calls':extras[sid]})
  check('trace_causality_and_conditional_calls',not violations,stage=stage,violations=dict(violations))
 csvout('window_call_audit.csv',details)
 # Verify declared timing calls and means/medians from actual serial records.
 rows=[a for a,_ in jl(R['frozen']/'qa_eval/direct_pipeline_cost_dev.jsonl')];groups=collections.defaultdict(list)
 for x in rows:groups[x['strategy']].append(x)
 for sid,aa in groups.items():
  su=S['timing']['frozen']['strategies'][sid];check('timing_sample_size',len(aa)==su['n_calls'],stage='frozen',strategy=sid,n=len(aa))
  for field in ['direct_pipeline_end_to_end_s','direct_pipeline_ttft_s']:
   if field in aa[0]:
    for stat,fn in [('mean',np.mean),('median',np.median)]:check('timing_recompute',math.isclose(float(fn([x[field] for x in aa])),su[field][stat],abs_tol=1e-9),stage='frozen',strategy=sid,field=field,stat=stat)
 for st in ['adaptive','gatefirst']:
  gr=collections.defaultdict(list)
  for x in S['timing'][st]['records']:gr[x['strategy']].append(x)
  for sid,aa in gr.items():
   for field in ['frontend_s','pipeline_total_s','pipeline_TTFT_s']:
    check('timing_recompute',math.isclose(float(np.mean([x[field] for x in aa])),S['timing'][st]['summary'][sid][field],abs_tol=1e-9),stage=st,strategy=sid,field=field,n=len(aa))
 # All sixteen actual QA results exist for every train/val IMU episode.
 ucount=collections.Counter()
 for p in sorted((R['imu']/'results/utility').glob('*.jsonl')):
  aa=[a for a,_ in jl(p)];check('complete_16_utility',len(aa)==16 and {x['policy_code'] for x in aa}==set(range(16)),path=rel(p));ucount[aa[0]['role']]+=len(aa)
 check('utility_train_val_counts',ucount=={'train':8000,'val':2656},counts=dict(ucount))
 # Frozen baseline summaries use a different nesting from dual-image summaries.
 su=js(R['frozen']/'summary.json')
 for x in [a for a in A['tables'] if a['stage']=='frozen' and a['strategy'] in ['V64','V196','A64','A64-null']]:
  st=su['baselines'][x['role']][x['strategy']]['all']
  for key,v in [('correct',st['correct']),('accuracy',st['accuracy']['mean']),('gold_nll',st['gold_nll']['mean'])]:check('frozen_baseline_summary',math.isclose(x[key],v,abs_tol=1e-9),role=x['role'],strategy=x['strategy'],metric=key)

def refine_mapping():
 rows=list(csv.DictReader((OUT/'strategy_name_mapping.csv').open()));iv=S['original_model_inventory']['data']['components']
 v0=rel(iv['Stage1_original']['checkpoint']);v1=rel(iv['V1_retrained']['checkpoint']);a0=rel(iv['Stage3_original_AudioRouter']['checkpoint']);ra=rel(S['gatefirst:locks/fusion_router_lock.json']['data']['selected_checkpoint'])
 bystage={v:k for k,v in ST.items()}
 for x in rows:
  stage=bystage[x['stage']];sid=x['raw_strategy_id'];native='绕过' in x['router_mode'];oldvis=sid in ['V0','V64','D0_RGB_legacy','D0_RGB_neutral','D1_OverlayOnly','D11_equal64_feature_mix'];audio=sid in ['A64','A64-null','A0','A1','Stage3_real','Stage3_null']
  cp=v0 if oldvis else a0 if audio else v1;router='旧Stage1：196→64原视觉值加权采样' if oldvis else '旧Stage3：BEATs控制查询，196→64原RGB值加权采样' if audio else 'V1：196→64原视觉值加权采样'
  x['evaluation_time_protocol']='按时长比例四帧，每帧此前2秒' if stage=='frozen' else '2秒起每2秒，最多32观察，每次此前2秒' if stage=='adaptive' else '8秒观察前缀内2/4/6/8；每时刻此前2秒'
  x['answer_scoring_protocol']='MUSIC规范化文本EM+整串NLL；Valor严格ABCD+四选项NLL' if not stage.startswith('imu') else '官方已完成动作动词衍生ABCD，四选项NLL'
  x['frozen_modules']='SigLIP、视觉投影器、Qwen2及原生特殊嵌入、BEATs、EZ-VSL均冻结；IMU无音频前端'
  x['gate_training_protocol']='无Gate训练' if stage=='frozen' else '见本阶段训练流水账；阈值只在验证集锁定'
  x['router_training_protocol']='旧checkpoint历史训练，元数据见原模型登记'
  # Each branch is described by the module actually invoked, not all models trained in its stage.
  if stage=='adaptive' and sid not in ['V0','V1','V2','A0','A1']:
   router='共享新V1查询库分别编码RGB／辅助图，再β=.25固定槽混合；OFF为V1';cp=v1;x['router_training_protocol']='V1在656QA上继续训练1968步；F0混合器参数训练0'
  if stage=='gatefirst' and sid.startswith('B') and sid not in ['B0_V1','B1_F0','B4_GA_F0']:
   router='同一R_A696320参数低秩双图残差融合；OFF为V1';cp=ra;x['router_training_protocol']='唯一R_A在656QA上首Token CE训练1968步；DUP/SHAM/ZERO是推理替换，非独立训练'
  if stage=='fixed' and sid!='T0_V1':
   router='固定R_A双图残差融合；OFF为V1';cp=ra;x['router_training_protocol']='复用Gate-first阶段R_A；本阶段Router更新0';x['gate_training_protocol']='固定R_A真实完整16组合QA→源均NLL轨迹分布；40Gate各360更新，验证335QA/96源选定'
  if stage.startswith('imu') and not native and sid!='T0_V1':
   model='R_DUP' if ('DUP_trained' in sid or 'R_DUP' in sid) else 'R_ZERO' if ('ZERO_trained' in sid or 'R_ZERO' in sid) else 'R_REAL'
   for suffix in ['scale025','scale050','scratch']:
    if suffix in sid:model+='_'+suffix
   root=R['imu_small'] if stage=='imu_small' or sid.startswith('Old_') else R['imu'];cp=rel(root/f'checkpoints/{model}/selected.pt');router='RGB196+辅助图196→rank32残差+V1基底→64；Gate关闭时只用V1'
   x['router_training_protocol']='同类型v3训练权重热启动，当前主模型500QA×3轮1500更新；scratch与scale按训练表' if root==R['imu'] else 'v2严格因果Router各96次实际更新，v3复用并B=1重验，不把v3重验算训练'
   x['gate_training_protocol']='冻结当前REAL，8000训练+2656验证完整QA→四Gate各1890更新；准确率／成本选相同ranking_e30阈值.5' if root==R['imu'] else '旧32训练QA的16组合→4Gate各120更新；没有新独立确认'
  if native:router='完全绕过所有64-token Router，原生单图／多图视觉插入';cp='冻结VLM主干（见model_inventory）；没有新Router权重';x['router_training_protocol']='本原生分支Router训练0；若使用Gate仍沿用相应Gate权重'
  if sid in ['V2','V196']:router='完整视觉视频196Token/帧及原生video newline；无压缩Router';cp='冻结VLM主干'
  x['router_mode']=router;x['checkpoint_path']=cp
  if sid in ['V0','V64','V1','V2','V196','T0_V1','A0_V1','B0_V1','N0_RGB','N5_V1_64'] or audio:
   x['train_input']='RGB-only历史训练' if not audio else '历史真实音频条件视觉路由训练；零波形仅本次推理干预'
   x['trainable_modules']='本策略评测全部冻结；旧活跃参数见对应模型登记';x['training_updates']='本阶段该旧/固定模型更新0（新V1训练所在阶段1968步另计）'
  if native and stage=='frozen':x['train_input']='冻结预训练视觉语言模型，无本轮训练';x['trainable_modules']='无';x['training_examples']='0';x['training_updates']='0'
  x['init_checkpoint']=cp+'；初始化与热启动上游见router_training_protocol／训练流水账'
 # Diagnostic strategies and aliases must also be discoverable by raw IDs.
 vg={'rgb_clean':('B0','原RGB视觉路由空间诊断'),'gt_overlay':('B1','GT框叠加图Oracle空间诊断'),'sham_overlay':('B2','GT框提示空间错移对照'),'uniform_overlay':('B3','全图均匀叠加对照'),'gt_routing_bias':('B4','GT框直接路由偏置Oracle'),'random_color_overlay':('C1','随机颜色叠加区域对照'),'bbox_outline_overlay':('C2','GT框轮廓提示Oracle'),'original_stage3':('B7','原始Stage3真实音频空间诊断'),'null_stage3':('B8','同Stage3零波形经BEATs空间诊断')}
 template=dict(rows[0])
 for key,(bid,nm) in vg.items():
  for raw in [key,bid]:
   z={k:'' for k in template};z.update(modality='Audio',stage='VGG-SS空间与类别代理诊断',dataset='VGG-SS旧开发125／新开发200／确认500',raw_strategy_id=raw,plain_chinese_name=nm,train_input='冻结旧Stage1或Stage3，未新训练',trainable_modules='无',training_examples='0',training_updates='0',init_checkpoint=a0 if bid in ['B7','B8'] else v0,inference_sensor_input='GT框仅Oracle；B7真实音频；B8零波形经BEATs；其他对照如中文名',gate_mode='未用Gate',router_mode='原始196→64加权重采样；B4直接Oracle路由偏置',view_count='单标注帧',visual_tokens_per_frame='64',visual_tokens_per_question='64',source_file='utility_gate/grounded_image_pilot/eval.py；utility_gate/grounding/scaleup_eval.py',verification_status='实际空间工件核验；不是正式QA',checkpoint_path=a0 if bid in ['B7','B8'] else v0);rows.append(z)
 for z0 in S['diagnostics']['LLP']['qa']:
  sid=z0['method'];z={k:'' for k in template};z.update(modality='Audio',stage='LLP共同事件代理与独立AVQA',dataset='LLP按秒标签；128新源AVQA衍生下游',raw_strategy_id=sid,plain_chinese_name=proxy_name(sid),train_input='LLP共同事件／AVS合成代理与QA收益分别mask监督；主干冻结',trainable_modules='仅对应的小双头网络；其他端点无新增训练',training_examples='开发480 LLP源、QA开发96源；有效标签数见监督工件',training_updates='六个小模型各30轮；逐步更新数原日志未保存，不推算',init_checkpoint='小网络同seed28102026重新初始化；Router冻结',inference_sensor_input='真实因果音频；null端点零波形经BEATs',gate_mode='相应对应／收益阈值在合法验证选择；或固定端点',router_mode='旧视觉路由或原始Stage3硬退回；每帧64视觉Token',view_count='每源最多五个到达帧',visual_tokens_per_frame='64',visual_tokens_per_question='按实际到达帧数',source_file='utility_gate/dual_head/proxy/evaluation.py',verification_status='原始逐点与逐题工件已核验');rows.append(z)
 csvout('strategy_name_mapping.csv',rows)

def append_material():
 # Record actual fitting histories for all tiny proxy models; no invented update count.
 rows=[];root=REPO/'artifacts/utility_gate/dual_head_proxy_20261028_v1/model'
 for p in sorted(root.glob('*/development_metrics.json')):
  a=js(p);hist=js(p.parent/'history.json');rows.append([proxy_name(p.parent.name),a['parameters_total'],a['parameters_trainable'],len(hist),'未记录逐步数',a['best_epoch'],num(hist[-1]['train_loss']),num(hist[-1]['validation_loss']),link(p)])
 extra='\n## 代理网络训练补表（不是下游Router训练）\n\n同一LLP／AVS／QA开发划分，各小网络从相同随机种子重新初始化；对应头用源加权二元交叉熵，收益头用按训练尺度归一化Huber，按可用标签mask分别计算，决策头冻结。基础VLM／Router不更新，验证mask损失选best_epoch。每模型30轮，逐更新日志／优化器step未保存，不能用行数机械补造实际更新。'+link('utility_gate/dual_head/proxy/training.py','fit')+'；'+link('utility_gate/dual_head/losses.py','dual_loss')+'。\n'
 extra+=table(['代理模型中文名','总参数','活跃参数','实际日志轮数','更新步数','选定轮','末轮训练损失','末轮验证损失','原始记录'],rows)
 # Full B0-B8 teacher/category diagnostics stay separate from QA.
 dg=S['diagnostics']['scaleup']['summary'];rows=[]
 for role in ['fresh_dev','fresh_holdout']:
  for mode,a in dg[role]['methods'].items():
   rows.append([role,{'rgb_clean':'RGB视觉参考','gt_overlay':'GT区域叠加Oracle','sham_overlay':'空间错移叠加','uniform_overlay':'均匀叠加','gt_routing_bias':'GT直接路由偏置Oracle','random_color_overlay':'随机颜色对照','bbox_outline_overlay':'GT框线Oracle'}[mode],a['n'],f'{a["category_proxy_correct_count"]}/{a["category_proxy_n"]}',num(a['teacher_agreement']['mean']),num(a['teacher_KL']['mean'])])
 extra+='\n### VGG-SS类别代理与视觉参考一致性\n\n开发200源／确认500源，标注中心帧、四选一类别映射代理；有效代理题数小于空间来源数。标准来自固定事件类别映射，不是官方独立问答；teacher是FullVision参考，不是真值。\n'+table(['来源池','策略解释','空间来源','类别代理正确／适用','teacher一致率','teacher KL ↓'],rows)
 extra+='证据：'+link('artifacts/utility_gate/vggss_feature_scaleup_20261028_v1/summary.json','fresh_dev/fresh_holdout.methods.*')+'；逐源列`category_proxy_correct`、`teacher_agreement`、`teacher_KL`位于'+link('artifacts/utility_gate/vggss_feature_scaleup_20261028_v1/per_source_results.csv')+'。B5/B6没有核验连续预测图来源，保持not_run；原始B0–B8工件未改。\n'
 rows=[]
 for role,v in dg['references_Stage3_independent_table'].items():
  x=v['same_stage3_real_minus_null'];rows.append([role,v['audio_valid_sources'],num(x['spatial']['mass']['mean'],6),ci(x['spatial']['mass']['ci95']),num(x['spatial']['lift']['mean'],6)])
 extra+='\n同权重Stage3真实音频减零波形空间参照（与Stage1不同训练历史单独列）：\n'+table(['来源池','音频合格源','ROI质量配对差 ↑','既有源CI','Lift配对差 ↑'],rows)
 extra+='真实波形可改变旧Stage3的空间路由分布；这不能替代Grounded图分支的真实QA收益。证据键`references_Stage3_independent_table.*.same_stage3_real_minus_null.spatial`。\n'
 p=OUT/'audio_results.md';p.write_text(p.read_text()+extra)
 # Reader-friendly complete split registry, including QA type counts.
 rows=[]
 for x in S['manifest_ledger']:
  if x['stage']=='frozen' or x['role'] in ['stage1_train','stage1_val','gate_train','gate_val','confirmation_music','confirmation_valor','train','val','holdout']:
   rows.append([ST[x['stage']],x['role'],x['qa'],x['sources'],x['families'],json.dumps(x['qa_types'],ensure_ascii=False),link(x['manifest'])])
 md('dataset_details.md','''# 数据集、角色与题型登记

本表由每个manifest实际重数。复用的stage1_train与gate_train不是两批独立样本；同一记录在不同任务目录复用也不能增加样本数。qa_types中Visual=视觉问题、Audio=音频问题、Audio-Visual=音视频联合问题；Holo的英文键是官方动作动词标签。原始类别不作为Gate输入。

MUSIC-AVQA保留官方模板与答案，但按可取得视频重新分配训练／验证／确认角色；合成视频ID排除，来源身份只到官方视频ID，无法核验原YouTube。Valor原AVQA问题／选项四选一，有可审计YouTube来源；每源两题。本地确认不是官方全量分数。HoloAssist以官方完成动作动词／名词生成已完成事件问题，录制家族隔离，参与者未知。VGG-SS与LLP见Audio明细的独立诊断段。
'''+table(['阶段','角色','QA','来源／前缀','保守家族（无专门字段时=来源）','问题类型／动词数量','manifest'],rows))
 p=OUT/'README.md';t=p.read_text().replace('[数据划分登记](dataset_split_ledger.csv)','[数据与题型解释](dataset_details.md)／[划分CSV](dataset_split_ledger.csv)');p.write_text(t)

def figures():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 (OUT/'figures').mkdir(exist_ok=True);fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 aa=[paid('holdout','Fixed_k_'+str(k)) for k in range(5)]
 ax[0].plot([x['mean_k'] for x in aa],[100*x['accuracy'] for x in aa],'o-',label='Fixed first k windows')
 ax[1].plot([x['mean_k'] for x in aa],[x['pipeline_s'] for x in aa],'o-',label='Fixed first k windows')
 for sid,label,marker in [('T2_accuracy','Learned gate','s'),('T8_DUP_trained_always','DUP-trained','^'),('T9_ZERO_trained_always','ZERO-trained','v')]:
  x=paid('holdout',sid);ax[0].scatter(x['mean_k'],100*x['accuracy'],label=label,marker=marker,s=65);ax[1].scatter(x['mean_k'],x['pipeline_s'],label=label,marker=marker,s=65)
 for a in ax:a.set_xlabel('Extra visual encodes / QA');a.set_xticks(range(5));a.grid(alpha=.25);a.legend(fontsize=8)
 ax[0].set_ylabel('Generated ABCD accuracy (%)');ax[1].set_ylabel('Measured pipeline seconds / QA');fig.suptitle('HoloAssist: 55 QA, 7 recordings, 4 families; 8s; 256 visual tokens')
 fig.savefig(OUT/'figures/imu_budget_curve.png',dpi=180);fig.savefig(OUT/'figures/imu_budget_curve.pdf');plt.close(fig)

if __name__=='__main__':
 trace_checks();refine_mapping();append_material();figures()
 write('audit_checks.json',CHECKS);write('audit_input_guard.json',GUARD);csvout('evidence_index.csv',EVIDENCE)
 print('ENRICH checks',len(CHECKS),'failed',sum(not x['passed'] for x in CHECKS),flush=True)
