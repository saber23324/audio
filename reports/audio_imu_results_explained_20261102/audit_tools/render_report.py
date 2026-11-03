"""Render Chinese reports from locally verified historical artifacts; no inference."""
from audit import *
import re
OUT.mkdir(exist_ok=True)
A=js(OUT/'audited_results.json');S=js(OUT/'supplement.json')
GUARD.update(js(OUT/'audit_input_guard.json'))
EVIDENCE.extend(list(csv.DictReader((OUT/'evidence_index.csv').open())))
ST={'frozen':'冻结模型迁移','adaptive':'新视觉路由器与单窗口收益 Gate','gatefirst':'先训练 Gate、再训练融合 Router','fixed':'固定融合 Router、重新训练整段策略 Gate','imu':'扩充数据后的 IMU 融合与 Gate','imu_small':'早期小样本 IMU 直接迁移'}
CODE={'frozen':'utility_gate/grounded_dual_image/experiment.py','adaptive':'utility_gate/adaptive_grounded64/eval_qa.py','gatefirst':'utility_gate/gate_first_native/evaluate.py','fixed':'utility_gate/fixed_router_gate/online_inference.py','imu':'utility_gate/imu_visual/scaleup_train/cost.py','imu_small':'utility_gate/imu_gate_router_transfer/evaluate.py'}
R=ROOTS
def link(p,label=None):
 p=Path(p);p=p if p.is_absolute() else REPO/p
 try:target='../../'+str(p.relative_to(REPO))
 except ValueError:target=str(p)
 return f'[{label or str(p.relative_to(REPO)) if p.is_relative_to(REPO) else label or str(p)}]({target})'
def src(stage,file,key=''):
 return link(R[stage]/file)+(f'，`{key}`' if key else '')
def md(name,text):
 (OUT/name).write_text(text.strip()+'\n')
def table(headers,rows):
 def cell(x):return str(x).replace('|','／').replace('\n','<br>')
 return '\n\n'+'| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(cell(c) for c in row)+' |' for row in rows)+'\n\n'
def num(x,d=4):return '未记录' if x is None else f'{x:.{d}f}'
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def ci(x,pp=False):
 if x is None:return '未存该项 CI'
 if isinstance(x,dict):x=x.get('ci95',x.get('accuracy_delta_ci95',x.get('CI95')))
 if not isinstance(x,(list,tuple)):return '见原始键'
 return (f'[{100*x[0]:.2f}, {100*x[1]:.2f}] pp' if pp else f'[{x[0]:.4f}, {x[1]:.4f}]')
def get(stage,role,sid):return next(x for x in A['tables'] if x['stage']==stage and x['role']==role and x['strategy']==sid)
def paid(role,sid):return next(x for x in S['imu_paid'] if x['role']==role and x['strategy']==sid)
def refs(x):
 rr=x.get('refs',{});keys=['correct','accuracy','gold_nll','on_rate','mean_k','visual_tokens_mean','pipeline_s'];vals=[rr[k] for k in keys if k in rr];return '、'.join(vals)
def named(stage,sid):
 if stage=='frozen':
  names={'V64':'原始视觉专用路由器，RGB 压缩至每帧 64 Token','V196':'完整视觉参考，每帧 196 内容 Token','A64':'原始音频条件路由器，使用真实因果音频','A64-null':'同一个原始音频路由器，改用零波形经 BEATs 编码','D0_RGB_legacy':'原始 RGB 64 Token 与原提示词','D0_RGB_neutral':'相同 RGB 64 Token 与中性提示词','D1_OverlayOnly':'只输入预测热图叠加图，经原视觉路由器压缩','D11_equal64_feature_mix':'RGB 与预测叠加图共享原视觉查询库，固定比例混合至 64 Token','D2_RGB_RGB':'原生重复 RGB 双图，使用双图引导提示词','D3_RGB_Grounded_neutral':'原生 RGB 与预测叠加图双图，中性提示词','D4_RGB_Grounded_guided':'原生 RGB 与预测叠加图双图，带不确定性说明的引导提示词','D5_Grounded_RGB_guided':'原生预测叠加图在前、RGB 在后的双图顺序对照','D6_RGB_Sham_guided':'原生 RGB 与空间错移热图双图对照'}
  return names[sid]
 if stage=='adaptive':
  names={'V0':'旧视觉专用路由器','V1':'在 MUSIC 训练源继续训练的新视觉专用路由器','V2':'完整视觉参考','A0':'原始音频条件路由器与真实音频','A1':'原始音频条件路由器与零波形编码','F0':'新视觉查询库的预测叠加图固定混合，所有时刻启用','G0':'旧双头代理 Gate 迁移到固定混合分支','G1':'新单窗口收益 Gate 控制固定混合分支，锁定后全关','G2':'冻结旧共同事件代理主干、仅训练新收益头的 Gate，锁定后全关','R_ONLY':'仅用共同事件代理分数的规则，验证后全关','ENERGY':'仅用音频能量的规则，验证后全关','G4':'沿用新 Gate Mask，把辅助图替换为重复 RGB','G5':'沿用新 Gate Mask，把辅助图替换为空间错移图','G6':'沿用新 Gate Mask，零音频波形通过定位器生成辅助图','G7':'始终启用固定预测叠加混合的复核分支'}
  if sid.startswith('G3_'):return '按验证启用概率进行随机选择，'+('三个种子的逐题均值（非新增样本）' if 'mean' in sid else '随机种子 '+sid.split('seed')[-1])
  return names[sid]
 if stage=='gatefirst':
  names={'V0':'旧视觉专用路由器','V2':'完整视觉参考','Stage3_real':'原始音频路由器与真实音频','Stage3_null':'原始音频路由器与零波形编码','A0_V1':'固定新视觉专用路由器，所有时刻关闭辅助分支','B0_V1':'同一固定新视觉专用路由器参照','A1_F0':'固定预测叠加图混合分支，所有时刻启用','B1_F0':'同一固定预测叠加混合参照','A2_u':'只根据新收益头选择固定混合分支','A3_dual':'新收益与冻结共同事件代理联合选择固定混合分支','A_MAIN_GA':'验证选定的主 Gate 控制固定混合分支','A5_old':'旧代理 Gate 控制固定混合分支','A_vision':'只看视觉特征的收益 Gate 控制固定混合分支','A_energy':'音频能量规则控制固定混合分支','A_cosine':'冻结特征余弦规则控制固定混合分支','B2_R_always':'训练后的预测叠加图融合 Router，所有时刻启用','B3_GA_R':'冻结的第一阶段 Gate 控制新训练融合 Router','B4_GA_F0':'同一冻结 Gate 控制原固定混合分支','B5_GA_dupR':'同一 Gate 与真实叠加图训练的融合 Router，推理改用重复 RGB','B6_GA_shamR':'同一 Gate 与真实叠加图训练的融合 Router，推理改用空间错移图','B8_dupR_always':'同一个真实叠加图训练的融合 Router，始终输入重复 RGB','B9_shamR_always':'同一个真实叠加图训练的融合 Router，始终输入空间错移图','B10_zeroR_always':'同一个真实叠加图训练的融合 Router，始终输入零波形定位图','C0_RGB_RGB':'完全绕过 Router，原生重复 RGB 双图','C1_always_ground':'完全绕过 Router，原生 RGB 与预测叠加图始终启用','C2_GA_native':'冻结 Gate 选择原生辅助图内容；关闭时仍输入重复 RGB','C3_GA_sham':'相同 Gate Mask 的原生空间错移图；关闭时重复 RGB','C4_GA_dup':'相同 Gate Mask 的原生重复 RGB 对照','C6_always_sham':'原生空间错移图双图，始终启用','C7_old_native':'旧 Gate 选择原生辅助图；关闭时重复 RGB','C9_GA_zero':'相同 Gate Mask 的原生零波形定位图','C10_always_zero':'原生零波形定位图双图，始终启用','C_DEPLOY':'原生动态单双图部署分支，关闭时只保留一张 RGB'}
  for prefix,label in [('A4_random','随机 Gate 控制固定混合分支'),('B7_randomR','随机 Gate 控制训练后融合 Router'),('C5_random','随机 Gate 选择原生双图内容')]:
   if sid.startswith(prefix):return label+'，'+('三个种子的逐题均值（非新增样本）' if 'mean' in sid else '种子 '+sid[len(prefix):])
  return names[sid]
 if stage=='fixed':
  names={'T0_V1':'固定新视觉专用路由器，关闭全部辅助分支','T1_RA_always':'固定已训练音频叠加图融合 Router，全部合法时刻启用','T2_old_gate':'旧的固定混合收益 Gate 迁移到当前融合 Router','T3_accuracy':'针对当前融合 Router 的完整 16 组合收益训练，准确率优先 Gate','T4_cost':'针对当前融合 Router 的完整 16 组合收益训练，成本优先 Gate','T6_visual':'仅用视觉特征训练的完整组合 Gate'}
  if sid in names:return names[sid]
  if sid.startswith('ABL_'):return '完整组合 Gate 输入消融：'+{'audio_r':'视觉、BEATs 与共同事件代理','audio_no_r':'视觉与 BEATs，不用代理','acoustic':'视觉与低成本声学统计','history':'视觉、BEATs、代理与历史决策'}[sid[4:]]
  if sid.startswith('BUDGET_prefix'):return f'固定顺序策略：仅最早 {sid[-1]} 个合法窗口启用融合 Router'
  if sid.startswith('T10_'):return ('准确率优先' if 'accuracy' in sid else '成本优先')+f' Gate，加因果累计上限 {sid[-1]} 张辅助图'
  if sid.startswith('T5_'):return ('准确率优先' if 'accuracy' in sid else '成本优先')+'验证预算的随机 Gate，'+('三个种子的逐题均值（非新增样本）' if 'mean' in sid else '种子 '+sid.rsplit('_',1)[-1])
  return ('准确率优先' if 'accuracy' in sid else '成本优先')+' Gate 的相同 Mask，对同一真实叠加图训练 Router 做推理替换：'+{'duplicate':'重复 RGB','sham':'空间错移热图','zero':'零音频波形经过定位器'}[sid.rsplit('_',1)[-1]]
 if stage in ['imu','imu_small']:
  names={'T0_V1':'固定视觉专用路由器，所有时刻只用 RGB','T1_REAL_always':'真实同步 IMU 曲线图训练的融合 Router，四时刻全部启用','T8_DUP_trained_always':'独立用重复 RGB 训练的融合 Router，四时刻全部启用','T9_ZERO_trained_always':'独立用零 IMU 曲线图训练的融合 Router，四时刻全部启用','T2_accuracy':'准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器','T2_cost':'成本优先选择名义分支；最新实验实际与准确率 Gate 相同','T10_REAL_wrong_always':'固定真实 IMU 训练 Router，始终换成错误来源 IMU 图','Old_R_REAL_always':'旧小样本真实 IMU Router，在本轮新队列始终启用','Old_R_DUP_always':'旧小样本重复 RGB 训练 Router，在本轮新队列始终启用','Old_R_ZERO_always':'旧小样本零 IMU 训练 Router，在本轮新队列始终启用','Old_v3_gate_accuracy':'旧小样本准确率 Gate 迁移，配旧真实 IMU Router','Old_v3_gate_cost':'旧小样本成本 Gate 迁移，配旧真实 IMU Router','R_REAL_scale025_always':'较少训练来源的真实 IMU Router，始终启用','R_REAL_scale050_always':'中等训练来源的真实 IMU Router，始终启用','R_REAL_scratch_always':'从零输出残差初始化训练的真实 IMU Router，始终启用','R_DUP_scratch_always':'从相同零输出残差初始化训练的重复 RGB Router，始终启用','Cheap_gyro_std_0p20':'陀螺仪标准差阈值 0.20 的固定规则选择真实 IMU Router','N0_RGB':'原生 RGB 单图输入，完全绕过路由器','N1_RGB_IMU':'原生 RGB 与真实 IMU 曲线双图，完全绕过路由器','N2_RGB_DUP':'原生 RGB 与重复 RGB 双图对照','N3_RGB_ZERO':'原生 RGB 与零 IMU 曲线双图对照','N4_RGB_WRONG':'原生 RGB 与错误来源 IMU 曲线双图对照','N5_V1_64':'固定视觉路由器 64 Token 参照','N6_IMU_only_diagnostic':'仅 IMU 曲线图的原生视觉诊断'}
  if sid in names:return names[sid]
  if sid.startswith('Fixed_k_'):return f'固定最早 {sid[-1]} 个时刻启用真实 IMU Router（其余用视觉路由器）'
  if sid.startswith('BUDGET_'):return ('准确率' if 'accuracy' in sid else '成本')+f' Gate 的因果辅助图上限 {sid[-1]} 张'
  if sid.startswith('T4_'):return ('准确率' if 'accuracy' in sid else '成本')+'验证预算的随机 Gate，种子 '+sid.rsplit('_',1)[-1]
  if 'same_mask' in sid:return ('准确率' if 'accuracy' in sid else '成本')+' Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成'+{'DUP':'重复 RGB','ZERO':'零 IMU 曲线图','WRONG':'错误来源 IMU 曲线图'}[sid.rsplit('_',1)[-1]]
 raise KeyError((stage,sid))

def mapping():
 out=[]
 allrows=A['tables']+S['imu_paid']
 for stage,sid in sorted({(x['stage'],x['strategy']) for x in allrows}):
  aa=[x for x in allrows if x['stage']==stage and x['strategy']==sid];nm=named(stage,sid);native=(stage=='frozen' and sid in ['D2_RGB_RGB','D3_RGB_Grounded_neutral','D4_RGB_Grounded_guided','D5_Grounded_RGB_guided','D6_RGB_Sham_guided']) or (stage=='gatefirst' and sid.startswith('C')) or sid.startswith('N')
  train='复用冻结的旧视觉／音频路由器，未新训练';mods='本阶段无新训练';examples=updates='0（本阶段）';init='见模型 checkpoint 登记'
  if stage=='adaptive':train='V1：MUSIC RGB；Gate：固定 V1/F0 的逐窗口真实问答 NLL 干预';mods='V1 的查询库与视觉键；两个新收益 Gate（对应时使用）';examples='V1 656 QA／192源；Gate 5697个有效训练窗口';updates='V1 1968；收益 Gate 各2700'
  elif stage=='gatefirst':train='先用固定 V1/F0 的完整16组合收益训练 Gate；再用 RGB+真实预测叠加图训练唯一 R_A';mods='Gate 576097等；唯一融合 Router 696320';examples='656 QA／192源，验证157 QA／48源';updates='Gate各360；R_A1968'
  elif stage=='fixed':train='固定 R_A，对656 QA的全部16组合生成真实QA收益后训练Gate';mods='仅新 Gate；R_A、V1、全部主干冻结';examples='656 QA／192训练源；335 QA／96验证源';updates='40个 Gate 各360；Router 0'
  elif stage.startswith('imu'):
   dup='DUP_trained' in sid or 'R_DUP' in sid;zero='ZERO_trained' in sid or 'R_ZERO' in sid
   train='RGB + '+('重复 RGB' if dup else '零 IMU 曲线图' if zero else '真实同步 IMU 曲线图')
   if 'same_mask' in sid:train='RGB+真实 IMU 图训练同一 REAL Router；DUP/ZERO/WRONG 仅推理干预'
   mods='Fusion Router 的 query/key/value/output；Gate 条件策略参数（按阶段分别训练）'
   examples='500训练QA／65录制／35家族；Gate用同训练集完整16组合' if stage=='imu' else '32训练QA／2录制；32验证QA／2录制，无独立确认'
   updates='主Router各1500；Gate各1890' if stage=='imu' else '复用v2 Router各96次更新；v3只重验Router并训练Gate各120次'
   if 'scale025' in sid:examples='91训练QA／9家族';updates='273'
   if 'scale050' in sid:examples='200训练QA／18家族';updates='600'
   if sid.startswith('Old_'):train='复用v3小样本相应Router／Gate，不在本轮重训该旧模型';updates='旧Router96；旧Gate120；本轮旧模型更新0'
   if sid.startswith('N') or sid in ['T0_V1','N5_V1_64']:train='冻结模型原生推理或视觉基线；无本轮训练';updates='0';mods='无'
  sensor='真实同步 IMU，六通道曲线图' if stage.startswith('imu') else '真实音频；预测 EZ-VSL 热图叠加在当前 RGB'
  if any(q in sid.lower() for q in ['null','zero']):sensor='零 IMU 数值经过固定坐标曲线绘图与冻结视觉编码器' if stage.startswith('imu') else '零音频波形；原AudioRouter经BEATs，Grounded分支经EZ-VSL；不是直接零特征'
  elif any(q in sid.lower() for q in ['dup','rgb_rgb']):sensor='重复同一 RGB；仍可能计算真实传感器特征以保持 Gate Mask'
  elif 'sham' in sid.lower():sensor='真实预测热图做确定性空间错移，非GT；传感器Gate Mask保持'
  elif 'WRONG' in sid or 'wrong' in sid:sensor='同划分另一来源的IMU；离线干预供体选择可用类别标签，非部署策略'
  elif sid in ['V0','V1','V2','V64','V196','D0_RGB_legacy','D0_RGB_neutral','A0_V1','B0_V1','T0_V1','N0_RGB','N5_V1_64']:sensor='无辅助信息进入路由输出；视觉专用基线'
  gate='始终启用' if 'always' in sid or sid in ['F0','G7','A1_F0','B1_F0','C1_always_ground'] else '随机选择（验证预算锁定）' if 'random' in sid or sid.startswith('G3_') else '固定因果顺序／预算规则' if 'Fixed_k' in sid or 'BUDGET' in sid or '_cap' in sid else '见中文策略名；学习阈值仅在验证集选择'
  if sid in ['V0','V1','V2','V64','V196','D0_RGB_legacy','D0_RGB_neutral','A0_V1','B0_V1','T0_V1','N0_RGB','N5_V1_64']:gate='无 Gate；视觉基线'
  if stage=='frozen':gate='未使用 Gate'
  if stage=='imu' and sid in ['T2_accuracy','T2_cost']:gate='同一 ranking_e30_p0.5 checkpoint，阈值0.5；两名称同一策略'
  tok=sorted({round(x['visual_tokens_mean'],6) for x in aa});out.append(dict(modality='IMU' if stage.startswith('imu') else 'Audio',stage=ST[stage],dataset='；'.join(sorted({x['dataset'] for x in aa})),raw_strategy_id=sid,plain_chinese_name=nm,train_input=train,trainable_modules=mods,training_examples=examples,training_updates=updates,init_checkpoint=init,inference_sensor_input=sensor,gate_mode=gate,router_mode='完全绕过Router，使用原生多图接口' if native else '视觉专用／原始AudioRouter／固定混合／低秩残差融合；以中文名及源码分支为准',view_count='原生四或八图；C_DEPLOY为4+k' if native else '每观察RGB一张；辅助分支开启时第二次视觉编码（DUP部分复用）',visual_tokens_per_frame='原生图729+newline或全视觉196，详见实际值' if native or sid in ['V2','V196'] else '64',visual_tokens_per_question='；'.join(str(x) for x in tok),source_file=CODE[stage],verification_status='逐题结果与策略实现已核对；旧模型未提供的训练元数据不补造'))
 csvout('strategy_name_mapping.csv',out);return out

def stage_tables(stage,roles):
 out=''
 for role in roles:
  aa=[x for x in A['tables'] if x['stage']==stage and x['role']==role];example=aa[0];music=example['dataset']=='music_avqa'
  window='每段视频按时长比例取四帧（12.5%、37.5%、62.5%、87.5%）；每帧只用此前2秒音频' if stage=='frozen' else '从2秒起每2秒一个观察，最多32帧；不按未来总时长重排' if stage=='adaptive' else '固定已观察8秒前缀，2／4／6／8秒四帧；每帧此前2秒音频'
  out+=f'\n### {"MUSIC-AVQA" if music else "Valor32k 衍生音视频问答"}：{role}\n\n'
  out+=f'独立队列表：{example["n_sources"]} 个来源、{example["n_qa"]} 道 QA。时间范围：{window}。答案与评分：'+('原生短答案文本规范化精确匹配；数字0–11转词；NLL为标准答案整串全词表Token负对数概率之和（不含EOS）。' if music else '原数据选项保持的四选一；贪心输出严格ABCD匹配；NLL为首Token在ABCD四选项内重新归一化。')+'\n\n'
  out+='下表每行中文名称说明实际分支，训练／推理完整对照见 [策略名称总表](strategy_name_mapping.csv)。视觉Token是每题实际均值；ON与k按来源去重，未使用Gate时的端点开关仅用于描述分支。随机均值保留小数正确数，不增加样本量。原始文件：'+src(stage,{'frozen':'summary.json','adaptive':'qa_eval/per_source_results.jsonl','gatefirst':'results/per_qa.jsonl','fixed':'per_qa_results.jsonl'}[stage])+'。\n'
  rows=[]
  for x in aa:
   rows.append([named(stage,x['strategy'])+f'（`{x["strategy"]}`）',f'{num(x["correct"],0) if float(x["correct"]).is_integer() else num(x["correct"],3)}/{x["n_qa"]}',pct(x['accuracy']),num(x['gold_nll']),pct(x['on_rate']),num(x['mean_k'],3),num(x['visual_tokens_mean'],2),num(x['pipeline_s']),refs(x)])
  out+=table(['完整策略名称与说明','正确／QA','准确率 ↑','Gold NLL ↓','ON率','额外窗口k／来源','LLM视觉Token／题','独立Pipeline秒／题','证据编号'],rows)
  if stage!='fixed':out+='本表 Pipeline 留空表示本轮主预测共用了前端或没有独立计时，不以语言模型生成时间冒充端到端时间；开发计时另表。\n'
 return out

def paired_table(stage,role,contrasts):
 rows=[]
 for aid,bid in contrasts:
  p=next((x for x in A['pairs'] if x['stage']==stage and x['role']==role and x['a']==aid and x['b']==bid and x['stratum'] in ['ALL','all','全部QA']),None)
  if not p:continue
  ss=p.get('stored',{});c=ss.get('accuracy_delta_CI95',ss.get('accuracy_cluster_CI',ss.get('delta_accuracy',{}).get('ci95') if isinstance(ss.get('delta_accuracy'),dict) else None))
  for key in ['accuracy_difference','accuracy_delta','delta_accuracy']:
   if c is None and isinstance(ss.get(key),dict):c=ss[key].get('ci95')
  rows.append([named(stage,aid)+' − '+named(stage,bid),p['n'],num(p['rescue'],3).rstrip('0').rstrip('.'),num(p['harm'],3).rstrip('0').rstrip('.'),f'{100*p["delta"]:+.3f}',ci(c,True),num(p['nll_improvement']),link(p['source_file'],'原统计')+' `'+p['source_key']+'`'])
 return table(['同队列配对：方法 − 基线','QA','Rescue ↑','Harm ↓','准确率差 pp','既有95%CI','NLL改善 ↑','证据'],rows)

def training():
 rows=[['原始视觉专用Stage1','VideoMME旧训练split，720源；QA数按旧manifest另见索引','旧训练中间权重续训；最终adbt_epoch_3.pt','总1991680，历史活跃933888：query_tokens与visual_key','配置3轮；最终global_step6480；元数据epoch=2按零索引解释，不能当只训练两轮','首答案Token全词表CE；AdamW lr1e-5, wd.01, clip1, seed1234','未核验旧checkpoint最优选择／旧优化器状态缺失','artifacts/imu_visual_next_phase_20261030/M0_snapshot/checkpoint_training_metadata.json'],['原始音频Stage3','同一旧VideoMME split','另一个音频训练checkpoint；非Stage1同权重音频开关','总1991680；历史活跃参数未核验','配置3轮；最终global_step6480；旧优化器状态缺失','旧phase4首答案Token CE；其他训练历史不外推到phase5','旧最优选择未核验；不混用论文摘要','artifacts/imu_visual_next_phase_20261030/M0_snapshot/checkpoint_training_metadata.json'],['新视觉专用V1','MUSIC训练656QA／192源；验证157／48','旧Stage1最终权重，fresh AdamW','933888，query_tokens+visual_key；VLM冻结','3轮／1968更新','首答案Token CE；lr1e-5, wd.01, clip1, QA batch1, seed1234','验证首标准答案Token NLL最低；选第3轮；不是Full-Vision KD','artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/locks/stage1_retrained_lock.json'],['自适应收益Gate两种','5697训练有效窗口／1436验证有效窗口；完整视频QA单窗口干预','一支重置收益网络；另一支冻结旧combined共同事件代理主干只换收益头','573953活跃；冻结主干支仅65活跃；各总574059','各30轮／2700更新','源加权Huber，训练分位尺度；lr.001, wd.01, batch64','验证MAE选第1／8轮；完整混合QA选阈值，均全关','artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/models/new_utility_gate/development_summary.json'],['先训练Gate阶段五模型','训练656QA／192源、验证157／48；全部16组合13008次QA','固定V1/F0；收益／排序／双头代理／视觉／分类候选','不同小Gate，主Gate576097','各30轮／360更新，共1800更新','完整mask收益与条件边际的Huber／排序／分类；不是简单相加；lr.001','合法验证生成EM、NLL及视觉保留约束；主dual_rank第30轮','artifacts/utility_gate/gate_first_router_second_native_ablation_v1/locks/gate_a_lock.json'],['音频融合Router R_A','同656训练QA／192源，验证157／48','V1查询／键截取初始化；低秩output为零；Gate先冻结','696320：query,key,value,output；全部主干/V1/Gate冻结','3轮／选定轨迹1968更新；中断续跑记录保留','首答案Token CE；AdamW1e-5,wd.01,clip1,seed1234；强制ON训练暴露','验证Always-on真实分支EM再NLL；第3轮；14零梯度样本，30未提交更新回滚','artifacts/utility_gate/gate_first_router_second_native_ablation_v1/locks/fusion_router_lock.json'],['固定R_A的新完整策略Gate','训练656QA／192源；新验证335QA／96源','R_A与V1固定；五种输入家族×四lambda×二温度','各465409–576161；所有Router更新0','40个模型各30轮／360更新，总14400','源等权的16轨迹软目标CE：q(S)∝exp[-(NLL(S)+λk)/T]；lr.001','977验证候选中锁准确率优先与成本优先；不是确认集选参','artifacts/utility_gate/fixed_router_gate_v1/train/catalog.json']]
 out=table(['模型／阶段','训练与验证数据','初始化','实际训练模块；冻结项','轮数与更新','目标及优化设置','模型选择与失败记录','原始元数据'],[x[:-1]+[link(x[-1])] for x in rows])
 out+='旧模型的6480是checkpoint记录的global_step，不等于6480个独立视频。原训练每段视频多道QA，精确QA分布与验证记录若无日志则不补造。原始Stage1/Stage3 checkpoint文件名与元数据不同索引，表中保留原值。\n'
 return out

def audio_report():
 txt='''# Audio 历史实验结果明细

本文件按实际队列逐轮整理。Grounded Image 指冻结声源定位器预测热图叠加到RGB形成的辅助图，不是标准答案热图。Gate 是决定是否计算辅助分支的小网络；Router 是把视觉特征重采样或融合成固定数量视觉Token的模块。所有VLM语言模型、视觉编码器、视觉投影器保持冻结；表中的重新训练指小型路由器／Gate。

## 原模型与训练流水账
'''+training()+'''
原始Stage1用RGB决定64个查询与196位置的加权采样；原始Stage3用BEATs音频状态改变查询，但输出值仍来自RGB。真实音频与零波形必须在同一个Stage3内比较；Stage1和Stage3不同训练checkpoint的差异不能当纯音频因果效应。零音频是零波形经冻结BEATs，非把最终768维状态设零。

## 冻结迁移：没有训练Gate或新Router

训练：本阶段新增0。MUSIC开发集715 QA／200视频ID（含32来源pilot），确认701 QA／500视频ID；Valor确认256 QA／128已知YouTube源。原MUSIC官方QA划分的视频有交叉，本项目重建来源角色；MUSIC缺原始YouTube映射，不能声称完全源血缘独立。四帧按视频时长比例选取，和后续8秒前缀协议不同。出处：'''+src('frozen','audit/music_metadata.json','selected、native_splits')+'；'+link('utility_gate/grounded_dual_image/data.py','数据与normalize函数')+'。\n'
 txt+=stage_tables('frozen',['music_confirm','valor_confirm'])
 for role in ['music_confirm','valor_confirm']:
  txt+=f'\n冻结迁移配对：{role}，队列／时间／评分同上；均为原生双图5840 Token（等64混合另列）。\n'
  txt+=paired_table('frozen',role,[('D4_RGB_Grounded_guided','D2_RGB_RGB'),('D4_RGB_Grounded_guided','D6_RGB_Sham_guided'),('D11_equal64_feature_mix','D0_RGB_legacy')])
 txt+='原生双图每题5840视觉Token，压缩路径256；FullVision196的四帧视频接口为785（含原生视频换行），并非原生729-patch图像接口。D0_RGB_legacy与V64是原提示词复现；提示词、图像顺序、重复图与空间错移均在本轮控制。D11只固定混合旧查询库的两组视觉特征，没有训练新的Fusion Router。\n'
 txt+='## 新视觉路由器与单窗口收益Gate\n\n训练与验证都只用官方MUSIC训练来源，冻结VLM后继续训练V1。F0沿用V1同一查询槽，固定β=.25混合RGB／预测图，β候选为.25与.5；β=0仅参照，不是候选。Gate标签为“其他帧保持RGB、只加一个窗口辅助图”的完整QA NLL干预，尚未建模16种组合相互作用。'+src('adaptive','locks/fuser_lock.json','beta、candidates_and_results')+'；'+src('adaptive','locks/utility_label_lock.json','distribution')+'。\n'
 txt+=stage_tables('adaptive',['confirmation_music','confirmation_valor'])
 txt+='新收益Gate验证MAE为0.008201，零预测参照0.007831；Spearman为−0.007306。冻结旧主干只训练新收益头验证MAE为0.007958。验证选择的G1/G2阈值无穷大，全部OFF，因而确认输出与V1相同。此结果表示没有学到可验证的收益选择，不能写成“自动过滤证明音频无价值”，也不能写成Gate没有训练。证据：'+src('adaptive','models/new_utility_gate/development_summary.json','validation_baselines、new_utility_only、new_utility_on_frozen_combined_corr_trunk')+'；'+src('adaptive','locks/policy_lock.json','policies')+'。\n'
 txt+='## 固定8秒：先Gate、后Router、再原生双图\n\nGate先针对固定V1/F0的16组合收益训练，冻结G_A，再独立训练唯一R_A。G_A没有按R_A重新生成标签，转到R_A及原生双图属于收益目标迁移，不是匹配训练。实际执行Router训练和确认推理，即使之前全OFF也没有停止。'+src('gatefirst','locks/fusion_router_lock.json','Gate_u_not_relabelled_for_new_Router、optimizer_steps')+'。\n'
 txt+=stage_tables('gatefirst',['confirmation_music','confirmation_valor'])
 txt+='### 这一轮可公平回答什么\n\nB2、B8、B9、B10使用同一R_A权重、同样每帧64 Token；B8并不是独立DUP训练。MUSIC下四者均206/288，而V1为173/288、固定F0为171/288，说明小Router训练能适配该队列，尚未识别真实音频特有收益。Valor下R_A为321/508、V1为326/508，未确认跨域改进。\n\n原生C0–C10完全绕过Router。多数C分支Gate OFF仍给第二张重复RGB，始终5840 Token，并不省第二张原生图编码；C_DEPLOY才动态减少图数，需单独按预算比较。原生RGB+预测图MUSIC190/288、重复RGB190/288；Valor348/508对344/508，需看同队列配对区间，不能和256 Token路径直接归因比较。出处：'+link('utility_gate/gate_first_native/evaluate.py','variants、run_source')+'；'+src('gatefirst','results/summary.json','strategies、paired_strata')+'。\n'
 for role in ['confirmation_music','confirmation_valor']:
  txt+=f'\n配对表：{role}；同上固定8秒、MUSIC文本EM／Valor严格ABCD协议。\n'
  txt+=paired_table('gatefirst',role,[('B2_R_always','B1_F0'),('B2_R_always','B8_dupR_always'),('B2_R_always','B9_shamR_always'),('B2_R_always','B10_zeroR_always'),('B3_GA_R','B2_R_always'),('C1_always_ground','C0_RGB_RGB'),('C2_GA_native','C4_GA_dup')])
 txt+='## 固定R_A：只训练新的完整组合Gate\n\n这是另一个新的确认队列，不能与上一轮288／508题合算提升。实际训练40个Gate，Router更新数为0。训练监督使用固定R_A全部16种ON/OFF组合的完整QA结果，先在来源内取QA平均NLL，再拟合因果轨迹分布，不把单窗口收益相加。新验证335 QA／96源选定：准确率优先为history_l0_t0.25第1轮、阈值.5；成本优先为acoustic_l1_t0.25第20轮、阈值.25。后者只用廉价声学统计，不调用BEATs。'+src('fixed','locks/selection_lock.json','selected')+'。\n'
 txt+=stage_tables('fixed',['confirmation_music','confirmation_valor'])
 for role in ['confirmation_music','confirmation_valor']:
  txt+=f'\n配对表：{role}；同上固定8秒、MUSIC文本EM／Valor严格ABCD协议。\n'
  txt+=paired_table('fixed',role,[('T3_accuracy','T0_V1'),('T4_cost','T1_RA_always'),('T3_accuracy','T5_random_accuracy_mean'),('T4_cost','T5_random_cost_mean'),('T3_accuracy','T7_accuracy_duplicate'),('T4_cost','T9_cost_zero')])
 txt+='准确率优先并未稳定胜过同预算随机Gate；成本策略减少了额外编码，但预设1个百分点非劣性没有得到独立确认。MUSIC282题的改善相对V1存在；Valor504题V1与R_A均310题正确，新Gate309／308题，且音视频相关主子集也未确认音频收益。相同Mask下重复图／错移图／零音频对照缺少可靠准确率差。保持原研究判断：选择能力证据不足，实际省算成立但准确率非劣性不足，真实音频独特收益NO-GO。'+src('fixed','judgment.json')+'。\n'
 txt+=timing_report()
 txt+=diagnostic_report()
 txt+='\n## 历史背景与未完成项目\n\n旧VideoMME训练及论文数字只作为模型来源：训练split见'+link('results/adbt_videomme_split.json')+'，训练实现见'+link('streamingtom/train_adbt_videomme.py','train：qa_loss')+'。本轮没有重新运行旧VideoMME基准，不将论文结果当本轮复现。此前Gate2、双头learnability及代理验证档案见'+link('reports/GATE2_EXPERIMENT_RESULTS_20261027.md')+'、'+link('reports/GATE2_GROUNDING_VALIDATION_20261027.md')+'、'+link('reports/DUAL_HEAD_GATE_20261027.md')+'。可信可见性负例缺失时语义对应头训练曾为not_run；后续被授权的LLP／合成负例训练解决的是代理任务，不能回填为真实可见性实验成功。\n'
 md('audio_results.md',txt)

def timing_report():
 txt='\n## 实际计算成本：只在同一轮计时协议内比较\n\n'
 t=S['timing']['frozen'];rows=[]
 for sid,z in t['strategies'].items():rows.append([named('frozen',sid),z['n_sources'],z['n_calls'],num(z['direct_pipeline_end_to_end_s']['mean']),num(z['direct_pipeline_end_to_end_s']['median']),num(z['direct_pipeline_ttft_s']['mean']),z['visual_tokens']['mean']])
 txt+='冻结迁移开发计时：16来源，每策略2次；与确认701／256题不是同一个计时样本。重新解码、处理原音频、定位、视觉编码、路由和生成；不含模型加载、Gold打分与播放等待。所有模型驻留，同顺序重复测量不是部署速度保证。原始'+src('frozen','qa_eval/pipeline_cost_summary.json','strategies')+'。\n'
 txt+=table(['策略','来源','调用','Pipeline均秒','Pipeline中位秒','首Token均秒','视觉Token'],rows)
 for st,count in [('adaptive',4),('gatefirst',3)]:
  rows=[]
  for sid,z in S['timing'][st]['summary'].items():rows.append([named(st,sid),num(z['frontend_s']),num(z['pipeline_total_s']),num(z['pipeline_TTFT_s']),pct(z['on_rate'])])
  txt+=f'\n{ST[st]}：{count}个预先指定开发来源，离线串行真实Pipeline；'+src(st,'qa_eval/pipeline_cost_summary.json' if st=='adaptive' else 'results/pipeline_cost.json','records、summary')+'。不与其他阶段耗时直接排名。\n'
  txt+=table(['策略','前端均秒','生成结束Pipeline均秒','Pipeline首Token均秒','ON率'],rows)
 txt+='自适应Gate全OFF仍要计算音频与BEATs/Gate特征，因此比直接V1慢；关闭辅助图不等于传感器前端免费。Gate-first计时只有3个开发源，早期预热与顺序有影响，不能把点估计当稳定加速。\n\n固定R_A阶段按来源／策略分别重跑前端，每个来源的问题共用该次前端。逐题`pipeline_total_s=frontend_wall_s+llm_total_s`是单题完整收费口径；不是每题真的重新执行前端。`frontend_s`按来源平均，`pipeline_s_per_QA`按题平均，两者权重不同。旧字段`amortized_video_all_QA_pipeline_s`真实公式为 mean_source(frontend + Σ该来源LLM秒)，单位是“秒／来源回答全部QA”，并不是摊销秒／题。原始结果保留，解释依据'+link('utility_gate/fixed_router_gate/paired_stats.py','summarize')+'；不得用该字段除错分母计算加速。表内确认Pipeline使用逐题收费口径。\n'
 return txt

def diagnostic_report():
 txt='\n## VGG-SS与LLP：诊断、代理结果单独呈现\n\n### VGG-SS声源区域诊断\n\nVGG-SS提供图像声源框，没有本项目需要的官方中心帧QA。最早125来源均为已检视开发集，只有109音频合格；类别映射代理适用108来源，不能称官方问答。初轮GT Overlay正确数与RGB同为105/108，框内质量均值0.347107→0.347845；这是Oracle提示诊断，不是可部署音频热图。'+link('artifacts/utility_gate/vggss_grounded_image_pilot_20261028/summary.json','最初summary')+'，`methods.*.spatial/category_proxy`；逐源'+link('artifacts/utility_gate/vggss_grounded_image_pilot_20261028/results.jsonl')+'。\n\n后续严格保留旧开发125、新开发200、新确认500来源。每源以标注时刻中心帧（通常相对5秒）及其此前因果音频诊断；按源bootstrap2000次，seed1234。所有模型冻结，未训练Gate／Router。下表仅为新确认500源的空间指标，不包含正式QA Accuracy。\n'
 sd=S['diagnostics']['scaleup']['summary'];names={'rgb_clean':'原RGB视觉路由','gt_overlay':'GT框区域叠加提示（Oracle）','sham_overlay':'GT提示的空间错移对照','uniform_overlay':'全图均匀叠加','gt_routing_bias':'GT直接加到路由logit（Oracle）','random_color_overlay':'随机颜色区域叠加','bbox_outline_overlay':'GT框轮廓提示（Oracle）'};rows=[]
 for k,z in sd['fresh_holdout']['methods'].items():
  sp=z['spatial'];rows.append([names[k],num(sp['mass']['mean'],6),num(sp['excess_mass']['mean'],6),num(sp['lift']['mean'],6),ci(sp['mass']['ci95']),link('artifacts/utility_gate/vggss_feature_scaleup_20261028_v1/per_source_results.csv',f'split=fresh_holdout;mode={k}')])
 txt+=table(['空间诊断策略（500来源，非QA）','ROI Mass ↑','超面积基线Excess ↑','Lift ↑','Mass既有95%CI','逐源证据'],rows)
 txt+='GT叠加虽能改变冻结视觉特征，路由质量仍接近ROI面积基线，而且错移及随机颜色也有相近变化；GT直接路由偏置才大幅增加框内质量，但这是答案位置已知的Oracle。特征层证据见'+link('artifacts/utility_gate/vggss_feature_scaleup_20261028_v1/features/fresh_holdout/summary.json','SigLIP→投影→路由矩阵→压缩特征诊断')+'。\n'
 f=S['diagnostics']['scaleup'].get('fresh_holdout_features',{}).get('methods',{}).get('gt_overlay',{}).get('features',{})
 keys=['encoder_patch_features.relative_l2_mean','projected_patch_features.relative_l2_mean','pooled_visual_values.relative_l2_mean','routing_matrix.relative_l2_mean','compressed_tokens_Z.relative_l2_mean']
 selected=[(k,z) for k,z in f.items() if 'relative_l2' in k and ('mean' in k or 'global' in k)][:14]
 if selected:txt+=table(['GT叠加相对RGB的特征指标（500来源）','均值','既有95%CI','原始键'],[[k,num(z['mean'],6),ci(z.get('ci95')),link('artifacts/utility_gate/vggss_feature_scaleup_20261028_v1/features/fresh_holdout/summary.json',f'methods.gt_overlay.features.{k}')] for k,z in selected])
 txt+='B5／B6来源核验的非GT连续预测图实验在该轮为not_run；Stage3真实／零音频只是独立参照。类别代理接近上限、teacher一致性只衡量模仿视觉参考，均不能推断真实问答收益。结论保留RGB Overlay NO-GO，GT-free独特增益及正式QA未证实。'+link('artifacts/utility_gate/vggss_feature_scaleup_20261028_v1/summary_interpretation.json')+'。\n\n### LLP音视频共同事件代理\n\nLLP为按秒事件标签；本轮官方Validation dense标签被明确授权用于自定义训练／校准，不能再称官方未触碰验证协议。开发480来源（训练394／验证86），新确认取得128来源。共同类别事件为正；有事件而无共同类别等为代理负；无标注／前一秒共同事件等未知行不当可靠负例。另加入AVS真实可见正例及授权合成代理负例，两个头按不同标签mask训练，没有每行同时具备两种可靠标签。六小模型各30轮，AdamW .001，batch64，hidden64，dropout .1，seed28102026；细节'+link('artifacts/utility_gate/dual_head_proxy_20261028_v1/model/training_configuration.json')+'，有效监督数'+link('artifacts/utility_gate/dual_head_proxy_20261028_v1/labels/supervision_summary.json')+'。\n'
 rows=[]
 for x in S['diagnostics']['LLP']['stats']:
  rows.append([x['grid'],{'corr_only_llp':'只训练LLP共同事件对应头','corr_only_robust':'只训练AVS合成代理对应头','corr_only_combined':'联合来源对应头','utility_only':'只训练QA收益头（用于代理诊断）','dual_llp':'LLP对应头与QA收益双头','dual_combined':'联合来源对应头与QA收益双头'}[x['method']],x['n'],x['sources'],x['positives'],num(x['AUROC'],6),num(x['AUPRC'],6),ci(x['AUROC_ci95']),x['evidence']])
 txt+=table(['时间网格','代理模型说明','有标签点','来源','正例','AUROC ↑','AUPRC ↑','AUROC既有源聚类CI','证据'],rows)
 txt+='1秒与2秒网格是相同视频上的不同诊断分母，不相加；2秒主分析571点／127来源，双头LLP AUROC约0.8251只能证明共同事件代理可学习，不能叫真实可见性概率、空间定位或QA Utility。\n\n独立AVQA代理阶段下游确认：128题／128来源，每源最多五个按2秒到达帧，每帧64 Token；四选一原选项ABCD生成，NLL四选项归一化。训练新QA来源64、验证32，未用确认标签训练。与MUSIC／Valor完全分队列。\n'
 txt+=table(['完整策略名称（代理阶段）','正确／128','准确率 ↑','Gold NLL ↓','音频启用比例','证据'],[[proxy_name(x['method']),f'{x["correct"]}/128',pct(x['correct']/128),num(x['gold_nll']),pct(x['on_rate']),x['evidence']] for x in S['diagnostics']['LLP']['qa']])
 txt+='固定视觉108/128、双头LLP108/128、联合双头107/128；没有确认QA增益。源聚类CI来自原报告2000次bootstrap；不同标注强度不能混合解释成可见对应真值。\n'
 return txt
def proxy_name(x):
 return {'qualified_audio':'仅音频可用窗口启用原始音频路由，否则退回视觉','random_rate_matched':'按联合双头验证启用率随机选择原始音频路由','always_off':'固定视觉路由，音频全部关闭','always_on':'音频全部启用','constant_utility':'常数收益选择策略','corr_only_llp':'LLP共同事件代理头选择音频','corr_only_robust':'AVS合成代理头选择音频','corr_only_combined':'联合来源代理头选择音频','utility_only':'仅QA收益头选择音频','dual_llp':'LLP代理与QA收益双头选择音频','dual_combined':'联合代理与QA收益双头选择音频','original_AudioRouter':'原始音频路由器全部启用','same_stage3_null_audio':'同Stage3直接置零音频特征（此代理阶段的实现）','frozen_Gate2':'冻结旧视觉锚定Gate2','constant_utility_transfer':'常数收益迁移'}.get(x,'已有代理策略：'+x)

def imu_report():
 txt='''# IMU 历史实验结果明细

IMU（Inertial Measurement Unit，惯性测量单元）在本项目中是同步加速度和角速度。六轴曲线画成图像后，与RGB共享同一个冻结视觉编码器；IMU数值没有作为单独Token送入语言模型。

## 数据与评估边界

最新HoloAssist实际可用94个录制前缀、721道QA，训练500／验证166／封存确认55；对应65／22／7个录制，保守录制家族35／12／4。96个传感器与时钟验收记录中2个无合格QA，不能把96写成94个完整视频。R／z编号用于合并任务变体与Part1/2，不等于参与者身份；身份未知，不能声称跨人泛化。题目由官方已完成动作Verb／Noun构建，问题中可明确对象，选项是动词，不是官方动作识别排行榜、一般视觉问答或未来动作预测。

每题使用以已知完成动作边界结束的8秒观察片段，片段内2／4／6／8秒四帧；每次只看此前2秒IMU。边界已给定是任务构造的Oracle时间条件，不是在线检测完成时刻。原始RGB／IMU时钟通过原始ticks对齐；只做因果零阶保持，原始采样来源也同时受上下界约束，禁止未来插值。答案ABCD严格生成匹配，Gold NLL是ABCD首Token重新归一化；全部压缩策略256视觉Token／题。

数据证据：'''+src('imu','data/qa_manifest.jsonl')+'；'+src('imu','locks/data_split.json')+'；'+link('utility_gate/imu_visual/scaleup_train/data.py','官方标签与sensor_window')+'。\n\nWEAR取得15人的传感器片段和标签，但没有原始时间时钟，官方预处理存在双侧插值，严格主训练为0对；没有补造陀螺仪。EgoExo4D权限未验证／未取得；UTD-MHAD只用作工程诊断。相关获取与同步限制见'+src('imu','M1_download_sync/ACQUISITION_REPORT.md')+'和'+src('imu','M0_audit/ARCHITECTURE_AND_HISTORY.md')+'。\n'
 txt+='## 实际训练流水账\n\n所有模型结构复用Audio的双图64-token低秩残差融合Router；REAL、DUP、ZERO真正分别训练。主干、视觉投影器、语言模型、BEATs及V1都不更新；IMU阶段不调用BEATs／EZ-VSL。三种主Router对应旧v3同类权重热启动，重置AdamW；训练3轮、lr1e-5、wd.01、clip1，首答案Token全词表CE。训练特征可缓存，正式成本验证不复用图表或视觉特征。验证真实贪心Accuracy优先、再Gold NLL选checkpoint。训练全ON、没有Gate参与Router训练。\n'
 txt+=table(['模型的训练输入与初始化','训练QA／来源／家族','轮数／更新','可训练参数','选定轮与验证正确／QA','checkpoint及SHA'],[[named('imu',{'R_REAL':'T1_REAL_always','R_DUP':'T8_DUP_trained_always','R_ZERO':'T9_ZERO_trained_always'}.get(x['model'],x['model']+'_always'))+'；'+x['init'],f'{x["qa"]}/{x["sources"]}/{x["groups"]}',f'3／{x["steps"]}',x['params'],f'{x["epoch"]}；{x["selected_val"]["correct"]}/{x["selected_val"]["qa"]}',link(x['checkpoint'])+'<br>'+x['sha256']] for x in S['imu_router_training']])
 txt+='7个Router合计8373更新，不是8373独立样本。9／18／35训练家族的规模实验分别91／200／500QA，273／600／1500更新，共享全训练集固定画图尺度并复用同类旧权重；数据量、更新数同时增加，不能单称纯规模效应。另有相同零输出残差初始化的REAL／DUP从头对照。原始每步梯度与冻结审计：'+src('imu','logs/R_REAL_train_steps.jsonl')+'；'+src('imu','M3_router_train/ROUTER_GRADIENT_AUDIT.json')+'。\n'
 cat=S['imu:locks/Gate_training_catalog.json']['data'];txt+=table(['Gate训练目标','参数','轮数','真实更新','非零梯度更新','选择规则'],[[{'ranking':'完整轨迹收益排序','classification':'完整轨迹最优Mask分类','cost_005':'每张成本惩罚0.05的软轨迹','cost_020':'每张成本惩罚0.20的软轨迹'}[x['name']]+'（'+x['name']+'）',x['parameters'],x['epochs'],x['steps'],x['nonzero_gradient_steps'],'验证真实QA，候选权重／阈值随后冻结'] for x in cat])
 txt+='Gate训练前冻结新的REAL Router，训练8000／验证2656次完整16组合QA，共10656次实际生成；这是666道QA的干预次数。训练295/500、验证98/166题至少一个组合改变答案，不再沿用旧32题全不变的结论。Accuracy目标为2×错误指示+NLL/(1+NLL)，排名／分类学完整轨迹；成本候选为NLL+λk，λ=.05/.2，温度.25。四Gate各30轮1890更新、合计7560。验证后准确率与成本名称均选ranking第30轮、阈值.5、相同SHA，不是两种不同决策。证据：'+src('imu','locks/utility_lock.json')+'；'+src('imu','locks/Gate_training_catalog.json')+'；'+src('imu','locks/validation_selection.json','selected')+'。\n'
 txt+='## 新封存确认：55QA／7录制／4家族，完整策略表\n\n所有行是同一HoloAssist确认队列、同样已观察8秒及ABCD评分；原生单图／双图仅作不同Token预算参考。Rescue/Harm均相对本表V1。ON比例=选中窗口数／220；额外编码为每题均数，4张基础RGB另计。原始文件在'+src('imu','results/cost')+'，主结果来自付费Pipeline真实生成，eval中的复制记录不能算第二次独立实验。均值与中位耗时均为秒／题。\n'
 rows=[]
 for x in [x for x in S['imu_paid'] if x['role']=='holdout']:
  rows.append([named('imu',x['strategy'])+f'（`{x["strategy"]}`）',f'{x["correct"]}/55',pct(x['accuracy']),ci(x.get('ci95')),num(x['gold_nll']),f'{x["rescue"]}/{x["harm"]}',pct(x['on_rate']) if not x['strategy'].startswith('N') else '无Gate',num(x['mean_k'],3),num(x['visual_tokens_mean'],0),num(x['pipeline_s']),num(x['pipeline_median_s']),refs(x)])
 txt+=table(['完整策略说明','正确／QA','准确率 ↑','既有95%家族CI','Gold NLL ↓','挽救／损害','ON率','额外编码图／题','视觉Token／题','Pipeline均秒 ↓','Pipeline中位秒','证据'],rows)
 txt+='原生参考的k若记录为辅助图计数只说明4张额外图，不意味着使用Gate；N0为2920 Token，N1为5840 Token。DUP训练分支在计时中仍第二次编码重复RGB，但无需绘制IMU图，其更短耗时不能全归因Gate。T2_accuracy／T2_cost同一Mask的不同耗时来自测量波动。\n\n### 同队列关键配对\n\n以下仍为55题／4家族、8秒、ABCD。原有CI是按保守录制家族10000次bootstrap，seed1234；4家族仅16种符号置换，双侧最小p=.125。没有预存CI的对照标明未存，不补造。\n'
 stored=S['imu:M6_confirmation/PAIRED_SUMMARY.json']['data']['paired'];rows=[]
 for a,b in [('T1_REAL_always','T0_V1'),('T1_REAL_always','T8_DUP_trained_always'),('T1_REAL_always','T9_ZERO_trained_always'),('T2_accuracy','T0_V1'),('T2_accuracy','T1_REAL_always'),('T2_accuracy','Fixed_k_2'),('T2_accuracy','T2_accuracy_same_mask_DUP'),('T2_accuracy','T2_accuracy_same_mask_ZERO'),('T2_accuracy','T2_accuracy_same_mask_WRONG'),('T2_accuracy','T4_random_accuracy_1234'),('T2_accuracy','T4_random_accuracy_2234'),('T2_accuracy','T4_random_accuracy_3234')]:
  p=next(x for x in S['imu_paid_pairs'] if x['role']=='holdout' and x['a']==a and x['b']==b);ss=next((x for x in stored if x['role']=='holdout' and x['a']==a and x['b']==b),{})
  if not ss and a=='T2_accuracy':
   ep=next((x for x in S['imu:M6_confirmation/GATE_VS_ENDPOINT_PAIRED.json']['data'] if x['role']=='holdout' and x['gate']=='accuracy' and x['reference']==b),None)
   if ep:ss={'accuracy_cluster_CI':ep['accuracy_delta_family_CI']}
  if not ss and b=='T0_V1':ss={'accuracy_cluster_CI':paid('holdout',a).get('delta_ci95')}
  rows.append([named('imu',a)+' − '+named('imu',b),int(p['rescue']),int(p['harm']),f'{100*p["delta"]:+.3f}',ci(ss.get('accuracy_cluster_CI'),True),num(p['nll_improvement'])])
 txt+=table(['配对比较','Rescue','Harm','准确率差 pp','既有95%CI','NLL改善 ↑'],rows)
 txt+='配对数值来源：'+src('imu','M6_confirmation/PAIRED_SUMMARY.json','paired')+'及本轮按原始付费记录重算的 [imu_paid_pairs](supplement.json)。随机Gate用验证ON概率生成，本确认每种种子167次ON，新Gate170次ON，是期望预算匹配，不是逐队列精确等额。Fixed-k2的39/55是预先列出的固定策略本次点估计，不能事后宣称验证选定或作为新的最优模型。\n'
 txt+='### Accuracy更好、Gold NLL却更差\n\nREAL为37/55、NLL1.779150；V1为16/55、NLL1.356172。Accuracy只统计最终对错，NLL还惩罚把标准答案概率压得极低的错误。不能单凭这组变化判定实现Bug，也不能声称概率校准改善。逐题尾部证据（同一55题）：\n'
 txt+=table(['QA与家族','标准／REAL输出／V1输出','REAL NLL','V1 NLL','NLL变化','原始记录'],[[x['qa_id']+'；'+x['family'],f'{x["gold"]}/{x["new_prediction"]}/{x["V1_prediction"]}',num(x['new_gold_nll']),num(x['V1_gold_nll']),num(x['nll_change']),link(x['raw_real'],'REAL')+' '+link(x['raw_V1'],'V1')] for x in S['imu_nll_tails'][:4]])
 txt+='## 预算曲线与真实计时\n\n付费计时含重新读取压缩视频、四帧解码、单帧B=1视觉编码、因果IMU裁剪统计、Gate、ON时绘图与第二次视觉编码、融合Router和贪心生成；不含网络获取、模型冷加载、物理等待8秒与Gold打分。媒体字节可在本地范围缓存，图像特征／图表不缓存。H800共享主机多GPU随机顺序，显存为PyTorch峰值allocated，未测能耗。四次采样是观察协议，不应把计算TTFT称实时答案等待时延。'+src('imu','M7_cost_final/MEASUREMENT_SCOPE.md')+'。\n'
 txt+='[确认预算与准确率曲线](figures/imu_budget_curve.png)；[逐策略完整数表](imu_paid_results.csv)。\n'
 rows=[]
 for sid in ['T0_V1','Fixed_k_1','Fixed_k_2','Fixed_k_3','Fixed_k_4','T2_accuracy','T2_cost']:
  x=paid('holdout',sid);rows.append([named('imu',sid),f'{x["correct"]}/55',num(x['gold_nll']),num(x['mean_k'],3),num(x['pipeline_s']),refs(x)])
 txt+=table(['同55题策略','正确／QA','NLL ↓','额外编码','Pipeline均秒','证据'],rows)
 c=S['imu:M7_cost_final/PAIRED_PIPELINE_COST.json']['data'];txt+=table(['与REAL Always-on配对的同策略计时','均值差秒','既有家族95%CI','节省比例','减少辅助编码图／题'],[[x['a'],num(x['pipeline_delta_s']),ci(x['pipeline_delta_cluster_CI']),pct(x['saving_fraction']),num(x['extra_image_reduction'],3)] for x in c if x['role']=='holdout'])
 txt+='来源：'+src('imu','M7_cost_final/PAIRED_PIPELINE_COST.json')+'。Gate少编码0.9091张／题，Accuracy与Cost两别名测得约18.19%／20.71%省时；这支持条件计算节省，但未证明Gate优于Fixed-k2或同预算随机。\n'
 txt+='## 小样本与失败历史：不改写为成功\n\n最初冻结IMU曲线诊断没有训练Router或Gate。UTD108工程样本，原生RGB66/108、曲线单图28/108；Holo64开发题，原生RGB24/64、曲线单图25/64、RGB+单张汇总曲线24/64，属于旧的不同视图／时钟协议。M3–M5当时not_run，不能写成完成。原始'+link('artifacts/imu_visual_next_phase_20261030/M2_frozen_imu_plots/utd_statistics_hierarchical.json')+'、'+link('artifacts/imu_visual_next_phase_20261030/M2_frozen_imu_plots/holo_statistics_hierarchical.json')+'，本轮仅聚合引用，未把它们当严格确认。\n\nv1直接迁移确实训练过，但原始采样来源下界检查失败，保留失败；v2修复后REAL／DUP／ZERO各96次更新，曾发现多帧批量与B=1细微数值差异，影响17个fp16位置及NLL。v3复用v2的真实训练权重、用B=1重新验证各epoch，重新生成1024次完整16组合QA，再训练4个Gate各120次更新；不能称v3重新训练了Router。\n\nv3仅32训练QA／2来源、32验证QA／2来源，无新独立确认：V1及三个训练后Router都是15/32，REAL NLL由1.226434变1.189613但生成答案不变。准确率Gate全ON，成本Gate全OFF，省时来自视觉退回，不证明学到片段收益。原生RGB+IMU17/32、原生RGB12/32是5840对2920 Token不同预算。'+src('imu_small','results/paired_summary.json')+'；'+link('utility_gate/imu_gate_router_transfer/revalidate_router.py')+'。\n\n最新扩充实验已实际训练并独立评估。REAL37、DUP36、ZERO38，Gate38、Fixed-k2为39，错误来源36，整体说明任务适配与条件省算；没有确认真实IMU独特理解贡献、强Gate选择能力或跨参与者泛化。\n'
 md('imu_results.md',txt)

def metric_report():
 txt=r'''# 指标解释词典

统一记号：题目 i 的标准答案 y_i，生成文本 a_i，二值正确性 c_i；方法 m、配对基线 b；总题数 N；每题／每来源合格窗口 T；窗口开关 g_t∈{0,1}。↓越小越好，↑越大越好；ON率和预算本身没有固定“越高越好”。所有公式先在完全相同的队列、问题和评分规则内计算。
'''
 rows=[['QA Accuracy／EM，问答准确率／精确匹配率','Acc=Σ_i 1[norm(a_i)=norm(y_i)]/N','题；比例或% ↑','MUSIC全短答案规范化、数字0–11映射词；严格规范化EM不做数字别名。Valor/Holo必须生成完整ABCD单字母，不能用logit最大值替代真实生成。','utility_gate/grounded_dual_image/data.py:normalize；runtime.py:answer'],['Gold NLL，正确答案负对数似然','整串：−Σ_j log p(y_j|x,y_<j)；四选项：−log[exp l_y/Σ_{z∈ABCD}exp l_z]','题均值；自然对数单位nat ↓','MUSIC主表整串全词表，不含EOS；Valor/Holo是首Token四选项重归一化，另存full_vocab。整串、首Token全词表和四选项三者数值不能混比。','utility_gate/grounded_dual_image/runtime.py:answer'],['Training Loss，训练损失','Router：−log p_vocab(首标准答案Token|视觉与问题)；Gate各目标另见训练流水账','优化样本均值；nat或自定义损失 ↓','训练首Token CE不是QA整串NLL，不是FullVision知识蒸馏；一次更新不是一个独立视频。','streamingtom/train_adbt_videomme.py:qa_loss；utility_gate/imu_visual/scaleup_train/train_gate.py:objective'],['Rescue／Harm，挽救／损害','R=Σ1[c_m=1,c_b=0]；H=Σ1[c_m=0,c_b=1]','同一道题配对；题数，R↑／H↓','必须注明基线；没有同QA配对不能相减。随机三种子的逐题平均正确率是分数，R/H为期望计数、可非整数。','utility_gate/adaptive_grounded64/report.py:paired'],['Paired Delta，配对准确率差','ΔAcc=(R−H)/N=mean(c_m−c_b)','百分点pp ↑','0.01=1pp，不是相对提升1%；不同队列不能计算配对提升。NLL改善在本报告定义mean(NLL_b−NLL_m)，正值更好；原IMU NLL_delta通常方法减基线，符号相反。','utility_gate/fixed_router_gate/paired_stats.py:paired'],['95% CI，置信区间','按来源／家族抽取有放回cluster，保留cluster内全部QA；重复统计取2.5%和97.5%分位','准确率或差值的原单位；区间而非方向分数','Audio大多数源聚类2000次seed1234；最新Holo4家族10000次seed1234。不是按220窗口独立抽样。4家族区间不稳，零翻转[0,0]不是等效性证明；多比较多为探索性无校正。','utility_gate/imu_visual/scaleup_train/report.py；M6_confirmation/EXACT_FAMILY_SIGN_PERMUTATION.json'],['Gate ON Rate，辅助分支启用比例','Σ g_t / 合格窗口总数；Holo每题4窗，55题分母220；Audio先按来源去重','合格决策窗口；% 无固定优劣','Gate决定开启、热图有效、实际第二编码是不同事件。Always-on是策略规则，不表示Gate判断正确。相同QA来源重复记录不可重复数Audio窗口。','utility_gate/gate_first_native/report.py:summarise；per_window输出'],['Added Image Count，额外编码图像数','k=Σ_t 实际第二视觉编码调用；平均k=Σ_i k_i/N；基础图为4张RGB','张／题或张／来源 ↓（给定精度）','Holo compressed编码数4+k，Token始终256；Audio无效图可回退。原生OFF仍重复双图时不省第二图。DUP编码是否复用按实现，不能仅按视图数猜。','utility_gate/imu_visual/scaleup_train/cost.py:counts'],['Visual Tokens，进入LLM的视觉Token','压缩：64×实际观察数；四观察为256；原生图729+image_newline=730','Token／题 ↓（给定精度）','原生四RGB为2920、四对图5840；旧全视觉视频4×196+newline=785。729→196是视觉池化中间层，64是压缩输出，文字／选项Token另计。不同接口不混算。','utility_gate/grounded_dual_image/native.py；runtime.py:answer'],['Pipeline Runtime／TTFT，实际计算耗时／首Token时间','wall_end−wall_start；TTFT=首次生成Token时刻−相同start','秒／题；均值／中位／p95 ↓','明确是否含解码、音频、BEATs、Gate、定位、画图、第二编码、Router、prefill、decode。主预测共享前端的LLM时间不能叫Pipeline。最新付费计时不含加载、网络、物理等待、Gold评分；不能跨阶段硬件／缓存／顺序比速度。','M7_cost_final/MEASUREMENT_SCOPE.md；utility_gate/fixed_router_gate/paired_stats.py:summarize'],['Cost Saving，耗时节省比例','saving=1−mean(time_gate)/mean(time_always)','相同队列／计时协议；% ↑','应同时给秒数、差值和配对CI；Gate少算不等于选择更有益传感器。两相同Gate别名不同耗时是测量变动。原字段amortized_video_all_QA_pipeline_s是平均每源回答全部QA总秒，不是秒/题。','M7_cost_final/PAIRED_PIPELINE_COST.json'],['AUROC／AUPRC，排序判别指标','AUROC=P(s_pos>s_neg)+0.5P(tie)；AP=Σ_n (Recall_n−Recall_(n−1)) Precision_n','带已知正负标签的诊断点；[0,1] ↑','LLP正负为共同事件／非共同代理，不是true visibility；AP受正例占比影响，必须报正例数。一个源有多个秒点，CI按源聚类。utility-only诊断用u分数，其他对应头用r。','utility_gate/dual_head/proxy/evaluation.py:methods'],['Spatial ROI Mass／Excess／Lift，区域路由质量','q_j=mean_k P_kj；ROI r_j为token与GT框重叠占比；Mass=Σ q_j r_j；面积A=mean r_j；Excess=Mass−A；Lift=Mass/A','源／标注帧；无量纲 ↑','ROI是Region of Interest；均匀路由Lift=1。框面积变化要做配对。GT叠加和直接logit偏置均为Oracle；无标注不能计算真实定位质量。质量高不自动等于QA好。','utility_gate/grounding/spatial_mapping.py:routing_metrics'],['Source Family，来源／录制聚类','把同一可审计来源的相关问题合并抽样','来源数、录制数、家族数','MUSIC仅确认本地官方视频ID，缺原始YT血缘；Valor使用已知YT源；Holo家族合并R/z号及任务变体，不能推断人。基础模型预训练重叠未知。','数据manifest与locks/data_split.json'],['Oracle Upper Bound，事后上限','根据标准答案／GT／完整16mask结果选max正确或min NLL','同题事后诊断；不作部署排名','测试答案不能参与Gate选择；真实部署只能当前与过去输入。16mask训练/验证标签可用，不能把测试16mask最优结果当学习Gate。','utility_gate/fixed_router_gate/enumerate_masks.py']]
 txt+=table(['指标中文定义','计算公式','统计单位／方向','不可比情况及注意事项','原始计算位置'],rows)
 txt+='\nAccuracy改善而NLL恶化不矛盾：多救回小优势题，同时少数错误把标准答案概率压到极低，即可让总NLL变差。NLL是正确答案概率的适当评分，但仅凭均值不能完整描述校准，需要进一步可靠性图／Brier等；本轮只展示原有NLL和逐题尾部，不补造未做的校准实验。证据见 [IMU逐题尾部](imu_gold_nll_tail_examples.csv)。\n'
 md('metric_dictionary.md',txt)

def main_report():
 txt='''# Audio 与 IMU 历史实验中文总报告

审计日期：2026-11-02。本次只读取历史源码、逐题预测、窗口行为、训练日志、锁与统计文件并重算指标；没有新训练、推理或调参，也没有改写旧结果。**请先读本页，再按链接看明细。**

## 一屏结论

- **Audio：有任务适配收益，真实音频独特收益未确证。** 固定8秒MUSIC-AVQA确认288题／200源：训练后融合Router为206/288（71.53%），固定视觉基线173/288；同一个Router改用重复RGB、空间错移图、零音频定位图也均206/288。Valor另一确认508题／254源，融合Router321/508对视觉326/508，未确认迁移改善。此结论只属于该队列，不能与冻结迁移701题60.20%的原生双图比较提升。
- **IMU：训练适配明显，但传感器特有价值未证明。** 最新HoloAssist确认55题／7录制／4保守家族：视觉16/55，旧真实IMU Router19/55，新真实IMU Router37/55；独立重复RGB训练36/55、零IMU训练38/55。学习Gate38/55，170/220窗口ON（77.27%），固定最早两窗39/55。不能声称真实IMU优于零信号，不能声称跨参与者泛化。
- **Gate：条件省算与“选得更好”分开。** IMU学习Gate平均额外编码3.091张，而Always-on为4张；相同付费协议约省18.19%计算秒数（同一策略Cost别名约20.71%）。尚未证明优于固定两窗或同预算随机。Audio固定R_A成本Gate确实少编码，但准确率非劣性及音频来源收益未确认。

关键数字原始依据：'''+src('gatefirst','results/summary.json','confirmation_music/confirmation_valor.strategies')+'；'+src('imu','M6_confirmation/PAIRED_SUMMARY.json','holdout metrics/paired')+'；'+src('imu','M7_cost_final/PAIRED_PIPELINE_COST.json','holdout')+'''。

## 阅读导航

| 文件 | 内容 |
| --- | --- |
| [Audio明细](audio_results.md) | 每轮MUSIC／Valor独立表、训练路径、原生双图、预算与VGG-SS／LLP诊断 |
| [IMU明细](imu_results.md) | HoloAssist全部确认策略、7个Router和4个Gate训练、同Mask配对、真实计时及失败历史 |
| [完整架构，重点Gate与Router](architecture_audio_imu.md) | 当前Audio／IMU从数据输入、因果窗口，到Gate、Router与LLM接口的逐层形状／公式／参数 |
| [指标中文词典](metric_dictionary.md) | 公式、统计单位、方向、NLL协议、Bootstrap及成本边界 |
| [所有策略中文对照](strategy_name_mapping.csv) | 原始ID、完整名称、训练输入与推理干预、Gate／Router／Token区别 |
| [数值证据索引](evidence_index.csv) | E编号→原始文件、结果键／选择条件、源码位置 |
| [数据划分登记](dataset_split_ledger.csv) | manifest逐行重数的QA／来源／家族与问题类型 |
| [自动核验](verification_report.json) | 通过／失败／缺失、原始输入SHA保护与可比性限制 |

## 不需要内部缩写的架构入门

RGB是普通彩色视频帧。VLM（Vision-Language Model，视觉语言模型）使用冻结的SigLIP视觉编码器、视觉投影器和Qwen2语言模型。视觉专用Router通过64个查询对196个视觉位置做加权重采样；不是挑64块的硬Top-K。原始音频Router用冻结BEATs音频特征控制查询，音频值不直接送入语言模型。

后续双图Fusion Router把RGB与辅助图各196个视觉特征融合，在固定视觉专用输出的基础上加低秩残差，始终每帧64视觉Token。Audio辅助图来自冻结EZ-VSL的预测声源热图；IMU辅助图来自真实同步加速度／陀螺仪曲线。Gate先看当前／过去RGB摘要与便宜的传感器特征，决定是否调用辅助绘图与第二次视觉编码；OFF严格退回V1。原生双图是另一个接口，绕过Router，让语言模型看到大量原生图像Token。

模型参数与冻结状态以原始审计为准：SigLIP397,747,744、视觉投影器16,980,992、语言模型及原生特殊嵌入7,612,631,040均冻结；原Router总1,991,680，视觉专用活跃训练933,888；后续Fusion Router696,320。不能把总参数量当本轮训练量。证据：'''+link('artifacts/imu_visual_next_phase_20261030/M0_snapshot/model_inventory.json','逐模块审计')+'；'+link('utility_gate/gate_first_native/fusion.py','GroundedFusionRouter')+'''。

## 数据与公平比较登记

以下是分队列索引，不是跨队列排行榜。MUSIC是原生短答案规范化精确匹配，Valor与HoloAssist是严格生成ABCD；Gold NLL亦不跨协议直接比较。
'''
 rows=[]
 for st,roles in [('frozen',['music_confirm','valor_confirm']),('adaptive',['confirmation_music','confirmation_valor']),('gatefirst',['confirmation_music','confirmation_valor']),('fixed',['confirmation_music','confirmation_valor'])]:
  for role in roles:
   x=next(x for x in A['tables'] if x['stage']==st and x['role']==role);rows.append([ST[st],x['dataset'],role,f'{x["n_qa"]}／{x["n_sources"]}','按时长比例四帧' if st=='frozen' else '2秒起最多32次因果观察' if st=='adaptive' else '已观察8秒，四次观察','MUSIC规范化文本EM／整串NLL' if x['dataset']=='music_avqa' else '原选项ABCD／四选项NLL',src(st,{'frozen':'summary.json','adaptive':'qa_eval/paired_summary.json','gatefirst':'results/summary.json','fixed':'paired_summary.json'}[st])])
 rows.append(['扩充后的IMU主确认','HoloAssist标签衍生QA','holdout','55／7录制／4家族','已知完成边界内8秒，四次观察','动作动词四选一ABCD；四选项NLL',src('imu','M6_confirmation/PAIRED_SUMMARY.json','cohorts')])
 txt+=table(['阶段','数据集','独立队列名称','QA／来源','时间范围','答案／评分','原证据'],rows)
 txt+='训练与验证来源统计、问题类别见 [完整划分CSV](dataset_split_ledger.csv)。Audio V1/Fusion训练656 QA／192源、验证157／48；固定R_A的新Gate沿用训练源但新验证335／96。最新IMU训练500 QA／65录制／35家族，验证166／22／12，确认55／7／4。MUSIC原始YouTube血缘不可核验；Valor已知YT去重不能证明基础模型预训练未见；所有这些已检视确认来源现已是历史，不允许未来调参后再称独立测试。\n'
 txt+='## 同一队列内哪些对照回答什么问题\n'
 txt+=table(['比较','控制住什么','可以解释什么','不能据此解释什么'],[['Stage3真实音频 vs 同权重零波形编码','同Router、Token、问题与音频编码器','该旧Router对真实波形是否敏感','Stage1与Stage3训练历史的纯因果分离'],['原生RGB+预测图 vs 原生重复RGB／错移图','同图数、同提示词／顺序需明确','辅助图内容是否带来增益','与压缩256Token模型的公平同预算排名'],['固定R_A真实图 vs 同权重重复／错移／零音频图','同已训练Router，仅推理干预','训练后对音频提示内容的依赖','独立重复图训练Router的效果'],['IMU REAL训练 vs DUP／ZERO独立训练','同结构、初始化类别／预算及训练来源，具体热启动不同','普通任务训练／额外视图可否解释收益','1题差异即真实IMU独特贡献'],['Gate真实图 vs 同Mask DUP／ZERO／WRONG','控制每题所选时间和64Token输出','辅助内容特异性','错误来源干预是一套可部署策略'],['Gate vs Always-on／固定k／随机Gate','同队列、Router与输出预算；额外图预算需实测','选择质量、预算效率与负迁移','少算自动等于识别了有益传感器片段']])
 txt+='## 训练与可复现性\n\n'+training()
 txt+='IMU全部训练更新与checkpoint SHA见 [IMU明细](imu_results.md)。基础VLM与V1始终冻结；原AudioRoutercheckpoint保持。预测试锁、随机种子、旧失败与回滚记录均原样保留。历史训练／推理命令见'+src('gatefirst','commands.json')+'、'+src('fixed','commands.json')+'和'+src('imu','locks/protocol.json')+'。前两个`.json`文件实际为逐行JSON命令日志，按JSONL读取。命令只是可复现记录，本轮没有执行它们。\n\n本次审计的可复现命令（只进行CPU读取／统计）：\n\n```bash\nPYTHONPATH=LLaVA-NeXT:lmms-eval:. PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=\'\' .venv-gate/bin/python reports/audio_imu_results_explained_20261102/audit_tools/audit.py\nPYTHONPATH=LLaVA-NeXT:lmms-eval:. PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=\'\' .venv-gate/bin/python reports/audio_imu_results_explained_20261102/audit_tools/supplement.py\nPYTHONPATH=LLaVA-NeXT:lmms-eval:. PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=\'\' .venv-gate/bin/python reports/audio_imu_results_explained_20261102/audit_tools/render_report.py\n```\n'
 txt+='\n## 结论的证据边界\n\n目前可以支持：小型融合Router经过真实训练后在部分标签衍生任务上适配更好；Gate能够减少第二次视觉编码及实测计算时间。尚不足以支持：音频／IMU比重复或零传感器提供独特理解信息；学习Gate稳定优于固定预算或随机选择；跨人、跨域、通用多模态理解或SOTA。四家族HoloAssist及不同Audio队列限制统计外推，Accuracy提升同时NLL可能恶化。VGG-SS空间Oracle、LLP共同事件代理与真实下游QA分别报告，不互相替代。所有缺失值、未重验聚合数据与成本命名问题均在核验报告中明示。\n'
 md('README.md',txt)

if __name__=='__main__':
 names=mapping();metric_report();audio_report();imu_report();main_report()
 # Fill evidence pointers with both the raw predictions and the actual implementation.
 for x in EVIDENCE:
  if not x.get('code_location'):x['code_location']=CODE.get(x['stage'],'见原始源码索引')
  if x['stage'] in ROOTS:
   x['result_key_or_selector']+='; raw_predictions='+str(next((a['raw_files'] for a in A['tables'] if a['stage']==x['stage'] and a['role']==x['cohort'] and a['strategy']==x['strategy']),[]))
 csvout('evidence_index.csv',EVIDENCE);write('audit_input_guard.json',GUARD)
 print('REPORT generated',len(names),'strategy mappings',flush=True)
