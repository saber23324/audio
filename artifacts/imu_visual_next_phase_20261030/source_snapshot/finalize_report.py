"""Generate the final report from saved actual outputs and an explicit decision."""
import json,math,subprocess,shutil,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .common import ROOT,DATA,REPO,read_jsonl,write_json,write_jsonl,sha256,journal,status,now

def finish():
    decision=json.loads((ROOT/'M7_final/training_decision.json').read_text())
    utd=json.loads((ROOT/'M2_frozen_imu_plots/utd_statistics_hierarchical.json').read_text());holo=json.loads((ROOT/'M2_frozen_imu_plots/holo_statistics_hierarchical.json').read_text());integrity=json.loads((ROOT/'M7_final/old_asset_integrity.json').read_text());assert integrity['all_unchanged']
    m=json.loads((ROOT/'M0_snapshot/model_inventory.json').read_text());clock=json.loads((ROOT/'M1_data/holo_rgb_clock_audit.json').read_text());utd_rows=read_jsonl(ROOT/'M1_data/utd_all_inertial_audit.jsonl');utd_dev=read_jsonl(ROOT/'M1_data/utd_probe_manifest.jsonl');holo_dev=read_jsonl(ROOT/'M1_data/holo_probe_manifest.jsonl');hlock=json.loads((ROOT/'M2_frozen_imu_plots/holo_renderer_lock.json').read_text());cost=read_jsonl(ROOT/'M6_pipeline_cost/resident_native_cost.jsonl')
    # This report is for the conditional stop branch, not an implementation of
    # a successful training run. Refuse to describe unexecuted work as trained.
    assert decision['M3']=='not_run' and decision['M4']=='not_run' and decision['M5']=='not_run'
    assert not list((ROOT/'models').glob('**/*.pt'))
    for stage,folder in [('M3','M3_router_train'),('M4','M4_gate_train'),('M5','M5_frozen_confirmation')]:
        write_json(ROOT/f'{folder}/execution_status.json',{'stage':stage,'state':'not_run','reason':decision['reason'],'checkpoint':None,'optimizer_steps':0,'fitted_models':0,'complete_mask_utility_QA_calls':0,'pretest_confirmation_lock':None,'not_zero_accuracy_or_zero_ON_rate':True,'taskbook_condition':'§6.3 and §11 upstream feasibility condition'})
        (ROOT/f'{folder}/NOT_RUN.md').write_text('# not_run\n\n'+decision['reason']+'\n\nTaskbook §6.3 allows only the predeclared small P1/P2 diagnostics when the frozen bridge is weak; §11 requires honest downstream not_run when mandatory upstream feasibility fails. No new IMU checkpoint, training curve, utility label, Gate ON rate, random Gate comparison, or independent confirmation is claimed.\n')
        status(stage,'not_run',reason=decision['reason'])
    # Honest source ledger: metadata-inspected sources are also marked seen.
    ud={r['source_id'] for r in utd_dev};hd={r['source_id'] for r in holo_dev};ledger={}
    for r in utd_rows:
        sid=r['source_id'];ledger[sid]={'source_id':sid,'dataset':'UTD-MHAD','participant_id':r['participant_id'],'session_id':r['session_id'],'role':'development_scored' if sid in ud else 'sensor_metadata_inspected','eligible_for_future_untouched_confirmation_claim':False}
    for r in read_jsonl(DATA/'holo/complete_annotation_records_from_partial.jsonl'):
        sid='holo:'+r['video_name'];ledger[sid]={'source_id':sid,'dataset':'HoloAssist','participant_id':None,'session_id':r['video_name'],'role':'development_scored' if sid in hd else 'annotation_metadata_inspected','eligible_for_future_untouched_confirmation_claim':False}
    for s in json.loads((ROOT/'M1_data/holo_sync_probe_source_lock.json').read_text())['sources']+hlock['calibration_sessions']:
        sid='holo:'+s
        if sid not in ledger:ledger[sid]={'source_id':sid,'dataset':'HoloAssist','participant_id':None,'session_id':s,'role':'paired_intake_or_scale_calibration','eligible_for_future_untouched_confirmation_claim':False}
        if s in hlock['calibration_sessions']:ledger[sid]['role']='scale_calibration_only'
    write_jsonl(ROOT/'all_seen_sources.jsonl',list(ledger.values()))
    assert not set(hlock['calibration_sessions'])&set(hlock['development_sessions'])
    write_json(ROOT/'train_val_lock.json',{'calibration_only_Holo_sessions':hlock['calibration_sessions'],'development_Holo_sessions':hlock['development_sessions'],'UTD_development_sources_sha256':sha256(ROOT/'M1_data/utd_probe_manifest.jsonl'),'UTD_scale':'subjects1/3/5, takes2/3/4; development trial1 includes some same subjects; no subject-generalization claim','model_training_QA_sources':[],'confirmation_sources':[],'confirmation_state':'not_run; no independent confirmation pretest lock was issued','Holo_participant_ID':'not available in inspected metadata; no subject-disjoint claim','no_source_overlap_between_Holo_scale_and_dev':True})
    data_report='''# 数据集与时钟报告

所有数据来自官方公开端点，访问、失败、Range 响应、条目 CRC 和文件 SHA 保存在本目录 JSON/JSONL。按范围取得的归档没有整包 SHA，不冒充完整下载。

|数据集|实际取得/检查|传感器与单位|严格因果可用性|本轮角色|
|---|---|---|---|---|
|UTD-MHAD|完整 Inertial.zip / Sample_Code.zip；861个MAT、8人；108个RGB AVI完整条目通过CRC并实际推理|50Hz，右腕(动作1–21)/右大腿(22–27)；acc g→m/s²，gyro deg/s|分发MAT仅d_iner[N,6]，无原始时间戳；名义时长2.14–6.52s，0段达到8s|已完成记录级开发 smoke，不能作为严格8s流式同步主数据|
|WEAR|官方目录、一个50Hz CSV头/片段、完整官方代码commit65cb6c4fda912dbce9b243e67d8a9bc73f4d87ba；未下载164GB整库|四腕/踝加速度计、12轴、50Hz、g；没有gyro|CSV没有时间列；官方预处理用首尾同步跳变建立时钟、先插值再20ms重采样，导出时丢掉时间。未取得插值前原始采样时钟，不能认证原始采集时间因果性|可下载≠已满足本协议；未运行QA/训练|
|HoloAssist|10段完整acc/gyro文本；覆盖完整视频时间线的稀疏RGB检查与前缀解码；12段IMU索引，40段视频索引；243条完整标注对象从12MB部分文件合法恢复，非完整117MB标注下载|HoloLens头部acc m/s² / gyro rad/s；原始.NET 100ns UTC ticks；已发布同步流约30Hz，非原始高频采样|10段中6段去重相机tick数与实际MP4帧数完全一致；4段差1帧，排除；可用标注与精确时钟交集4段，64道开发QA|2段单独校准固定量程；4段开发；无独立确认|

HoloAssist 不能直接用压缩MP4播放PTS当采集时间。以首段为例，完整时间线缩略图播放203.07s对应相机采集198.12s。采用 VideoMp4Timing 起点，在整数域减去.NET原点后再换秒；对 Pose_sync 的单调原始相机ticks去重，要求与MP4帧数精确一一对应；解码时逐帧验证CFR，按采集时钟选不晚于t的RGB。同步IMU行的原始传感器时刻可能晚于其第一列的RGB名义时间，必须按第二列原始ticks筛选。相差1帧不通过猜测偏移修复。

acc/gyro不同步时，在原始时刻并集上做只取过去值的zero-order hold；保存每个值的原始时刻，断言原始时刻≤并集时刻≤决策t。不做双向滤波或未来插值。发布数据的30Hz同步过程已经丢失的高频样本不能恢复；本轮不声称重建了原始高频IMU。量程只由另外2段校准来源估计，不用测试/当前窗口min-max。

HoloAssist派生QA来自官方 Fine grained action 的已完成Verb/Noun区间，不使用未来纠错结果。任务构建给定标注事件边界：把标注媒体时间映射到不早于结束边界的媒体帧，再取其原始相机时刻作为观察终点，四次观测为该终点前6/4/2/0秒。此为给定查询到达/裁剪边界的派生QA，不是官方QA，也不是在线自主事件检测或提前预测。候选动词词表由其它标注来源固定，选项位置由固定哈希随机化；各方法共享相同问题、候选、RGB、时间。身体位置/物理单位之外的标签不进入图表。受试者身份未提供，不把R编号当参与者编号，不声称跨参与者隔离。

UTD所有试验均为development，非官方27类排行榜。量程校准来源与测试take分离，但部分受试者重合，故不作跨人确认。所有已检视记录登记为seen；历史AudioRouter来源账本仍只读保留。未证明跨库人员重合或基础模型预训练无重合。

10段HoloAssist完整传感器文本和全时段RGB缩略图已检查；没有下载/逐帧解码10段完整视频，也没有独立标定残余同步误差。时间戳计数一致及定性运动检查不能被称为硬件同步误差为零。4段开发不足以建立大型主训练/验证/确认队列。

来源与许可：[UTD官方](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)（本次未找到单独明确的SPDX数据许可，不擅自称CC0）；[WEAR官方](https://mariusbock.github.io/wear/) / [预处理源码](https://github.com/mariusbock/wear/blob/65cb6c4fda912dbce9b243e67d8a9bc73f4d87ba/data_creation/utils.py)（CC BY-NC-SA4.0）；[HoloAssist官方](https://holoassist.github.io/)（CDLA Permissive2.0）；[Microsoft PSI导出格式](https://github.com/microsoft/psi/blob/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter/Readme.md)（原始时钟及IMU单位）。WEAR论文22人，当前目录含24个subject条目，数据发布差异未混称同一版本。
'''
    (ROOT/'M1_data/DATASET_REPORT.md').write_text(data_report)
    # Measured cost curves: never call these a trained 64-token IMU frontier.
    grouped=defaultdict(list)
    for x in cost:grouped[(x['path'],x['k'])].append(x)
    cost_summary={};lines=['|Path / k|Measurements / sources|Visual tokens|Render count / encoded images|Mean total s [source CI]|Plot s|Vision/native prepare s|LLM s|Peak GiB|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    from .summarize import ci
    for (path,k),rows in sorted(grouped.items()):
        values=[r['pipeline_generation_s'] for r in rows];sources=[r['source_id'] for r in rows];summary={'path':path,'k':k,'measurements':len(rows),'sources':len(set(sources)),'mean_pipeline_s':float(np.mean(values)),'source95CI':ci(values,sources),'median_pipeline_s':float(np.median(values)),'mean_plot_s':float(np.mean([r['plot_render_s'] for r in rows])),'mean_frontend_s':float(np.mean([r['vision_and_native_prepare_s'] for r in rows])),'mean_llm_s':float(np.mean([r['llm_total_s'] for r in rows])),'max_peak_GiB':max(r['peak_allocated_bytes'] for r in rows)/1024**3,'visual_tokens':sorted({r['visual_tokens'] for r in rows})};cost_summary[f'{path}_k{k}']=summary
        lines.append(f"|{path} / {k}|{len(rows)} / {len(set(sources))}|{summary['visual_tokens']}|{k} / {4+k}|{summary['mean_pipeline_s']:.4f} [{summary['source95CI'][0]:.4f},{summary['source95CI'][1]:.4f}]|{summary['mean_plot_s']:.4f}|{summary['mean_frontend_s']:.4f}|{summary['mean_llm_s']:.4f}|{summary['max_peak_GiB']:.2f}|")
    write_json(ROOT/'M6_pipeline_cost/cost_summary.json',cost_summary)
    (ROOT/'M6_pipeline_cost/COST_REPORT.md').write_text('# 实测成本（冻结原生参考，不是Gate结论）\n\n'+'\n'.join(lines)+'\n\n每来源1道QA、3次重复，重复不视为新的独立QA；顺序随机，6次各形状warmup。GPU常驻、无特征或图表缓存，PNG读入计费；网络/原始视频解码、模型加载与Gold评分额外开销明确排除。Vision/native prepare含SigLIP、projector、预处理与原生插入，未伪造各层独立计时。Gate和新IMU Router未训练/运行，其成本及收益为N/A。k为预定first-k因果成本日程，不是学习选择或随机Gate效果。原生token随k增长，与拟议始终256token的压缩系统不同。有限来源、共享主机计时只能作工程测量，不能称实时速度或质量保持。\n')
    fig=Figure(figsize=(10,4),dpi=160);canvas=FigureCanvasAgg(fig)
    for i,(field,title) in enumerate([('mean_pipeline_s','Resident native pipeline seconds'),('visual_tokens','Visual tokens')]):
        ax=fig.add_subplot(1,2,i+1);xx=list(range(5));yy=[cost_summary[f'native_k{k}'][field] for k in xx];yy=[v[0] if isinstance(v,list) else v for v in yy];ax.plot(xx,yy,'o-');ax.axhline(cost_summary['V1_k0'][field][0] if isinstance(cost_summary['V1_k0'][field],list) else cost_summary['V1_k0'][field],linestyle='--',color='gray',label='V1 RGB64 reference');ax.set_xlabel('Extra IMU plot images (fixed schedule)');ax.set_title(title);ax.set_xticks(xx);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();canvas.print_figure(str(ROOT/'M6_pipeline_cost/native_budget_curve.png'))
    model_lines=['|模块|实际参数总数|当前可训练|历史激活参数|','|---|---:|---:|---:|']
    for name,c in m['components'].items():model_lines.append(f"|{name}|{c['parameter_count']:,}|{c['requires_grad_count']}|{c.get('historical_active_count','未推断')}|")
    report='''# IMU-as-Image 下一阶段实际执行报告

本轮完成M0旧阶段归档、M1三库获取/同步诊断、M2冻结模型开发推理、M6原生成本实测及M7条件性研究结论。**没有训练IMU Router或IMU Gate；M3/M4/M5为not_run，而不是已完成或零分。** 严格按任务书§6.3/§11的前提约束处理；不把负结果改写为成功训练。

## 1. 旧阶段模型快照

'''+ '\n'.join(model_lines)+'''

SigLIP输入384，原生729 patch tokens、hidden1152；projector到3584，经14×14池化为196；V1的P[64,196]与原RGB V[196,3584]相乘得到64 tokens，每4帧256。原Stage3使用BEATs条件查询做加权重采样，非Top-K。现有R_A为跨两图392输入、key128/rank32残差融合，共696,320参数；与音频仅调权、值全来自原RGB的旧路径不同。现有Gate拒绝走固定V1，接受才调用EZVSL/额外图像编码；当前Cost-aware用廉价声学统计而非BEATs。F0是固定beta=.25的V1双图差分融合，无新训练参数。

完整参数级形状、dtype、device、checkpoint路径/SHA、冻结/历史激活映射、严格加载结果及实际P/Z/生成ID回归见[M0快照](../M0_snapshot/CURRENT_MODEL_SNAPSHOT.md)、[参数清单](../M0_snapshot/MODEL_PARAMETER_INVENTORY.csv)、[训练元数据](../M0_snapshot/checkpoint_training_metadata.json)、[历史结果](../M0_snapshot/PHASE_HISTORY_AND_RESULTS.md)。旧Stage1实际first-answer-token CE，不伪称Full-Vision KD；旧原始checkpoint缺失的优化器细节明确记未发现。新V1历史激活933,888，R_A696,320，Accuracy-first Gate576,161，Cost-aware478,529，过去u-only激活573,953与模型总574,059分开。

上一阶段固定R_A的最新独立来源（MUSIC282QA / Valor504QA）V1为153/310、R_A175/310、Accuracy Gate174/309、Cost Gate171/308。新的known-YT AV252为154/147/149/148。历史Cost Gate减少额外编码并节省实测计算，但1pp质量非劣未成立；重复/Sham/零音频未建立音频特异性。不混同更早MUSIC288/Valor508队列。所有旧结果保留。

## 2. Google论文映射与本轮方法

采用[Plots Unlock Time-Series Understanding in Multimodal Models](https://arxiv.org/html/2410.02637v2)的数值曲线图→现有视觉编码器思路。该论文评估Gemini1.5/GPT4o等2024模型，包含不同任务/提示条件；本轮是现有冻结LLaVA-OneVision零样本开发适配，不是其模型、少样本设置、15秒fall数据或分数的复现。[Google说明](https://research.google/blog/unlocking-the-power-of-time-series-data-with-multimodal-models/)的图像相对数值文本费用结论不表示新增图像编码免费。

本轮不加载BEATs/EZVSL，不使用音频。原始IMU值只能经物理坐标图像进入冻结视觉VLM；I7数值文本仅为单独替代基线。固定训练/校准来源量程，P1轴曲线、P2轴+norm、固定物理位置标签，不打印活动/答案/来源；不用每窗min-max、未来样本、双向滤波或IMU生成的RGB空间热图。通过实际hook计数核验视觉编码。正式新64-token双图融合没有实现/训练，不能把原生多图视为已训练R_IMU。

## 3. 数据来源、合法标签与划分

详见[数据集报告](../M1_data/DATASET_REPORT.md)、时钟审计JSON、全时间线图库与manifest。UTD108条完成记录开发smoke无可认证原始同步时间，不声称8s流式；WEAR可公开取得，但缺插值前原始时钟，本轮未运行QA；HoloAssist10段审计、6段相机计数匹配、4段有标注用于64道派生QA。只做session隔离，参与者身份未知。另2段仅拟合量程，与开发无session重合；没有模型训练集或独立确认集。Holo任务给定完成事件/查询边界，不声称自主事件发现或提前预报。

## 4. Table A：实际冻结模型QA

准确率按完整greedy生成的严格ABCD判分，最多16新token；Gold NLL为ABCD归一化首答案token，另存full-vocabulary NLL，不跨数据集汇总。Macro-F1按本队列有gold的动作类计算，不能称官方27类/Holo排行榜。UTD按受试者→来源两层bootstrap；Holo按session整体保留相关QA，2000次、seed1234；每库只有4个父簇，CI为探索性。

### UTD-MHAD（108来源，4人）

'''+(ROOT/'M2_frozen_imu_plots/utd_paired_hierarchical_results.md').read_text()+'''

UTD最初25条件×108=2700次真实推理；补充无图先验/单图零信号/单图错误来源是观察主结果后加入的开发诊断，5×108=540，绝不称预注册确认。P1/P2、系统提示/中性提示、W2/W4为事前小集合，保留每一项。最初单来源分数帧采样不同的工程smoke单独归档，不并入主108结果。

### HoloAssist（64QA，4session）

'''+(ROOT/'M2_frozen_imu_plots/holo_paired_hierarchical_results.md').read_text()+'''

Holo每QA20条件，共1280次完整实际推理。Holo图单独zero/wrong/prior控制在任何Holo分数产生前锁定；UTD分数此前已看过，因此Holo仍是开发诊断而非独立确认。所有方法相同RGB、问题/候选、确定性解码；实际图取最后2秒，RGB4张为8秒观察窗口。原生RGB2920 tokens，RGB+单图3650，图单独730，V1四帧256；数值文本为不同输入路径。真实图引起logit或文字变化不等于有益理解。

## 5. Table B / C：训练和确认状态

|阶段/策略|实际状态|新checkpoint / optimizer steps|QA/成本/ON率|
|---|---|---|---|
|R_REAL / R_DUP / R_SHAM 64-token训练|not_run|无 / 0|N/A|
|冻结R_IMU全16-mask真实Utility|not_run|无 / 0|N/A，不用旧R_A标签替代|
|新IMU G2/G3/G4/G8训练与选择|not_run|无 / 0|N/A|
|同预算Random Gate≥3seed及同mask反事实|not_run|无|N/A，没有假造Gate策略|
|Accuracy-first / Cost-aware pretest策略|not_run|无|未锁定，不用本开发队列冒充确认|
|新独立来源/跨域确认|not_run|无|未评分；本轮已检视来源全部加入seen账本|

'''+decision['reason']+'''

## 6. M6：真实成本和预算曲线

'''+ '\n'.join(lines)+'''

成本曲线见[PNG](../M6_pipeline_cost/native_budget_curve.png)和[逐次测量](../M6_pipeline_cost/resident_native_cost.jsonl)。本表是**冻结原生多图参考**的first-k固定因果日程，k=0–4不是学习Gate。每来源1道QA×3次重复；序列顺序随机，形状warmup6次。实际图渲染k次、图像编码4+k张，V1始终4张/256token。无特征/图表缓存；计入PNG读取/绘图/原生视觉处理/LLM生成；模型初始化、网络与原始视频解码、额外Gold评分排除并单独说明。Vision/native prepare未进一步分离SigLIP、projector与原生插入计时。不声称完整在线视频管线加速；没有新Router/Gate，不能给出其0–4准确率前沿、Gate ON率或实际节省比例。Gate缺失是N/A，不能写成0%。

## 7. 六项判断及统计能力

'''
    report+='|判据|判断|依据|\n|---|---|---|\n'
    for key,value in decision['six_go'].items():report+=f"|{key}|{value['status']}|{value['reason']}|\n"
    report+='''
本轮操作结论：不扩大64-token Router/Gate训练，保留桥接开发结果与可重现失败诊断。这不是对所有IMU表示或传感器的否定；UTD腕/腿运动、Holo头运动与细手动作含义不同；当前冻结模型的零样本图表读取不代表Google测试模型的能力，也不支持雷达/事件相机或跨VLM推广。

1pp非劣没有完成预注册的独立检验，不给出quality-preserving结论。即使108个独立样本零伤害，单侧95%二项上界仍约2.74%；64个约4.57%，而实际还只有4个簇。零伤害要排除1%风险，理想独立条件至少299例；存在救援/伤害与聚类时需要更多。零翻转bootstrap[0,0]不能证明等效。开发中的多提示/窗口比较存在多重比较风险，未据此挑独立测试赢家。

## 8. 完整性、复现与限制

'''
    report+=f"最终新读字节SHA复核：**{integrity['old_assets_freshly_rehashed']:,}个旧资产 + {integrity['backbone_files_freshly_rehashed']}个主干文件，0变化**。所有旧checkpoint和旧实验结果未覆盖。实际大模型参数均requires_grad=False；本轮训练步数0。新模型目录无权重文件。全仓测试341 passed（含8个IMU测试）；保留既有LLaVA SyntaxWarning，未修改旧代码。\n\n"
    report+='[环境](execution_environment.json)、[完整性](old_asset_integrity.json)、[命令/事件](../commands.json)、[源码快照](../source_snapshot/)、[划分锁](../train_val_lock.json)、[来源账本](../all_seen_sources.jsonl)及各阶段原始JSONL/日志可复查。下载中的SSL/接口失败、被主动停止的慢速整包获取、IO缓存重启与保留的部分文件均记录；未把其称为成功完整下载。传感器时钟与索引对齐假设仍有误差风险，数据集既有重采样和基础模型预训练重合无法由本轮消除。\n'
    (ROOT/'M7_final/FINAL_IMU_STAGE_REPORT.md').write_text(report)
    status('M2','completed_development_diagnostics',UTD_QA_generations=3240,Holo_QA_generations=1280,no_model_training=True)
    status('M6','completed_native_reference_only',actual_measured_pipeline_runs=len(cost),trained_IMU_Gate_pipeline='not_run')
    status('M7','completed_conditional_report',independent_generalization='not_run',overall=decision['operational_decision'])
    for p in (REPO/'utility_gate/imu_visual').glob('*'):
        if p.is_file():shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    shutil.copy2(REPO/'tests/test_imu_visual_contract.py',ROOT/'source_snapshot/test_imu_visual_contract.py');shutil.copy2(REPO/'AGENTS_IMU_AS_IMAGE_CAUSAL_GATE_ROUTER_NEXT_PHASE.md',ROOT/'source_snapshot/AGENTS_IMU_AS_IMAGE_CAUSAL_GATE_ROUTER_NEXT_PHASE.md')
    journal('final conditional IMU report saved; no Router or Gate was trained',actual_scored_QA_generations=4520,cost_measurements=len(cost),historical_assets_unchanged=True)
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in ROOT.rglob('*') if p.is_file() and p.name not in ('final_audit.json','artifact_sha256.json')};write_json(ROOT/'M7_final/artifact_sha256.json',hashes)
    write_json(ROOT/'final_audit.json',{'utc':now(),'old_asset_integrity':integrity['all_unchanged'],'fresh_old_asset_count':24584,'fresh_backbone_count':17,'new_trained_models':0,'new_optimizer_steps':0,'UTD_QA_generations':3240,'Holo_QA_generations':1280,'cost_measured_runs':len(cost),'tests':341,'stage_states':json.loads((ROOT/'STATUS.json').read_text()),'artifact_manifest_sha256':sha256(ROOT/'M7_final/artifact_sha256.json'),'old_results_overwritten':False,'independent_confirmation':'not_run'})
    print('FINAL REPORT SAVED',ROOT/'M7_final/FINAL_IMU_STAGE_REPORT.md',flush=True)
if __name__=='__main__':finish()
