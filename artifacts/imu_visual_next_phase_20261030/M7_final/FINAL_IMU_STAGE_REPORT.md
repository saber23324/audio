# IMU-as-Image 下一阶段实际执行报告

本轮完成M0旧阶段归档、M1三库获取/同步诊断、M2冻结模型开发推理、M6原生成本实测及M7条件性研究结论。**没有训练IMU Router或IMU Gate；M3/M4/M5为not_run，而不是已完成或零分。** 严格按任务书§6.3/§11的前提约束处理；不把负结果改写为成功训练。

## 1. 旧阶段模型快照

|模块|实际参数总数|当前可训练|历史激活参数|
|---|---:|---:|---:|
|SigLIP|397,747,744|0|None|
|projector|16,980,992|0|None|
|LLM_and_native_special_embeddings|7,612,631,040|0|未推断|
|Stage1_original|1,991,680|0|933888|
|V1_retrained|1,991,680|0|933888|
|Stage3_original_AudioRouter|1,991,680|0|None|
|R_A|696,320|0|696320|
|BEATs|90,717,055|0|None|
|EZVSL|22,872,064|0|None|
|old_LLP_AVS_proxy|574,059|0|None|
|old_G_A_F0_target|576,097|0|None|
|latest_accuracy_first|576,161|0|576161|
|latest_cost_aware|478,529|0|478529|
|historical_u_only|574,059|0|573953|

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

|Variant|Correct/QA|Accuracy [cluster 95% CI]|Gold NLL|Macro F1|Visual tokens|
|---|---:|---:|---:|---:|---:|
|I0_native_RGB|66/108|0.6111 [0.4907, 0.7315]|1.373050|0.5608|[2920]|
|I0_V1_64|56/108|0.5185 [0.4074, 0.6206]|1.163469|0.4653|[256]|
|I1_plot|28/108|0.2593 [0.1667, 0.3704]|1.833826|0.1899|[730]|
|I2_real|63/108|0.5833 [0.4907, 0.6759]|1.352251|0.5445|[3650]|
|I3_duplicate|66/108|0.6111 [0.5000, 0.7130]|1.407409|0.5634|[3650]|
|I4_zero|63/108|0.5833 [0.4722, 0.6944]|1.354489|0.5496|[3650]|
|I5_wrong_source|64/108|0.5926 [0.5000, 0.6852]|1.319170|0.5528|[3650]|
|I6_past_shift|63/108|0.5833 [0.4907, 0.6759]|1.335285|0.5458|[3650]|
|I7_numeric_text|35/108|0.3241 [0.2130, 0.4444]|1.471858|0.2628|[0]|
|S_P1_neutral_only|27/108|0.2500 [0.1667, 0.3519]|2.051680|0.1815|[730]|
|S_P1_neutral_real|57/108|0.5278 [0.4352, 0.6204]|1.427030|0.5164|[3650]|
|S_P1_neutral_zero|58/108|0.5370 [0.4352, 0.6481]|1.349373|0.5113|[3650]|
|S_P1_neutral_wrong|61/108|0.5648 [0.4630, 0.6667]|1.375031|0.5521|[3650]|
|S_P2_sensor_instructions_only|27/108|0.2500 [0.1574, 0.3519]|1.788247|0.1825|[730]|
|S_P2_sensor_instructions_real|62/108|0.5741 [0.4722, 0.6759]|1.338314|0.5404|[3650]|
|S_P2_sensor_instructions_zero|62/108|0.5741 [0.4722, 0.6667]|1.352512|0.5414|[3650]|
|S_P2_sensor_instructions_wrong|63/108|0.5833 [0.4907, 0.6759]|1.352010|0.5459|[3650]|
|S_P2_neutral_only|33/108|0.3056 [0.2130, 0.4074]|1.974050|0.2588|[730]|
|S_P2_neutral_real|60/108|0.5556 [0.4630, 0.6481]|1.385933|0.5286|[3650]|
|S_P2_neutral_zero|58/108|0.5370 [0.4259, 0.6481]|1.352043|0.5048|[3650]|
|S_P2_neutral_wrong|58/108|0.5370 [0.4444, 0.6296]|1.393927|0.5079|[3650]|
|S_W4_only|34/108|0.3148 [0.2222, 0.4167]|1.748555|0.2341|[730]|
|S_W4_real|64/108|0.5926 [0.4907, 0.6944]|1.343612|0.5627|[3650]|
|S_W4_zero|64/108|0.5926 [0.5000, 0.6852]|1.358891|0.5497|[3650]|
|S_W4_wrong|64/108|0.5926 [0.4907, 0.6944]|1.332977|0.5618|[3650]|
|A_prior_no_image|32/108|0.2963 [0.2130, 0.3889]|1.496860|0.2416|[0]|
|A_plot_zero_W2|36/108|0.3333 [0.2315, 0.4444]|1.839756|0.2561|[730]|
|A_plot_wrong_W2|31/108|0.2870 [0.1944, 0.3889]|1.835581|0.2122|[730]|
|A_plot_zero_W4|33/108|0.3056 [0.2130, 0.4167]|1.833775|0.2416|[730]|
|A_plot_wrong_W4|30/108|0.2778 [0.1944, 0.3796]|1.803112|0.2061|[730]|

|Paired comparison|Delta accuracy [cluster 95% CI]|Rescue / Harm|Delta NLL|
|---|---:|---:|---:|
|I2_real minus I0_native_RGB|-0.0278 [-0.1389, 0.0833]|6 / 9|-0.020799|
|I2_real minus I3_duplicate|-0.0278 [-0.1111, 0.0556]|5 / 8|-0.055158|
|I2_real minus I4_zero|0.0000 [-0.0556, 0.0463]|2 / 2|-0.002237|
|I2_real minus I5_wrong_source|-0.0093 [-0.0463, 0.0000]|0 / 1|0.033081|
|I2_real minus I6_past_shift|0.0000 [0.0000, 0.0000]|0 / 0|0.016966|
|I1_plot minus A_prior_no_image|-0.0370 [-0.1389, 0.0648]|6 / 10|0.336966|
|I1_plot minus A_plot_zero_W2|-0.0741 [-0.1667, 0.0093]|3 / 11|-0.005930|
|I1_plot minus A_plot_wrong_W2|-0.0278 [-0.0648, 0.0000]|0 / 3|-0.001756|
|S_W4_only minus A_plot_zero_W4|0.0093 [-0.0741, 0.1111]|10 / 9|-0.085220|
|S_W4_only minus A_plot_wrong_W4|0.0370 [-0.0093, 0.0926]|5 / 1|-0.054557|

Development only. Few independent clusters: intervals are exploratory. Zero flips and degenerate intervals do not establish equivalence/noninferiority. Macro-F1 uses observed gold classes; four-choice QA is not official multiclass benchmark accuracy.


UTD最初25条件×108=2700次真实推理；补充无图先验/单图零信号/单图错误来源是观察主结果后加入的开发诊断，5×108=540，绝不称预注册确认。P1/P2、系统提示/中性提示、W2/W4为事前小集合，保留每一项。最初单来源分数帧采样不同的工程smoke单独归档，不并入主108结果。

### HoloAssist（64QA，4session）

|Variant|Correct/QA|Accuracy [cluster 95% CI]|Gold NLL|Macro F1|Visual tokens|
|---|---:|---:|---:|---:|---:|
|H0_RGB|24/64|0.3750 [0.2969, 0.4375]|1.864270|0.4396|[2920]|
|H0_V1|22/64|0.3438 [0.1875, 0.5004]|1.498790|0.3408|[256]|
|H1_plot|25/64|0.3906 [0.3438, 0.4375]|1.713900|0.3350|[730]|
|H2_real|24/64|0.3750 [0.3281, 0.4219]|1.965642|0.4856|[3650]|
|H3_duplicate|23/64|0.3594 [0.2812, 0.4375]|1.936488|0.4314|[3650]|
|H4_zero|24/64|0.3750 [0.3281, 0.4219]|1.945589|0.4856|[3650]|
|H5_wrong|24/64|0.3750 [0.3281, 0.4219]|1.964939|0.4856|[3650]|
|H6_past_shift|24/64|0.3750 [0.3281, 0.4219]|1.954738|0.4856|[3650]|
|H7_numeric_text|23/64|0.3594 [0.2500, 0.4375]|1.437789|0.3832|[0]|
|H1_zero_plot|24/64|0.3750 [0.2969, 0.4375]|1.704421|0.3387|[730]|
|H1_wrong_plot|25/64|0.3906 [0.2500, 0.4844]|1.683888|0.3224|[730]|
|H1_prior_no_image|23/64|0.3594 [0.2812, 0.4219]|1.567146|0.3694|[0]|
|Hs_P2_sensor_instructions_only|25/64|0.3906 [0.3438, 0.4375]|1.719350|0.3350|[730]|
|Hs_P2_sensor_instructions_real|25/64|0.3906 [0.3438, 0.4375]|1.938839|0.4959|[3650]|
|Hs_P2_sensor_instructions_zero|24/64|0.3750 [0.3281, 0.4219]|1.938481|0.4891|[3650]|
|Hs_P2_sensor_instructions_wrong|24/64|0.3750 [0.3281, 0.4219]|1.941415|0.4856|[3650]|
|Hs_P1_neutral_only|21/64|0.3281 [0.2344, 0.4062]|1.853250|0.2788|[730]|
|Hs_P1_neutral_real|25/64|0.3906 [0.3438, 0.4375]|2.255096|0.4851|[3650]|
|Hs_P1_neutral_zero|22/64|0.3438 [0.3125, 0.3750]|2.234198|0.4282|[3650]|
|Hs_P1_neutral_wrong|23/64|0.3594 [0.3125, 0.4066]|2.249568|0.4545|[3650]|

|Paired comparison|Delta accuracy [cluster 95% CI]|Rescue / Harm|Delta NLL|
|---|---:|---:|---:|
|H2_real minus H0_RGB|0.0000 [-0.0625, 0.0781]|3 / 3|0.101372|
|H2_real minus H3_duplicate|0.0156 [-0.0469, 0.0938]|4 / 3|0.029154|
|H2_real minus H4_zero|0.0000 [0.0000, 0.0000]|0 / 0|0.020053|
|H2_real minus H5_wrong|0.0000 [0.0000, 0.0000]|0 / 0|0.000703|
|H2_real minus H6_past_shift|0.0000 [0.0000, 0.0000]|0 / 0|0.010904|
|H1_plot minus H1_prior_no_image|0.0312 [-0.0312, 0.0938]|6 / 4|0.146754|
|H1_plot minus H1_zero_plot|0.0156 [-0.0312, 0.0625]|3 / 2|0.009479|
|H1_plot minus H1_wrong_plot|0.0000 [-0.0938, 0.0938]|2 / 2|0.030012|

Development only. Few independent clusters: intervals are exploratory. Zero flips and degenerate intervals do not establish equivalence/noninferiority. Macro-F1 uses observed gold classes; four-choice QA is not official multiclass benchmark accuracy.


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

M2没有建立冻结视觉骨干可利用的稳定IMU特异桥接，任务书§6.3(a)推进前提未满足。UTD主条件真实图63/108、重复RGB66/108、零图63/108、错误图64/108；图单独28/108，零图36/108，先验32/108。W4有弱logit/个例趋势，但精度差的受试者→来源CI跨0，RGB融合无稳定独特收益。Holo正确性真实/零/错误均24/64且逐题完全一致，真实Gold NLL1.965642高于RGB1.864270与零图1.945589；P1/P2、提示诊断未提供稳健证据。两库都只有4个父簇；UTD缺原始同步时钟，WEAR缺插值前时间，Holo只有4个精确时钟且有标注的开发session，尚未建立足量独立训练/验证/确认队列。依据§6.3的弱桥接诊断约束及§11的条件性not_run规则，停止扩大本配置训练；不训练任意Router/Gate或拿旧音频Utility代替新IMU监督。此为本配置的研究推进判断，不证明所有IMU任务/表示无效。

## 6. M6：真实成本和预算曲线

|Path / k|Measurements / sources|Visual tokens|Render count / encoded images|Mean total s [source CI]|Plot s|Vision/native prepare s|LLM s|Peak GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|V1 / 0|12 / 4|[256]|0 / 4|0.1160 [0.1156,0.1163]|0.0000|0.0552|0.0405|15.16|
|native / 0|12 / 4|[2920]|0 / 4|0.1823 [0.1765,0.1933]|0.0000|0.0450|0.1218|15.57|
|native / 1|12 / 4|[3650]|1 / 5|0.3450 [0.3408,0.3512]|0.1258|0.0599|0.1430|15.71|
|native / 2|12 / 4|[4380]|2 / 6|0.5464 [0.5063,0.5865]|0.2926|0.0766|0.1602|15.85|
|native / 3|12 / 4|[5110]|3 / 7|0.6966 [0.6695,0.7341]|0.3977|0.0917|0.1893|15.98|
|native / 4|12 / 4|[5840]|4 / 8|0.9335 [0.8614,1.0121]|0.5883|0.1073|0.2192|16.12|

成本曲线见[PNG](../M6_pipeline_cost/native_budget_curve.png)和[逐次测量](../M6_pipeline_cost/resident_native_cost.jsonl)。本表是**冻结原生多图参考**的first-k固定因果日程，k=0–4不是学习Gate。每来源1道QA×3次重复；序列顺序随机，形状warmup6次。实际图渲染k次、图像编码4+k张，V1始终4张/256token。无特征/图表缓存；计入PNG读取/绘图/原生视觉处理/LLM生成；模型初始化、网络与原始视频解码、额外Gold评分排除并单独说明。Vision/native prepare未进一步分离SigLIP、projector与原生插入计时。不声称完整在线视频管线加速；没有新Router/Gate，不能给出其0–4准确率前沿、Gate ON率或实际节省比例。Gate缺失是N/A，不能写成0%。

## 7. 六项判断及统计能力

|判据|判断|依据|
|---|---|---|
|GO-IMU-BRIDGE|INCONCLUSIVE|主条件未超过正确的零/错误/先验控制；W4有弱信号，父簇仅4，未建立可泛化桥接|
|GO-IMU-ROUTER|INCONCLUSIVE|not_run；无R_REAL/R_DUP/R_SHAM checkpoint或训练比较|
|GO-GATE-SELECT|INCONCLUSIVE|not_run；无固定R_IMU真实16-mask Utility、新Gate或同预算随机选择|
|GO-GATE-EFFICIENCY|INCONCLUSIVE|not_run；仅原生多图成本实测，未验证受质量约束的Gate节省|
|GO-IMU-SPECIFIC|NO-GO|本轮配置的收益归因不成立：UTD真实不优于零/错误；Holo真实/零/错误逐题正确性一致；不视为等价性证明|
|GO-GENERALIZATION|INCONCLUSIVE|仅开发诊断，无冻结独立确认或跨参与者、同标签跨域验证|

本轮操作结论：不扩大64-token Router/Gate训练，保留桥接开发结果与可重现失败诊断。这不是对所有IMU表示或传感器的否定；UTD腕/腿运动、Holo头运动与细手动作含义不同；当前冻结模型的零样本图表读取不代表Google测试模型的能力，也不支持雷达/事件相机或跨VLM推广。

1pp非劣没有完成预注册的独立检验，不给出quality-preserving结论。即使108个独立样本零伤害，单侧95%二项上界仍约2.74%；64个约4.57%，而实际还只有4个簇。零伤害要排除1%风险，理想独立条件至少299例；存在救援/伤害与聚类时需要更多。零翻转bootstrap[0,0]不能证明等效。开发中的多提示/窗口比较存在多重比较风险，未据此挑独立测试赢家。

## 8. 完整性、复现与限制

最终新读字节SHA复核：**24,584个旧资产 + 17个主干文件，0变化**。所有旧checkpoint和旧实验结果未覆盖。实际大模型参数均requires_grad=False；本轮训练步数0。新模型目录无权重文件。全仓测试341 passed（含8个IMU测试）；保留既有LLaVA SyntaxWarning，未修改旧代码。

[环境](execution_environment.json)、[完整性](old_asset_integrity.json)、[命令/事件](../commands.json)、[源码快照](../source_snapshot/)、[划分锁](../train_val_lock.json)、[来源账本](../all_seen_sources.jsonl)及各阶段原始JSONL/日志可复查。下载中的SSL/接口失败、被主动停止的慢速整包获取、IO缓存重启与保留的部分文件均记录；未把其称为成功完整下载。传感器时钟与索引对齐假设仍有误差风险，数据集既有重采样和基础模型预训练重合无法由本轮消除。


### 实际检查与未执行检查

- `python -m pytest -q tests -p no:cacheprovider`，最终341项通过；新增8项验证未来NaN/噪声不影响当前统计与像素、物理幅值/极性、固定量程、时钟精度、缺失/间隙、WEAR无虚构gyro、异步IMU只取过去值、压缩PTS与原始相机时钟不可混用。M0实际GPU旧P/Z/生成ID回归通过。
- [原始输出集成审计](actual_output_contract_audit.json)核验全部4,520次QA的条件数量、真实/零/错误图像确实不同、Holo全部256个时刻的传感器原始值因果界限、RGB采集时刻、实际SigLIP图像数和视觉token数；72次成本测量的0–4渲染/图像计数通过。保留一次验证脚本PYTHONPATH缺失报错及修正后的通过日志。
- [Prefill/TTFT/forward计数](../M6_pipeline_cost/prefill_TTFT_forward_counts.json)提供实际LLM计时与串行前端+LLM推导的常驻输入到首token时间；不含视频/网络解码/模型加载，未伪称在线播放实测。
- 未执行的新IMU Router零残差64-token身份测试、Gate ON/OFF条件编码集成、Router/Gate训练互斥更新检查、完整16-mask Utility、随机Gate≥3seed、质量非劣与独立确认，均随M3–M5 `not_run`；没有用旧音频测试替代新IMU部署验收。
- 本轮新增文件仅位于`utility_gate/imu_visual/`、`tests/test_imu_visual_contract.py`与新实验目录；数据位于新的NVMe目录。旧框架源码与已审计checkpoint/结果SHA不变。源码快照、原始错误与负结果、量程和时钟锁均保留。

补充关闭审计：所有实际活跃checkpoint路径（含符号链接别名与NVMe EZVSL）再次逐个SHA复核通过，见[active_checkpoints_final.json](active_checkpoints_final.json)。新取得和派生数据的逐文件字节/SHA已保存到M1，未把部分tar/zip文件冒充整包下载。
