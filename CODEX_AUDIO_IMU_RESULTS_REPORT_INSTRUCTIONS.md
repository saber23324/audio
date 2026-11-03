# Codex 任务：审计并整理 Audio 与 IMU 的全部已完成实验结果（面向科研人员的可读报告）

> 任务类型：**只读审计 + 指标重新核对 + 中文报告整理**。不是新的实验计划。必须实际检索本仓库中既有训练日志、checkpoint 元数据、模型实现、数据 manifest、预测文件、统计输出、计时日志与研究判断。**不得进行新训练、修改模型权重、重新调参、重新选测试样本或更改任何已锁定策略**。如果发现历史摘要与原始结果不一致，保留原始结果，标明差异及证据路径；不要凭摘要猜测未知配置。

## 0. 核心目标与写作风格

用户希望一次看懂：**Audio 方向和 IMU 方向分别在什么数据集上训练、采用什么网络、到底训练了哪些模块、每个实验策略的完整含义、测的是什么指标、结果是多少，以及能得出什么结论。**

强制要求：
1. **每个表都写清数据集、数据划分、确认来源数、问答数、评测时间窗口、答案格式、评分规则。** 不同测试队列、不同提问与不同时间窗口的结果绝对不能混为一张可直接排名的表。
2. **默认使用完整中文名称**。代码策略 ID 可以放括号中供检索，不允许只写 `T1_REAL_always`、`B2_R_always` 之类没有解释的列名。首次出现缩写必须紧跟完整英文或中文含义；以后每张独立结果表仍需有简洁的“策略说明”列或紧邻表的图例。
3. 对每一种策略都说明五件事：**训练阶段**（从哪组初始化、哪组样本训练、训练目标、可训练参数和步数）、**推理输入**（原始 RGB / 真实音频或 IMU / 错配或零值 / 辅助可视化图）、**Gate 决策方式**（无 / Always-on / 学习得到 / 固定规则 / 随机）、**是否使用 Router 及每帧输出 token 数**、**冻结了哪些主干**。如果一个字段无法由源码或工件验证，填“未核验”，不准猜。
4. 一律把**真正使用真实传感器信息的收益**与**普通任务训练收益、双图/额外计算收益、选择性节省成本**拆开。不能把“训练后比未训练强”写成“辅助模态有效”；不能把“Gate 比 Always-on 快”写成“Gate 能辨别有益传感器片段”。
5. 对于同一策略的**训练时输入**和**推理时干预**单独列示。特别区分：真实传感器训练 + 错传感器测试，与从头使用重复图/零传感器训练的独立 Router；两者不是同一实验。
6. 不得将 16-mask Oracle、GT 热图或 QA 标准答案参与测试时选择的分析上限写成部署系统。解释因果音频/IMU 时钟处理和严格的当前/历史窗口。
7. 报告中每一个数值、比率、参数数量和结论都附上**具体相对路径 + 结果键 / CSV 列名 / 日志行或源码函数**。附表中提供可追溯原始数据，不准编造显著性或不存在的置信区间。

## 1. 先做结果资产定位，不要直接撰写

优先审查已知 Audio 工件/实现（以实际存在路径为准，若不存在在报告中说明）：
- `artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/GO_NO_GO.md`
- `utility_gate/grounded_dual_image/README.md`
- `artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/final_audit.json`
- `utility_gate/adaptive_grounded64/`、上一轮 M0–M7 训练、确认及 `qa_eval/paired_summary.*`
- `utility_gate/gate_first_native/`、Gate-first → Router-second → Native Gate-only 的 M0–M7 报告、`results/summary.json`、`results/per_qa.jsonl`、`results/per_window_gate_decisions.jsonl`、`results/pipeline_cost.json`、`commands.json`、`pretest_lock.json`
- 原 Stage1 Vision-only、原 Stage3 AudioRouter 的 checkpoint 元数据、实际训练源码，以及 VGG-SS、LLP 等诊断报告（单独标注为**诊断/代理任务**，不得混入正式 QA）。

查找 IMU 相关的 HoloAssist 最新实验根目录、M0–M6、`RESEARCH_DECISIONS.json`、`M6_confirmation/PAIRED_SUMMARY.json`、`FINAL_ASSET_SHA_AUDIT.json`、训练/验证源 manifest、Router/Gate 训练日志、逐题与逐窗口输出、原生多图对照及 `Fixed-k2` 策略定义。IMU 目录未知时由 Codex 在仓库中搜索关键字 `HoloAssist`、`T1_REAL_always`、`T8_DUP_trained_always`、`T2_accuracy`、`Fixed-k2`、`REAL_wrong`，不能擅自虚构目录。

优先级为：**逐题预测/逐窗口行为与实际运行配置 > 可复现 JSON/CSV > 运行日志 > 阶段报告 > 人工总结**。对于冲突，记录版本与来源并标出待核验。确认文件身份、数据 split 与历史已经检视的 source registry；只读保护现有全部权重、代码和结果。

## 2. 必须覆盖的数据集与实验阶段

### Audio（音频辅助视觉理解）
- 原始 AudioRouter 的 Stage1 Vision-only 与 Stage3 Audio-conditioned Router（包括真实音频 / 零波形经过同一个冻结音频编码器的路径）。说明它们是否为**不同训练 checkpoint**；禁止将不同训练历史的差异直接视为纯音频因果效应。
- VGG-SS：声源框区域的 routing mass、excess/lift、Oracle GT / 预测图；这是一种**空间诊断**，非官方问答准确率。LLP：音视频共同事件/代理对应关系 AUROC、AUPRC；并非下游 Utility，也不是精确可见声源概率。按原始结果表核验样本量和版本。
- MUSIC-AVQA / Valor32k AVQA 的**冻结迁移**：单图、重复 RGB 双图、Grounded 双图、错移/Sham 双图、等 64-token 融合、原 AudioRouter；逐队列写 701 / 256 及其协议，不将这轮与下一轮的准确率直接比较。
- **Adaptive Gate + 新 Stage1**：重训 Vision-only Router，冻结的 Grounded-64 固定融合分支，再独立训练新 Utility Gate；主确认 MUSIC 439 题、Valor 512 题，说明 Gate 全关与 V1 相同的含义，并说明原 Stage1 训练损失是首答案 token CE，而非未经核验的 Full-Vision KD。
- **Gate-first → Router-second → Native Gate-only**：固定 8 秒已观察前缀，t=2/4/6/8，MUSIC 288 题与 Valor 508 题；区分 Gate-first、冻结 Gate 后训练 64-token Fusion Router、完全绕过 Router 的原生双图三个阶段。解释 `A0_V1`, `A1_F0`, `A_MAIN_GA`, `B2_R_always`, `B3_GA_R`, `B8_dupR_always`, `B9_shamR_always`, `B10_zeroR_always`, `C0_RGB_RGB`, `C1_always_ground`, `C2_GA_native`, `C3_GA_sham`, `C4_GA_dup`。将 MUSIC 上 Router 适配提升与 Duplicate/Sham/Zero 几乎相同、Valor 未确认泛化并排说明。
- 旧 AudioRouter 在视频问答数据集上的早期论文/项目结果可置于“历史背景”一章，但必须标为不同数据、训练或评测协议；不可与本次确认表混排。

### IMU（Inertial Measurement Unit，惯性测量单元）
- HoloAssist：当前实际有 **94 个可用录制前缀 / 721 道 QA；训练 500 / 验证 166 / 确认 55**；确认 7 个录制、4 个保守录制家族。注意 94 个是**可用前缀**而非 94 部完整视频；参与者身份未知，不能说跨人泛化。QA 是官方完成动作 Verb（动词）/Noun（名词）标签衍生的问题，不是官方动作识别指标、未来动作预测或一般视觉问答。
- 训练阶段：7 个 Router、共 8,373 次更新；REAL（真实 IMU）、DUP（重复 RGB）和 ZERO（零 IMU）各 1,500 次更新；4 个 Gate 各 1,890 步、总 7,560 步；冻结 REAL 后 10,656 次完整 16-mask QA。通过源码核对可训练参数与不同 Router 的定义；不能只凭数字把这些误说成不同的独立样本数。
- WEAR 实际有 15 人传感器片段但缺原始时钟且含非因果插值，**未进入严格主训练**；EgoExo4D 受限数据未取得；UTD 仅工程参考。明确不把它们列作有结果的外部测试集。
- 必须逐项定义并列出：`T0_V1`、`Old_R_REAL_always`、`T1_REAL_always`、`T8_DUP_trained_always`、`T9_ZERO_trained_always`、`T2_accuracy`、`T2_cost`、`T10_REAL_wrong_always`、`Fixed-k2`、原生单图/原生双图对照。**例如 `T1_REAL_always` 在报告里要写成类似“使用真实 IMU 辅助视图训练的融合 Router，在所有四个时刻均无条件启用该路径，最终每时刻输出 64 个视觉 token”（具体可视化编码、Gate/Router 调用与加载的 checkpoint 必须以实际代码核验）**。不能只写“REAL”。
- 最新确认参考数值须与 JSON 原始记录核对：V1=16/55；旧 REAL always=19/55；新 REAL always=37/55；DUP-trained always=36/55；ZERO-trained always=38/55；已训练 Gate（accuracy/cost 实际同策略）=38/55、170/220 个窗口 ON=77.27%；REAL_wrong=36/55；Fixed-k2=39/55。不要把 Fixed-k2 事后更高点估计写成新的调参选择。

## 3. 对每个策略提供“人能读懂的名称对照表”

产出 `strategy_name_mapping.csv`，列顺序建议：
`modality,stage,dataset,raw_strategy_id,plain_chinese_name,train_input,trainable_modules,training_examples,training_updates,init_checkpoint,inference_sensor_input,gate_mode,router_mode,view_count,visual_tokens_per_frame,visual_tokens_per_question,source_file,verification_status`

要求对每个 raw ID 给出完整中文解释。特别注意：
- `REAL` = 训练/推理中使用真正同步的音频或 IMU 信号（具体代表什么要按代码明确），**不等同“GT 热图”**。
- `DUP` = 重复 RGB 视觉输入；区分**从头 DUP 训练** vs **固定 REAL Router 在推理时把辅助图换成 DUP**。
- `ZERO` = 零传感器输入经过什么预处理或编码器，还是直接零热图，必须源码核查。
- `WRONG` / `SHAM` = 同样 RGB 下的错配/错移/伪提示；说明干预是训练时还是测试时。
- `always` = 每个有资格的时刻无条件使用融合分支，不是 Gate 预测正确。
- `G_A`/`T2` = 学到的 Gate 策略，说明是基于哪个 Router 的收益标签训练、何时阈值锁定、实际 ON 率以及是否改变原生图数量。
- `R`/`F0`/`V1` = 逐项解释具体的训练 checkpoint / 固定融合器 / Vision-only 基线，禁止写“模型 1/2”。
- `T2_accuracy` 与 `T2_cost` 如果是同一个 checkpoint 和同一个阈值，必须明确写“实际相同策略”，不可作为两条不同模型贡献。

## 4. 指标词典：每个指标必须定义**统计对象、公式、方向、单位、注意事项**

独立提供 `metric_dictionary.md`，正文中首次出现时用一句中文解释。至少包括：

| 指标 | 必须解释的内容 |
| --- | --- |
| QA Accuracy / Exact Match (EM，准确率 / 精确匹配率) | 正确题数/实际题数；MUSIC 生成文本规范化 EM 与 Valor/HoloAssist 四选项 ABCD Accuracy 规则不同；若不一致需写清。 |
| Gold Negative Log-Likelihood (Gold NLL，正确答案负对数似然) | 真实答案的负对数概率；越小越好。明确是整串答案、首答案 token 全词表概率，还是四选项重新归一化后计算；这些数字跨协议不能比较。 |
| Training Loss (训练损失) | 首答案 token 全词表交叉熵（CE）或其他实际损失；与报告的四选项 Gold NLL 不能混淆。 |
| Rescue / Harm（挽救/损害题数） | 对同一道题：基线错而方法对/基线对而方法错；两者差/样本数即准确率差；说明与哪个基线对比。 |
| Paired Delta (配对准确率差) | 同一题、同一评测协议下方法减基线，单位百分点 pp，不等于百分比相对提升。 |
| Confidence Interval (CI，置信区间) | 95% CI，按什么来源/录制家族聚类 Bootstrap、重采样次数与家族数。4 家族时 bootstrap 与显著性极不稳健；零翻转 [0,0] **不是**等效性证明。 |
| Gate ON Rate（Gate 启用比例） | 分子=启用的时刻/合格时刻，区别逐题 Gate 选择、平均额外图数和 16-mask 离线干预。 |
| Added Image Count（额外图像数） | 相对基础 RGB 每题额外编码多少张图；不是 LLM token 数，不能替代真实成本。 |
| Visual Tokens（视觉 token） | 路由后每帧 64、四时刻 256；原生单图/双图 2,920/5,840 视具体协议核实；分清 SigLIP/Projector 中间 token、LLM 最终视觉 token。 |
| Pipeline Runtime / Latency（计算时延） | 平均还是中位、秒/题；前端/TTFT/端到端各包含哪些步骤，模型冷启动、Gold 打分、缓存是否包含；多轮小样本顺序测量不可跨实验比较“加速”。 |
| Cost Saving（成本节省比例） | 相对相同计时协议 Always-on 的实际秒数比例；给原始数、差值及来源家族配对 CI；省时不自动证明有用模态选择。 |
| AUROC / AUPRC | 若包含旧 LLP correspondence 代理诊断，定义正负类别、类别占比及衡量对象；不能称真实 QA Utility 或声源定位质量。 |
| Spatial ROI Mass / Lift | 若包含 VGG-SS，解释 ROI、面积基线、GT 是否仅用于诊断；非 QA 提升证据。 |
| Source Family / Recording Cluster | 以实际可审计身份聚类的单位和不确定之处（MUSIC 原始 YouTube 血缘不可核验、IMU 参与者身份未知）。 |
| Oracle Upper Bound（Oracle 上限） | 用答案或未来信息选择的事后诊断上限，不可部署或用于验证策略。 |

请另外解释 Gold NLL 可能在 Accuracy 提升时仍变差的原因：正确题数与预测概率校准/极端错误置信度是不同目标。不得仅凭此判定实现有 Bug，但应指明可核查的逐题尾部证据。

## 5. 报告结构与比较纪律

报告首页先给 **一屏摘要**：Audio 与 IMU 各自的数据集、训练路径、最好但必须指明来源/协议的数值、相对视觉基线是否有益、相对 DUP/ZERO/SHAM 是否有额外传感器独特收益、Gate 是否超过固定策略，以及证据强度/结论。不要跨任务计算统一“平均准确率”。

正文结构建议：
1. **快速读懂模型架构**：RGB → 视觉编码器 → 64-token Vision-only Router；音频/IMU → 传感器编码与图像化/辅助视图 → Fusion Router；Gate 控制分支与条件计算；原生多图如何绕过 Router；清楚标注冻结/训练参数。
2. **数据集与划分登记表**：MUSIC-AVQA、Valor、HoloAssist、VGG-SS、LLP 等；样本、时间协议、答案、来源身份、用在哪个阶段、哪些未能确认独立性。
3. **Audio 实验按时间与训练阶段分表**：冻结迁移、Stage1/Gate、Gate-first Router-second、Native Gate-only，逐项写策略全名 + raw ID + 实际训练 + 结果 + 公平对照。
4. **IMU 实验完整表**：55 题每个策略的正确数/准确率、Gold NLL、Rescue/Harm、Gate ON 率、额外图像均数、视觉 token、Pipeline 秒；另外列数据规模与训练家族数扩展的消融，注明数据量和更新步数同时变化，不能声称“纯数据规模效应”。
5. **每个模型训练阶段流水账表**：初始权重、数据 split、是否训练真实/伪传感器、模型参数、轮数与 updates、损失、选择 checkpoint 的验证指标、冻结项、监督标签如何产生。
6. **关键同队列配对表**：Audio Grounded Router vs Duplicate-trained / Sham / Zero，Audio Gate vs 相同 mask 的 DUP，IMU REAL vs DUP/ZERO、IMU Learned Gate vs Always-on / Fixed-k2 / 同预算随机，全部提供逐题 Rescue/Harm 和可验证的 CI（若存）。
7. **指标释义、局限与研究结论**：能证明的是 Router 任务适配、条件计算节省；没有证明的是音频/IMU 独特收益或优于简单规则的 Gate 选择；指出数据少、来源家族少、域限制、未知预训练重叠等。
8. **可复现性**：实际训练和推理命令、checkpoint 相对路径、运行配置、随机种子、源码入口、数据 manifest、报告/原始文件相对链接。

特别要说明：
- AUDIO 上 MUSIC 288 题训练后 Router 206/288（71.53%），但 Duplicate/Sham/Zero 同为 206/288；Valor 508 题没有稳定提升。与另一轮 MUSIC 701 题 Grounded 原生双图 60.20% 不能直接比较。
- IMU 的 55 题中 REAL Router 37/55，DUP 36/55，ZERO 38/55；Gate 为 38/55（77.27% ON）、Fixed-k2 为 39/55。REAL 并没有证明优于零 IMU 或重复图；只有 4 个确认家族，不能写显著优越或跨参与者泛化。
- Audio Gate-first → Router-second 训练顺序及 Utility 的“相对于**哪一版 Router**的收益”必须写清。旧 Gate 迁移到后来训练的新 Router 并非同目标。
- 原生 Audio 双图 5,840 token 与 64-token 路由四时刻共 256 token 的成本不相同；既要展示准确率也要展示代价。

## 6. 精确交付与自动校验

在仓库内新建 `reports/audio_imu_results_explained_20261102/`（若重名则保留已有文件、创建新版本），至少输出：
- `README.md`：**给人的中文完整报告**，含完整策略名称、两模态结果、训练与指标定义、结论与表内引用；这是主交付。
- `audio_results.md`：Audio 各历史阶段独立数据集/队列的详细结果，不混排。
- `imu_results.md`：HoloAssist 完整策略结果与训练过程。
- `metric_dictionary.md`：指标释义，包含具体公式、单位及不可比情况。
- `strategy_name_mapping.csv`：每一个 raw 策略 ID → 中文解释及 train/inference 路径。
- `evidence_index.csv`：每个关键数值映射到原始文件、字段、代码/运行信息。
- `verification_report.json`：完整性、核验通过/失败/缺失值、可比性检查、原始资产未修改验证。

执行核验脚本：对每个可追踪 Accuracy 核验 `correct/n`；每个 Rescue/Harm 核验两策略配对题目的答案变化；检查 ON 次数和 eligible window 总数；区分总 LLM 视觉 token 和额外编码图像数；检查时间统计样本数、均值中位定义；核对数值 rounding 与 CI 聚类单位。无法取得逐题记录时明确标注“仅报告聚合结果、不可重新验证”，**不可补造数据**。

最终在终端用中文回答用户：产物相对路径及“请先读 README.md”；列最重要 Audio、IMU 对比各一条；说明哪些结果属于任务适配、哪些尚不足以证明真实传感器价值。禁止没有证据地写“取得 SOTA”“显著超越”“大幅提升音频/IMU 理解能力”。
