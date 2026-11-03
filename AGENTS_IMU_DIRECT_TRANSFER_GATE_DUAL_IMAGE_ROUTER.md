# AGENTS — IMU 直接迁移 Audio 阶段最佳架构：Gate + [RGB, IMU Plot] + 64-token Router

> **Codex 执行任务书（本轮要求实际训练和验证，不是只做零样本可行性）**。本文件是仓库原 `AGENTS.md` 的专项补充，不覆盖旧文件。必须阅读现有 `AGENTS.md`、当前 Audio 阶段最优配置及本轮 IMU 开发报告后执行。以实际 checkpoint、源码和日志为准，不凭历史文件名推断。不得声称已完成未执行的阶段。

## 0. 用户的明确目标与上轮偏差纠正

**本轮主路线只有一条：沿用已经跑通的 Audio 阶段系统，将辅助模态从音频生成的 Grounded 图替换成 IMU 生成的时间序列图。**

```text
时间 t：已观测的 RGB_t + IMU_[t-Δ,t]
             │
        轻量 Causal Gate（先决策；可用低成本 IMU 统计 + RGB/历史状态）
             │
       ┌─────┴─────┐
       │ OFF       │ ON
       ▼           ▼
 RGB→冻结 V1      RGB + IMU 曲线图（在 ON 后才生成第二图）
 →64 tokens        →冻结视觉骨干分别编码→训练好的 IMU Fusion Router→64 tokens
       └─────┬─────┘
             ▼
  Frozen Visual-only LLM，每时刻严格 64 tokens
```

- **不能再把 `IMU-only Plot → LLM` 当作主模型，不能以 IMU-only 零样本低分为由停止 M3/M4/M5。** 该结果只属于输入诊断。目标是 RGB 主证据、IMU 图辅助证据、Gate 控制额外编码、Router 压缩至固定预算。
- **不是**直接复用 AUDIO `R_A` 的权重就认为其学会 IMU；应迁移其**成功架构、接口、训练程序及必要时的初始化**，训练新的 `R_IMU`。同样 Gate 网络可迁移结构或初始化，但 Utility 必须由 `R_IMU` 的真实整段 QA 干预重新生成，不能沿用 Audio/F0 标签。
- 首先产出清晰的冻结端点和可训练模块参数表。旧模型全部冻结、只读、校验 SHA；新权重保存到独立 IMU 目录。大模型主干不训练。
- **必须真正执行 IMU Fusion Router 训练和 Gate 训练**，即使早期无音频/IMU特异性增益也要跑预先限定的最小训练与诊断；若数据物理缺失、授权或无法核验的同步时钟造成实际阻断，记录明确阻断，不得伪造训练。
- 研究目标为**有效地保留视觉 QA 表现并减少第二张图的编码计算**；真实 IMU 是否贡献额外可用信息由相同输入规模的对照判断，不强行预设一定有益或 SOTA。

## 1. 不变的 Audio 阶段实现参考（M0：实际审计）

Codex 首先定位 Audio 阶段可运行的 `V1`, `R_A`, `accuracy-first Gate`, `cost-aware Gate`，保存快照：路径、SHA、配置、参数数目和形状、optimizer、训练步数、输入预处理、固定帧时刻、完整测试结果及 GPU 实测回归。历史审计记录：SigLIP 原生 729 patch → projector → 196视觉token；V1 的 `P[64,196] @ V[196,3584]`；Audio `R_A` 是额外 696,320 参数、双图特征392输入/key128/rank32的残差融合器；两种历史 Gate 分别约576,161和478,529参数。**以上只是历史参考，必须以实际参数清单核实。**

已知 Audio 历史现象：R_A 在 MUSIC 域内强于 V1，Valor 迁移有限；Grounded/重复图/Sham的准确率常接近；Gate 减少额外编码数量，但 Valor 质量非劣未确认。这些不能当作 IMU 新结果。

**本阶段按以下优先级迁移**：
1. 固定 `V1` 64-token OFF 路径与模型主干。
2. 将成功的 `R_A` 残差双视图64-token结构复制为**新** `R_IMU`，从零残差/可审计初始化开始，仅训练新增融合模块；先跑必需 identity、形状、梯度检查。
3. 复制 Audio Gate 里的因果观测接口、历史聚合、候选策略训练及完整16-mask评测思想，换用**IMU可用的轻量特征**与**新R_IMU的效用监督**。绝不悄悄调用 BEATs/EZVSL。
4. 分阶段训练：**先训练 R_IMU（Always-ON、Gate不参与梯度），冻结它后再训练匹配的 G_IMU**。可补测旧 Gate 结构初始化对比，但不能同时更新新 Router 与新 Gate 导致归因不明。

## 2. IMU 图像与因果数据接口（M1）

参考 Google Research《Plots Unlock Time-Series Understanding in Multimodal Models》中的**多通道数值时间序列曲线 → 视觉模型**思路，而非声源定位热图。来源：<https://arxiv.org/abs/2410.02637>；官方解读：<https://research.google/blog/unlocking-the-power-of-time-series-data-with-multimodal-models/>。

- 输入 IMU：尽可能真实原始三轴加速度、三轴角速度（有则使用；例如 WEAR 仅加速度时不得凭空创造 gyro）；标清各轴单位、传感器安装部位、轴向定义、采样率和缺失情况。
- 固定规范画法 `IMU_PLOT_V1`：横轴相对时间（秒），每轴固定物理纵轴范围/分面（仅训练集拟合或设备文档确定）；颜色/轴标签固定；可加入加速度模长作为并排子图；图中不得写活动名、正确答案、来源 ID 或未来统计量；严禁每窗单独自适应 min-max 造成泄露/抹去幅度。
- `t=2,4,6,8 秒` 为首个可运行四时刻协议。`IMU_[max(0,t-2),t]` 只能取已到达的原始或已验证时间戳；对齐用真实可认证时钟，不能把重采样后时间索引假装原始同步时钟。图像只在 Gate ON 后生成并送视觉编码器。
- 若某库无法支持 **8s可证实的逐时刻同步**，可换为经审计的较短前缀和固定时刻，但必须全策略一致并将该库标为独立协议；不得与8s队列合并或伪称8s。
- 原生双图参考 `[RGB, IMU_PLOT]` 通过冻结 LLaVA-OneVision 的多图视觉路径运行；**这是验证双图输入有无潜力的主端点，不是 IMU-only**。它可使用更多LLM视觉token，必须和64-token压缩路径分列。
- IMU 图绘制开销必须测量；可实施同样像素、尺度、单位一致的快速栅格化版本并逐像素/语义回归；不能只报告 Token 节省忽略 CPU/GPU 绘图成本。

## 3. 数据集：先修复来源/时钟，再选合法QA（M1）

候选：**WEAR、HoloAssist、UTD-MHAD**，数据与已有 `utility_gate/imu_visual/` 资产尽量复用。优先 WEAR 的 RGB+穿戴加速度和真实时间线（若已提供可靠对应关系），HoloAssist 的相机+IMU+时间边界，UTD 作为开发smoke。数据集正式主页、使用条款与下载策略由 Codex 实际查证并写入 `DATA_PROVENANCE.md`。

**不要再次将 108 条 UTD 开发队列/64道 Holo 开发题伪装成独立测试。** 上轮已看结果；只用于技术回归或训练/开发，严格依照许可和身份划分。新训练、验证、确认按 participant→session→source/video 层级去重，至少报告参与者、session数、物理同步可证性和 QA 原生/派生标签的区别；不足以完成跨被试确认时如实标 INCONCLUSIVE，不伪造新的独立样本。

任务选择必须优先真实可验证的 RGB+IMU 相关任务：动作类别、起停、运动节律、身体/头部转动、手持装置运动、操作阶段；问题标注必须从合法已有标注确定，且不将答案或任务类别刻到IMU图。可添加视觉不足/运动歧义的预先定义子集，训练验证按源划分，不能按测试答错事后筛选。先固定统一4选项（ABCD）任务和生成/Gold NLL口径，避免与原生开放式EM混算。若标签仅能支持分类，诚实称为视频动作QA派生评测，不声称官方AVQA成绩。

## 4. M2 — 先实测双图端点，不受IMU-only阻断

同一 RGB、IMU、问答与帧时刻、Prompt、解码：

| ID | 路径 | 作用 |
|---|---|---|
| N0 | RGB native | 原生视觉基线 |
| N1 | **RGB + real IMU Plot native** | 双图可能的非压缩上限 |
| N2 | RGB + RGB duplicate native | 额外图数量/视觉路径对照 |
| N3 | RGB + zero-IMU Plot native | 图形与Prompt对照 |
| N4 | RGB + wrong-source IMU Plot native | 有效IMU信息的特异性 |
| N5 | V1(RGB) 64/frame | 严格相同LLM token预算的OFF基线 |
| N6 | IMU-only Plot | **仅诊断**，低分不停止Router训练 |

必须保留等形态、同图序、固定Prompt；训练数据里任何“真实IMU来自哪里”的字段不能进入模型。对 N1 vs N2/N3/N4做配对 Rescue/Harm、Gold NLL；可能得出无 IMU 特异性结论但照样推进 M3/M4。

## 5. M3 — 单独训练 IMU Fusion Router（必须实际执行）

固定 V1、视觉编码器/Projector、LLM。模仿 AUDIO `R_A` 的双视图残差融合结构，初始为恒等OFF：

\[
Z_t^{V}=R_{V1}(I_t)\in\mathbb R^{64\times d},\quad
Z_t^{G}=R_{V1}(\operatorname{Plot}(IMU_t)),\quad
Z_t^{R}=Z_t^V+F_{\phi}(Z_t^V,Z_t^G)\in\mathbb R^{64\times d}.
\]

若实际 `R_A` 需要 196×2 特征而不是两个先压缩的64 tokens，**严格复用实际审计的392输入结构**，不得为了符合上式而悄悄换模型。式子只表示输出语义。默认仅 `F_phi` 可训练，旧模型全冻结。训练时以 **always ON** 强制曝露真实IMU双图，Gate不参与、不更新。以已审计 Audio R_A 训练程序为首选（损失、优化器、训练步数、scheduler、早停和首答案 CE/Gold NLL须据源码核实），不得凭想象称 KD。训练 source 严格在训练集，不用确认真值选择checkpoint。

**最关键的归因消融必须同步独立训练**：相同模型、同训练源、同seed、同更新预算的 `R_REAL`, `R_DUP`, `R_ZERO`（资源允许再加 `R_WRONG`），而不是只在训练好的R_REAL上测试时换图。必须报告各自Always-on与V1、N1原生双图的同源配对对比；如果 R_REAL 不超过 R_DUP，也允许继续 Gate 实验，但不得声称 IMU 信息被学会。

训练产出：实际梯度/参数更新、checkpoint、优化器状态、epoch/step、train/val完整QA、失败样本、每帧64token/4帧256token合同、OFF identity、安全NaN/缺失规则。不能以“IMU-only≈chance”提前终止。

## 6. M4 — 冻结 R_IMU，再训练 Gate（必须实际执行）

Gate动作：`g_t=0`仅RGB→V1；`g_t=1`才构造IMU图、第二次SigLIP+Projector→冻结R_REAL，生成64 tokens。Gate在生成图之前必须依据已到达、**廉价** IMU统计+先前RGB/历史状态决策。可以比较 Audio 历史的 accuracy-first（带历史/条件表示）与cost-aware（廉价IMU）结构，但不允许把Audio模型输入BEATs特征硬套IMU。

固定 `R_REAL` 后，在**训练来源**上的每个合法QA真实执行全部16个因果ON/OFF mask（四时刻）并获得**完整QA** Accuracy/Gold NLL，构造条件边际/sequence-level preference；不可将单时刻 Utility 相加当完整QA真值。训练新的G_IMU：至少 `ranking`, `classification`, `cost-aware` 三类可行配置；每类在预定训练步数下真实更新，有参数差异记录。开发集挑accuracy-first和cost-aware策略；不强制16%、32%或其他ON率。若最优全关，仍执行训练与固定预算策略诊断，不假装有效选择。

**要求消除门控开销混淆**：低成本 Gate 在调用 IMU 图绘制和额外视觉Encoder 前必须完成决策；若 Gate 先用冻结 SigLIP 编码第二张图作为特征，不能宣称“省去第二图编码”。测量同一视频轨迹 k=0…4的额外图数量、真实总延迟与能耗/显存（有测量能力时）。

## 7. M5 — 主评测配置及必要反事实

| 策略 | Router | Gate | 第二视图 | 总LLM视觉token |
|---|---|---|---|---|
| T0 | V1 | Always-OFF | 无 | 256 |
| T1 | R_REAL | Always-ON | Real IMU Plot | 256 |
| T2 | R_REAL | 新Accuracy Gate | Real IMU Plot | 256 |
| T3 | R_REAL | 新Cost Gate | Real IMU Plot | 256 |
| T4 | R_REAL | Random Gate，匹配开发ON率，≥3 seeds | Real IMU Plot | 256 |
| T5 | R_REAL | 与T2/T3**完全相同Gate mask** | Duplicate RGB | 256 |
| T6 | R_REAL | 与T2/T3完全相同Gate mask | Zero IMU Plot | 256 |
| T7 | R_REAL | 与T2/T3完全相同Gate mask | Wrong-source IMU Plot | 256 |
| T8 | R_DUP | Always-ON | Duplicate RGB | 256 |
| N1 | 绕过压缩 | Always-ON | Native RGB + IMU Plot | 实测较大，单列 |
| N2 | 绕过压缩 | Always-ON | Native RGB + RGB | 实测较大，单列 |

正式确认前冻结配置、开发阈值、预算规则、source分组、打分方式、权重SHA并写 `pretest_lock.json`。所有确认来源不得出现在此前任何预调试/检视/训练/验证来源；若新可用来源不足，完成train/dev，独立确认诚实 not_run，不要拿已看过结果复验来当独立确认。

输出：完整准确率、NLL、source-cluster CI、逐题Rescue/Harm、按任务类别分层、Gate每帧ON分布、每条QA额外图数量、IMU曲线绘制耗时、第二视觉编码耗时、Router与Gate计算耗时、VLM Prefill/Decode与完整pipeline耗时。单图/原生双图/64-token比较要明确Token与路径差异，不能从跨队列总体数字断言提升。

## 8. M6 — 明确区分成功标准

1. **BRIDGE**：N1 vs N2/N3/N4，IMU图是否比重复/错误图提供可观察的额外QA价值；未成立不阻止有意义的路由/成本分析。
2. **ROUTER**：T1 vs T0及独立训练的T8，输出256token是否保留了有用视觉信息；注意R_REAL≈R_DUP只能说明一般双图适配，不能声称IMU信息收益。
3. **GATE-SELECT**：T2/T3 vs T0/T1、同预算Random，是否有可靠选择优势，不能只因为选择了非零ON称学会Utility。
4. **GATE-EFFICIENCY**：在预先声明的准确率非劣界（建议开发集固定1pp，若样本不足报告insufficient-power）下，是否减少额外编码与实际延迟；即使省时也不能无证据称quality-preserving。
5. **IMU-SPECIFIC**：T2/T3 vs T5/T6/T7，真实IMU在完全相同mask与token成本下是否有额外收益。
6. **GENERALIZATION**：至少跨参与者/录制/session来源；不同数据集的物理传感器位置和标签定义不能直接混合成同一指标。

判定分项 GO/NO-GO/INCONCLUSIVE，不能强迫所有维度一起通过；**必须实际交付训练工件，即使所有确认项未GO**。

## 9. Codex 交付物及工作纪律

在新目录（可根据仓库实际结构调整）创建：

```text
utility_gate/imu_gate_router_transfer/
artifacts/utility_gate/imu_gate_router_transfer_<date>/
  M0_audio_best_config_and_checkpoints.md
  M1_imu_sync_and_dataset_manifest.md
  M2_native_rgb_imu_qa.md
  M3_imu_router_training.md
  M4_imu_gate_training.md
  M5_frozen_independent_confirmation.md
  M6_cost_and_paired_analysis.md
  FINAL_JUDGMENT.md
  checkpoints/{R_REAL,R_DUP,R_ZERO,G_IMU_accuracy,G_IMU_cost}/
  results/{per_qa.jsonl,per_window_gate.jsonl,paired_summary.json,cost.jsonl}
  locks/{data_split.json,pretest_lock.json,all_seen_sources.jsonl}
  final_audit.json
```

不得覆盖旧`AGENTS.md`、旧实验源码、checkpoint或曾经的确认结果；运行前后SHA比较；尽量复用已有实现并写专项单测：因果IMU截窗、不同设备时钟对齐、缺失通道、图像真实不同、OFF identity、ON分支调用计数、梯度仅更新新模块、64-token输出、逐时刻不读取未来数据、所有16 masks完整QA真实性、测试来源隔离、Token与实测延迟。实际命令、seed、配置、GPU、版本、训练中断/恢复和失败原因都存档。

**硬执行顺序：M0/M1审计与可用数据 → M2原生RGB+IMU双图基线 → M3实际训练R_REAL和R_DUP等 → 冻结Router后M4实际训练Gate → M5测试 → M6成本与判断。不要再次因IMU-only Plot零样本弱而跳过Router/Gate。** 若客观数据来源、时间轴、标签缺失使训练无法合法开展，明确列出精确阻断和需要的数据，不虚构训练。
