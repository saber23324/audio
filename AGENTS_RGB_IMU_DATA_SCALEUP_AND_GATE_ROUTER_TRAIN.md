# Codex 专项任务书：扩充真实同步 RGB+IMU 数据，并立即开展 Fusion Router → Gate 训练与验证

> 本文覆盖上一版 `CODEX_RGB_IMU_SYNC_DATA_EXPANSION.md` 中的 **DATA-ONLY / 禁止训练** 限制。新任务是**一个连续的完整研究工作流：更多真实同步数据 → 构建可监督 QA → 训练 IMU Fusion Router → 固定 Router 后训练 IMU Gate → 真实推理与效率/泛化验证**。不得只交付下载清单就结束。原仓库 `AGENTS.md` 的安全、审计、不可覆盖旧资产与因果原则仍有效。
>
> 重要：所有训练、评价必须**实际执行**，无法下载、没有授权、训练失败或独立确认不可做则精确报告，不得臆造步数、成绩、独立来源或 SOTA。不可因旧 IMU-only 零样本表现差、旧 32 QA 的 16 mask 答案不变，而停止这次有监督的双图 Router/Gate 训练。

## 0. 研究目标：用户真正要做的事

目前已验证的 v3 架构如下，**不改科研主线，不另起大型模态模型**：

```text
                       RGB_t + 原始同步 IMU_[t-2,t]
                                   |
                      cheap IMU stats + causal RGB history
                                   |
                           trainable Gate G_IMU
                        /                         \
                 OFF                               ON
                  |                                |
            frozen V1                       render IMU plot
                  |                                |
          RGB -> 64 tokens             RGB + IMU plot -> frozen SigLIP/projector
                                                   |
                                         trained R_IMU Fusion Router
                                                   |
                                              64 tokens
                        \                         /
                            append causal Z_t
                                   |
                    frozen visual-only LLaVA-OneVision LLM
```

**本轮真正要验证的是扩大数据后的训练是否改善：**

1. `R_IMU_REAL` 相比冻结 V1、同容量的 `R_IMU_DUP`/`R_IMU_ZERO`，是否学到可泛化的有效视觉+惯性信息；
2. 冻结已经训练完成的 `R_IMU_REAL` 后，`G_IMU` 是否能选择值得额外编码 IMU 图的时刻；
3. 同为 `4×64=256` LLM 视觉 tokens 的条件下，能否用更少的 IMU 图像编码保留或改善真实 QA；
4. 分别明确说明同任务新录制/新参与者（若 ID 可核验）与跨数据集迁移能力。不要用未经核实的成绩宣称 SOTA。

**以下约束必须保留：**

- 四个决策时刻优先固定 `t=[2,4,6,8]s`，每帧 IMU 只读 `[max(0,t-2),t]`；只根据已观察数据当场选择，不回看/修改过去已生成的视觉 tokens。
- Gate OFF：当前 RGB 走冻结 `V1`；Gate ON：仅此时生成/编码 IMU Plot，当前 RGB 与 Plot 走**新训练的** `R_IMU`；每时刻严格 64 visual tokens。**不删除其他 84%/任意比例的 RGB 帧，也不固定 16% 或 32% Gate ON。**
- **冻结** SigLIP、Projector、LLM、原始 Stage1/Stage3、当前 V1、Audio R_A、旧 Gate；只训练 IMU Fusion Router、后续与其收益匹配的新 IMU Gate。不得覆盖旧 checkpoint。
- 保留原 `GroundedFusionRouter` 结构与 696,320 参数量附近的实现（实际以加载审计为准），典型 `196 RGB + 196 IMU Plot → 64`，零残差身份初始化；保留廉价 IMU Gate 的 Accuracy-first / Cost-aware 两套方案。
- **训练次序：先 Router，冻结并选好 Router，再生成该 Router 的完整 Mask Utility，再训练 Gate。** 不以旧 Audio/F0/G_A Utility 作为新监督。
- Gate 不得读取问题、选项、Gold、未来 IMU、source ID、任务类别文本；离线训练损失可以依赖训练集真实答案，推理决策必须完全因果。
- 所有数值监督与缓存评估都复用已修复的 **online B=1 单样本/逐帧路径**；在有代表性的样本上做缓存与真实付费推理的 logits/生成 ID 对齐。旧 v1 窗口越界和 v2 batch 数值不一致禁止复发。

## 1. M0 — 首先锁定旧资产、运行环境和数据状态

实际定位并检查（不要照抄猜测路径）：

- 仓库根 `AGENTS.md`、`utility_gate/imu_visual/`、当前 v3 `M0_audio_best_config_and_checkpoints.md`、`locks/protocol.json`、`results/paired_summary.json`、`MODEL_SHA256.csv`、`REPRODUCTION_COMMANDS.md`；
- v1 边界失败记录、v2 数值路径审计、v3 Router (`R_REAL`, `R_DUP`, `R_ZERO`) 与 4 种 Gate 的真实 checkpoint、参数名、训练过程、sha256；
- 历史来源账本（用户已有检视的全部 source/session/clip）。旧 4 个有标注 HoloAssist session 的 2train/2val、32/32 QA **不能被重新命名为新的独立测试集**。

记录历史 v3：3 个 Router 各 96 步，4 个 Gate 各 120 步，训练只有 2 个 session/32QA，验证 2 个 session/32QA，验证 16 mask 没有改变任何题目的 greedy 答案；原生 RGB+IMU 17/32，但 V1/压缩 Router 都 15/32。**这些数字只作为历史研究动机，不作为新模型的预设成绩或停训条件。**

新目录建议：

```text
utility_gate/imu_visual/scaleup_train/
artifacts/utility_gate/imu_visual_scaleup_train_v1/
  M0_audit/
  M1_download_sync/
  M2_qa_manifest_splits/
  M3_router_train/
  M4_router_eval/
  M5_gate_train/
  M6_confirmation/
  M7_cost_final/
  checkpoints/ logs/ manifests/ figures/ audits/
```

所有历史文件与 checkpoint 只读；新 checkpoint、缓存和预测写入新目录，并保存运行命令、随机种子、包版本、显存与磁盘预算、断点恢复信息。先检查官方许可、下载权限、剩余 NVMe 空间、网络情况；不要未经授权直接启动 TB 级下载。

## 2. M1 — **实际扩大**真实成对 RGB + IMU 数据（必须与训练接续）

### 2.1 数据来源与实施优先级

**P0 HoloAssist（优先主域）**

- 官方入口：<https://holoassist.github.io/>；结构说明 <https://github.com/holoAssist/holoassist.github.io/blob/main/data_links/README.md>。
- 扩充尽可能多**新的完整录制 session**：RGB 原始/官方视频、acc、gyro、官方 `_synced` 与相机时序映射、动作/步骤标注必须成套。实际审查 `Accelerometer_sync`、`Gyroscope_sync`、`VideoMp4Timing` 等对应实际文件；不能只依据文件名假定原始时钟精度。
- 建议以**至少 50 个合格 session 为目标**，条件允许进一步扩大；这是目标不是承诺。优先更多真实动作、更多来源和不同操作者（身份只有官方元数据证实才能标记）。

**P1 WEAR（扩大参与者和运动域）**

- 官方入口：<https://mariusbock.github.io/wear/>；数据说明 <https://ubi29.informatik.uni-siegen.de/usi/data_wear.html>。
- 扩充同步头戴 RGB、穿戴式加速度计、动作区间标签，优先 raw 与真实对齐元数据；尽量覆盖 ≥15 个可核验 participant，依据真实可用性调整。
- **WEAR 不可凭空填陀螺仪**。字段采用 sensor availability mask；多传感器位置、轴与采样率、单位必须按实际记录。记录 processed 数据是否为重采样及其精度。WEAR 与 HoloAssist 标签空间不同，应分任务训练/评价并谨慎解释跨域。

**P2 Ego-Exo4D（有权限才扩）**

- 文档：<https://docs.ego-exo4d-data.org/overview/> 与 <https://docs.ego-exo4d-data.org/data/takes/>。
- 从受许可、含 Aria RGB 和 IMU 的 take 开始，核查 VRS、采集时间戳、`takes.json`、`timesync.csv` 和实际 RGB/IMU 同时可解码；分小批真实下载。没有权限就记录阻断，不绕过登录或许可。

**UTD-MHAD**：保留工程 smoke；因历史原始严格时间映射不足，除非拿到可认证时钟关系，不纳入“已证明严格同步”的主训练/独立确认。

### 2.2 下载/验收不可偷换

每一 recording 都需真实检查：视频可解码、IMU 样本存在且对应同源记录、时间戳/时基与采样率、sensor 单位/位置、单调性/间隙/NaN、可重采样区间、动作标签与片段覆盖。记录原始路径、官方 URL/版本、许可、文件大小及 SHA256。支持断点续传、幂等下载；禁止用生成的随机/零 IMU 代替下载失败。

按同步证据分级：

- **A**：原始或可追溯时间戳与跨设备 clock mapping 可核验；
- **B**：官方 RGB 同步/插值结果可核验，但缺原始设备精度证明；可用于标清限制的训练与开发；
- **C**：仅弱推测、时钟不可说明或缺关键信号；只用于诊断，不能当高可信度训练/holdout；
- **Rejected**：错误配对、严重损坏或数据缺失。

将历史 `all_seen_sources` 与新的 source identities 合并为只读追踪 registry（不覆盖历史）；按官方 `participant`（若真实存在）/原始录制/`session`/`take`/重编码 parent 分组。**先锁定 train/val/sealed_holdout，再运行任何新训练或开发 QA**。同一原始录制的各种窗口、图像、题目绝不能跨组；历史已经看过答案的来源不得充作 fresh sealed holdout。参与者无法证明时仅声称 session-disjoint。

### 2.3 不能只下载不训练

M1 数据达成可用状态后，**必须继续执行 M2–M7**；不要输出“数据准备完成，请下一轮再训练”的结果作为本任务结束。若无法达到建议数量，仍应在所有实际有效 A/B 数据上完成**明确标注为 pilot** 的训练与评估，报告独立统计不足；不可把不存在的来源硬凑到目标规模，也不可为追求训练步数使用 C/Rejected 数据作为严格同步主证据。

只有实际授权、无可下载训练样本、数据不可解码/同步等**真实阻断**才可停止对应数据集；其他已可用数据集继续训练。将阻断、预算和实际产出逐一记录。

**M1 实际交付**：`ACQUISITION_REPORT.md`、`RAW_PAIR_INVENTORY.jsonl`、`SYNC_AUDIT.jsonl`、`ACCEPTED_RECORDINGS.jsonl`、`SOURCE_LICENSES.md`、`FAILED_DOWNLOADS.jsonl`、`SPLIT_LOCK.json`。

## 3. M2 — 从扩大后的真实数据构建可监督 QA 和因果输入

### 3.1 QA 标注必须真实可复核

对所有可用 A/B 录制：

1. 从**官方动作/步骤/事件标注**提取答案，基于已观察的 `[0,8]s` 或明确标注的其他合法前缀构造有确定答案的任务。问题及选项固定且不泄露答案；程序化模板可使用官方标注，但需保留逐题 `gold_provenance`、原始 annotation ID、起止时间、定义和 QA 模板版本。
2. QA 仅询问到提问时刻 **已经可判定** 的运动/操作事件，如已观察动作类别、动作是否发生、相对运动、已完成步骤；严禁用未来才发生的动作/标签事实回答当前问题。不用未来视频、后续传感器或完整动作结束标签在前缀中泄漏结果。
3. 题目可依据官方 activity/step 标签离线生成，但标签不得进入 Gate 特征、IMU Plot、模型视觉输入、系统提示隐藏字段。相同任务下选项名称和顺序应固定种子、校验平衡；若多段标注模糊或时间覆盖不足应排除该 QA 而不是猜答案。
4. 额外生成可独立解释的 `IMU-relevant` 任务层（强运动、旋转/加速度突变、动作起止等），**任务语义事先定义**，不得以 V1 答错或者 Grounded 答对的结果反向筛选主测试集。
5. 不得把 WEAR 标签映射成 HoloAssist 同类标签而没有官方定义依据；按 dataset/task 输出指标，混训时使用显式 task identity（可以在 QA 文本中存在任务描述，但 Gate 不能读取 QA）。

如果录制超过8秒，可预先按真实标注/固定时间采样规则建立更多**非重叠或明示重叠**的8秒前缀，训练中可产生多个 QA，但统计有效独立样本数必须按 parent recording/participant 分组，不得把同录制多段当独立。不能在原始 clip 全局量程或未来片段上拟合绘图归一化。保留原始时钟与相机PTS到传感器的真实映射。

### 3.2 与 v3 输入完全兼容

```yaml
online_numerical_path: B1_stepwise_batch1
observation_horizon_s: 8
sample_t_s: [2, 4, 6, 8]
imu_window_s: [t - 2, t]
future_access: forbidden
rgb_frames_per_question: 4
gate_off_encoder_images: 1
gate_on_encoder_images: 2
llm_visual_tokens_per_frame: 64
llm_visual_tokens_per_question: 256
plot_axis_range: fitted_on_train_only
imu_missing_axes: explicit_mask_no_fabrication
```

保留与 v3 一致的 IMU P1/P2 画布尺寸、量程策略及合法单位转换；在新训练来源确定量程后**锁定**，不能用 val/holdout 未来全局最大值。Gate 只读取廉价数值摘要和已观察的视觉历史；Plot 只在 ON 时渲染。实际写出 `qa_manifest.jsonl`、`imu_windows.jsonl`、`frame_timestamps.jsonl`、`source_lineage.jsonl`、`train_val_holdout.json`。

### 3.3 检查任务是否为有效监督

生成 `LABEL_QC.md`：每任务 gold 来源与标签可判定时间、QA 选项均衡、训练/验证/holdout 各类分布、重复题/同源重复比例、传感器时钟等级、自动抽检例子。若受授权/标签缺失影响，明确区分监督 QA 和仅有未标注同步数据；不允许将没有真值的来源伪造 Gold。未标注数据可以保留，不计入有监督 QA 的训练样本。

## 4. M3 — **在新数据上实际训练 IMU Fusion Router**（本轮不能略过）

### 4.1 模型与初始化

复制 v3 经过回归测试的 `GroundedFusionRouter` 结构（参考音频 R_A，`RGB196 + IMU196 → 64`），冻结 V1 与 SigLIP/Projector/LLM。仅训练残差融合参数。**主线采用上轮 v3 `R_REAL` checkpoint warm-start，在扩大后的训练来源上继续训练；同时用与主线匹配的 `R_DUP`、`R_ZERO` 原有 checkpoint warm-start 继续训练，保持各自独立模型、同源数据、更新预算与评估口径。**

另记录一个重要混杂：由于 3 个旧模型使用不同输入训练过，不可把它们的未来差距直接全归因于此次增量数据；如资源允许增设**同一零残差新初始化**的 matched-from-scratch `R_REAL`/`R_DUP` 对照，使用同一数据/随机 seed/步数、相同学习率和训练制度。若 warm-start 方案优于 scratch，需区分增量微调收益与新数据训练收益。

**训练时 Route 强制 ON**，保证新融合器确实暴露于 RGB+IMU；这是训练条件而非最终在线 Gate 的选择策略。训练原始任务 QA CE/选项 CE（具体按 v3 代码实际实现），保存完整生成 QA 验证；不因为 32QA 旧基线不变就停止。以新验证来源上的既定主指标选 Epoch，不看封存确认集。

### 4.2 匹配训练预算和训练正确性

- 保留上一轮实际可复现的优化器、lr schedule、权重衰减、混合精度策略与零残差初始化；允许因扩大数据而增大 epoch/实际更新步数，**按样本量而非无意义固定96步**安排迭代。开始前明确记录 `epoch`, effective source count, QA per epoch, optimizer steps, warmup, seed, checkpoint selection`，避免因微小数据而假称“大规模训练”。
- `R_REAL` / `R_DUP` / `R_ZERO` 的更新**每次使用相同原始问题、次序、loss 口径、总样本暴露**；trainable tensor 名、Grad Norm、训练 NLL、验证 NLL、Accuracy 和逐题变化必须记录。
- 单独跑 `V1` 冻结基线，以及 `R_REAL always`, `R_DUP always`, `R_ZERO always`, `R_REAL` 相同权重换 `wrong source IMU Plot` 的推理反事实；必要时加入 native `[RGB,IMU Plot]`（原生高token）参考，但必须和 256-token 路径分表。
- 新 Router 的 `ON=0` 与 V1 P/Z/generated IDs 严格一致；在 B1 在线前向下核对所有 Router checkpoint 的 token 数量、时间边界、低精度回归、数值稳定性、loss 与生成一致性。
- **不可用 val/holdout 来更新 Router**；早停/最佳epoch 只在训练/验证集上选定。实测 R_REAL 是否具备 QA 互补性（Rescue/Harm vs V1）、R_REAL vs独立训练 DUP/ZERO，以及 16-mask 的 gold NLL/答案可辨性。

**M3交付**：`ROUTER_TRAINING.md`、`ROUTER_CHECKPOINTS.csv`、`ROUTER_GRADIENT_AUDIT.json`、`ROUTER_VALIDATION.md`、`ROUTER_COMPARISON_REAL_DUP_ZERO.csv`、训练曲线和 SHA。

## 5. M4 — 固定新 Router，重新构建真实 16-mask Utility（绝不复用旧标签）

对**新合法训练及验证的 QA**：

1. 冻结新 `R_REAL`；四个时刻每个 mask 为 `0000..1111`。实际运行完整在线 B=1 QA 生成与 Gold NLL，得到每题每 mask 的答对/错、NLL、路由图数 `k`、延迟及当前 source group；至少对预先确定且覆盖任务/参与者的训练子集完整枚举16 mask，若资源允许覆盖全体。实际执行数和遗漏原因必须透明记录。
2. 计算**完整策略收益** `U(S) = L(V1) - L(R_IMU(S))`，不把单窗 add-one 效用简单相加；可以辅助构造条件边际，但 Gate 的主目标是序列/策略结果。
3. 统计 `QA_with_any_16mask_answer_change`、16 mask 能救回的 QA 数量、既有 QA 中所有 mask 都正确/都错误、负效用比例与组合交互。如果验证集仍 16 mask 均不改答案，**继续训练/比较 NLL 和成本目标**，但必须如实标明不存在可证明的 Accuracy 选择空间，不以多训练几轮营造新提升。
4. Gate 特征可用原始 `acc/gyro` 的幅值、变化率、能量、缺失率与历史 RGB 的廉价摘要；**不能使用 ground truth 的窗口标签、以后时刻的全段 IMU 统计或 QA 答案推理时选 mask**。离线训练标签可包含训练集 Gold，只作为监督。

数据独立单位以 participant/recording 为基准，重复QA与相邻窗口只视作相关样本。实际 manifest 中标记已枚举 vs未枚举；不能将预测收益代替已执行的16mask效果。

**M4交付**：`MASK16_UTILITY_ACTUAL.jsonl`、`MASK_INTERACTION_ANALYSIS.md`、`UTILITY_DISTRIBUTION.json`、`CACHED_ONLINE_PARITY.md`。

## 6. M5 — **实际训练新 IMU Gate**（严格 Router 已冻结）

训练至少两套，尽量复用 v3 音频→IMU 已成功跑通的结构和训练逻辑，重新选择阈值但不反向使用 holdout：

| 模型 | 特征与目标 | 主要报告 |
|---|---|---|
| `G_IMU_ACCURACY` | 历史 RGB 摘要 + 因果 IMU 低成本统计，`ranking/classification` 或序列级 utility 评分 | QA Accuracy、Rescue/Harm、自然ON率、平均k |
| `G_IMU_COST` | 更廉价的 IMU 统计；训练 `Utility - λ·ExtraImageCost`、对应实际成本单位 | QA/NLL 与额外图像数、计算成本的 Pareto 曲线 |

复用已有工程实现与初始化，但**不复制旧音频的 r 头当作 IMU 真值**；也不可把 v3 旧 IMU Gate 的阈值、全开/全关策略视作本次最优。新 Gate 的 epoch、神经参数、温度、阈值和成本 λ 在合法训练/验证中选择。不预设 ON 配额。至少记录一个 cheap rule gate、旧 v3 gate 的冻结迁移和同实际预算 random ≥3 seeds、固定 `k` 可行的 causal baseline。

训练过程中严禁更新固定 R_REAL/V1/主干；明确记录 Gate 的前向输入时刻、反向梯度流与更新参数，任何新 Router 权重变化都触发验收失败。Train mask 表只作为训练监督，val mask 用于验证/策略锁定。验证选择目标事先给定且应能区分：

- Accuracy-first：优先实际完整 QA Accuracy，Gold NLL 仅作预定义 tie-breaker；
- Cost-aware：在预定义质量边界/损失 tradeoff 下最小化额外编码图数及实际 pipeline cost；不得把固定OFF直接记作学会选择。

**M5交付**：`GATE_TRAINING.md`、`GATE_VALIDATION.md`、`GATE_POLICY_SELECTION_LOCK.json`、`GATE_CHECKPOINTS.csv`、`BUDGET_FRONTIER_VAL.csv`。

## 7. M6 — 锁定后在新独立来源运行完整 QA 和同-mask 控制

**只有 M1/M2 时实际预先封存且从未用过模型输出选取的来源，才可称新独立确认**。如果可用录制不足，就执行尽可能完整的 train/val development 评估，标记 `holdout_not_available`，绝不把旧 v3 32 个验证 QA、同源新模板 QA 或已看过结果的片段当 fresh test。

在每个数据集/任务相同 QA 与相同四时刻上比较：

1. `V1` Always-off：4张原 RGB 经冻结 V1 = 256 tokens；
2. 新 `R_IMU_REAL` Always-on：4张额外真实 IMU Plot = 256 LLM tokens；
3. 新 `R_IMU_DUP` Always-on、新 `R_IMU_ZERO` Always-on：**同训练预算独立训练**；
4. `G_IMU_ACCURACY` + 固定 R_REAL，`G_IMU_COST` + 固定 R_REAL；
5. 同 Gate mask 下，真实/重复 RGB/零 IMU/错源 IMU 输入到同一个固定 R_REAL；
6. `Random` ≥3个预先固定seed（按 validation 预定的期望预算分布选取，不用测试分布反向匹配）、Fixed-k 0..4 因果策略、廉价规则基线；
7. 原生 RGB、原生 RGB+IMU Plot 作为高 token 参考，单独报告其实际 token 数。Plot-only 只做诊断，不得阻止训练。

指标：每任务准确率、Gold NLL（按官方任务评分协议，不合并不同 NLL 定义）、Rescue/Harm vs V1、Gate ON 率、每题额外编码 IMU 图像数 `k∈[0,4]`、来源聚类 bootstrap CI、按动作类别/传感器类型/时钟 A/B 分层、全 mask 互补性/Oracle（仅描述理论上限，不是部署策略）。报告原生 vs压缩差异，但不把 token budget 差当作音频/IMU 的独特贡献。

如果同-mask REAL 与 ZERO/WRONG/DUP 没有可靠差异，应坦诚说明没证明 IMU 信息增量，哪怕 Gate 节省了额外编码。Gate 在相同额外图像预算下未超过 Random 也不能称“学到选择”。若跨数据集存在不同标签本体，应使用对应任务的同指标跨域或预定义迁移协议，不硬合并准确率。

锁定后的 holdout 只运行一次主要政策（失败恢复允许做严格同一配置的确定性重放，不因分数修改阈值/绘图/采样/训练数据）。**不宣布 SOTA，除非有真正可比的官方任务设置与标准基线。**

**M6交付**：`HOLDOUT_QA_RESULTS.jsonl`、`PAIRED_SUMMARY.csv/json`、`PER_WINDOW_ACTIONS.jsonl`、`SPLIT_INDEPENDENCE_AUDIT.md`、`M6_EVAL_REPORT.md`。

## 8. M7 — 实际流水线成本、对照与归档

测量模型已加载状态下的真实在线 B=1 路径，**仅 Gate ON 才调用 IMU Plot 绘制、第二次 SigLIP、Projector、融合 R_IMU**。成本至少包括：合法原视频解码、IMU裁剪/必要重采样、廉价Gate统计及决策、Plot绘制、额外视觉编码、V1/R_IMU、LLM prefill/decode。视频捕获等待、权重首次加载、下载/网络和 Gold 打分额外前向应另表说明。要区分 

- 追加编码张数/视觉编码工作量（4+k张编码图，**LLM永远256视觉tokens**）；
- 原生 `RGB+IMU` 多图是不同数量的 LLM 视觉 tokens，不可放在同预算的单一排行榜；
- 平均/中位延迟、分部件耗时、峰值显存、若能真实计量才有 GPU 能耗；
- 训练/验证/测试不同输入源的成本不能混为独立部署统计。

新数据量变大后，务必分析更多训练数据是否让 `R_REAL` 的真实 IMU-specific 改善、Mask 选择空间、Gate 预算效率、session/participant 泛化有实质变化。若只得到 NLL 优化而生成未改变，应清楚区分概率校准与实际答题成功，不把固定OFF代价零认作学习能力。

最后提供：`FINAL_RESEARCH_JUDGMENT.md`、`TRAINING_SCALE_CURVE.md`（按 train recordings/QA 有效样本规模的合法子集训练，资源允许则至少小/中/全三个尺度，**不能为了曲线窃用 holdout**）、`PIPELINE_COST.jsonl`、`ENVIRONMENT_COMMANDS.md`、`FAILURES_AND_RECOVERIES.md`、`FINAL_ASSET_SHA_AUDIT.json`、`ALL_SEEN_SOURCES_UPDATE.jsonl`，附新增训练源码与所有 checkpoint 路径及 SHA。

## 9. 必需的结果矩阵（所有条目必须真实执行或明确 not_run 原因）

| 阶段 | 训练/实施 | 必须回答的问题 |
|---|---|---|
| M0 | 旧资产核对与来源排重 | 旧模型、v3数值路径是否被正确继承？ |
| M1 | 大规模增量获取真实 RGB/IMU | 实际增加了多少合格 session、参与者、小时？ |
| M2 | 严格同步、真实 QA 监督与分组锁 | 新增多少独立可训练 QA？有无未来泄漏？ |
| M3 | **实际训练 REAL / DUP / ZERO Router** | 训练是否让 IMU 分支比视觉/对照更有效？ |
| M4 | 固定 Router 的真实16 mask QA | 是否出现新的可选窗口/Rescue？ |
| M5 | **实际训练 Accuracy / Cost Gate** | 选择策略是否比端点与同预算随机更好？ |
| M6 | 新独立确认（可得时）+全部模型对照 | 提升可否跨来源复现？ |
| M7 | 真实在线Pipeline成本与规模归因 | 多编码多少张图，花多少成本，有无质量保持证据？ |

**不得仅因 M1 下载成功就结束；不得只跑 IMU-only 的零训练测试；不得用历史 32QA 的 16-mask 不变来阻止扩大数据后的训练。**

## 10. 推荐 Codex 执行入口指令

将本文件置于仓库根目录，执行：

> 阅读项目原有 `AGENTS.md` 和 `AGENTS_RGB_IMU_DATA_SCALEUP_AND_GATE_ROUTER_TRAIN.md`，执行 M0–M7：先实增真实同步 RGB+IMU，审核时钟和官方 QA 标签，完成按录制/参与者隔离的 train/val/封存holdout，然后在新增训练来源上**继续训练 R_IMU_REAL/DUP/ZERO**，冻结 R_IMU_REAL 并实际枚举 train/val 完整16-mask QA，再训练新 Accuracy-first/Cost-aware Gate，最后用新独立来源跑完整相同256-token QA 与真实成本/同mask反事实。所有大模型与 V1/旧 Audio checkpoint 冻结，在线 B=1、2秒因果、Gate ON 才绘图/第二视觉编码。即使未达到建议数据量也尽最大合法可用数据执行真实训练并标注pilot；只有真实授权或数据阻断才可跳过相应步骤。不得把旧结果改名为新独立验证，不得虚构达到SOTA。
