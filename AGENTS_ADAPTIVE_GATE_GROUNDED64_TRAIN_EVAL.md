# AGENTS — AudioRouter：自适应音频选择 Gate、Vision-only Router 训练与 Grounded-64 验证

> **适用场景**：直接放入现有 AudioRouter 仓库，供 Codex 执行。它是本轮专项研发任务书；仓库已有 `AGENTS.md`、所有历史结果和 checkpoint 应保留不变。
>
> **最新变更（必须遵守）**：**不预设音频启用比例**，不以历史观察到的任何启用率为目标或模型选择条件；**Vision-only Stage1 Router 也要实际训练**，训练过程优先严格复用仓库原先 Stage1 的实现、教师、数据处理、损失及优化配置。随后以**新训练并冻结的 Stage1** 为视觉基线，重新构造 Grounded-64 的 Utility 监督，训练 Gate，并在独立来源上真实评估。所有最终输入下游 VLM 的视觉 token 数仍为每采样帧 64。
>
> **任务不是**动态 token 预算、并非 native 双图 5,840-token 推理，也不是将 QA 答案或音频 token 加入 LLM。整个项目的主要研究问题是：**在冻结 Visual-only VLM 条件下，能否由 Gate 自动识别哪些时刻引入视觉化音频空间提示可提升视觉证据保留和 QA 表现？**

---

## 0. 非谈判约束与成功定义

1. **基础模型冻结**：SigLIP / Visual Encoder、Visual Projector、LLM、BEATs、已有 Stage3 AudioRouter、已训练的外部 AV Localizer / EZ-VSL 权重均冻结。允许训练：**一份新 Stage1 Vision-only Query Bottleneck**、**新的轻量 Utility Gate**；可选独立训练一个可归零的小型 Fuser，但必须作为独立消融，并与固定 Fuser 主实验分开。
2. **旧 Stage1 保留**：已有 Stage1 checkpoint 和历史生成文件永久只读。新 Stage1 写入新目录，不覆盖旧权重；旧 Stage1 仍是历史基线。
3. **严格因果**：任何时间 `t` 的输入只能包含已到达的图像、历史信息及原始音频 `a[max(0,t-2),t]`；音频先裁剪再解码/重采样/编码。不得基于完整视频未来帧、尚未出现的问题或正确答案决定早期选择；已保存的历史视觉 token 不得被未来重写。
4. **统一视觉 token 预算**：主要可比较策略每采样帧固定输出 `Z_t:[64,d]`。Grounded 图片只是为生成视觉证据所用，不能变成额外的 LLM 音频 token；原生双图 5,840 token 只作非等预算历史参考。
5. **Gate 自动选择、允许全关和全开**：按合法开发集上的任务目标和阈值选择选择策略，不指定启用率，不强制每段至少打开一次。**真实 ON 率是结果，不是训练标签或预设约束**。必须单独报告 ON 率以及错误启用/遗漏有益窗口。
6. **训练标签必须匹配被选分支**：新 Gate 的 Utility 来自“**新 Stage1 基础路径** + **新 Grounded-64 路径**”的同条件干预；不得把旧 Stage3 AudioRouter 的 Utility 标签改名复用。若 Stage1、Fuser 或 Grounder 发生变化，受影响的 Utility 标签必须重新生成。
7. **对照公平**：同时报告固定旧 Stage1、训练后 Stage1、新 Stage1 始终 Grounded、Gate 选择 Grounded、随机选择、**同 Gate ON mask 的重复 RGB 和错误热图**。只有比这些控制更好，才有理由把收益归于音频空间提示或选择策略。
8. **不能为结果改协议**：已检视 MUSIC 701 QA、Valor 256 QA、VGG-SS、LLP 等历史确认来源不能在新增训练/调参后继续宣称独立确认；新确认来源及所有模型选择应在测试前锁定。所有负结果及失败必须保存。

### 成功证据应分层报告

- **训练有效性**：新 Stage1 的视觉压缩训练真的执行，报告 train/val 目标、冻结模块、可训练参数、与旧 Stage1 的配对 QA 结果；不以单一 train loss 下降代替泛化。
- **Gate 有效性**：新 Gate 对 `add-one-grounded` 的 Utility 排序或有益/有害选择优于零值、中位数、恒关及随机控制，并在真实完整混合路由推理中有效。
- **音频提示贡献**：同一 Gate ON mask、相同 prompt 和相同 64-token 预算下，真实 predicted-grounded 至少能相对重复 RGB / Sham 提供有价值的差异；仅超过旧 Stage1 或原始 AudioRouter 不足以证明音频特异性。
- **真实下游结果**：在未见来源报告逐题 Accuracy/EM、Gold NLL、Rescue/Harm 和按原始来源聚类的置信区间，同时记录计算成本。GO/NO-GO/INCONCLUSIVE 必须依照预先冻结的判断规则，不追求必然正结果。

---

## 1. 当前项目基线：禁止混淆历史路径

以下数字为先前已检视结果，仅供工程对照，不是新确认结果：

| 历史策略 | MUSIC 701 QA | Valor 256 QA | 真实推理路径 |
|---|---:|---:|---|
| Vision-only 64（旧 Stage1） | 54.78% | 61.72% | 每采样帧 `196 -> 64`，旧 Stage1 冻结 |
| Full Vision | 55.63% | 63.67% | 每帧 196，旁路 Router |
| Original AudioRouter（Stage3 real） | 50.50% | 56.25% | 旧 Stage3，真实音频 |
| AudioRouter-null（Stage3） | 51.50% | 56.25% | **同一 Stage3 checkpoint**，全零波形经 BEATs |
| 原生重复 RGB 双图 | 60.06% | 64.06% | 原生多图路径，约 5,840 tokens |
| 原生 Grounded 双图 | 60.20% | 64.06% | 原生多图路径，没有 Gate |
| D11 等 64-token 融合 | 56.35% | 60.94% | 两视图经旧 Stage1，固定比例融合 |

历史比较的限制：

- 旧 Stage1 和 Stage3 不同训练历史；不能把相互差异解释成“唯一改变音频”。
- 原生双图远多于 64 tokens，且视觉前端使用不同多图/视频路径；不能作为等预算改进。
- 曾有约一成多窗口启用音频的结果，但不能因此认定存在通用最优启用比例；**本轮禁止据此固定 ON 比例**。
- 旧 Gate 的 Correspondence 头可识别 LLP 的共同事件代理（历史 AUROC 约 0.82），但旧 Utility 预测较弱，且是对**旧动作空间**学习的；旧 Gate 只能作为冻结零样本迁移基线。
- `extension.enabled=False` 通常意味着恢复旧 Stage3 AudioRouter，不等同于研究策略 `always_off`；所有路径应显式命名与记录。

潜在旧资产（必须通过实际仓库审计，不可推断文件必然存在）：

```text
utility_gate/runtime.py
utility_gate/grounded_dual_image/
artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/
artifacts/utility_gate/dual_head_proxy_20261028_v1/
artifacts/utility_gate/gate2_20261027/
/nvme_data/pkt/streaming/dataset/
```

本轮新工件根目录建议为：

```text
artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/
```

存在同名目录时必须使用不可覆盖的新版本，报告实际目录。

---

## 2. 数据、训练协议与来源隔离

### 2.1 预先审计数据角色

- 读取 MUSIC-AVQA、Valor、AVQA、VGG-SS、LLP 的现有完整 train/dev/test/seen manifests；按能核验的原始 YouTube ID、clip 时间、媒体 SHA、路径/别名和其他 source metadata 分组去重。
- MUSIC 中 `video_id` 并不一定等于原始 YouTube source ID；当无法严格确认时记录 `unknown_source_lineage`，保守披露，不能捏造完全独立。
- 已经检视的 MUSIC 701、Valor 256、VGG-SS 等结果只可作为历史回归或明确标记的开发分析；新训练后的正式确认必须使用此前未参与训练、选模、分析和人工挑选的来源。
- **新 Stage1、Fuser（若训练）、Gate、阈值** 都只能使用合法训练或验证角色；确认集 Gold、QA 以及标签文件不得被训练 loader 访问。
- 记录下载与解码失败、视频无音轨、PTS gap/overlap、源别名不明等情况。不得按照结果正确性挑选来源。

### 2.2 训练任务

- 优先使用具有官方 QA 标注的合法 MUSIC-AVQA **训练划分**；必要时以 Valor 合法未见训练来源补充。尽可能保留 `Visual`、`Audio`、`Audio-Visual` 任务类型供分层分析，但不能改写标签以求提高准确率。
- 对 Stage1 训练，**视觉内容就是唯一输入**；QA Prompt 与训练目标要和旧 Stage1 训练协议一致；允许问题用于**训练期**蒸馏/QA 监督，但推理 Gate 不得读取未来问题。
- 对 Gate 训练，允许合法训练 QA gold 仅用于**离线 Utility 监督生成**；部署阶段不使用问题、答案、source ID、教师输出或 QA 类型。
- 对空间定位评估，GT Bounding Box 或 Sound Mask 仅能用于**事后评价**；必须使用 Frozen Localizer 预测图作为正式输入，禁止 GT 充当可部署热图。

---

## 3. 优先恢复并实际训练 Vision-only Stage1（本轮新增核心任务）

### 3.1 必须先查明旧 Stage1 的训练过程

在仓库中寻找旧 Stage1 训练脚本、配置、日志、checkpoint metadata、数据构造和随机种子；记录而非猜测：

- 视觉编码器/Projector/LLM 的模型版本和冻结情况。
- 视觉 `196 -> 64` query bottleneck 结构、初始化、温度、相似度、`Z=P@V`、视觉值路径和位置处理。
- 训练目标（原有 Full-Vision teacher KD / QA 等）、损失定义、归一化方式、权重和 Prompt。
- 数据集及 split、采样时间、帧格式、优化器、学习率、epoch、batch/accumulation、精度、梯度裁剪、恢复状态等。
- 如果发现旧 Stage1 的代码/配置与论文或既有说明不一致，以**可复现的实际仓库训练实现**为优先，并在 M0 报告中逐项披露差异；不得默默套用 Stage3 的音频训练配置。

### 3.2 训练约束

- **必须真实训练一个新的 Stage1 Vision-only Router**。优先复用原有 Stage1 训练命令/数据管线/损失/优化策略，不引入新方法模块以免混淆。若旧训练脚本不可运行，先按可核验旧实现进行等价复原并运行测试，明确无法复原的部分；不能用加载旧 checkpoint 冒充新训练。
- 冻结 `SigLIP, Projector, LLM`，视觉 Only 查询模块是主训练对象；不得输入真实或 null audio；无任何多模态 loss。若旧 Stage1 的训练损失本身包含独立的视觉 KD、视觉 QA，按其实现保留。
- 当旧 Stage1 以 Full Vision 为 teacher 时，用相同问题/答案格式在训练期比较 196-token teacher 与 64-token student 的**原输出分布**；按旧代码定义使用 KD（例如 KL），避免自行添加新标签或 future-aware 推理输入。**不得未经核实直接设置新的损失权重/训练超参数。**
- 训练数据仅使用合法 train；验证仅用于 epoch/超参选择。Stage1 训练 checkpoint 保存为新名称，例如 `models/stage1_vision64_retrained/`；含完整 optimizer/config/data/SHA/seed。
- 如果选择从旧 Stage1 warm-start/继续训练，必须记录起始 checkpoint 与其旧训练语料角色及可能重叠。为区分“复现原训练”与“继续微调”，建议报告 `from_scratch` 与 `warm_start` 为不同实验配置；至少真实执行一条并明确命名，不得混用。
- 训练必须支持断点续跑，但不得覆盖已完成的历史 checkpoint 或旧输出。

### 3.3 Stage1 训练后验收

- 报告可训练参数数、训练梯度与冻结主干无梯度/权重 SHA 验收，train/val 学习曲线和选模依据。
- 同源配对评估 `legacy_stage1_rgb64`、`retrained_stage1_rgb64`、`full_vision196`，显示 QA Accuracy/EM、Gold NLL、视觉 token 数与训练成本；不要只报告训练 Loss。
- 检查 64 个 slot、`P`/`Z` 形状、因果输入不读取未来、错误/空输入、确定性重跑。
- **只有 Stage1 新 checkpoint 冻结并锁定之后，才能生成本轮 Grounded-64 Utility 标签和训练 Gate。** Stage1 训练后任何变更都将使已生成的效用标签失效。

---

## 4. Grounded Image 的 64-token 融合：使用新训练的 Stage1

### 4.1 基本约定

- 冻结视觉前端分别处理 `I_rgb` 与 `I_grounded_pred`，得到 `V_rgb:[196,d]` 与 `V_ground:[196,d]`。后者仅在 Gate ON 且音频有效时生成。
- Grounded Image 来自已冻结的 EZ-VSL / Localizer，使用当时可用的 RGB 与因果 `[t-2,t]` 音频；不能读取未来或 GT 框。
- 所有正式路径最终每帧仅向 VLM 输出 `Z:[64,d]`。
- 将**新训练的 Stage1**（下称 `Stage1*`）作为主要视觉基线，旧 Stage1 留作独立历史控制。

### F0 — 新 Stage1* 视觉路径（必须支持 identity）

```python
V_rgb = frozen_video_frame_encoder(rgb_frame)
Z_base = retrained_stage1(V_rgb)              # [64,d]
# Gate OFF -> Z_out = Z_base, no extra Localizer or second visual encoding
```

Gate OFF 必须与独立直接调用 `Stage1*` 的 `P/Z/logits/generated IDs` 在固定环境中逐值或规定数值容差一致；`extension.enabled=False` 的旧行为另做保护回归。

### F1 — 双图 Slot-level Resampling 后融合（第一版，固定轻量策略）

```python
Z_rgb = retrained_stage1(V_rgb)              # [64,d]
if gate_on:
    pred_map = frozen_localizer(rgb, causal_audio)
    grounded_rgb = compose(rgb, pred_map)
    V_ground = frozen_video_frame_encoder(grounded_rgb)
    Z_ground = retrained_stage1(V_ground)      # [64,d]
    Z_out = Z_rgb + beta * (Z_ground - Z_rgb)  # [64,d]
else:
    Z_out = Z_rgb
```

- 与历史 D11 的两路后融合相似，但本轮**使用新训练 Stage1***，并且 Gate 决定哪些窗口启用。
- 主实验先固定一个由**合法开发集**确定的 `beta`，所有训练 Gate 的标签和推理都保持该 `beta` 不变。不能在已见确认集按结果调 `beta`。
- `beta=0`、Gate OFF、audio unavailable 分别进行 identity 测试。`RGB+RGB`、Sham / 错位图在**完全相同 ON 窗口**对照。
- 两路 Stage1* 虽共享查询权重，slot 的具体语义对齐仍应诊断，不能认为加权平均天然有效。

### F2 — 可选 Joint 64-slot Resampling（单独消融）

如果 F1 的机制诊断明确存在 slot-level 混合失真，可对 RGB 和 Grounded 视觉值做 64-slot 联合匹配，而不是直接两路 `Z` 相加：

\[
R^I=\mathrm{match}(Q,V_{rgb}W),\quad R^G=\mathrm{match}(Q,V_{ground}W),
\]
\[
P=\mathrm{Softmax}_{392}([R^I+\log(1-\beta),R^G+\log\beta]/\tau),
\quad Z=P[V_{rgb};V_{ground}].
\]

`beta=0` 时走严格 Stage1* Identity Path，避免 `log(0)`；对完全重复的 RGB 两视图进行等价性测试。F2 不得在测试后回头替代预声明的 F1 主结果。若学习额外 Fuser 参数，训练/验证必须独立记录，F1 Utility 标签不得挪用。

---

## 5. Gate 训练：从原有的“固定启用比例”改成任务驱动的选择

### 5.1 Gate 输出和目标

- `r_t`：Correspondence Head 输出的跨模态共同事件/对应程度代理；不等同于有益概率。
- `u_t`：为**当前新 Stage1* + F1 Grounded** 动作学习的下游 Utility 预测值；不能沿用旧 Stage3 或旧 Stage1 路径的 Utility target。
- 最终 `g_t=0/1`：由合法开发集校准的策略在每个时刻独立作出决定，可以全关、全开或者部分启用。不要强制每个视频选固定比例，也不在推理时看完整未来视频再排序。

### 5.2 在合法训练集自动构造新 Utility 标签

令 `Z^0_t` 为**新 Stage1*** 的纯 RGB 64-token 表示，`Z^G_t` 为 F1 的 Grounded-64 表示。对于合法训练问题 `q`：

\[
L^0(q)=\ell(\{Z^0_1,\ldots,Z^0_T\},q,y_q),
\]
\[
L^{+t}(q)=\ell(\{Z^0_1,\ldots,Z^G_t,\ldots,Z^0_T\},q,y_q),
\]
\[
\boxed{U^{add}_t(q)=L^0(q)-L^{+t}(q)}.
\]

- 正值表示“其它窗口仍走新 Stage1* 时，在 t 引入 Grounded 能降低训练问题的 Gold NLL”；负值表示干预有害；接近零需保留不确定区间。记录正确性整数差、source ID 和所有实际前向配置。
- **不要把整视频 QA Utility 复制到每帧**；可在合法 train 上通过逐窗口 `add-one`、块级干预和多 QA 稳健聚合构造目标；多个窗口间相互作用另作诊断。
- 只允许合法训练 QA 标准答案参与标签计算；Gate 部署输入不能包含问题、答案、QA type、教师结果、任何未来信息。
- 固定 Stage1*、F1 `beta` 和预测图生成方法后再产生标签，若变更任一组件需重算受影响的 `U_add`。
- 当 Utility 标签极稀疏时可进行**训练期**分层采样及损失权重，但验证/确认必须使用自然分布；不能在标签生成上人为把错配音频当作有害真值。

### 5.3 Gate 训练方案（主干冻结）

**T0：旧 Gate 冻结迁移基线**
- 直接加载旧 `Dual-combined`，使用它原有的冻结 r/u 和阈值，只统计作为迁移策略的结果；旧 u 不代表新 F1 Utility。

**T1：新的 Utility-only Gate（必做）**
- 使用当前/过去 BEATs、当前/过去视觉特征及 quality / timing 特征，采用轻量 pooled+MLP（或仓库现有 Gate2 结构），原则上只优化小模型。
- 使用 Huber / 可选高置信 beneficial-harmful 分类或排序监督，以合法 train 构建的 `U_add` 训练，用合法 val 选择 checkpoint。
- 与常量零、训练集中位数、旧 Gate 冻结转移、Energy Gate 和 Random Gate 比较 Utility MAE、Spearman、分组收益、实际 QA。

**T2：Correspondence + 新 Utility（必做消融）**
- `r` 可来自合法训练的 LLP 对应 Head（冻结迁移），或由本轮合法训练来源单独更新；不可使用已检视 LLP test 调参。
- 分别测试 `r-only`、`u-only`、`r AND u`；如果使用多任务训练，允许两头在不同合法来源上使用 masked loss，不强制同一窗口具备双标签。
- 真实选模必须比较 `T1` 与 `T2` 的下游收益，不能仅凭 LLP AUROC 判定双头更优。

### 5.4 阈值与选择策略：不设置固定 ON 率

- 主要的 `u` 决策形式：`g_t=available_t AND (u_t > tau_u)`；双头则为 `available AND (r_t>tau_r) AND (u_t>tau_u)`；如果学习选择器，必须保存可复核的开发数据及 OOF 训练记录。
- `tau_u, tau_r` 由合法 val 的任务目标、配对 QA Accuracy / Gold NLL 和视觉保真约束选择，或按预注册成本敏感目标选择，**不得将启用率固定为某个历史百分比**。
- 候选阈值/目标、Tie-breaking、主要指标优先级必须在 confirmation 之前预注册；开发集可以出现“全部 OFF”，但应诚实报告：这种策略等于视觉回退，未证明主动使用音频。
- 不允许按测试来源后验调阈值，也不允许逐视频先看全部未来 Gate scores 再选择 Top-K 时刻。
- 开发集可以画 Gate 开启率—QA 性能曲线作为**诊断**；仅报告一条事前选定的确认策略。随机控制用**val 观测到的实际 Gate ON 率**提前固定 Bernoulli 参数，多 seed 运行并报告确认集实际 ON 率；不使用确认集启用率反向匹配。
- 如果 Gate 在验证中总关/总开，仍要和恒关、恒开、旧 Gate、随机与同 mask Grounded/Sham 做实际对照；不能通过强制规定启用比例隐瞒退化。

---

## 6. 实验矩阵：同时检验 Vision-only 重训与 Gate 的真实增益

| ID | 策略 | Gate ON 路径 | Gate OFF 路径 | 主要作用 |
|---|---|---|---|---|
| V0 | **旧 Stage1 RGB64** | — | Legacy Stage1 | 历史回归基线 |
| **V1** | **新训练 Stage1* RGB64** | — | Retrained Stage1* | **本轮主要视觉基线** |
| V2 | Full Vision 196 | — | Full 196 | 非等预算教师/参考 |
| A0 | 原 Stage3 real audio | 原 Stage3 | — | 旧音频路由参考；不同训练历史 |
| A1 | 同 Stage3 null audio | 原 Stage3 null | — | 同 checkpoint 音频干预 |
| F0 | Always Grounded | F1 Grounded64 + Stage1* | — | 新融合分支无选择的作用 |
| G0 | Old Gate frozen transfer | F1 Grounded64 + Stage1* | Stage1* RGB64 | 旧 Gate 零样本迁移 |
| G1 | New u-only Gate | F1 Grounded64 + Stage1* | Stage1* RGB64 | **主策略** |
| G2 | New r AND u Gate | F1 Grounded64 + Stage1* | Stage1* RGB64 | Correspondence 增量 |
| G3 | Random (dev-rate-matched) | F1 Grounded64 + Stage1* | Stage1* RGB64 | 同成本随机选择 |
| G4 | **与 G1 同一 ON mask** | RGB + RGB F1 → Stage1* | Stage1* RGB64 | 消除第二视图数量因素 |
| G5 | **与 G1 同一 ON mask** | Sham/Wrong-map F1 → Stage1* | Stage1* RGB64 | 检验真实声源空间信息 |
| G6 | **与 G1 同一 ON mask** | Zero-audio Map F1 → Stage1* | Stage1* RGB64 | 检验实际音频依赖 |
| G7 | 新 Stage1* 的既有固定 mix | F1 Grounded64 | — | 与历史 D11 等结构重新比较 |
| G8 | 可选 F2 Joint64 | F2 Grounded64 | Stage1* RGB64 | 扩展消融，另立标签/控制 |
| H0 | 旧 Native 5,840 双图 | Native multi-image | — | 非等预算历史参照，不作为主比较 |

**主要比较（预注册后使用）**：

1. `V1 - V0`：新 Stage1 训练本身是否改善基础视觉压缩？
2. `G1 - V1`：Gate + Grounded 是否带来超出**新训练视觉路由器**的任务收益？
3. `G1 - G3`：是否优于开发集锁定的随机启用？
4. `G1 - G4`、`G1 - G5`、`G1 - G6`：同 ON mask 下，真实声源提示是否提供额外价值？
5. `G2 - G1`：Correspondence Head 是否提高选择质量？
6. `G1 - F0`：Gate 是否能避免始终启用 Grounded 带来的损害？

上述比较统一下游 Prompt、答案评分、时间采样和最终 64-token 预算。对每个真实策略都报告 QA Accuracy/EM、Gold NLL、Rescue/Harm、前端与 LLM 成本、实际 Gate ON 率。Full 196 与 Native 双图仅作不等预算参考。

---

## 7. 真实执行阶段与阶段性验收

### M0 — 仓库、旧 Stage1 训练过程与数据角色审计

- 审计旧 Stage1/Stage3/BEATs/SigLIP/LLM/Localizer 的版本、冻结状态、输入前端、模型 SHA 和训练脚本。
- 核查 Stage1 原有训练数据、KD/QA 损失、超参数和训练命令，不从 Stage3 配置猜测 Stage1。
- 审计 train/val/seen/new_confirmation 来源；生成去重表和不能核验的来源列表；旧工件只读。
- 输出 `reports/M0_stage1_training_audit.md`、`manifests/source_roles.json`、`locks/old_assets.sha256`。

### M1 — **实际重新训练 Vision-only Stage1***（必做）

- 恢复之前 Stage1 的训练管线；只训练 Vision-only query bottleneck，Full Vision teacher 与冻结 VLM/视觉骨干按旧流程执行。
- 记录本轮新 checkpoint 的完整可复现信息和曲线；不得将旧 Stage1 的 load 当作新训练。
- 比较旧 Stage1、新 Stage1*、Full 196 在合法 train/val 的视觉 QA/Gold NLL；锁定 Stage1* 最终权重，之后不再更改。
- 输出 `models/stage1_vision64_retrained/`、`reports/M1_stage1_train.md`、`reports/M1_stage1_val.md`、`locks/stage1_retrained_lock.json`。

### M2 — Grounded-64 F0/F1 接口与原路由兼容

- 以 Stage1* 实现 F0 和 F1；Gate OFF 和 `beta=0` 逐值回归到 Stage1*。
- 所有输入合法情况下 `Z` 必须为 `[64,d]`。预测热图来自因果 Localizer，不使用 VGG-SS GT。
- 在合法 dev 上进行有限固定 Fuser 参数诊断并锁定；必要时用保留的旧 Stage1 做历史 D11 复现，但不能将两套结果当同 checkpoint。
- 输出 `reports/M2_fuser64.md`、`tests/test_stage1_retrained_fuser.py`、`trace/identity_results.json`。

### M3 — **重新计算相对 Stage1* 的 add-one-grounded Utility**

- 只使用合法 `gate_train`/`gate_val` 的现成答案和因果视频；利用冻结 Stage1* 与 F1 的真实 VLM 前向生成窗口 `U_add`，保留原问题及窗口来源以便审计。
- 绝不从旧 Gate2/Stage3/旧 D11 复用 Utility 标签；Stage1* 改动后旧标签视为过期。
- 输出 `labels/utility_add_one_{train,val}.jsonl`、`reports/M3_label_distribution.md`、`logs/label_commands.json`。

### M4 — **新 Gate 训练：Utility-only、Dual-combined 与冻结转移**

- 训练轻量 u 预测器；对照 zero/median/旧 gate，并测试 Corr-only、u-only、r AND u。
- 使用开发来源计算 MAE、Spearman、训练收益与有害接受率；真实 `val` QA 检查完整时间窗口的混合路由。
- 输出 `models/new_utility_gate/`、`reports/M4_gate_train.md`、`reports/M4_gate_val.md`。

### M5 — **任务驱动、无预设启用比例的策略选择与测试前锁定**

- 用合法 val 选择主 Gate 策略、阈值、固定 Fuser 参数与 QA prompt；可以优化 Accuracy、Gold NLL 的预声明排序或成本敏感目标，但**不把任何音频启用百分比作为约束或主目标**。
- 在 val 上报告真实 ON 率和是否发生模式塌缩；固定随机控制的 Bernoulli 概率来自 val 的 ON 率，而非历史观测比例。
- 在新 confirmation 推理前生成 `locks/pretest_lock.json`，包含所有数据来源、模型 SHA、参数与分数/损失定义。

### M6 — 新独立来源上进行真实 VLM QA

- 真实运行 V0/V1/相关 Gate 策略、Always Grounded、same-mask RGB-duplicate/Sham/zero-audio、随机与其他预声明控制。
- 每个时刻必须实时按因果状态决定是否启动 Localizer/第二次视觉编码；保留历史 token，不回头更改。
- 每题真实生成/评分、日志标注 source/video/clip/QA, prompt, `r/u`, ON/OFF, grounded map provenance, 64 token 核验及失败原因。
- 没有足够独立来源时声明 `confirmation_limited`，仍保留已完成的实际运行，不虚构准确率或置信区间。

### M7 — 统计、系统成本、研究结论

- 使用原始视频来源聚类 Bootstrap（建议至少 2,000 次），报告主配对 Accuracy 差、Gold NLL 差、Rescue/Harm；同一 source 多 QA 不得当独立样本。
- 分层报告 `Visual`、`Audio`、`Audio-Visual`、空间/时间任务；不以 Audio-only 当作视觉信息保留的充分证据。
- 分开报告 Stage1 新训练收益、Gate 选择收益、Grounded 图片空间信息收益，不把 `V1-V0` 误写成 Gate 提升。
- 测试真实 Encoder/BEATs/Gate/EZ-VSL/Projector/LLM 的前端与完整时延，拒绝音频时仍计已运行模态编码成本，缓存推理不宣称完整部署加速。
- GO 需预先写明“新来源上 Gate 对 Stage1* 的任务收益”和“same-mask Grounded 相对 duplicate/Sham 的音频特异性价值”等独立条件；不满足时报告部分支持或 INCONCLUSIVE。

---

## 8. 必须实现的测试与反例

1. `frozen_originals`: 原 Stage1/Stage3/BEATs/SigLIP/Projector/LLM/Localizer checkpoint 和保护源码 SHA 始末一致；仅新 Stage1 与新 Gate 的独立目录可写。
2. `stage1_trainable_only`: 新 Stage1 训练时只有允许的 Bottleneck 获得梯度及参数变化；teacher/full vision 永远冻结。
3. `stage1_new_identity`: Gate OFF 与新 Stage1* 直接路径 `P,Z,logits,generated_ids` 等价；旧 `extension.enabled=False` 恢复旧 Stage3 行为不受影响。
4. `no_future`: 时间 t 注入未来音视频，不能改变当前 Stage1* 特征、r/u/ON、预测图及历史 P/Z；验证前端先截音频再编码。
5. `fixed_visual_budget`: 主策略每个采样帧 64 visual tokens，拒绝 native 多图 5,840 作为等预算成绩。
6. `no_fixed_on_rate`: 不存在硬编码历史启用比例、按完整未来序列的全视频 TopK，也不因音频可用就强迫 ON；支持全关、全开与部分开启。
7. `role_isolation`: 训练 loader 无法读取 confirmation/test 答案；用于 M3 的 Gold 只能来自合法 train/val。
8. `same_mask_controls`: 重复 RGB、错误热图、零音频图使用与主 Gate 完全一致的开关时间戳及 Prompt；记录样本级生成差异。
9. `cache_invalidation`: 新 Stage1、Localizer、beta、Fuser 策略任何改变都会触发依赖的 Utility 标签/特征缓存失效或重新计算，不可沿用旧预测标签。
10. `negative_results`: 没有提升、Gate 全关、Stage1 不优于旧 Stage1 时也须完成可行的对照与报告，不允许改 QA 类别或确认集阈值谋求正结果。
11. `reproducibility`: 数据与 source SHA、seed、训练配置、优化器、日志、命令、模型版本、Prompt、计时环境、预测试冻结哈希全部可审计。

---

## 9. 建议目录和产物布局

```text
utility_gate/adaptive_grounded64/
    data.py                    # dataset sources, roles, exact time windows
    train_vision_router.py     # reproduce original stage1 train process
    teacher_kd.py              # reuse original full-vision teacher distillation
    fuser64.py                 # F0/F1/(optional F2) with retrained Stage1*
    generate_utility.py        # add-one-grounded labels relative to Stage1*
    gate_model.py              # small u head, optional frozen r head
    train_gate.py              # fit on train, checkpoint select on val
    online_policy.py           # causal threshold, NO fixed ON proportion
    eval_qa.py                 # real frozen VLM generation and gold NLL
    controls.py                # original/retrained stage1, same-mask controls
    report.py                  # paired QA, source bootstrap, real pipeline cost

tests/
    test_stage1_retraining.py
    test_adaptive_gate64.py
    test_adaptive_causality.py
    test_adaptive_source_roles.py
    test_adaptive_fuser_identity.py
    test_adaptive_budget_and_controls.py

artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/
    reports/M0_...M7_....md
    manifests/
    models/stage1_vision64_retrained/
    models/new_utility_gate/
    labels/
    features/
    qa_eval/per_source_results.jsonl
    qa_eval/per_window_actions.jsonl
    qa_eval/paired_summary.csv
    qa_eval/pipeline_cost_summary.json
    locks/old_assets.sha256
    locks/stage1_retrained_lock.json
    locks/pretest_lock.json
    logs/commands.json
    final_audit.json
    STATUS.json
```

### 逐窗口日志（结构示例，数值仅示意）

```json
{
  "source_id": "original_video_source",
  "timestamp_s": 4.0,
  "audio_window_s": [2.0, 4.0],
  "audio_available": true,
  "base_router_checkpoint": "stage1_vision64_retrained",
  "corr_score": 0.61,
  "utility_predicted_delta": 0.08,
  "gate_policy": "val_locked_utility_threshold",
  "gate_on": true,
  "fuser": "F1_fixed_slot_mix",
  "map_source": "EZVSL_PRED_CAUSAL",
  "llm_visual_tokens_this_frame": 64,
  "historical_tokens_unchanged": true
}
```

### 逐时刻执行草案（伪代码，须适配实际仓库接口）

```python
def run_one_step(rgb_frame, causal_audio, history, policy):
    V_rgb = frozen_video_frame_encoder(rgb_frame)
    Z_rgb = retrained_stage1_router(V_rgb)      # [64,d]
    a_features = frozen_beats(causal_audio)      # decode only observed audio
    corr, utility = policy.predict(V_rgb, a_features, history)
    on = policy.apply_locked_thresholds(corr, utility)
    if not on:
        return Z_rgb
    heatmap = frozen_localizer(rgb_frame, causal_audio)
    grounded = compose_grounded_image(rgb_frame, heatmap)
    V_ground = frozen_video_frame_encoder(grounded)
    Z_ground = retrained_stage1_router(V_ground)
    return Z_rgb + policy.locked_beta * (Z_ground - Z_rgb)
```

---

## 10. 可以直接发给 Codex 的执行指令

> 阅读仓库根目录原有 `AGENTS.md` 和本文件 `AGENTS_ADAPTIVE_GATE_GROUNDED64_TRAIN_EVAL.md`。**本次移除所有预设音频启用比例，不再将任何历史启用率当作训练目标或验证阈值约束。** 首先 M0 审计旧 Stage1 真实训练脚本与数据角色；在 M1 **严格按照原 Stage1 训练流程实际训练一份新的 Vision-only 64-token Router**，冻结 VLM、视觉 Encoder 和 Projector、保持旧权重及所有历史工件只读。锁定新 Stage1 后，完成 M2 的 64-token Grounded Fuser，M3 重新生成相对新 Stage1 的 add-one-grounded Utility 标签，M4 训练新的轻量 Gate，并在 M5 用合法验证集按 QA 收益/保真及预声明目标选择阈值，由 Gate 自主决定实际音频启用率。M6 在未见来源上真实运行 Vision-only 新旧 Router、Always-grounded、Gated-grounded、随机与同 mask 重复 RGB/Sham 控制，M7 报告 Accuracy、Gold NLL、Rescue/Harm、配对置信区间、实际 ON 率及完整 Pipeline Cost。主要比较 `Gate - retrained Vision-only`，次要比较 `retrained - legacy Vision-only`；不得将 Vision-only 重训增益算作 Gate 增益；禁止未来信息、测试答案和原生 5,840-token 双图混入 64-token 主实验。无提升也必须如实保存完整报告和旧工件 SHA。
