# AGENTS — AudioRouter: 先训练 Gate、再训练 Grounded Fusion Router，并做无 Router 的原生双图 Gate 对照

> **使用方式**：本文件是现有 AudioRouter 仓库的**专项执行任务书**，不是重新搭建项目，也不要覆盖仓库原有 `AGENTS.md`。Codex 必须先阅读项目根目录已有规范、当前运行实现及历史报告，随后按本文件顺序实际开展实验。所有结果（包括负结果、未运行项目和失败记录）必须留存。
>
> **此次用户明确要求的顺序**：**(A) 固定现有 Router、重新训练 Gate → (B) 冻结 Gate、单独训练新的 Grounded Fusion Router → (C) 完全旁路 Router，只保留 Gate 控制原生双图输入，和 Always-grounded native 双图比较**。不得将 B 和 A 偷偷改成联合训练；可在完成主实验后将“针对新 Router 重新训练 Gate”作为单独附加消融。
>
> **目标**：区分失败原因究竟是“选择器没有学会何时使用音频”，还是“固定 Grounded-64 融合器没有能力利用音频空间提示”，抑或“原生双图本身更强只是因为视觉输入规模/接口不同”。不强制 16% 或其他 ON 比例。

---

## 0. 不可混淆的事实和硬约束

### 0.1 已知事实（历史已检视，不是本次新测试）

- 上轮真正**先训练了新 Stage1 Vision-only Router（V1），再训练 Utility Gate**，没有联合反传；**但没有专门训练一个 Grounded Fusion Router**。Fixed F0 以冻结 V1 和固定 `beta=0.25` 融合双视图，故问题可能在融合适配不足。
- 已有新的 MUSIC 队列 **439 QA**：旧 Stage1 V0 `282/439`，新 V1 `279/439`，旧 Gate G1（全关）`279/439`，Always-grounded F0 `275/439`。
- 已有新的 Valor 队列 **512 QA**：V0 `332/512`，V1 `333/512`，G1（全关）`333/512`，F0 `334/512`。
- 这两队列的 Gate 都由合法验证集选择为**0% ON**；Gate 与 V1 的生成及 NLL 相同。Utility 验证排序不可靠（历史 Spearman 接近 0）。
- 之前原生 Grounded 双图在**另一批** MUSIC 701 QA 中 `422/701=60.20%`，原生重复 RGB 双图 `421/701=60.06%`；仅 `+0.14pp`，不支持声源提示优于重复 RGB 的稳定结论。
- 原生双图在历史四时刻配置约 **5,840 个视觉 token**，64-token/frame 路径在四时刻为 **256 个视觉 token**。两轮数据、Prompt、视觉输入接口和 Token 数不能直接通过 `63% > 60%` 宣称进步。
- 原始 Stage1 与 Stage3 的训练历史不同；`extension.enabled=False` 在旧实现中恢复原 Stage3，不等同于研究策略 `always_off`（V1）。
- 旧窗口监督采用过 `add-one` 干预，并观察到部分问题存在显著**多窗口交互**：单窗口 Utility 相加无法准确表示同时启用多个窗口的收益。

**研究假设，不是既定结果**：现有 Grounded-64 路径可能因融合模块欠适配而没有发挥 Grounded Image 的价值。不得提前声称“Grounded 一定最好”或“Gate 理应启用音频”。

### 0.2 研究目标必须分开陈述

1. **Gate 是否学会选择**：在固定 F0 的条件下，所选窗口是否比恒关/随机/恒开带来更好的真实 QA 收益？
2. **Fusion Router 是否学会融合**：固定 Gate、单独训练新 Router 后，是否改善同一 Gate mask 下的 QA？在 Always-on 下是否也有所改进？
3. **没有 Router 时 Gate 是否有效**：在原生视觉双图接口，Gate 选择是否优于恒用 Grounded 双图、恒用重复 RGB 双图、同 Gate mask 的错误提示图？
4. **可部署音频证据的作用**：真实 EZ-VSL predicted map 是否优于错配图、零音频图和重复 RGB？不能把原生双图增益自动归因于音频。

### 0.3 非谈判约束

- 冻结 **SigLIP、视觉 Projector、LLM、BEATs、EZ-VSL**，冻结旧 Stage1/Stage3、当前 V1 和所有历史 Gate checkpoint；只在独立新目录训练**新的 Gate**和**新的融合 Router/Adapter**。
- 原样保留旧实验配置、模型、预测、结果与 SHA；任何已有 `artifacts/utility_gate/` 历史目录不得覆盖。
- 每个流式时刻 `t` 只允许已到达视觉及 `[max(0,t-2),t]` 原音频；原音频必须先裁剪再解码/重采样/编码。Gate 推理不能看问题、答案、Gold NLL、目标事件真值或未来观测。已生成的历史视觉 Token 不能回写。
- 所有 **64-token 路径每采样帧固定输出 `[64,d]`**；原生双图是另一种输入/Token 成本，单独成表。
- 本轮**不预设 16%**；Gate ON 比例必须成为训练/验证自然产生的结果。不得在测试/确认时强制某比例、按正确答案调整 mask 或替换最优 seed。
- 真实音频提示必须来自预测的 EZ-VSL Heatmap；**GT VGG-SS Box/Mask 仅事后评价或明确标为 Oracle 的开发诊断**，不得进入正式 QA。
- 必须做到：`gate off` 与对应视觉参考路径实际 `P/Z/logits/generated IDs` 等价；原生双图 Gate OFF 的重复 RGB 对照必须维持原生双图输入形态和相同 Prompt。
- 测试集的 Gold、样本来源、Question Type 不得进入推理 Gate；训练期可用合法 train QA 标注计算监督或 QA Loss。
- 新确认来源必须与全部已知训练、调参、历史确认和人工检视来源隔离。MUSIC 的原始 YouTube 血缘若无法核验，必须单独说明，不能称完全独立。

---

## 1. 输入资源与审计（M0）

Codex 首先**实际查找**以下资产；路径只是已知候选，不要假设均存在：

```text
utility_gate/runtime.py
utility_gate/adaptive_grounded64/
utility_gate/grounded_dual_image/
artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/
artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/
artifacts/utility_gate/dual_head_proxy_20261028_v1/
artifacts/utility_gate/gate2_20261027/
/nvme_data/pkt/streaming/dataset/
```

必须恢复并锁定：

- 当前 V1 Router checkpoint、F0 `beta=0.25` 及所有输入处理、Prompt、QA 评分与四时刻采样配置。
- 旧 `Dual-combined` 的 Correspondence `r` 头、Utility `u` 头、阈值/版本；明确以前 `dual_llp` 在某实验中 `r` 阈值为 `-inf`，实际只由 `u` 决策，不允许将其称为有效双头选择。
- EZ-VSL 的预测图生成、质量掩码、音频条件裁剪、RGB→196 tokens 的原生图/双图接口。
- 可训练参数边界、原 Stage1 的实际训练方式（已有审计发现使用首答案 token CE，**不要凭历史纸面计划冒充 KD**）、优化器和梯度路径。
- 已见来源清单 `all_seen_sources.jsonl`、各批次 `pretest_lock`、完整 train/val/test 角色；核对已检视 MUSIC 701/439、Valor 256/512、VGG-SS、LLP 等交叉重复。
- 先生成一张**端点模型逐题四象限**：在已允许用于开发分析的对应队列，计算 V1 与 F0 的 `both_correct / F0_only_correct / V1_only_correct / both_wrong`、Oracle 上限、Gold NLL；不要只比较汇总准确率。

**M0 验收**：提交 `M0_audit.md`、训练/测试划分清单、训练入口、保护 SHA、完整旧模型端点表；未完成审计不得训练。

建议本轮输出目录：

```text
artifacts/utility_gate/gate_first_router_second_native_ablation_v1/
```

如果存在，则创建新的不可覆盖版本。训练源、模型、结果、日志、图、锁、失败信息分别保存。

---

## 2. 第一阶段：先训练 Gate，固定已有 V1/F0（M1–M2）

### 2.1 严格定义 Gate 的作用

本阶段 **V1 和 F0 完全冻结**：

- `V1`: `RGB -> frozen Stage1* -> 64`，`always_off`。
- `F0`: `RGB + EZ-VSL Grounded RGB -> frozen Stage1* + fixed beta=0.25 -> 64`，`always_on`。
- `G_A`: `Gate(V_current,V_history,A_causal,availability) -> {ON,OFF}`，每个采样窗口只选择 V1 或 F0 输出；严禁把旧 Stage3 权重、不同训练历史的 Query/P 进行未经验证的插值。
- 可使用旧冻结 Correspondence `r` 作为额外特征/消融，但它预测 LLP 同类同秒事件代理，**不是**可见声源概率、也不是 F0 的 QA Utility。

**关键限制**：此处新 Gate 是为**固定 F0**训练的。其 Utility 标签不能未经重新评估直接描述后续重新训练的 Router。

### 2.2 不再只拟合单窗口 NLL

在**合法训练集**上构造训练期候选因果选择策略（只用于离线给标签，不在推理时使用未来信息）：

- `S_empty`: 全部窗口 V1；`S_full`: 全部窗口 F0。
- `S_one(t)`: 仅在 `t` 启用 F0。
- `S_remove(t)`: 除 `t` 外全部启用 F0。
- `S_mix`: 若窗口数较少（例如不超过 5）且计算允许，可枚举全部组合；否则由固定种子预生成数量受控的多窗口掩码（包含相邻、相隔与稀疏组合），并保存生成规则。
- 全部策略**实际执行完整混合前向/生成**，记录对同一真实 QA 的正确性、Gold NLL、输入 Token 数；**不能**用各窗口 NLL 差相加假装完整混合的 Utility。

定义训练期完整策略的经验增益：

\[
U(S) = \ell(\text{all-off}) - \ell(\text{mixed}(S))
\]

真实问答标签仅用于离线评分；同一视频/同一来源的多条 QA 需要分组管理、报告有效独立来源数。不存在充分的任务条件监督时，必须如实说明。

### 2.3 Gate 训练方案，低成本且可归因

优先复用当前 `r/u` 双头框架，**重新训练 Utility/决策部分**，而不是把旧 AudioRouter 的 Utility 标签直接使用：

- A1：`u-only`，当前/历史 SigLIP + BEATs 特征预测 F0 相对 V1 的效用。
- A2：`r (frozen LLP proxy) + new u`，双头独立判别；r 仅是辅助特征和可解释诊断。
- A3：常量、零值、冻结旧 Gate、视觉-only Gate（去掉音频输入）、Energy/Cosine 非学习控制。
- 至少比较 `masked Huber`、**成对策略排序损失**、有足够正负互补时的有益/有害分类；在标签接近零时显式标记 tie，避免用大量弱标签压垮信号。
- 可以通过平衡**训练样本的监督权重**和训练期候选策略覆盖，缓解类别不平衡；禁止因此强行要求推理 Gate ON 或拿测试集挑出容易有益的样本。
- Gate 不访问未来问题。对一个视频多问题的离线 Utility 可以在训练期聚合成任务分布上的目标，但不得把同一 QA 的正确性直接复制到全部帧。
- 验证选择依次关注：真实完整混合策略的 QA 准确率、Gold NLL、Rescue/Harm、相较随机/恒关的增量，并报告 Utility 排序；不要只凭最小 Huber 作为成功证据。
- 阈值、配置、seed 在**合法验证集**确定并锁定；允许验证集最优是 0% ON，必须如实记录，**不许为了出非零结果调测试阈值**。

要报告预测 `r/u`、ON/OFF 时间轨迹、不同问题类别的完整策略收益、常量/随机基线、训练/验证源计数及失效模式。

### 2.4 第一阶段必须真实运行的对照

| ID | 视觉分支 | Gate/Mask | LLM Tokens | 目的 |
|---|---|---|---|---|
| A0 | V1 | all-off | 64/frame | 新视觉基线 |
| A1 | F0 | all-on | 64/frame | 固定 Grounded 融合参考 |
| A2 | V1/F0 | 新训练 u-only | 64/frame | 预测效用 |
| A3 | V1/F0 | 新训练 Dual gate | 64/frame | 比较 r 的增量 |
| A4 | V1/F0 | random（开发集 ON 比例） | 64/frame | 防止“少开即好”误判 |
| A5 | V1/F0 | frozen old gate | 64/frame | 零样本迁移对照 |
| A6 | V1/F0 | Oracle full-policy（训练/开发诊断） | 64/frame | 仅估计可选择空间；不得部署 |

**M2 产出**：`gate_a_best.pt`、冻结策略与阈值、所有候选配置、paired QA、完整真实混合结果、`M2_gate_fixed_router.md`。Gate 的失败允许继续进入第二阶段，以定位 Fusion Router 的问题；不可以以“全关”就跳过 Router 训练。

---

## 3. 第二阶段：冻结 Gate，真正训练 Grounded Fusion Router（M3–M4）

### 3.1 与之前固定 beta 融合的本质差异

此前 F0 没有专门为 Grounded 图像训练；Stage1* 只在普通 RGB 上训练，固定 `beta=0.25` 不保证能理解新的视觉提示。

现在训练一个**新、独立的 64-slot Grounded Fusion Router** `R_F_phi`。**Gate 仍使用第一阶段锁定的 G_A，权重、阈值均不更新**。

### 3.2 推荐主实现：零初始化的 64-slot 融合 Router

- 输入：`V_rgb:[196,d]`、`V_ground:[196,d]`，来自冻结同一 Vision Encoder / Projector。
- `Z_base = frozen V1(V_rgb) -> [64,d]`；对于 Gate OFF，直接返回 `Z_base`，不得因新 Router 介入而改变。
- Gate ON 时：以 V1 查询或压缩槽为初始化，学习 Grounded 分支提供的路由残差或低秩特征增量；输入包含原 RGB 与 Grounded RGB，输出必须是 `[64,d]`。可采用下述结构之一：

**主路线 R-A（建议先实现）**：trainable 64-query cross-view resampling / residual fusion：

\[
Z_{on}=Z_{base} + F_{\phi}(V_{rgb},V_{ground},Z_{base})
\]

初始化令 `F_phi ≡ 0`，便于从 V1 Identity 开始；轻量 Query / Key 投影、跨视图 Cross-Attention、零初始化输出或低秩适配可训练。不要直接无约束地把 3,584 维 64×64 大 MLP 堆得比原主干还复杂。记录 trainable params 与成本。

**可选 R-B**：joint 64-slot routing over `[V_rgb; V_ground]`，学习在两个视图的视觉值之间分配 64 个槽。必须报告其输出值可以包含 Grounded Image 的视觉编码特征，不能声称只处于**原始 RGB** visual-value convex hull。

默认至少真实训练 R-A。若后续改成 R-B，应为独立消融，不替换主路线的预测试锁。

### 3.3 训练 Router 的监督和梯度路径

- 冻结 V1、Gate G_A、SigLIP、Projector、LLM、BEATs、EZ-VSL 和旧 Stage3；只训练新 `R_F_phi`（和其明示的轻量附属投影），在仓库核实训练参数及反向传播路径。
- 主损失沿用项目已审计的 Stage1 首答案 token CE / Gold NLL 约定；可将轻量保真或残差正则作为**独立消融**，不得擅自把原协议改名为 Full-Vision KD。
- 尽量使用合法训练集上生成的预测 Grounded Map，不能使用测试 GT 框/Mask；训练问答和来源严格隔离。
- **防止 Gate 全关导致 Router 无梯度**：训练时可对 train-only 样本增加明确标记的 `forced_on_exposure` / 训练期随机 ON 增广，使新 Router 真正见到 Grounded 信息。它只是训练分支覆盖策略，**不允许在验证/测试中强制 ON**，也不能把这些训练期掩码记录为 Gate 自己学到的选择。
- 同时必须保留一份纯粹 G_A 选择下的训练/验证轨迹，报告哪些训练梯度来自 `gate_selected_on`、哪些来自 `forced_on_exposure`。若 Gate 永远全关，训练一个 always-on Router 可以诊断融合潜力，但 `Gate+Router` 的部署主结果仍应显示全关。
- 训练/验证期必须**真实执行**混合视频序列推理，评估多窗口交互；不要只用单窗口 Utility 相加替代真实生成。
- 所有 Router 超参数、epoch、候选模型仅用训练/验证决定；测试前锁定权重、Prompt、阈值、beta、采样配置。

### 3.4 第二阶段必须运行的对照

| ID | Gate | Fusion 路径 | 预算 | 主要问题 |
|---|---|---|---|---|
| B0 | OFF | V1 | 64/frame | 视觉基线 |
| B1 | ON | 固定 F0 | 64/frame | 未训练的 Grounded-64 |
| B2 | ON | **训练后的 R_F** | 64/frame | 新 Router 是否有融合能力 |
| B3 | 冻结 G_A | **训练后的 R_F** | 64/frame | Gate + 新 Router 是否可改善最终 QA |
| B4 | 同一个 G_A mask | 固定 F0 | 64/frame | 在固定 Gate 下测试 R 的增量 |
| B5 | 同一个 G_A mask | `RGB+RGB` 经新 R_F | 64/frame | 新 Router 的收益是否仅因多一视图 |
| B6 | 同一个 G_A mask | `RGB+Sham` 经新 R_F | 64/frame | 排除任意提示造成的收益 |
| B7 | Random mask（开发集锁定概率） | 训练后 R_F | 64/frame | 排除仅凭 ON 率取得的收益 |

所有可行对照必须同 Prompt、同帧、同输出 64/frame、同 Grounder 质量规则、同 QA decode。

**B2 vs B1** 检验融合训练本身；**B3 vs B4** 检验同一 Gate 下新 Router 的增量；**B3 vs B0** 检验系统整体价值；**B3 vs B5/B6** 才检验音频空间提示独特贡献。

### 3.5 可选的第三步刷新 Gate（不属于主实验）

Router 训练后，第一阶段 Gate 对固定 F0 的 Utility 可能失效。允许在主结果保存并锁定之后，用**合法 train/val** 重新为 R_F 生成 Utility 标签，训练新 `G_B`，对比 `G_A+R_F` 与 `G_B+R_F`。必须把 `G_B` 明确命名为**重新适配 Gate**，不能说是 Gate-first 的自然结果；不能使用已经读过的确认集调参，也不能冒充端到端联合训练。

---

## 4. 第三阶段：完全没有 Router，只使用 Gate + 原生双图（M5）

此阶段**彻底旁路 V1、F0、R_F 和任何 64-token 压缩 Router**。Gate 仅判断当前时刻第二张图应该是什么，真正进入冻结 VLM 的是原生双图接口。

### 4.1 两种 Gate-only 形式必须分开报告

**C-MAIN — 固定双图形态（主要可比对照）**：

```python
if gate_a_on:
    second_image = predicted_grounded_rgb    # EZ-VSL; 只有实际ON时可运行
else:
    second_image = original_rgb              # 重复原图；不隐藏第二图
vlm_input = native_multimage(original_rgb, second_image, same_prompt)
```

这样所有样本和窗口保持相同图像数量/原生多图格式，不会因 Gate OFF 时少了一张图而混入明显的 Token 数量差异。参考历史四时刻原生双图为约 `5840` tokens，必须**实测**当前输入真实 token 数，不能硬编码。

**C-DEPLOY — 动态单图/双图（部署成本补充）**：Gate OFF 时仅原始 RGB；ON 时原始 RGB + Grounded RGB。此策略会改变视觉 Token 数、采样处理路径和耗时，只作部署代价结果，不能取代固定双图的音频贡献比较。

### 4.2 统一 Prompt，避免指令混杂

固定双图及各控制设置使用完全相同的 system prompt 和解码参数，建议：

> You will receive an original visual observation and a second visual view. The second view may be identical to the first or may contain a noisy visual cue derived from an auxiliary sensor. Treat the cue as uncertain. Preserve the original scene facts, use the extra view only when it provides clear supporting evidence, and answer the question using the visible observations. Do not assume a highlighted region is always correct.

若原实现以视频多时刻 image-pairs 展开，必须核实顺序和时间标签：`(t1_original,t1_second,t2_original,t2_second,...)`；另可选 `second_first` 顺序对照，但必须在开发集预先确定，不要在确认集选更好的顺序。不能将音频类别或 QA Gold 文字嵌入 Prompt。

### 4.3 必做 Gate-only 对照

| ID | Gate/Mask | 原生第二视图 | 是否 Router | 核心对比 |
|---|---|---|---|---|
| C0 | always-off | RGB 重复 | 否 | Native repeated RGB |
| C1 | always-on | Grounded RGB | 否 | Native Grounded 双图（原 D4） |
| C2 | **冻结 G_A** | ON=Grounded、OFF=RGB | 否 | **Gate-only 主结果** |
| C3 | C2 **相同 ON mask** | ON=Sham、OFF=RGB | 否 | 音频特异性 |
| C4 | C2 **相同 ON mask** | ON=RGB、OFF=RGB | 否 | 控制模态与输入数量 |
| C5 | random（验证 ON 率） | ON=Grounded、OFF=RGB | 否 | 是否真的学到了选择 |
| C6 | always-on | Sham/错位 Grounded | 否 | 无条件错误提示 |
| C7 | frozen old Gate2 / 双头 | ON=Grounded、OFF=RGB | 否 | 零样本旧 Gate 迁移 |

`C2 - C1` 测 Gate 相对原生 Always-Grounded；`C2 - C0` 测 Gate 在相同双图成本下的整体增量；`C2 - C3/C4` 测选中窗口的 Grounded 提示是否提供音频特异性。**不要只报 C2 高于单图 V1 就归因于 Gate**。

额外记录每个视频的 ON 时间、实际 Grounder 调用次数、原生视觉 token 数、真实完整流水线耗时。对 C2 与 C4，因同 mask 且 Grounded vs RGB，能尽量隔离声源图信息；若语义效果为零，不能仅凭更高成本或 Prompt 变化声称收益。

### 4.4 旧 Gate 不适配原生双图的风险

A 阶段 Gate 针对 F0 64-token Utility 训练，其输出**不保证**预测原生双图的 Utility。因此必须如实将 C2 标记为**Gate zero-shot transfer**。若 C2 没提升，仅能得出该 Gate 未成功迁移，不能直接宣称 native Gate-only 永远无效。

可选 C8：在合法训练来源上针对原生 `C0 ↔ C1` 重新生成任务效用标签，**另外训练 native-specific Gate**；保存新的模型、选择阈值和独立结果，不将其混入 C2 主实验。这样能进一步区分“Gate 不可迁移”与“原生双图提示确实缺少额外收益”。

---

## 5. 公平比较、样本角色和测试规范（M6）

### 5.1 应当先核对的三类预算

1. **64-token 等预算实验**：`A0–A6`、`B0–B7`，每采样帧 64 LLM 视觉 token。若两视图分别编码，完整编码成本要额外计入。
2. **原生双图等输入形态**：`C0–C8`，同类设置必须视觉输入数量、视频时刻、Prompt、顺序相同；比较 C2 与 C0/C1/C3 等。
3. **跨预算诊断**：`B3` vs `C2`、`V1` vs 原生双图，只能说明整体体系不同，不可直接归因为某一组件更有效。

### 5.2 数据源与禁止泄漏

- 优先使用官方 MUSIC-AVQA 合法 `train`/内部 `val` QA 训练新的 Gate/Router；必须核对现有训练 192/48 MUSIC 视频及历次来源，扩充时按可核验 video ID、YouTube ID（若有）、时间点、媒体 SHA 分组。
- 旧 MUSIC 701/439 与旧 Valor 256/512、历史 VGG-SS/LLP 和其他已检视来源可**明确作为历史回归或开发诊断**，不得在新一轮训练后再次作为“全新独立确认”。
- 新独立确认从可用的合法未见源中预先抽样并锁定；如只剩开发数据，仍应跑完整开发诊断并报告 `independent_confirmation=not_run/insufficient_new_sources`，不得虚构检验结果。
- MUSIC 源 YT 映射不全的限制必须陈述；Valor 的已知 YouTube ID 按源聚类，未知 ID 不要强行写成已核验。
- 标准答案可离线用于候选 Utility 或 Gate/Router 训练损失，但推理 Gate 不能使用未出现的问题、Gold、教师分数、source ID、未来图像/音频。
- 所有实验种子、参数、checkpoint、测试清单必须在新确认集预测前锁定。旧的负结果不得被擦除或用新的命名替代。

### 5.3 报告完整统计，不能只给 Accuracy

**三条主线逐一报告：**

- Gate-first：`A2/A3` vs `A0/A1/A4`，是否发生真正的 Rescue，是否只是全关？实际 Gate ON 率、按源视频 TPR/误启用、Utility Spearman/MAE、配对 Accuracy/Gold NLL。
- Router-second：`B2-B1`、`B3-B4`、`B3-B0`、`B3-B5/B6`；逐题 `both_correct/rescue/harm/both_wrong`；完整序列 QA 而非窗口效用求和。
- Gate-only-native：`C2-C0`、`C2-C1`、`C2-C3/C4/C5`，报告按视频来源聚类 bootstrap CI（建议 2000 次）、原生 Token 数与 Pipeline Cost。

按问题类别（Visual / Audio / Audio-Visual、空间 / 时间 / 普通）分层，但真实分母不足时要写 `insufficient_data`，不得宣称稳定收益。Gold NLL 需遵守各数据集官方答案格式，MUSIC 字符串 NLL 与 Valor 四选一 NLL 不可混为一个均值。

**必要基线**：V0、V1、原 Stage3 Real/Null、固定 F0、重复 RGB、Sham/错位图、Full Vision、Native D2/D4（历史/本次，分清同批数据）、固定 Gate-mask 的重复图。Null Audio 必须是零波形经过 BEATs，不是随意置零 embedding。

### 5.4 成本与技术测试

- 真实端到端统计解码→因果音频→BEATs→Gate→（条件 EZ-VSL）→SigLIP/Projector→Router（若有）→VLM Prefill/Decode；Gate 全关也应报告 BEATs 等预付成本。
- Gate OFF、`alpha/beta=0`、重复 RGB、无音频、异常/过期音频，必须做等价性/回退测试。
- 对无未来访问测试：向输入追加 `t+Δ` 的音视频，当前 `gate/r/u/P/Z` 和选择/生成不能改变。所有历史 Token 只追加不重写。
- 训练新 Router 的有效梯度、Gate 冻结哈希、VLM 冻结哈希；每阶段保证 checkpoint 分离与结果目录不可覆盖。
- 原生双图 D2/D4 与压缩 64 的总视觉 token 不一致，严禁用原生 D4 的准确率作为“仅加 Gate 或 Router”的同预算成绩。

---

## 6. 最小代码接口（按现有项目适配，不要求重写框架）

建议提供下述语义接口和测试，实际文件路径以审计后为准：

```python
class FrozenVision64:
    def forward(self, rgb, history):
        # return z64 [B,64,D], reference stage1 checkpoint
        ...

class GateA:
    def decide(self, current_visual, visual_history, causal_audio, availability):
        # return on_mask, r_score, u_score; no future/q/answers
        ...

class GroundedFusionRouter:
    def forward(self, visual_tokens_rgb, visual_tokens_grounded, base_z64):
        # learned, outputs z64 [B,64,D]; trainable parameters only
        ...

class NativeGateOnly:
    def build_inputs(self, original_rgb, predicted_grounded_rgb, on_mask):
        # ON: RGB+Grounded, OFF: RGB+RGB; no 64-token router
        ...
```

对应的配置文件须显式区分：

```yaml
experiment_name: gate_first_router_second_native_ablation
causal_audio_seconds: 2
visual_tokens_per_sampled_frame: 64
fixed_gate_on_rate: null  # 不强制比例
phase_order:
  - audit
  - train_gate_on_frozen_v1_f0
  - freeze_gate_train_grounded_fusion_router
  - eval_gate_only_native_dual_image
  - paired_analysis

branches:
  vision: retrained_stage1_v1_frozen
  grounded_fixed: frozen_f0_beta_0.25
  grounded_trainable: new_fusion_router
  native_gate_only: true

training:
  freeze_backbones: true
  use_gold_only_for_training_objective: true
  train_gate_and_router_jointly: false
  train_time_forced_on_exposure: true  # 仅为防止Router无梯度，须显式记录
  refit_gate_after_router: false       # 可选后续独立消融

controls:
  same_mask_duplicate_rgb: true
  same_mask_sham_grounded: true
  native_double_rgb: true
  native_always_grounded: true
  matched_random_gate: true
```

这些为**建议语义配置**，不是声称仓库已经支持此 YAML。Codex 应在检查原工程后映射成真实可运行的 CLI/配置，并将实际命令保存到 `commands.json`。不允许只创建空接口后声称完成训练。

---

## 7. 必须提交的工件与验收门槛（M7）

至少提供：

```text
artifacts/utility_gate/gate_first_router_second_native_ablation_v1/
  STATUS.json
  commands.json
  final_audit.json
  all_seen_sources.jsonl
  pretest_lock.json
  reports/
    M0_audit.md
    M1_utility_labels_and_interactions.md
    M2_gate_training_and_validation.md
    M3_router_training.md
    M4_router_qa_ablation.md
    M5_gate_only_native_qa.md
    M6_paired_analysis_and_cost.md
    M7_final_judgment.md
  models/
    gate_a/
    fusion_router/
    optional_gate_b_refit/
  results/
    per_qa.jsonl
    per_window_gate_decisions.jsonl
    per_policy_correctness.csv
    paired_comparisons.csv
    lineage_and_splits.jsonl
    failed_samples.jsonl
  logs/
  figures/
```

每份报告必须标记 `completed / not_run / insufficient_data / failed`，绝不把未运行实验写为 0 accuracy 或用历史成绩冒充当前预测。

### 7.1 成功/失败必须按阶段独立判断

- **Gate-first 成功**：固定 F0 下，Gate 真正产生有用的非平凡 ON 决策；优于恒关/恒开及对应随机对照，显示实际 Rescue 且 Harm 可控。若 Gate 全关，不能宣称学会利用音频。
- **Router-second 成功**：与同一 frozen Gate mask 的旧 F0 比，新训练 Router 提供更好真实 QA；Always-on 也应报告以区分“融合能力”和“Gate能力”。不能仅靠 train loss 降低做结论。
- **Gate-only-native 成功**：在同样 native 双图 token/Prompt/图序下，Gate 策略优于 Always-on Grounded 或重复 RGB/错配图；若仅超过单图，无法归因于 Gate 或声源位置。
- **总体科研 GO**：上述阶段至少有清楚、可归因的主动音频收益，并在未见独立来源上复现；不强制三阶段都达到正结果，需如实说明在哪一层有效。
- **INCONCLUSIVE**：训练来源有限、来源血缘未知、QA 子类别过少或区间过宽；不能将未验证判断为方法失败。
- **NO-GO（当前锁定方案）**：在足够有效的数据与预注册对照下，Router 与 Gate 均无可复核增量。此结论不得推广为所有视觉化传感路线不成立。

### 7.2 必要自动测试

- 原 checkpoint/源码与历史产物 SHA 不变。
- Trainable parameter 白名单：Gate 阶段只 Gate 有梯度；Router 阶段只新 Router 有梯度。
- Gate OFF + 新 Router 路径与直接 V1 的 `Z/输出` 完全一致（容差与确定性明确）。
- 原生 Gate-only 确认**没有调用** Stage1、Stage3、F0、R_F 或其他 64-token resampler。
- 同 ON mask 下 Grounded、重复 RGB、Sham 的原生图片数量、顺序、Prompt 相同。
- 同一帧 `t` 添加未来音视频不改变其 Gate 决策及其即时压缩表示。
- 所有确认集标签与模型训练/调阈值 loader 严格隔离；新样本来源锁定前不读取推理结果。
- 完整流水线真实运行，checkpoint load 后实际生成，保留逐题预测和执行失败日志。

---

## 8. Codex 的第一条执行指令

> 请阅读项目原有 `AGENTS.md` 与 `AGENTS_GATE_FIRST_ROUTER_SECOND_NATIVE_GATE_ABLATION.md`。不要覆盖旧模型、旧实验和旧结果。先完成 M0，核对 V1、F0、EZ-VSL、现有 Gate、各项训练/评估脚本及来源隔离；随后**先真实训练 Gate（固定 V1/F0），再冻结 Gate 真实训练一个 Grounded Fusion 64-slot Router**。即使 Gate 验证全关，也要在训练集用明确标记的强制 ON 暴露训练 Router，并把训练暴露与实际 Gate 选择区分。完成后进行**完全绕过 Router 的 Gate-only 原生双图实验**：OFF=RGB+RGB，ON=RGB+Predicted Grounded，和原生 Always-grounded、重复 RGB、Sham、同 mask 控制做真实 QA 配对比较。全程保持两秒因果音频、冻结主干、不设固定音频启用率、训练/确认来源隔离。输出各阶段真实运行记录、Gate ON 率、QA Accuracy/Gold NLL、Rescue/Harm、源聚类 CI、视觉 Token 开销与真实成本；不得以训练曲线或旧确认结果取代新的证据。最后分别判断 Gate、Fusion Router、Gate-only Native 的作用。
