# AGENTS — 固定已训练 Fusion Router，只训练 Gate；寻找最优的额外 Grounded Image 编码数量

> **Codex 专项执行任务书 / 接入现有 AudioRouter 仓库**  
> 不覆盖仓库原有 `AGENTS.md`；本文件仅定义新一轮 **Fixed-Router / Gate-only** 实验。先阅读原有 `AGENTS.md`、Gate-first→Router-second 最新 M0–M7 报告、实际代码和模型资产，发现与文档不一致时以经过审计的**实际运行实现**为准并留下差异报告。  
> **用户本轮唯一要求：固定当前已训练好、表现最好的 Grounded Fusion Router `R_A`，只研究 Gate 能否更好地选择在何时使用第二张 Grounded Image、需要额外编码多少张；不要再次训练 Router、SigLIP、BEATs、EZ-VSL、Projector 或 LLM。**  
> **不能预先固定 16%、32% 或任何音频启用率。** 最佳数量由合法开发集上的真实整段 QA 和额外编码代价决定。成功标准不仅是比原始 AudioRouter 好，也要与 `Vision-only V1`、`R_A always-on`、`同 Gate mask 的重复 RGB / Sham` 比较。

---

## 0. 研究假设、已有事实与结论边界

### 0.1 研究目标

现有 **V1（冻结的 Vision-only Stage1）** 和 **R_A（已完成训练的 696,320 参数残差 Grounded Fusion Router）** 构成两个确定的端点：

- `OFF`: 仅编码本时刻原始 RGB；由冻结 V1 产生 `64` 个视觉 token。
- `ON`: 在原始 RGB 之外，额外生成并编码一张 **EZ-VSL predicted Grounded RGB**；由冻结 R_A 融合双视图，仍产生 `64` 个视觉 token。
- **Gate 的唯一动作**：在每个已到达时刻决定 `OFF/ON`，以及通过决策的频度决定全段共增加多少张图像进入视觉编码器。它不能调 Router 权重，也不能把 GT 框、音频或频谱 token 接入 LLM。

当前四个采样时刻 `t=2,4,6,8s`，总有 `4` 张原始 RGB 需要编码；如果 Gate 启用 `k∈{0,1,2,3,4}` 个时刻，则该段视频**额外进入视觉编码器的 Grounded 图像为 k 张**，总编码图像为 `4+k` 张；最终 LLM 的视频视觉输入仍为 `4×64=256 tokens`。训练/测试必须验证具体模型实际行为，而不是硬编码计数。  
**核心科学问题**：在 R_A 已冻结的条件下，Gate 能否选择少数但确实有帮助的辅助图像，在保留（或改善）完整问答表现的同时，减少不必要的第二次视觉编码？

### 0.2 冻结的历史事实（不得冒充新实验结果）

最新一次 Gate-first→Router-second 确认数据：

| 策略 | MUSIC 288 QA | Valor 508 QA | 角色 |
|---|---:|---:|---|
| V1（OFF） | 173/288 = 60.07% | 326/508 = 64.17% | 纯视觉端点 |
| F0 固定 beta 融合 | 171/288 = 59.38% | 326/508 = 64.17% | 旧融合，**不是**这次 Gate 的目标 |
| R_A Always-on | **206/288 = 71.53%** | 321/508 = 63.19% | 这次固定的增强端点 |
| G_A + R_A | 186/288 = 64.58% | 323/508 = 63.58% | **旧 Gate 仅为迁移基线** |
| 同一 R_A + 重复 RGB | 206/288 | 322/508 | 控制一般视觉适配 |
| 同一 R_A + Sham / Zero | MUSIC 各 206/288 | 320、321/508 | 音频特异性尚无证据 |

- `G_A` 训练时的 Utility 对象是**旧 F0**，随后 R_A 才被训练，故旧 G_A 的标签与 R_A 不匹配；本轮不再沿用 F0 Utility 作为新 Gate 训练真值。
- 最新队列 G_A ON 率约 `32%`，但没有在确认集上带来优势；本次 **不固定 32%**。
- R_A 在 MUSIC 的提升可能来自一般 QA / 视觉融合适配，因为相同权重在 Grounded、重复 RGB、Sham、Zero 上的正确题数基本一致。**不能提前宣称 R_A 已经利用了音频。**
- MUSIC 与 Valor 使用不同答案格式和 NLL 口径，不合并两组 NLL，也不把不同确认来源的准确率直接横向比较。
- 既往检视/确认来源（最新登记至少 6,471 个身份，须由实际 `all_seen_sources.jsonl` 核对）已经被看过，不能继续作为“新的独立测试集”进行调参或模型选择。

### 0.3 硬约束

1. **只允许更新本轮新 Gate 的小型编码投影、门控头、可选轻量历史状态。** V1、R_A、Stage1/Stage3、EZ-VSL、SigLIP、Projector、BEATs、LLM 及旧 Gate 全冻结，权重/配置/旧目录不可覆盖。训练前后执行 SHA 与可训练参数核验。
2. 冻结的 `R_A` 必须加载本轮审计定位到的**已训练 Fusion Router**；不能错载旧 F0、旧 Stage3、Native 5,840-token 双图或未训练适配器。
3. 视频流严格因果：在决策时刻 `t`，可看当前/历史 RGB 与 `[max(0,t−2),t]` 的真实原音频；先按时刻裁剪原始 PCM，再重采样与编码；**不可访问未来帧、未来音频、未知未来用户问题或答案**。
4. Gate 必须在 `EZ-VSL → 第二张 Grounded Image → 第二次 SigLIP` **之前**作出 ON/OFF 决定，不能先算第二图特征再决定是否调用它。若需要 BEATs 特征用于 Gate，预付 BEATs 成本如实计入。
5. Gate OFF 的实际模型输出（`P/Z/logits/generated IDs`，按所比较的 V1 接口）必须与冻结 V1 全关路径等价；Gate ON 需与冻结 R_A 的单窗口 ON 路径等价。禁止重算和修改历史视觉 token。
6. 所有策略使用相同帧时刻、QA Prompt、解码参数、输入图像预处理与 `64/frame` 的 LLM token 预算；**只允许改变第二图是否额外经过编码，以及 Gate 本身**。
7. 不用 VGG-SS GT 热图/框训练正式可部署 Gate；正式第二图只能来自 EZ-VSL 预测。旧 LLP `r` 分数可以作为冻结的辅助特征/对照，但其含义只是共同事件代理，不是 QA Utility 真值。
8. 全部自动生成的标注必须记载 QA / source / 选择 mask / 打分协议；按原视频来源划分训练验证测试，训练过程中不得用开发集以外的 Gold 对 Gate 产生梯度/阈值选择。
9. 不允许因为希望得到非零 ON 率而强行调测试阈值。若最佳政策全关/全开，如实报告；同时进行**预先定义的不同成本点诊断**，用来画边界，而不是把某个测试点替换成主结果。
10. 不得在结论中预写 “SOTA”；任何 SOTA 主张必须先核对官方同任务、同输入约束、同数据划分/采样的公开方法，且有独立确认和显著增益。目前是研究原型，目标为可复现的策略改进。

---

## 1. M0 — 只读审计、端点与训练资源恢复

**先审计仓库，不能凭文件名猜测运行版本。** 尝试定位（路径为历史候选，不保证完全一致）：

```text
utility_gate/gate_first_native/
utility_gate/adaptive_grounded64/
utility_gate/grounded_dual_image/
utility_gate/runtime.py
artifacts/utility_gate/gate_first_router_second_native_ablation_v1/
artifacts/utility_gate/dual_head_proxy_20261028_v1/
artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/
/nvme_data/pkt/streaming/dataset/
```

检查、记录：

- 真实 R_A checkpoint 的路径、SHA、训练源码 snapshot、输入 `[V_rgb,V_ground]` 维度、残差输出、ON 时精确函数调用。
- 真实 V1 checkpoint 与独立 Stage1 历史；`enabled=False` 曾表示关闭 Extension 恢复旧 Stage3，**不等于**研究中 `Gate OFF=V1`，必须显式使用策略名而非模糊布尔量。
- 冻结音频特征、视觉特征获取方式；每个时刻 ON 需要新增哪些计算；`64/帧` 对齐检查。
- 训练/验证合法来源与模型/评估确认集，核对所有已检视 ID（数字 MUSIC ID、可得的原 YouTube ID、片段起点、媒体 SHA、路径别名）。MUSIC 原 YT 血缘未知必须如实说明；不许假定“数字视频 ID 不重叠”就意味着互联网来源完全独立。
- `per_qa.jsonl`、`per_window_gate_decisions.jsonl` 与端点生成的 QA 配对四象限；重建 `V1_only / R_A_only / BothCorrect / BothWrong` 和整段 **Oracle** 上限。Oracle 仅诊断，不能用于推理或正式方法分数。
- R_A 在 Grounded、重复 RGB、Sham、Zero 的特征差异与原始 QA 结果；需要明确旧确认样本只能做**只读诊断**。

**交付** `reports/M0_audit.md`、`data/role_manifest.json`、`locks/model_sha_before.json`、`reports/oracle_ceiling_existing.md`。审计不通过不得拟合。

---

## 2. M1 — 用固定 R_A 自动构造 Gate 监督（整个 Mask 的真实收益）

### 2.1 为什么不能继续用旧的单窗口 Utility

此前已经观察到：某些 QA 上，多个单窗口预估有益，但同时启用后完整推理变差。所以上一轮 `G_A` 不能直接迁移，本轮必须为**固定 R_A**重建实测收益。

对每条合法训练/验证视频有 `T=4` 个因果时刻，令 `m=(m_1,...,m_4)∈{0,1}^4`，`m_t=1` 表示在 `t` 额外生成/编码一张 Grounded Image 并使用 R_A；`m_t=0` 表示只用 V1。共有 `2^4=16` 种完整策略。

**训练标签阶段**允许将合法训练视频上全部 16 个 Mask 真正执行完整冻结 VLM 前向/生成，评价 Gold NLL 与正确性；**推理阶段 Gate 不能使用这个 Mask 查表或未来信息**。可缓存合法 train/val 的已计算前端特征以降低枚举成本，但必须另测真实条件调用的成本。

对 QA `q` 和策略 `m` 记录：

```text
source_id, qa_id, question_type, dataset_role, timestamps, model_hash,
mask_bits, k_extra_images=sum(mask_bits), source_quality_mask,
real_generated_answer, exact_match_or_abcd, gold_nll,
encoded_image_count, vlm_visual_tokens, prompt_hash, valid_flag
```

- `k_extra_images = sum_t m_t`，范围 0…4；`OFF` 不编码第二图。
- `q` 的正确答案只允许参与**训练/验证的离线损失计算**，不传入 Gate 的 input tensors。
- 必须记录不同策略**真实完整视频 QA**，不能用 `sum(single_window_utility)` 伪造结果。
- 同一原视频的多条 QA 共用 video split，Gate 不依赖未来问题时，应在训练期按该视频的训练问题分布聚合标签/收益，防止模型学到某道题的偏置。
- 对合法训练数据报告：真实最优 `k` 分布、R_A 的 Rescue/Harm、`k=0…4` 最优 Mask 上限、简单随机 Mask 结果。**最优 k 是训练期答案辅助 Oracle 诊断，不是可部署策略。**

### 2.2 每张额外图片的代价

对每条视频定义 `C(m)`，优先由开发来源实测：

```text
C = cost_gate + cost_prepaid_audio_encoder
    + sum_t m_t * (
        cost_EZVSL_t + cost_second_image_transform_t
        + cost_second_SigLIP_projector_t + cost_R_A_increment_t
      )
    + cost_primary_rgb_and_LLM
```

主要以**增量前端计算时间**、实际第二图编码次数和**完整 Pipeline 时间**记录；缓存特征训练时测得的运行速度不能当作真实部署速度。预训练 EZ-VSL 与 SigLIP 如有缓存/复用应分别说明。

**注意**：R_A 与 V1 最终都是 `64` visual tokens/frame，改变 ON 率通常并不改变 LLM token 数；这里优化的是 **第二图视觉编码开销**，而不是重新研究 LLM 视觉 Token 压缩率。

---

## 3. M2 — 只训练 Gate：让它预测何时多编码一张 Grounded Image

### 3.1 模型接口（保持极轻量）

```python
class FixedRouterGate(nn.Module):
    def forward(
        self,
        current_rgb_features,       # 已存在的第一视图因果特征
        past_rgb_summary,           # 仅到 t 的可用历史
        causal_audio_summary,       # BEATs 或便宜的声学统计，严格[t-2,t]
        history_audio_summary,      # 严格历史
        availability_mask,
        previous_actions,           # 过去的ON/OFF决策，可选
        encoded_extra_count,        # 过去已多编码多少张，可选
    ) -> dict:
        # 不允许读取第二张Grounded图特征、未来画面、问题/答案或loss
        return {"logit_on": ..., "utility_score": ..., "prob_on": ...}
```

1. **G0（必要基线）**：旧冻结 `G_A`；用于比较旧 F0 Utility 门控迁移失败的影响。
2. **G1（主模型）**：以旧 r/BEATs/视觉因果特征作为输入，训练小型 `u + decision` 网络；`r` 可冻结提供辅助，不能把它解释为可见声源概率。保留不含 r 的版本作为消融。
3. **G2（必要对照）**：仅用第一视觉特征/时间上下文的 Visual-only Gate，控制同一个 R_A；检验音频是否真的用于选择。
4. **G3（低成本对照）**：声学能量/突发变化等可算统计 + 第一视觉特征小 MLP；给 BEATs Gate 一个计算成本更低的参考。
5. **G4（实验性）**：带少量历史状态（例如最近 1–2 次 ON/OFF、已使用的额外图像次数）的因果 GRU/MLP，处理多窗口交互。不得引入完整视频编码器或偷偷看未来。

只训练 Gate 新建目录中的参数；使用冻结 R_A、V1 和音频视觉 Encoder。所有候选参数量、训练轮次、GPU 开销、数据来源逐项记录。必须包含 all-off、all-on、constant / random 基线。

### 3.2 强烈推荐：16 Mask 的**序列级**策略蒸馏

因为 `T=4`，可以直接利用 M1 对**全部 16 种完整 Mask**的真实反馈训练，而不再用单窗口 NLL 做独立阈值回归。

对一段合法训练视频 `i` 和可部署 Mask `m`：

\[
L_i(m)=\operatorname{GoldNLL}_i(m)+\lambda C_i(m)
\]

形成离线训练目标分布：

\[
q_{i,\lambda}(m)=\operatorname{softmax}_{m}\big(-L_i(m)/T_{\mathrm{distill}}\big)
\]

其中 `C_i(m)` 可以先用 `k_extra_images` 作为代价代理，再用真正的增量耗时校验；`λ`、蒸馏温度等**只可在训练/验证集**上预先指定网格。

让 Gate 对当前时刻产生因果决策概率，构成完整 Mask 的预测概率：

\[
\pi_\theta(m\mid x_i)=\prod_t \pi_\theta(m_t\mid x_{i,\le t},m_{<t})
\]

只使用 `t` 及其以前的特征与历史动作，优化：

\[
L_{\mathrm{gate}}=-\sum_{m\in\{0,1\}^4}q_{i,\lambda}(m)\log\pi_\theta(m\mid x_i)
\]

这是**建议的新 Gate 训练设计**，不是既有实验已验证的算法。离线全局 Gold 可以监督目标 Mask，但不能作为推理输入；若此方法训练稳定性较差，可退回到“按多个其他窗口状态平均的 conditional add-one utility + masked Huber/pairwise rank”方案，并将两者独立报告，不在测试集挑优。

为了避免一个视频多条 QA 带来虚假样本数，先按源视频聚合 `GoldNLL` 及策略效用或对 QA 分组加权；用源视频采样 bootstrap/置信区间。不同 QA 数据集的 NLL 口径不得直接混合。

### 3.3 目标不是强制全开、全关或固定比例

分别优化并报告两个任务：

**任务 I — 最佳 QA 策略（Accuracy-first）**：先不惩罚额外图像数量（`λ=0`），在合法验证上评估是否能优于 V1 与 R_A always-on。若只为准确率，全部启用 R_A 最优，也必须如实报告。

**任务 II — 最佳效益/开销策略（Cost-aware）**：扫描预先列出的若干 `λ` 或验证期目标增量成本范围，让 Gate 自动选择更多或更少的 ON；输出准确率/Gold NLL—额外图像张数—Pipeline 时间三维 Pareto。`k` 不是一个提前写死的研究超参数。

**严禁**依据新确认集结果补设 `16%`、`32%` 或其他启用目标；也不能根据“R_A always-on 的 MUSIC 增益很大”预先宣称少数额外图就够。

### 3.4 开发集选择与锁定

- 所有新 Gate 的网络结构、训练配置、随机种子、`λ`、阈值由合法训练/验证来源决定。
- 验证必须**运行真正因果逐时刻 Mask**进行完整 QA 生成，不得只比较预测的单窗口效用。
- 报告 Accuracy、Gold NLL、`Rescue/Harm`、`mean k`、`P(k=0…4)`、各时刻 ON、实际 Pipeline Cost、模型复杂度，不能只报 AUROC/MAE。
- 选择一个固定 `Accuracy-first` 策略；一个固定 `Cost-aware` 策略，后一策略应采用验证集预先指定的非劣容忍界，例如“允许不超过预定的微小准确率损失，最小化多编码图像开销”。容忍界必须事先声明且仅使用验证集选择。
- 若验证最佳为全关，说明在当前主干/数据下选择模块尚未增益；若最佳全开，它可能缺乏节省额外图像成本的价值。都应如实报告。
- `pretest_lock.json` 必须保存 Gate checkpoint、R_A/V1 SHA、Prompt、时间网格、选择阈值、λ、任务口径、训练源/确认源名单及校准规则；测试前锁定且测试后不更改。

---

## 4. M3 — “最好额外编码多少张双图”的正式实验设计

当前视频有 4 个已观察时刻，ON 意味着在该时刻额外一张 Grounded Image；研究变量是每视频的 `k_extra_images∈0..4`。

### 4.1 两种数量实验必须区分

**A. 可部署、因果的 Learned Gate（主结果）**

- Gate 看当前/过去，逐时刻立即决定 ON/OFF；不要求固定 k。整段完成后统计 `k=0,1,2,3,4` 的经验分布和均值。
- 可选 `online_cap(K)` 诊断：K=0/1/2/3/4 时，按**已经到达的时刻**运行相同 Gate，若过去已用尽配额则拒绝新第二图；不可先看完整视频后挑得分最高的 K 个窗口。
- 所有 `online_cap(K)` 的规则必须在验证集固定；它测试一个因果预算上限，不等于全局 Oracle Top-K。

**B. 离线 Oracle 上限（仅开发/分析）**

- 使用 M1 的 16 个真实完整 Mask，对每个 K 离线计算 `best_gold_nll_mask_at_k` 与 `best_correct_mask_at_k`；这个步骤读取训练/开发 Gold，只能用于判定“是否存在可被选择的增益”。
- 必须突出它是**事后知道答案**的上限，不可以混入 Learned Gate 的正式准确率曲线，也不能拿它当真正在线选择结果。

### 4.2 优先汇报的图

1. `QA Accuracy vs mean k_extra_images`（同数据/Prompt、同冻结 V1/R_A）。
2. `QA Gold NLL vs mean k_extra_images`（MUSIC 与 Valor 分开）。
3. `Pipeline total/TTFT vs mean k_extra_images`（模型加载/缓存协议统一，包含预付 BEATs）。
4. `k=0..4` 选择分布图与各时刻选择率；不同问题子型仅用于事后评价。
5. `Gate vs Random-matched-rate` 配对准确率和原视频源聚类 CI。
6. `selected_on` 的音频替换敏感性：同 ON mask 下真实 Grounded、重复 RGB、Sham、Zero 产生了什么不同答案/Gold NLL。

### 4.3 必须防止的两个虚假结论

- 当 Gate 通过**额外视图数量**改善准确率时，不能把它直接解释为“选择到了有用的音频”：必须检验同 mask 的 `RGB+RGB` 与 Sham。当前 R_A 在 MUSIC 上的优势曾几乎不依赖 Grounded 提示。
- `mean k` 降低不等于 LLM 推理加速：LLM 一直是 `256 visual tokens/4 frames`。效率主要来自少调用 Localizer/额外视觉 Encoder；必须测全链路，并明确拒绝时 BEATs/Gate 的成本。

---

## 5. M4 — 正式对照矩阵（完全冻结 V1 / R_A）

本表全部走每帧 64 tokens 的接口，除显式注明的 Native 参考之外：

| ID | 选择策略 | ON 时第二图 | Router | 作用 |
|---|---|---|---|---|
| T0 | Always-OFF | 无 | V1 | 纯视觉参考；额外图数 0 |
| T1 | Always-ON | predicted Grounded | R_A | 增强端点；额外图数 4 |
| T2 | 旧 G_A | predicted Grounded | V1/R_A | 旧 Gate 在新 R_A 的零样本转移 |
| T3 | **新 Gate Accuracy-first** | predicted Grounded | V1/R_A | **主效果** |
| T4 | **新 Gate Cost-aware** | predicted Grounded | V1/R_A | **主成本—效果** |
| T5 | Random@同验证ON率（≥3 seeds） | predicted Grounded | V1/R_A | 区分学习选择与随机少开 |
| T6 | Visual-only Gate | predicted Grounded | V1/R_A | 判断音频是否影响选择 |
| T7 | 同 T3/T4 的精确 mask | 第二图改为重复 RGB | V1/R_A | **音频专属证据关键对照** |
| T8 | 同 T3/T4 的精确 mask | 第二图改为 Sham/错位提示 | V1/R_A | 检查空间提示特异性 |
| T9 | 同 T3/T4 的精确 mask | 第二图改为 Zero-audio 地图 | V1/R_A | 真实声学内容干预 |
| T10 | 可部署 online_cap(K)，K=0..4 | predicted Grounded | V1/R_A | 不窥视未来的数量上限曲线 |
| T11 | 离线完整 Mask Oracle | predicted Grounded | V1/R_A | **只限训练/开发诊断，不能作为部署得分** |
| N0（参考） | 原生双图 Always-ON | predicted Grounded | **无64 Router** | 单独比较，不混淆 visual token 数 |

`T7/T8/T9` 必须复用 **T3/T4 的同一实际 ON 位置**，保持第一视图、Prompt、输出 Token 数和融合器权重一致；只有第二张图的内容变化。不允许因为对照结果不好就重新选 Gate mask。

随机对照建议使用验证集冻结的 ON 概率，确认集的实际随机 ON 率可能有偏差，应如实报告；若需要严格同数量的描述性随机对照，则只能使用**在线预算计数器**，不能在看完所有时刻后重新选择随机位置冒充在线策略。

---

## 6. M5 — 数据使用与独立确认

1. 使用之前**合法的 MUSIC 训练/验证来源**进行新策略训练，但需核实已用来训练 R_A 的原始训练样本及是否存在多次验证超参搜索；若验证资源已反复使用，建议另设未使用的开发验证来源用于选择策略，不得把确认集拿来补验证。
2. **已检视确认样本**（例如 MUSIC 的 200 数字视频 ID / 288 QA、Valor 的 254 已知 YouTube 来源 / 508 QA，以及此前 439/512、701/256、VGG-SS、LLP 等）只可用于只读机制分析。若在其结果指导下开发新 Gate，则不能把在同一来源上的新评估称为独立 confirmation。
3. 从合法可获得的剩余媒体中寻找新来源；创建 `all_seen_sources.jsonl` 排重脚本，按原 YouTube ID（有则用）、数字视频 ID、片段起点、媒体 SHA256 和路径别名检查，记录不能证明的跨数据集源身份。
4. 对 Valor 优先要求可核验的 YouTube 来源；MUSIC 无法核验的原始血缘必须单独列为残余不确定性。若无法获得新独立来源，仍可运行旧来源作**已检视开发诊断**，但最终结论为 `independent_confirmation=not_run`，不能假装新的测试结论。
5. 预先声明主指标。MUSIC 使用规范化生成 EM + 完整 Gold 字符串 NLL；Valor 使用严格 ABCD 准确率 + 四候选归一化 Gold NLL。不同口径不求一个混合 NLL。
6. 不允许从正式确认结果中再挑 `λ`、Gate checkpoint、ON 率、主指标或最好的随机 seed。配对 CI 采用原视频来源聚类 bootstrap（例如 2000 次，固定种子），报告各来源/QA 数和所有失败。

---

## 7. M6 — 代码结构、工程验收与复现

建议在现有仓库**新增**：

```text
utility_gate/fixed_router_gate/
  audit.py                  # 老资产身份与隔离
  collect_causal_features.py
  enumerate_masks.py        # train/val完整16个真实策略
  policy_dataset.py
  gate_models.py            # 只训练Gate
  policy_distill.py         # 完整Mask策略蒸馏
  gate_train.py
  online_inference.py       # 真正按时刻条件调用第二图
  evaluate.py
  budget_sweep.py           # 额外图片0..4/在线预算上限
  paired_stats.py
  timing.py
  report.py

tests/test_fixed_router_gate.py
scripts/run_fixed_router_gate.sh

artifacts/utility_gate/fixed_router_gate_v1/
  audit/
  data/
  labels/
  train/
  models/
  validation/
  confirmation/
  timing/
  reports/
  locks/
  figures/
  final_audit.json
```

如果输出目录已存在，创建不可覆盖的新版本；不允许覆盖历史工件。

### 7.1 自动单测必须覆盖

- `gate.disabled / always_off`: P/Z/generated IDs 与 V1 逐值一致。
- `always_on`: 路由生成与 R_A 旧 fixture 逐值一致；`k=4` 时最多 4 次新增 Grounded 视觉编码。
- 严格模拟 `t=2/4/6/8` 到达，注入未来音视频不影响当前 Gate/mask/P/Z；不回写旧 token；历史动作和计数均因果。
- Gate ON 之前没有发生 EZ-VSL/第二 SigLIP 执行；OFF 时 Grounder/第二视觉编码调用为 0。
- 不存在 `QA prompt/gold/question_type/source_id` 进入 Gate tensors；没有 QA Gold 出现在新确认 `training_loader`。
- 对固定 Mask 的 `T7/T8/T9` 与主策略在每个来源都**完全相同的 ON 位置**。
- 同一视频多条 QA 的来源分组、完整策略收益与 16 mask 记录一致；没有把窗口独立样本当成视频独立来源。
- `k_extra_images`、实际视觉编码调用、64-token/帧与可用性掩码一致。
- 所有旧 checkpoint、旧源文件、历史实验输出 SHA 比对保持不变；新增参数只有 Gate。

### 7.2 真实运行验收

1. GPU 上真实加载 V1/R_A/EZ-VSL/LLM 与新 Gate，运行 OFF、ON、混合、重复图/错位图的实测生成，而不是仅跑单元 mock。
2. 预先锁定一个有音频的训练/开发来源，验证 `k=0,1,2,3,4` 的**因果在线选择上限**与真实第二视图编码次数；在 Gate 全关时仍能跑完完整视频 QA。
3. 训练必须记录 Gate 的真实梯度、更新、最佳 epoch、loss 曲线与策略对照，不可把“接口已实现”当作“Gate 已训练成功”。
4. 每个确认策略均实际执行完整生成，记录逐题预测、Gold NLL、配对 Rescue/Harm、聚类 CI 和原始耗时；除可核验的输入与权重完全相同策略外，不以缓存旧答案替代新策略推理。
5. 软件、CUDA、模型加载时间是否计入、预热与中位数计算方式须一致；只用少数来源计时必须标为描述性测量。

---

## 8. GO / NO-GO / INCONCLUSIVE 规则（测试前锁定）

**GO-SELECT (主要科研目标)**：在同一新独立来源、同一冻结 R_A/V1 下，新 Gate 的实际完整视频 QA 表现，在明确的统计区间中优于预声明的固定端点/匹配随机策略之一，并且不能靠简单全开/全关解释。至少给出可复核的 ON 时 Rescue、Harm、节省编码图片数和实际 Pipeline Cost。

**GO-EFFICIENCY (实用目标)**：新 Gate 在 Accuracy / Gold NLL 容忍界内，相比 R_A always-on 明确减少额外图像编码数和实际前端成本，且收益大于 Gate（BEATs/决策）预付开销。容忍界必须在开发期声明，不得测试后选择。

**GO-AUDIO-SPECIFIC (更强的研究主张)**：在**相同 Gate mask、相同 R_A、相同双图编码数量**下，真实 Grounded 比重复 RGB / Sham / Zero 音频对照获得稳定的额外 QA 收益；如果不满足，最多支持**选择性第二视觉路径/一般模型适配**，不能称音频辅助证据有效。

**NO-GO**：全流程真实运行、模型已训练，但没有表现优于合理固定基线或只出现同 Mask 的重复图收益；针对**当前固定 R_A 的 Gate 方案**不继续无限调参。

**INCONCLUSIVE**：缺少可训练的独立来源、QA 监督质量不足、模型执行失败、主要 CI 太宽或没有合法新确认；不得把 `not_run` 填成 0，也不得将旧已检视结果包装成新测试。

必须同时给出上述三个 GO 状态，不得只给一个模糊总分，也不因 GO-EFFICIENCY 成立就写“新 SOTA”。

---

## 9. 最终交付清单

**必须提交**：

- `reports/M0_audit.md`：模型、checkpoint、旧结果、数据角色、来源隔离、四象限。
- `reports/M1_complete_mask_utility.md`：合法 train/val 的 16 Mask、每个 `k=0..4` 的真实收益/代价与 Oracle 上限。
- `reports/M2_gate_training.md`：只训练 Gate 的配置、真实运行、损失/梯度/曲线、不同训练方案对照。
- `reports/M3_budget_selection.md`：预声明 Accuracy-first 和 Cost-aware 规则，开发集 ON 率、额外图片分布和阈值。
- `reports/M4_frozen_confirmation.md`：新独立来源的主/控制结果与配对区间；若无独立来源，明确 `not_run`。
- `reports/M5_audio_specificity.md`：相同 mask 下真实/重复/Sham/Zero 的差异。
- `reports/M6_system_cost.md`：实际新增 Encoder 图片数、EZ-VSL 调用数、视觉/音频编码耗时、LLM TTFT、整 Pipeline 时间。
- `reports/FINAL_JUDGMENT.md`：GO-SELECT、GO-EFFICIENCY、GO-AUDIO-SPECIFIC、NO-GO、INCONCLUSIVE；逐一说明证据等级。
- `per_qa_results.jsonl`、`per_window_gate.jsonl`、`policy_mask_results.jsonl`、`paired_summary.json`、`pareto.csv`、`pareto.png`、`commands.json`、`pretest_lock.json`、`final_audit.json` 和完整 `all_seen_sources.jsonl`。

如出现异常、负结果或来源不足，请**保持现有数据与预测不被覆盖，报告阻碍并完成所有仍然可合法执行的阶段**，不因第一轮 Gate 出现全开/全关就停止机制诊断。

---

## 10. 交给 Codex 的第一条执行命令（自然语言）

> 阅读当前仓库 `AGENTS.md` 和本文件 `AGENTS_FIXED_ROUTER_GATE_BUDGET_SEARCH.md`。**本轮只训练 Gate，不再训练 Router。** 首先定位并锁定最新已经训练的 Grounded Fusion Router R_A 和 Vision-only V1；读取旧 M0–M7 报告与已见来源。使用合法 train/val QA 对四时刻的 16 种 ON/OFF 完整 Mask 执行真实冻结 VLM 推理，建立针对 R_A 的序列级 QA 效用监督。然后只训练轻量因果 Gate，分别寻找 Accuracy-first 和 Cost-aware 策略，在验证集上确定其额外 Grounded 图像编码次数/启用规则并提前锁定。最终在未参与训练、调参、历史结果分析的新来源上运行完整配对对照：V1、R_A always-on、旧 Gate、新 Gate、同 mask 的重复 RGB/Sham/Zero、matched random 和在线 k=0..4 数量曲线。保持每帧 64 visual tokens，严格音频 [t−2,t] 因果输入，训练时不可修改任何旧权重。对音频特异性、模态选择和额外编码成本给出分开的 GO/NO-GO 判断，禁止提前声称 SOTA。

## 11. 主要参考（方法思想，不构成实验完成证据）

- MuseVLA, adaptive on-demand sensor selection and grounded sensor image: <https://arxiv.org/abs/2606.17598>，<https://github.com/microsoft/MuseVLA>。
- DynamicViT, differentiable importance prediction and sparsification: <https://papers.neurips.cc/paper_files/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html>。这里仅借鉴**可学习选择与计算代价评估**的思想，不修改视觉 backbone 或训练 token pruning ratio。

