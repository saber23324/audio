# AGENTS — AudioRouter / VGG-SS 特征层诊断与数据规模扩展

> **执行对象**：Codex / 工程 Agent。将本文件放在现有 AudioRouter 仓库根目录，按阶段执行；不要新建一个与原项目无关的实现。  
> **任务性质**：冻结模型的机制诊断 + 预注册的数据规模扩展 + 独立来源确认。**不是**新 Gate 训练，不是大模型微调，也不是声源定位器训练。  
> **优先级**：先理解原项目与现有 B0–B8 工件，再写代码。保持原有 `AGENTS.md` 的通用约束；发生冲突时，以本文件的实验隔离、历史保护和 Oracle 限制为准，并在报告中记录冲突。

## 0. 研究背景与本轮目标

项目已经有 AudioRouter、Stage1 Vision-only 64、Stage3 AudioRouter 64、Gate2 / Dual-Head 等研究资产。本轮围绕 VGG-SS 声源框生成的 **Oracle Grounded-Audio Image** 进行低成本验证。已有 2026-10-28 试验：125 个来源的视频视觉特征测试，类别代理仅 108 个有效来源，已经看过全部结果。

**已观察到的结果（历史观察，不能重新解释为本轮独立确认）**：

| 方案 | 有效来源 | ROI mass | 相对 B0 的 Δmass [95% CI] | 类别代理 | 限定解释 |
|---|---:|---:|---:|---:|---|
| B0 清洁 RGB | 125 | 0.3471 | — | 105/108 | 固定 Stage1 Vision-only 基线 |
| B1 GT bbox overlay | 125 | 0.3478 | +0.0007 [-0.0018, 0.0033] | 105/108 | 像素叠加未显示稳定空间增益 |
| B2 sham overlay | 125 | 0.3487 | +0.0015 [-0.0010, 0.0043] | 104/108 | 非真框区域控制 |
| B3 uniform overlay | 125 | 0.3474 | +0.0003 [-0.0012, 0.0018] | 105/108 | 全局扰动控制 |
| B4 GT routing bias | 125 | 0.4639 | +0.1168 [0.1082, 0.1247] | 105/108 | 直接注入 Oracle 空间先验，mass 提高具有机械性 |
| C1 random color | 125 | 0.3525 | +0.0054 [0.0025, 0.0080] | 105/108 | 颜色/纹理扰动可能比 GT overlay 更影响路由 |
| C2 bbox outline | 125 | 0.3489 | +0.0018 [-0.0001, 0.0037] | 105/108 | 轮廓对照 |

另外 B1−B0 的 Teacher-KL +0.0311 [0.0172, 0.0487]（保真变差）；B4 的类别代理 Gold-NLL 差约 -0.0026 [-0.0054, -0.0008]，但正确数未增加，Teacher-KL 差区间跨零。B5/B6（**模型预测**热图）因缺失可信预测资产，尚未运行。现有 B7/B8 是单帧 Stage3 参考，样本数与 Stage1 不同且训练历史不同，**不得与 B0–B4 冒充同 checkpoint 消融**。

**本轮要回答的两个问题**：

1. **机制诊断**：B1 的 RGB 热图叠加究竟在哪一级失效？是像素提示无法明显改变冻结 SigLIP 的特征、改变了特征却没有改变 Router、改变了 Router 但没有保留额外视觉证据，还是下游类别代理接近饱和？
2. **样本扩展**：将先前 125 来源的开发观察扩展至更多有合法媒体和框的 VGG-SS 来源。对预先冻结的 B0/B1/B2/B3/B4/C1/C2 进行配对评价，并且将**已检视开发集**与**未检视独立确认集**严格分开。

**本轮不允许**仅凭 VGG-SS GT box 获得的性能宣称可部署音频推理收益；不将 Source-category Proxy 或 Teacher Agreement 称为官方 QA Accuracy；不将 B4 大幅提高 ROI mass 称为独立的科学发现。

## 1. 必须保留的科学边界

1. **视觉主模态 / 冻结接口**：视觉编码器、视觉 projector、Stage1/Stage3 原 checkpoint、LLM 保持冻结；当前每采样帧仍只传入 64 个视觉接口 token。保留原始 `Z=P@V` 对照，不增加 native audio token。
2. **Oracle 明示**：由 VGG-SS **标注 bbox** 生成的热图标记 `gt_oracle=true`；这是上限/机制诊断，不能作为真实音频推理输入或可部署模型。
3. **评估对象**：VGG-SS 的官方标注主要为视频片段中心帧的可见声源 bbox，不是全时间连续热图，也不自带标准 Video QA；不要将框推断到未经标注的其他帧当真值。
4. **时间协议**：若启用音频路径，只有 `[max(0,t-2s), t]` 的音频可用；图片测试仅在官方注释帧和其合法可用时间点执行；视频时间轴与原 10s clip 截取位置显式记载。
5. **训练历史混杂**：Stage1 与 Stage3 不同 checkpoint；Stage3 real-vs-null 才能视为同 checkpoint 音频输入干预。所有 Stage3 参考独立报告。
6. **既有结果永久只读**：不得覆盖此前 `artifacts/utility_gate/gate2_grounding_20261027/`、旧 VGG-SS pilot、Gate2、Dual-Head 工件、checkpoint 和训练脚本。新结果使用独立、不可覆盖的目录。
7. **独立性**：所有此前开发/测试过的 VGG-SS、Gate2、Dual-Head、AVQA、LLP 来源视为已检视，不能变成新的 confirmation 来源。至少按原始 YouTube ID + 原始剪辑时间 + 内容 hash 排重，保留映射表。
8. **不能用测试框调参**：新 holdout 的 GT box 只在模型/融合参数/分析口径冻结后用于最终报告。使用 GT bbox 生成 Oracle 输入是被定义的测试干预，并不是自动拥有独立评价；参数必须先固定。
9. **不以调高准确率为由删异常**：实际媒体不可用、损坏、时长不足、坐标非法等按预先定义理由记录；任何结果相关的筛样禁止。
10. **无虚构执行**：没有媒体、注释或预测图时，写 `not_run`、具体原因、尝试路径，不得填 0 或声称已完成。

## 2. 权威数据与参考实现

优先检查仓库和已有 `/nvme_data/pkt/streaming/dataset`、`artifacts/utility_gate/` 中的实际资产，先复用现有 source manifest、SHA 与下载日志。

- **VGG-SS 官方页面**：https://www.robots.ox.ac.uk/~vgg/research/lvs/ —— 下载/核查 bbox 注释，约 5k 视频片段、220 类；标注单位一般是带 class 与 bbox 的剪辑名，不等于自动提供所有源视频文件。
- **原始 VGGSound 元数据**：https://github.com/hche11/VGGSound ，页面 https://www.robots.ox.ac.uk/~vgg/data/vggsound/ —— CSV 为 YouTube ID、起始秒数、类别、split；**官网目前提示视频下载链接不再提供**，不能假设有现成全量视频直链。
- **EZ-VSL 官方实现与模型库**：https://github.com/stonemo/ez-vsl —— 可供以后预测热图实验参考；本轮不必安装它。
- **DenseAV 官方实现与预训练权重**：https://github.com/mhamilton723/DenseAV —— 以后生成无 GT 的预测热图可参考；本轮不把未经验证的预测图作为输入。
- **MuseVLA**：https://arxiv.org/abs/2606.17598 —— 只借鉴 Grounded Sensor Image 视觉化思想；其声学传感、训练和空间标定与 VGG-SS 的普通音轨不同。

下载时遵守数据许可、平台访问限制、重试/速率规则；不得绕过鉴权，不得将未经验证的第三方镜像标记为官方媒体。镜像使用需记录 URL、revision、文件 hash、授权说明；下载失败不能悄悄换成别的来源。

## 3. M0 — 仓库审计与数据清单（必须先完成）

**代码/资产审计**：

- 找到原始 pilot 的 B0/B1/B2/B3/B4/C1/C2 实际实现与 alpha/lambda 固定值；找出 B0 的 Stage1 路由 `R, P, Z`、SigLIP feature、投影和池化路径及生成接口。
- 找到旧 125/109/108 源来源 ID、剪辑时间、图片/音频 hash、caption/类别 proxy、GT bbox 和排除原因；扫描其他已检视工件中出现的 VGG-SS 或相同 YouTube 视频。
- 审查原视觉链：输入图像 resize 到 384；valid patch14 产生 27×27 空间格；projector 与空间 pooling 后产生 14×14=196 tokens；确认双线性插值 `align_corners=False`、row-major 排列以及边界有效域。**以实际代码核验，不根据该描述强制更改模型。**
- 确定各层张量是 patch/grid、projected token、LLM 输入嵌入的哪一种，避免不同空间分辨率之间简单 reshape。
- 核验 B0 的 `V` 是否在图像加工前后均使用同一图像路径；B4 使用的应是 clean RGB 的原始 `V`，其改变来自 routing bias。

**新数据索引**：

- 读取官方 VGG-SS bbox 元数据，全量去重并校验 x/y 坐标约定、多个 bbox、非方形 resize 与中心帧时间点。
- 构造排除集 `seen_sources.jsonl`，同时存 source video ID、clip offset、hash 别名。确认旧 125、旧 128 以及其他已检视视频均被排除在新确认集之外。
- 对全量候选逐个标记媒体状态：`existing_verified`、`downloadable`、`missing`、`restricted`、`failed_transient`、`invalid_duration`、`invalid_bbox`、`duplicate`；保留失败数与失败日志。
- 对可用媒体计算 frame timestamp、实际 frame dimensions、decoded audio presence、文件 hash，**不根据模型预测结果筛选媒体**。

**交付**：`reports/M0_audit.md`、`manifests/all_candidates.jsonl`、`seen_sources.jsonl`、`media_acquisition.csv`、`tensor_layout.json`、原始受保护资产 SHA 清单。

## 4. M1 — 已检视 125 来源上的特征层诊断（不重新拟合）

先在旧 125 来源复现冻结 B0/B1/B2/B3/B4/C1/C2 的一致性，**不能修改主表结果**。本阶段可以增加中间特征抽取，不改变 forward。若需要 sweep，必须明确标记为已经检视数据的**开发性敏感性分析**，不能作为独立性能结论。

### 4.1 需要捕获的流水线层级

为每个来源、每个策略保存一致命名的以下输出（推荐压缩 npz / torch tensor，巨大张量抽样存）：

```
image_clean / image_modified / overlay_mask / gt_bbox
encoder_patch_features          # SigLIP patch/grid spatial features，例如 27×27
visual_projector_features       # 投影后的视觉特征
visual_tokens_196              # 原 Stage1 进入 Router 的视觉值
router_queries                 # 使用中的 query bank / conditioned queries
router_logits_R                # softmax 之前匹配分数
routing_matrix_P               # 64×196 的权重
compressed_tokens_Z            # P @ V
llm_gold_option_logprobs       # 仅在具有合法 proxy label 时
llm_generated_ids / gold_nll / teacher_kl
```

严格记录张量定义/shape/dtype/normalization/空间格映射。不得把平均 64 个 Query 之后的 `P` 当成 SigLIP 原生 attention。

### 4.2 特征响应度量（配对 Bx − B0）

对 B1/B2/B3/C1/C2，按**清洁图像**与被修改图像进行相同索引对齐，计算：

- 输入像素差：全图 L1/L2、Mask 内/外差、遮挡比例、alpha=0 的逐像素等价。
- Encoder patch 变化：每 patch 的 `1−cos(f^B,f^0)`、相对 L2 变化、均值/中位数/90分位，变化是否集中在 overlay ROI。
- 投影后 196-token 变化：同样计算 `1−cos` 与相对 L2，分为 GT box 内、外、边界；检查是否在 pooling / projector 后被削弱。
- Router 原始 logit 变化：`R_B−R_0` 的尺度、每 query 方差、top1/topk 匹配位置变化；注意尺度/温度。
- Router softmax 变化：每 query 的 JS divergence、平均 routing map 的 JS、top-K ROI rank/重叠、routing entropy 与 spatial concentration。
- 压缩结果变化：`Z_B−Z_0` 的 cosine、归一化距离、token 范数；可附上 LLM Gold NLL/Teacher KL 的配对差，但不得把 category proxy 叫 QA。

**面积校正**：给出 `Area(B)`、`Raw ROI mass`、`Mass excess = mass−area`、`Mass lift = mass/area`（area>0），并使用原始像素覆盖到 27×27，再通过已验证的真实池化映射评估 196-token 位置；多个 bbox 取 union，不能重复计权。对于 bbox 覆盖的区域，应在 token 级使用真实重叠/投影系数而非硬中心点近似。若当初的定义不同，保留旧指标，同时新增新版本并明确不得直接比较数值。

**对 B4 的单独分析**：B4 直接添加 GT 先验，所以 ROI mass 提高具有机械性。重点分析它是否改变了 `Z`、Gold NLL、Teacher KL，以及改变的区域与 QA 错误是否相关。不要用 GT ROI mass 增长为 B4 的独立成功标准。

### 4.3 必须作图

- 典型成功/失败案例：原图、GT 框、B1 叠加图、B0/B1/B2/C1/B4 路由热图、patch feature delta heatmap，布局相同。
- 特征变化量随层级变化图：pixel → patch → projector → 196-token → R/P → Z → LLM NLL。不同单位须分别呈现或无量纲归一化，不能误连成同尺度曲线。
- box 面积 × Feature Delta / Δmass 散点图；按类别/框面积/音频质量分层。
- B1 与 Sham、Random color 的配对差分布，明确显示 0 附近及负差样本。
- ROI mass 提高但 NLL 未改善、Teacher KL 变差的案例页。

### 4.4 预定义诊断分支（不是新结果保证）

| 观察 | 合理推断 | 推荐下一步 |
|---|---|---|
| Pixel 已明显改动，SigLIP/196 tokens 几乎不动 | 叠加信息被视觉前端忽略或归一化抵消 | 考虑 feature-level conditioning |
| 视觉特征明显改变，`P` 几乎不动 | Router 对叠加特征不敏感 | 检查 query-similarity / 归一化与空间信息 |
| `P` 明显改变，但与 bbox 无稳定正关联 | 纹理/颜色噪声影响大于语义定位 | 不再把像素热图当可靠提示 |
| B4 改变空间选择且 Gold NLL 更好，B1 无效 | 先验位置有潜力，像素传递路径失效 | 优先验证预测热图的 routing bias |
| ROI mass 与教师保真好，但类别准确率不动 | 任务天花板或空间选择与类别任务错位 | 使用更难且合法的视觉证据任务 |

上述都是**待证伪假设**，不允许 Agent 根据观察跳过必需对照或事后改写成功标准。

**M1 交付**：`features/per_source.*`、`reports/M1_feature_diagnostics.md`、`figures/*.png`、特征层指标 CSV、逐例失败分析、shape/等价性测试。

## 5. M2 — 扩展 VGG-SS 数据量：三层数据池

官方 VGG-SS 约 5k 有 bbox 的片段，但可获取完整视频的数量取决于当前合法访问和媒体存活情况。**不许把“5k 标注”写成“5k 已下载有效配对视频”。**

| 池 | 来源 | 初步目标 | 角色 | 是否可调参 |
|---|---|---:|---|---|
| `legacy_dev` | 已检视原 125 等 | 现有全量 | 机制诊断 / 复现 | 只能做诊断，不能称独立确认 |
| `fresh_dev` | 未检视且合法媒体 | 若可得 200–400 个独立源 | 开发、敏感性、选择最终设置 | 可以，但记录所有尝试 |
| `fresh_holdout` | 与一切开发来源不重叠 | 尽量 ≥500；有条件扩到 1000–2000 | **冻结后的一次性确认** | 绝对不允许 |

上述数量是**建议的目标，不是承诺**。若可得数量不足，仍对所有合法来源完成透明报告；统计效能应基于新开发池的配对差方差估算后预先锁定。不要因为结果不明显而不断添加确认样本并反复查看显著性。

**数据抽样与代表性**：

- 在查看任何新确认预测之前，固定随机种子、原始 source ID 去重、官方 split 和可用性筛选规则。
- 按 VGG-SS 类别/大类、bbox 面积（例如小/中/大，阈值需预先锁定）、多 bbox、视频获取来源进行分层；稀有场景可以适量超采样，但同时报告原始分布与加权总体估计。
- 尽可能避免只采集容易定位/容易分类的大框或单声源视频；列出“完整候选—获得媒体—成功编码—最终评估”漏斗与每步失败数量。
- 保持同一视频在各策略中配对执行，禁止某方案出错时只删除该方案而保留其他方案在主表中。
- 防止同一视频不同 clip、同字节镜像、多平台别名跨池；与之前 Gate2/LLP/AVQA、VGG-SS 实验的已检视 ID 做比对。

**媒体缺失策略**：已有合法缓存优先；官方元数据可用于定位来源，但不保证视频可下载；仅对公开、授权、当前可取得的资源进行重试。`429/403/TLS/timeouts` 必须记录，不能绕过访问限制。第三方镜像如果使用，记录媒体 hash 与来源声明，不能把未经核验的文件当官方版本。

**分布变化报告**：新池与旧125 在 bbox 大小、声源类别、音频长度、压缩后 ROI 面积上的差异必须报告；对新池不应简单声称“与旧数据同分布”。

**M2 交付**：`manifests/fresh_dev.jsonl`、`manifests/fresh_holdout_LOCKED.jsonl`、`reports/M2_data_scale.md`、可用率及失败图、SHA/源去重审计、统计效能预估、独立性检查结果。

## 6. M3 — 新开发集上的敏感性分析（全部开发性）

只在 `fresh_dev` 中尝试图像注入强度或几何形式。**不得**利用 `fresh_holdout` 选择参数。

### 6.1 固定原实验主配置为第一行

首先完全复用原 pilot 里 B0/B1/B2/B3/B4/C1/C2 的实际 alpha、lambda、map smoothing、插值/overlay/colormap 参数；没有的参数从源代码审计得到，不能臆造。

之后可以进行有限、明确命名的开发性 sweep：

- Overlay intensity：`alpha ∈ {0, 0.05, 0.1, 0.2, 0.4}`（**示例开发网格**，原来固定 alpha 的值必须单独保留）；
- Overlay 表达：半透明颜色填充、灰度/亮度调制、框边线、RGB+heatmap 的两幅拼接图（如做，标记会改变图像布局和 token 接口，不能与单图混为同条件）；
- GT bias strength：有限 `lambda` 网格并报告 ROI/NLL 与 Teacher KL；注意 bias 用 GT 本身，不能当可部署方案；
- GT spatial map：hard box / blurred Gaussian / soft-edge，Blur 不能改变目标为可见声源 bbox 的解释。

每种新方案须有等覆盖面积/等颜色能量的 sham 和 random control。对于原已检视旧 125 的开发曲线单独保存，不允许其成为选择阈值的唯一依据。

### 6.2 预先冻结正式主对照

至少保留：

```
B0: clean_RGB -> Stage1 Vision64
B1: GT bbox overlay -> same frozen Stage1 Vision64 (Oracle)
B2: spatially matched sham overlay -> same Stage1
B3: uniform overlay -> same Stage1
B4: GT routing bias over clean RGB V/P -> same Stage1 (Oracle)
C1: randomized color, same mask geometry/strength -> same Stage1
C2: GT bbox outline -> same Stage1
```

`B5/B6 predicted heatmap` **只在**已经有可核验、未使用确认 GT 输入、按因果时间可生成的定位器/预测资产时运行，作为独立的后续可部署路线；不能因本轮没有预测图而失败停工。

同时保留原 `Stage3 real/null` **独立参照表**，不得利用不同 109 来源的历史数值拼接为主表；若新增 Stage3 测试，必须使用相同新来源与一致的视觉/时间协议，仍标明训练历史混杂。

### 6.3 选择标准

在新开发集上根据事先固定的优先级选择正式 Overlay 配置（如要选择）：

1. 非 Oracle 视角下不允许出现 bbox label 泄漏；
2. 与 B0、Sham 相比具有稳定的 ROI 相关性（**仅机制信号**）；
3. Teacher-KL / Gold NLL 不恶化或改善（只作为合法的 proxy）；
4. 类别代理准确率可以补充但不作为唯一选模标准，因为此前 B0 105/108 接近饱和。

如果所有 Overlay 在开发集都表现差，**仍按锁定的原主配置跑一次新的 holdout 确认**，或者按事先明确的资源中止规则停止，并输出 `NO_GO`。不能临时发明更复杂方法使结果变好再叫独立确认。

**M3 交付**：`reports/M3_dev_sensitivity.md`、所有开发参数与结果表、最终预注册 `pretest_lock.json`（必须在 confirmation 前写入 model/code SHA、manifests、metrics、参数、决策/停止规则）。

## 7. M4 — 扩展数据上的独立确认与统计

**前置条件**：`fresh_holdout` 尚未被任何相关运行读取标签来选模型/阈值；`pretest_lock.json` 已创建且经自动校验不可覆盖。

主结果同一 `N` 个有效视频配对比较 B0/B1/B2/B3/B4/C1/C2。

### 7.1 主空间指标

- `ROI_mass`, `ROI_mass_excess`, `ROI_mass_lift`，包括 GT bbox area 的修正；
- `ΔROI_mass(B1−B0)`、`ΔROI_mass(B1−B2)`、`ΔROI_mass(B1−C1)`、`ΔROI_mass(B4−B0)`；
- Routing JS、TopK overlap、in/out-of-box feature change 等机制指标，评估是否出现同方向变化；
- 按 box 面积、声源类别、多框、音频质量分层，报告每组 n 及不确定性。

### 7.2 下游与保真指标

- 若仅有 VGG-SS 类别：输出 `Source-category proxy` 正确数/准确率/Gold NLL，**绝不称官方 QA**。类别来自 VGGSound，不以原本 105/108 的高基线作为方法有效性的唯一依据。
- `Teacher agreement`、`Teacher KL`、gold-answer NLL 都必须精确说明 teacher 的来源、比较内容及 prompt；Teacher Agreement 不是 Ground-truth Accuracy。
- 若另有经核验的、合法且**与 GT box 不同源构造**的视觉问答数据，另立独立任务协议与 source overlap 排查，不与 VGG-SS category proxy 混成一列。
- 必须统计配对答案 `both correct / B1-only rescue / B0-only win / both wrong`；如果只有 3 个 baseline 错误，不夸大“超过基线的空间”。

### 7.3 统计方案

- 分析单位：原始视频来源 ID；如一个视频有多个剪辑/问题，按来源聚类 bootstrap；不要把所有 frame、query、patch 当独立样本。
- 主对照使用至少 2000 次原视频 source-cluster paired bootstrap，固定种子；报告均值/中位数、95% CI、有效样本数与分层支持数。
- 对多个不同对照/参数给出探索性标记，不能挑一个最显著的对照声称主结果；预定义 B1−B0 与 B1−B2 为首要 Overlay 对照。
- 可以在开发集通过 paired discordance 与样本方差估计功效/目标样本量，但不可用确认集结果逐次算显著性并决定继续追加确认样本。
- 若需要论证“没有有意义的改善”，预先定义最小重要差值（例如 ROI mass +0.01 或视觉 QA 提升 +1pp，数值须作为设计假设写入预注册）；不显著不等于零效果，等效性检验另需方案。

### 7.4 结果结论必须分开

| 结果 | 可允许的结论 |
|---|---|
| B1 feature/P 差几乎不变 | 当前冻结视觉前端 + RGB Overlay 没有形成有效路由输入变化（仅限测试配置） |
| B1 feature 改而 P 不改 | 问题更可能在 Router/表示交互 |
| B4 ROI mass 大增 | 注入 Oracle 空间 bias 可以机械性重定向空间权重 |
| B4 Gold NLL 轻微改善 | 存在 Oracle 先验在类别 proxy 上的有限连续指标信号，非可部署音频收益 |
| B1 vs B0 的类别代理持平 | 在当前高基线任务下尚未观察到 Accuracy 增益；不能证明所有更难任务无效 |
| B5/B6 预测热图有效（若另立新确认协议） | 才有机会讨论 GT-free 的推理可行性，仍须计入 Localizer 开销 |

**M4 交付**：`reports/M4_fresh_confirmation.md`、`per_source_results.parquet/csv`、配对统计/CI/可视化、源ID独立性审计、不可覆盖的原始生成记录。

## 8. M5 — 机制汇总与未来分支决策（不重训）

写出一张“像素 → 编码 → 路由 → 压缩 → 下游”责任定位表，并给出：

- **GO-A**：仅当新独立来源的 Overlay 在空间映射和合理下游指标上均有稳定改善，且不是颜色/框形态 sham 造成，才支持继续研究 RGB Grounded Image。
- **GO-B**：如果 B4 Oracle bias 在空间层有效，B1 overlay 不强，优先后续开展 `Predicted Heatmap → Routing Bias`；本轮只给出可行性线索，不声称已经部署成功。
- **NO-GO**：如果新确认中的 B1 与对照都没有稳定改进，明确记录“冻结 Stage1 + 这种 GT Overlay 无法形成可靠增益”；这不会否定所有 Audio-to-Visual 方案。
- **INCONCLUSIVE**：若有效新来源不足、类别 proxy 高度饱和、媒体条件不足或不确定性很大，给出具体限制与下一步条件，不编造显著性。

如果以后接入 DenseAV/EZ-VSL 等 Localizer，必须另立 `predicted_map` 实验：冻结 Localizer，声源图根据**真实音频+图像**预测，禁止 GT box 输入；如果 Localizer 预训练使用 VGGSound，与 VGG-SS 的来源交集/数据集内标签问题需在实验限制中披露。

## 9. 建议代码与运行接口（Codex 需检查实际仓库，不可假定已存在）

建议在既有工程增加，保留旧入口不变：

```
utility_gate/grounding/feature_diagnostics.py
utility_gate/grounding/scaleup_manifest.py
utility_gate/grounding/scaleup_eval.py
utility_gate/grounding/feature_stats.py
utility_gate/grounding/feature_viz.py
tests/test_vggss_feature_scaleup.py
```

所有脚本支持：`--config`、`--split`、`--output-dir`、`--resume`、`--dry-run`（按现有 CLI 习惯落地）；示意调用必须在新增实现与 `--help` 自测成功后写入 README，不得先假装命令有效。

推荐新输出根目录（绝不覆盖已存在运行）：

```
artifacts/utility_gate/vggss_feature_scaleup_20261028_v1/
├── audit/
├── manifests/
├── features/
├── per_source/
├── figures/
├── reports/
├── logs/
├── config/
├── pretest_lock.json
├── STATUS.json
├── summary.json
└── final_audit.json
```

其中 `STATUS.json` 逐 M0–M5 记录 `pending/running/done/not_run/failed`，带真实来源分母、时间、命令、错误原因；`summary.json` 不许把 `not_run` 填零；原始生成答案、路由张量、特征张量如缓存过大，按校验 hash 分块保留。

## 10. 强制自动测试

1. `alpha=0` 时 B1 与 B0 像素、feature、P、Z、generated IDs/logits 逐值等价（允许明确的数值 tolerance）。
2. `lambda=0` 时 B4 与 B0 的 P、Z、生成结果逐值等价。
3. B1/B2/C1 使用相同 masking geometry/透明度/面积或明确记录无法严格匹配之处；randomization 可由 seed 复现。
4. Multi-box union、bbox 坐标系、非方形像素与 27×27→196 的实际映射均有 toy 和真实媒体测试。
5. 旧 Stage1 与 extension-disabled Stage3 fixture 完全保持；所有旧 checkpoint 与保护代码 SHA 不变。
6. `seen_sources` 中任何 ID/原视频 alias/hash 不进入 fresh_holdout；媒体损坏/下载失败不替换样本而不记录。
7. `fresh_holdout` 在锁定前不得访问 GT 注释进行调参；锁定后只按预注册统计一次。
8. 旧 125、109、108 来源不可以重命名成新确认来源；同视频多次采样不会增加独立 n。
9. B4 必须标记 `gt_oracle`；B5/B6 缺预测资产时应标记 `not_run:no_predicted_map_asset`。
10. 评价表中的 `category_proxy_accuracy`、`teacher_agreement`、`localization_metric` 字段互不覆盖，防止 schema 或图表混列。
11. 保存 CUDA/torch/transformers 环境、GPU、dtype、seed、模型 hash、source code hash、媒体 hash、实际分母与阶段耗时。
12. 必须在一个真实 GPU 样本上进行 feature-hook 流水线回放，证明 hook 不改变原始输出。

## 11. 阶段验收与停工标准

- **M0 验收**：找到真实旧代码与模型，全部旧 hash 保持；官方框与现有图像坐标核实；至少输出全量候选/已检视排除/媒体可用状态。若无法确认 bbox 到 token 映射，停止正式空间指标但继续报告技术障碍。
- **M1 验收**：旧结果配对复现；逐层 feature/P/Z 诊断和可视化可复现；测试均通过；不依据旧数据宣称新确认。
- **M2 验收**：产生明确分开的 `legacy_dev/fresh_dev/fresh_holdout`，源去重可审计；媒体规模不足时如实报告而非造数据。
- **M3 验收**：所有参数尝试都保存在开发报告，`pretest_lock.json` 在确认前落盘；不追求测试集调参。
- **M4 验收**：有且仅有一次正式新确认报告（必要技术重跑需记录修复而非丢弃旧结果）；所有配对同来源、同预算；空间、proxy、Teacher 指标区分。
- **M5 验收**：GO/NO-GO/INCONCLUSIVE 与证据相符；结论不越过 GT Oracle 和 dataset source 独立性限制。

如果数据访问受限，本次仍应完成 M0/M1，并交付 M2 的候选/失败台账以及如何合法扩大规模的具体建议。不可因为不能凑足 1000 来源而跳过机制诊断。

## 12. 首条给 Codex 的执行指令

> 阅读本 `AGENTS_VGGSS_FEATURE_DIAGNOSTICS_SCALEUP.md`，复用现有 AudioRouter、Stage1 Vision64 和此前 VGG-SS B0–B8 实验资产。**先完成 M0 和 M1**：核实旧 125 来源、重现并增加 SigLIP patch/196 token/Router logits/P/Z 逐层诊断，解释 GT Overlay 为什么没有稳定提高 ROI mass；不修改模型。然后从官方 VGG-SS 元数据建立去重候选池，合法获取更多媒体，分离新开发集和未检视独立确认集，按 M2–M5 的参数锁与配对统计规则执行。保持 Oracle 标签、代理任务、模型预测图三者严格区分；任何失败、数据不足或未运行结果必须如实落盘。
