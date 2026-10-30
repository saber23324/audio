# AGENTS — AudioRouter：更合适的下游 QA Benchmark 与 Grounded Audio 双图输入实验

> **执行对象**：Codex / 代码 Agent；以已有 AudioRouter / Gate2 / Dual-Head / VGG-SS 项目为基础开发，不新建脱离原仓库的实现。  
> **任务目标**：先摆脱 VGG-SS 类别代理问答的饱和问题，建立真正具有视觉或音视频联合证据需求的下游 QA；再比较 **原图单输入**、**热图替换原图**、**原图与 Grounded Audio Image 双图输入**，检验冻结 Visual-only VLM 是否能利用附加的视觉化音频证据。  
> **本轮性质**：数据调研与获取、评估协议搭建、冻结模型推理及输入形式消融；**不训练 LLM，不训练新 Gate/Router/Localizer**。如果后续确有收益，另外立项轻量 Adapter。  
> **文档定位**：在原有 `AGENTS.md` 之外使用本文件，不直接覆盖项目全局约束；冲突时优先遵守数据隔离、因果性、不可变资产和结果真实性要求。

---

## 0. 背景、旧结果和本次必须回答的问题

### 0.1 继承的事实，不得改写

原始 AudioRouter 使用冻结的 SigLIP / Projector / Qwen2-7B 等视觉语言主干，采用音频控制视觉值聚合；原视觉输入按每采样帧 196 tokens，经 Stage1 Vision-only 或 Stage3 AudioRouter 输出 64 个视觉接口 tokens。历史 Stage1 与 Stage3 来自**不同训练历史**，不可把跨 checkpoint 性能差异直接归因为音频。`extension.enabled=False` 恢复的是 Stage3 原路由，并非 Stage1 的 `always_off`。

已有 VGG-SS 开发与扩展实验在 **500 个新来源、139 道有效 source-category proxy** 上得到：

| 路径 | ROI mass | Source-category proxy | Teacher KL | 限定解释 |
|---|---:|---:|---:|---|
| B0 Clean RGB Stage1 | 0.32384 | 137/139 | 0.06257 | 分类接近饱和 |
| B1 GT-Box Overlay（单图替换）| 0.32648 | 134/139 | 0.10592 | 该冻结配置未显示可靠收益 |
| B2 Sham Overlay | 0.32610 | 137/139 | 0.09183 | 同等视觉扰动控制 |
| B4 GT Routing Bias | 0.44068 | 136/139 | 0.06043 | Oracle 位置先验直接增加 ROI 权重，具有机械性 |

特征诊断确认 B1 在 Pixel / SigLIP / Projector / 196 Token / Routing Logits / P / Z 全链条均有可测变化，**不能说 SigLIP 完全无响应**。但空间改变不等于正确定位，教师分布偏离不等于必然错误。VGG-SS 官方提供可见声源框，**不提供原生 QA**，由框渲染的热图属于 Oracle，不能声称可部署 Audio Grounding。

旧 125、500 以及 Gate2/LLP/AVQA 等已检视来源全部继续作为 `seen_sources`，不得再次作为新的独立 confirmation。

### 0.2 本轮四个主问题

1. **下游可检测性**：换成非饱和、原生带答案的视觉/音视频 QA 后，当前冻结 Vision-only Stage1 的表现如何？是否存在足够的配对 Rescue/Harm 空间？
2. **双图是否优于单图**：在保持原 RGB 不被覆盖的情况下，向视觉模型同时提供原图与一张 Grounded Audio Image，是否比直接替换原图更有效？
3. **Prompt 与视觉表示是否混淆**：收益来自附加图像内容，还是来自新 system prompt、输入顺序、重复图片、Token 数增多？必须用消融拆分。
4. **可部署与 Oracle 边界**：真实主结果只能使用冻结模型从合法音视频输入预测出的 heatmap；GT Box 可用于空间诊断及开发性 Oracle 上限，不能当作模型预测。

**第一阶段不以压缩率为创新**。但原生双图会增加视觉 token 和编码开销，必须报告并进行等 64-token 对照。

---

## 1. 权威研究依据与数据集优先级（先核对官方 README）

### 1.1 首选：MUSIC-AVQA（主要下游实验）

- 官方项目：https://gewu-lab.github.io/MUSIC-AVQA/
- 官方代码和标注：https://github.com/GeWu-Lab/MUSIC-AVQA
- CVPR 2022，约 **9,288 段视频、45,867 QA**；包含真实与合成演奏视频，问题包含 `Audio`、`Visual`、`Audio-Visual`，以及空间、时间等类别。
- 原始视频官网提供 Google Drive / 百度网盘下载入口；官方页面记录真实视频约 36.67 GB、合成视频约 11.59 GB。**不要假装已下载或全量可用**；先检查媒体和授权。
- 官方标注使用 `video_id`、`question_id`、`type`、`question_content`、`templ_values`、`anser`（原作者字段拼写即如此）；务必使用仓库中的问句模板展开代码，不能仅按字符串直接评分。
- **为什么首选**：示例问题包括“第一个发声的乐器是什么”“第二个发声乐器左侧的乐器是什么”“画面中正在发声的乐器数量”，这些任务同时需要音频引导对象选择和视觉空间/时间判断，更接近本项目的能力假设。
- 主实验在官方可用划分中，预先筛出 `Audio-Visual`，再按 `Spatial / Temporal / Counting / Existence`（以官方实际标签为准）分层；`Visual-only` 用作原视觉能力保真对照，`Audio-only` 仅用于能力边界分析，不作为主成功标准。
- **媒体有风险**：Google Drive、百度网盘权限/速率、旧 YouTube 片段不可访问、Synthetic 视频音频与画面组合方式。所有不可用项目记录，不得按模型预测结果筛选视频。
- **评分有风险**：原生多为开放式回答，必须优先复用官方评估的答案词典/标准化逻辑；不得随意生成四个选项并当作官方任务。可附加固定候选词表版本，但须明确是自定义评估。

### 1.2 第二优先：AVQA-Hard（独立一般场景）

- 数据索引：https://huggingface.co/datasets/gwkrsrch/avqa_hard
- 构造与评估代码：https://github.com/naver-ai/unimambamia-av
- 原始 AVQA 标注：https://github.com/AlyssaYoung/AVQA
- 已有音视频媒体缓存 / 合法公开媒体的来源示例：https://huggingface.co/datasets/juyil/AVQA-videos （第三方镜像，须审计许可/版本，不当作官方媒体）
- AVQA-Hard 有 **1,696 道**由 GPT-4o 单张静音中央帧回答错误筛出的题，**不自带视频、问题全文或答案**，必须通过原 AVQA `val_qa.json` 的题目 ID 与原始片段映射表连接。
- 不能把“单张中央帧答错”直接视作“音频一定有用”；也不能把 `Both` 直接当做视觉证据保留真值。报告 `View`、`Sound`、`Both` 和题目类型的真实分母。
- 由于项目之前反复使用 AVQA 来源，这里必须按原 YouTube ID、片段开始时间、媒体 hash/别名排除所有旧训练、开发、测试与已检视来源。**不能保证剩余未见来源数足够**；不够则仅报告样本获取不足，不可复用旧结果宣称新 confirmation。
- 多选题保留 AVQA 原四选项与官方答案，复用现有 Gold-letter NLL 与真实生成答案解析协议。

### 1.3 第三优先：Valor32k-AVQA v2.0（大规模和跨域补充）

- 项目：https://inesriahi.github.io/valor32k-avqa-2/
- 数据：https://huggingface.co/datasets/inesriahi/valor32k-avqa-v2
- ACM MM 2025；HF 版本约 28.9k 视频、225k QA，包含 `modality=visual/audio/audio-visual` 与 `description/action/count/temporal/location/relative-position` 类别；有四选项字段 `options` 和 `correct_answer_idx`。
- **重要限制**：该 HF 版本包含测试视频，但 train/val 视频未完全在同一发布包中；题目与答案由自动化生成，只有代表性子集经过人工核验。主研究中将其作为补充，必须报告可用媒体、标签质量风险和测试协议。
- 优先考虑 `audio-visual` + `location/relative-position/action/temporal`，同时保存 `visual` 作为无音频保真分层。不要使用 `caption`、`source_tags` 或 QA Gold 生成 Grounded Image（会造成答案泄漏）。

### 1.4 其他可选数据，不能自动并入主结果

- MUSIC-AVQA-R：https://github.com/reml-group/MUSIC-AVQA-R （NeurIPS 2024，基于 MUSIC-AVQA 试题改写/分布偏移；**不是来源独立的新视频验证**）。
- EgoSound：https://groolegend.github.io/EgoSound/ （CVPR 2026，约 900 视频、7315 音视频 QA，空间/时间/因果等问题；较多题依赖纯声音，适合作为扩展和能力边界，不是首个成功标准）。
- AVHBench：https://avhbench.github.io/ （音视频错配/幻觉鲁棒性；适合辅助评估热图错误提示）。
- VGG-SS：https://www.robots.ox.ac.uk/~vgg/research/lvs/ （仅空间定位/Oracle 诊断，不得当作新 QA）。
- MUSIC-AVQA-Hard：https://huggingface.co/datasets/gwkrsrch/music_avqa_hard （需要核查筛选规则，其标注可能偏向 Audio-only，不直接等价于视觉保留任务；不是第一优先）。
- NExT-GQA 等纯视频 Grounded QA 需先确认真实可用音轨与合法配对，不假设天然符合音频实验。

### 1.5 选择原则与阶段性目标

**建议主线：MUSIC-AVQA → AVQA-Hard → Valor32k（可得时）**。它们的任务、评分和媒体来源不同，**不能不加区分地将 Accuracy 混成一个总分**。

预声明数据范围建议：

| 数据池 | 来源 | 建议目标 | 用途 |
|---|---|---:|---|
| `music_dev` | MUSIC-AVQA 官方 train 内按原视频划分 | 200–400 独立来源起步 | 选 prompt、选择表示形式、验证输入实现 |
| `music_confirm` | MUSIC-AVQA 官方 test 且未见来源 | 若可得争取 500–1000 来源 | 一次性确认；数量不是承诺 |
| `avqa_hard_confirm` | AVQA-Hard 与旧项目所有来源去重后的可用视频 | 全部合法未见来源 | 跨场景确认；不够则明示 |
| `valor_confirm` | Valor32k 官方 test 中合法可用且未见来源 | 视媒体和协议而定 | 第三数据集验证 |

**不能为了制造足够的题目数量，拿同一视频的几十个 QA 当作几十个独立来源**。主要置信区间按视频来源聚类；可以报告 QA 总数但不能以题数冒充独立样本数。

在开发集上先做**固定全量 Baseline**，检查 Headroom：如果 Vision-only 准确率接近满分，则该类别只做保真/诊断，不以其作为主要提升主张。不可通过先运行 Baseline 再挑选其错题创建“独立测试子集”；允许按原数据集题目类型、预先固定的外部困难题索引选题。

---

## 2. 强制执行的实验边界

1. **所有核心模型冻结**：Stage1 Vision-only 64、Stage3 AudioRouter 64、SigLIP、Projector、LLM、现有 Dual-Head Gate、预训练 Localizer（如使用）均不训练、不覆盖 checkpoint。
2. **保持 Visual-only VLM 架构**：仅通过原生图像/视频输入通道提供 `RGB` 与 `Grounded RGB`；不把额外音频 token、答案文字、Sound Class 名称送进 LLM。通过固定、非答案相关的文字说明告诉 VLM 第二视图是一种提示，仍属于原有视觉-文本接口。
3. **严格时间因果**：采样帧在 `t`，Audio Grounder 只可使用 `[max(0,t-2s),t]` 音频；必须先裁剪原始音频，再按 Localizer 要求处理。不能用整段未来音频为当前帧生成图像。QA 问题只在回答时送入冻结 LLM，不能给在线预压缩和 Grounder 偷偷选择未来相关帧。
4. **热图来源分级**：`GT_ORACLE_BBOX`、`DENSEAV_PRED`、`EZVSL_PRED`、`OTHER_FROZEN_LOCALIZER_PRED`、`LEGACY_ROUTING_PRED`、`NO_MAP`。GT 不参与主部署结果；旧路由均值热图不能伪称独立声源定位器。
5. **数据干净**：音视频媒体完整、帧时间合法、音轨有质量状态、原 source ID/时间偏移/hash 与所有旧 manifest 做交集；第三方镜像需记录 revision、源与许可证/用途限制。
6. **旧结果全只读**：`artifacts/utility_gate/gate2_*`、`dual_head_*`、VGG-SS pilot、原 M5/新确认工件均禁止覆盖或重写。
7. **不据确认集调超参数**：prompt、图像顺序、Alpha、融合比例、候选数据类别、采样帧率、输出评分器先在 dev 锁定；确认集只执行已锁定的预声明主比较。确认结果不理想也不得事后换样本或顺序挑最优重测。
8. **不虚构支持**：没有真实双图处理接口、数据、预测图或官方评估实现时写 `not_run`+原因；不要用伪多图/未来帧/GT 混进主结果来填空。
9. **音频可用性不是语义对应**：PTS gap、silence、解码异常只决定数据质量；不自动当作音视频无关联真值。
10. **任务协议明确**：这是冻结模型的下游 QA 评估。若后续在训练集中为新模型学习 Gate，必须另立计划；本轮不拿测试 Grounding 标注生成真实部署输入。

---

## 3. M0：仓库/数据/推理接口审计（必须先完成）

### 3.1 审计当前项目，输出真实存在性

首先查找并核验，不假设任意目录已存在：

```text
artifacts/utility_gate/gate2_20261027/
artifacts/utility_gate/gate2_grounding_20261027/
artifacts/utility_gate/dual_head_proxy_20261028_v1/
artifacts/utility_gate/dual_head_20261027_v1/
artifacts/utility_gate/vggss*/
utility_gate/
/nvme_data/pkt/streaming/dataset/
```

- 记录旧 Stage1、Stage3、Gate checkpoint SHA、推理 Prompt、Image/Video Processor、sample FPS、gold NLL 与 QA 生成评分器；确认 `extension.enabled=False` 与 Stage1 `always_off` 的不同开关行为。
- 查找现有 VGG-SS 500 个确认来源及旧的 AVQA/LLP 896+ / 128+ / 197+ 来源等所有可见 manifest（具体数量以审计为准），建立 `all_seen_sources.jsonl`；核对原始 YouTube ID、片段开始时间、hash、镜像别名。
- 记录原 VLM 的可用输入接口：原生 `image`、`multi_image`、`video` 以及必要的聊天模板；验证 Processor、模型是否原生支持同时间点两张图片。
- **特别关注** LLaVA-OneVision 官方资料：原生多图通常每张图片产生约 729 visual tokens，而视频模式每帧通常约 196 tokens。实际依赖模型配置和 Processor 实现；请在当前环境上测量实际 shape，**不要凭文档假设双图等于两个 196-token 视频帧**。
- 检查从 `image1 RGB`、`image2 Grounded RGB` 进入 Vision Encoder 后，视觉 token 是否真正保留为两个不同视图，以及相关时刻标记是否被错误当成两个连续时间帧。

### 3.2 数据集资产审计与获取

**必须按顺序执行**：

1. 先只克隆官方音乐数据的标注/README，列出原生 train/val/test 及视频来源 ID；若发布包不提供 val，则在官方 train 内按源划分 `dev`，官方 test 只用确认。
2. 对建议主测类别统计题量、唯一视频量、标准答案词表、真正媒体可用率、音轨可用率、时长、视频中发声对象出现情况；查看 dev 中允许检查的个例，检查 question 类型与视频有无正确对应。
3. MUSIC 媒体优先下载 dev pilot 小批量（例如 32 来源），不先盲目下载全部 48 GB；媒体下载按官方链接，Google Drive/网盘需要认证就报告/使用授权方式。不要对需要授权的资源写破解脚本。
4. AVQA-Hard 先下载 HF **索引/元数据**，并与官方 AVQA 标注按题目 ID 关联；检查与已有 AVQA 历史 source 的交集。未见源不足时只将其列为外部开发诊断而非独立确认。
5. Valor32k 先下载 metadata，仅在确认合法可读的 test media、评分脚本与 prompt 规则后再建立正式任务。
6. 记录每个文件的下载地址、repo revision/commit、SHA256、媒体解码时长、分辨率、音频 sample rate、clip offset、权限/许可及失败原因。

**建议命令（使用当前环境可用工具，先 `--help` 核对接口，不假定 CLI 已安装）**：

```bash
# 只获取公开源码/标注；这是示意命令，不包含需要登录的实际视频
mkdir -p /nvme_data/pkt/streaming/dataset/audio_grounded_dual_image
cd /nvme_data/pkt/streaming/dataset/audio_grounded_dual_image
git clone https://github.com/GeWu-Lab/MUSIC-AVQA.git music_avqa_official_repo

# 仅 metadata，用于审查独立性和可用视频，不代表原始音视频已到手
# pip install huggingface_hub datasets 仅在独立依赖环境里按需进行
hf download gwkrsrch/avqa_hard --repo-type dataset --local-dir metadata/avqa_hard
# 对 Valor 先从 HF 官方页面/manifest 判断存储规模、媒体权限，再决定范围
```

**交付**：`reports/M0_repo_audit.md`、`reports/M0_dataset_research.md`、`dataset_inventory.csv`、`dataset_link_validation.md`、`seen_sources.jsonl`、`tensor_layout.json`、`sha_locks.json`、`download_failures.jsonl`。

---

## 4. M1：先更换下游 QA — 原始 RGB Baseline

> **这一阶段先不要实现双图。** 必须先确认新 Benchmark 能解决原 VGG-SS proxy 98% 饱和的问题。

### 4.1 数据适配器和统一样本格式

新建独立 QA Adapter，每个样本至少包含：

```json
{
  "dataset": "music_avqa",
  "source_video_id": "00000272",
  "question_id": "50",
  "source_split": "test",
  "video_path": "<resolved_existing_path>",
  "question": "Where is the first sounding instrument?",
  "answer_text": "right",
  "modality_type": "Audio-Visual",
  "reasoning_type": "Temporal",
  "allowed_video_prefix_end": null,
  "media_has_audio": true,
  "score_protocol": "official_music_avqa",
  "oracle": false
}
```

字段值为 **schema 示意**，不是实际数据行，不要把示例当作标注复制入训练或评价。Music 模板中的 `<FL>` 必须根据官方 `templ_values` 展开；视频中若没有明确的时间截止，则只以任务定义下允许的完整观察结束时刻作为回答时间，仍按每帧因果处理音频，不允许用未来声音修正早先生成的提示图。

### 4.2 原基线强制执行的路径

- `V64`: 原 Stage1 Vision-only，每采样帧 64 tokens。
- `V196`: Full Vision（不压缩），用于衡量视觉能力上限，不用它隐含证明音频有用。
- `A64`: 原 Stage3 AudioRouter 64（真实可用因果音频）。
- `A64-null`: 同一 Stage3 的合法 2s 零波形输入经 BEATs 处理（不是直接清空 BEATs embedding）。
- `G2`: 已冻结的 Dual-combined 或 Gate2（可选系统参照，不让 Gate 影响第一阶段数据选择）。

所有策略：同一媒体、同一视频采样时间戳、同一题目文本、同一个评估器、同一个 greedy decoding / generation length / answer normalizer。每条 QA 实际运行，不能复制旧 AVQA 结果。

### 4.3 QA 评分

- **MUSIC-AVQA**：使用官方 answer vocabulary / 预测与评分器；如模型输出开放文字，先核验官方或可复现的规范化逻辑，再另外报告严格 normalized exact match。不得事后手工把错误答案改为正确，也不得将 `question_type` 转成测试时额外提示。如果使用自定义 fixed-label constrained decoding，作为次指标并公开 label 集合与训练来源。
- **AVQA-Hard**：严格四选项 A/B/C/D，保存确定性生成正确性、四候选归一化 Gold NLL；禁止把测试答案放进 system prompt / 图片。
- **Valor32k**：根据 `options` 与 `correct_answer_idx` 评价，分别按 `visual` / `audio-visual` / `audio`、推理类别和质量评级报告。
- Gold NLL、Teacher KL、VGG-SS ROI Mass 都是不同定义，不得混用成真实问答正确率。

### 4.4 必须产出的基线诊断

- 每个数据集、每个官方模态子类、空间/时间/计数子类的 `n_questions`、`n_unique_sources`、accuracy / EM / NLL（按适用的评分器）；同源 bootstrap 95% CI。
- `V64` vs `A64` 的配对四象限：Both correct / Audio only win / Vision only win / Both wrong；输出 `Oracle two-endpoint accuracy` 只作诊断，不作为可部署分数。
- 按 QA 类型区分“声源作为视觉定位线索”与“纯音频语义问题”。视觉保留贡献的主评估以 `Visual` 和 `Audio-Visual` 中需要视觉空间/对象证据的题为主，不用 Audio-only 子集来证明视觉保留。
- 在 dev 识别是否存在足够的不饱和空间；如果某子类 `V64` 准确率超过 95%，继续透明报告但不能把它当主改善任务。

**输出**：`reports/M1_dataset_baselines.md`、`qa_eval/per_question_*.jsonl`、`qa_eval/paired_endpoint_matrix.csv`、`qa_eval/baseline_aggregate.json`。

---

## 5. M2：准备真正无 GT 的 Predicted Grounded Audio Image

本实验的 MUSIC-AVQA / AVQA-Hard **通常没有 VGG-SS 式逐帧可见声源 Box**。因此正式 QA 的提示图必须来自**冻结 Localizer 的预测**，不能复制 VGG-SS GT Box，更不能用 QA 答案决定高亮位置。

### 5.1 优先模型（不训练）

| Localizer | 官方项目 | 优先用途 | 限制 |
|---|---|---|---|
| DenseAV | https://github.com/mhamilton723/DenseAV | 第一候选，生成连续预测的声源区域图 | 核验官方权重和 2s 因果输入兼容性 |
| EZ-VSL | https://github.com/stonemo/ez-vsl | 第二候选，轻量对照 | 核验模型原生时长和预测预处理 |
| Legacy AudioRouter routing map | 现有项目 | 零新下载的代理 spatial map | **不是独立 Localizer，不可当成声源定位正确性** |

初始化必须记录模型名称、commit、checkpoint hash、音频采样率、音频窗口原生长度、图像 resize、输出热图分辨率、归一化和运行耗时。

### 5.2 音频与视频的时间匹配

对每个采样视频帧 `frame_time=t`：

```python
# 伪代码，仅表达时间契约
observed_pcm = decode_audio_until(media, t)
audio_window = observed_pcm[max(0.0, t - 2.0):t]
heatmap = frozen_localizer(rgb_frame_at_t, audio_window)
```

实现中不能按 Python sample arrays 用秒直接 slice；必须根据真实 sample rate/PTS 严格换算。不要先解码 `[0,video_end]` 后错误使用未来 context；缓存允许离线生成，但每张 heatmap 的因果来源必须可追溯。若模型要求 10 秒音频，只允许使用**历史音频或零左填充**，先评估这种分布变化是否使预测图退化；不能为了匹配模型默认条件而读取未来的 8 秒音频。

- 音频无效或缺失：记录可用性掩码；主部署分支回退原 RGB，不产生伪“可信预测”。
- 输出连续数值 `H_t ∈ [0,1]^{H×W}`，不是仅保存带图例、标题的 JPEG。
- 无可用预训练权重或合法 2s 输入失败：该预测分支标记 `not_run` 并说明原因，可先完成 M1 和双图输入工程单测；不能把 GT 或随机热图代替真实预测结果写进部署主表。
- Oracle VGG-SS 可在**已检视开发来源**上帮助检查投影与输入拼接，必须单列 `oracle=true`；不得将预测模型是否好/坏在旧测试 Box 上调完参数后把旧来源称为新确认。

### 5.3 Grounded Image 构造

- 原图 `I_t` 完全保留，不修改像素。
- 预测图 `H_t` 在对应原图 ROI 内生成高亮版 `I_t^G`，允许开发集上预声明测试：轻度透明热图、细线轮廓、背景抑制三种形式；**不能在确认集挑胜者**。
- 颜色和 Alpha、阈值、平滑、缩放、最大高亮面积、是否标注小图角标（无任何 sound class 名称）都要在配置中固定。
- 所有地图均附 `map_origin, source_id, frame_time, audio_start, audio_end, localizer_checkpoint, alpha, oracle, sha`，与输出 PNG/NPZ 一同保存。
- 禁止使用 QLabel/GT `answer`、GT Bounding Box 和包含正确声源文本的颜色图例生成正式预测提示。

**输出**：`maps/predicted/*.npz`、`images/grounded/*.png`、`map_provenance.jsonl`、`reports/M2_localizer_and_preprocess.md`、`figures/map_examples.png`。

---

## 6. M3：原生双图输入 + System Prompt 消融（核心）

### 6.1 主要科学对照

| Run ID | 视觉输入 | 提示设置 | 输入预算性质 | 核心用途 |
|---|---|---|---|---|
| `D0_RGB_legacy` | 只有原 RGB | 原始项目 prompt | 64 token/frame | **主要 Vision-only 基线** |
| `D0_RGB_neutral` | 只有原 RGB | 新中性 prompt | 64 token/frame | 控制 prompt 更改 |
| `D1_OverlayOnly` | 仅预测 Overlay 替换 RGB | 中性 prompt | 64 token/frame | 单图替换对照 |
| `D2_RGB_RGB` | 原 RGB + 原 RGB 拷贝 | 双图说明 prompt | 原生双图 | 排除仅增加一张图的收益 |
| `D3_RGB_Grounded_neutral` | 原 RGB + **预测** Grounded RGB | 双图中性 prompt | 原生双图 | 纯第二视图内容收益 |
| `D4_RGB_Grounded_guided` | 原 RGB + **预测** Grounded RGB | **双图提示解释 prompt** | 原生双图 | **主双图方法** |
| `D5_Grounded_RGB_guided` | Grounded RGB + 原 RGB | 同 D4 | 原生双图 | 图像顺序偏置 |
| `D6_RGB_Sham_guided` | 原 RGB + 伪随机/等面积 Sham | 同 D4 | 原生双图 | 避免把多图扰动当声音信息 |
| `D7_RGB_HeatmapOnly` | 原 RGB + 纯预测灰度热图图片 | 同 D4 | 原生双图 | 对比 Grounded RGB vs Raw Heatmap |
| `D8_RGB_PredCrop` | 原 RGB + 预测区域 crop | 同 D4 | 原生双图 | 对比 ROI 细节与 Overlay |
| `D9_OldAudioStage3` | 原始 Stage3 | 旧 prompt | 64 token/frame | 保留旧 AudioRouter 参照 |
| `D10_OracleVGGSS` | VGG-SS 原图 + GT 框生成 Overlay | 双图 prompt | Oracle，仅开发诊断 | **禁止进入正式 QA 主表** |

`D2` 与 `D6` 很重要：如果 D4 只是因为额外图像、prompt 或色彩干扰提高分数，这两个控制可能得到相似收益。**D5 调换图像顺序时必须同步改写 View 1 / View 2 的角色描述**，不能让原 prompt 错误声称第一张是原图；此时记录 `S2_swapped`，比较中明确“交换输入顺序 + 同步更新视图标号”的共同变化。`D7` / `D8` 也必须分别用准确的 `S2_heatmap` / `S2_crop` 说明第二张是灰度图还是裁剪图，不得错误称为原始画面的 RGB Grounded View；这两个设置属于形式对照而不是纯内容同 prompt 对照。

阶段优先顺序：先实现 `D0/D1/D2/D3/D4/D6` 的真实前向与答案评分；再补顺序/heatmap/crop；不允许为了追结果擅自添加大量变体。

### 6.2 System Prompt：只轻微改变，不添加音频答案

必须把 `system_prompt_version` 写入结果文件。最终模型可使用英语（数据集 QA 主要为英文），不要改变选项与答案本身。

**S0：原始 Baseline**

- 原样继承当前代码中已经固定的 System/User Prompt；不能为了让 B0 变弱而简化其指令。

**S1：新的中性视觉 QA Prompt（单图/双图均可使用）**

```text
You are a visual question-answering assistant.
Use the provided visual evidence to answer the question accurately.
Do not invent details that are not supported by the images.
Give only the answer in the format required by this benchmark.
```

**S2：双图 Grounded Visual Cue Prompt（主实验）**

```text
You may receive two views for the same video moment.
View 1 is the original, unmodified RGB frame.
View 2 is a derived view of that same frame with a visual highlight that
may indicate a region associated with an observed sound event.
The highlight is only a hint and may be incorrect. Use the original view
for object identity, color, counts, and scene context. Consult the second
view when spatial focus may help. Do not treat the highlighted region as
a guaranteed sound source or as the answer to the question.
Reason from the available visual evidence and respond only in the required
answer format.
```

**S3：双图中性 Prompt（用于 D3）**

```text
Two images may describe the same instant from two visual renderings.
They are not two consecutive time steps. Use both as visual evidence,
prioritizing the original scene content. Follow the question and return
only the required answer format.
```

**Prompt 与图像顺序的一致性检查**：任何图像数量或角色变化均要使视图标签准确；`D5` 的 `S2_swapped` 示例：“View 1 is the derived highlighted view; View 2 is the original RGB frame.”；其余内容尽量不变。不能为了固定 prompt 字面相同而故意给模型错误角色说明。`D7/D8` 的角色描述同理准确，并在分析中将这些跨形式对照标为“prompt 也随输入格式适配”。

**必须有 Prompt 对照**：

- `D0_RGB_legacy` vs `D0_RGB_neutral`：判断新的通用 QA 文字是否本身改变准确率；
- `D3_RGB_Grounded_neutral` vs `D4_RGB_Grounded_guided`：判断“解释第二图含义”是否额外有效；
- `D2_RGB_RGB` vs `D4_RGB_Grounded_guided`：判断真正的 Grounded Image 是否优于仅增加重复图。

无论哪种提示，不可说“红色区域一定对应正确发声物体”，不可额外提供正确乐器名称、声源坐标、事件时间或来自 Benchmark 的标签。

### 6.3 必须核验原生多图接口，不能把第二图误当未来视频帧

LLaVA-OneVision 公开文档指出：视频帧与原生多图的视觉 Token 布局不同（视频通常每帧 196、原生多图每张图片通常更多）。因此：

1. 用一个有真实媒体的 dev fixture 验证 `processor`/聊天模板对两图输入的张量 shape、image count 和生成 `input_ids`，保存原始 token 统计。
2. 若接口为 `image,image` 双图，须在模型支持的 **native multi-image** 模式下真正分别编码，且明确这两幅图是同一时刻；不可把第二图伪装为 `t+1` 的下一视频帧。
3. 如果现有 video runtime 不支持每个时刻 2 个 image slots：先在 single-time-step / 中央帧的原生多图接口完成双图 QA feasibility；全视频方法另外实现可信时序打包，不能把单帧双图结果称为流式视频主结果。
4. 对多采样帧，记录每个时间点的双图顺序、时刻、视觉 token 数及 LLM 位置编码；如果工程无法在不改变时序含义下实现，则标记 `not_run: multi_frame_pairing_interface_unavailable`。
5. `D5` 只改变双图顺序，不改变图像对应关系。任何图片顺序变化造成的回答差异独立报告。
6. 禁止在 Prompt 中插入假音频 transcriptions 或用数据集 `type=Both` 直接告诉 LLM 当前答案需要声音。测试时只能使用题目原本包含的文字。

### 6.4 必须有 **同 64-token 输出** 的对照（第二阶段）

原生双图可能显著增加 LLM visual token，因此先观察其 feasibility，但不能将收益全部归因于融合。至少实现一个输出恒定为每采样帧 64-token 的冻结对照：

```text
RGB -> Frozen Vision Encoder -> Stage1 64-slot Resampler -> Z_rgb [64,d]
Grounded RGB -> Same Frozen Encoder -> Same Stage1 64-slot Resampler -> Z_g [64,d]
Z_mix = (1-alpha) * Z_rgb + alpha * Z_g     # [64,d]
Frozen LLM receives only Z_mix
```

- `alpha=0` 必须逐值复现 Stage1 RGB 基线；其余 alpha 只在 dev 预声明候选网格（例如 0.25、0.5）中比较，冻结后才能测确认集；数值仅为待试选择而非已验证最优值。
- 使用**相同 Stage1 Query Bank** 逐图编码，保证 output slots 定义一致；如果当前 Stage1 Resampler 无法稳定处理两张图，不能拿两套独立训练的 Query Bank 生硬相加。
- 这个方案是 **dual-image feature mix into 64 tokens**，**不是**原生“两个 image token 串联”的等价重现；结果必须单独命名 `D11_equal64_feature_mix`。
- 它保持 LLM 视觉 token 预算为 64，但额外运行了第二次视觉编码，必须分别报告 encoder 成本和端到端延迟。此时 `Z_mix` 含 Grounded Image 的视觉特征，不能再声称所有值都来自未修改 RGB 的凸包。
- 如果同预算实现对原模型其他组件有侵入，先交付原生双图和完整测量，不可静默跳过而仍声称等预算公平。

---

## 7. M4：运行顺序（先 Dev，后冻结 Confirm）

### 7.1 Pilot 先行

1. `MUSIC-AVQA` 合法 dev 来源中固定 seed 1234，先选 32–64 个独立视频来源进行接口冒烟测试；只要有真实音视频和可解析 QA，**不根据模型正确性挑样本**。
2. 运行 `D0/D1/D2/D3/D4/D6` 和 `D11`（若可行），验证时间窗、prompt、真实生成、图像顺序、输出解析、有效音频/地图状态及统计。
3. 先对模型是否真的看见第二图做输入对照：`D4` vs `D2`、随机化第二图、交换顺序；如果多图 token 的记录没有差异，首先修工程问题。
4. 不以 32–64 来源 pilot 的准确率作为统计结论，只输出真实错误案例与张量一致性。

### 7.2 Dev 调参限定范围

- 只允许在 `music_dev` 上选择**表示类型**（Grounded RGB、Raw Heatmap、Predicted Crop）和**一次轻量 prompt 配置**。最多保留预声明的 2–3 个候选，不进行无界的 Prompt Search。
- `alpha`/热图阈值、降噪与背景颜色不能从 confirmation 反向调节。
- 可以做按题目类型的结果诊断，但不得根据 dev 某一模型最擅长的类别在 confirm 中选择性报告主指标。
- 对已曝光过的旧 VGG-SS 500 来源只作 Oracle 工程回归，不转为新的 QA Confirmation。

### 7.3 确认集锁定

运行前生成 `pretest_lock.json`，必须锁定：

```json
{
  "experiment_id": "audio_grounded_dual_image_qa_v1",
  "primary_dataset": "music_avqa",
  "primary_comparison": ["D4_RGB_Grounded_guided", "D0_RGB_legacy"],
  "primary_subtask": "Audio-Visual spatial-temporal",
  "model_checkpoints": {},
  "localizer_checkpoint": null,
  "processor_version": "<actual_version>",
  "system_prompt_versions": ["S0", "S2"],
  "selection_manifest_sha256": "<actual_sha256>",
  "code_sha256": "<actual_sha256>",
  "seed": 1234,
  "frozen_hyperparameters": {},
  "score_protocol": "<actual_official_or_declared_custom_protocol>"
}
```

此 JSON 是字段模板，不得将 `<placeholder>` 写作实际完成值。选定后才运行新来源 confirmation，一次性评估预声明主对照与必要控制。若继续研究新增候选，则建立新的实验/数据来源版本，不得复用已看过的确认集充当首次检验。

---

## 8. M5：评估指标与失败归因

### 8.1 下游主指标（首先看是否真正提分）

- `QA_Accuracy`：AVQA-Hard、Valor32k 等四选项任务；`Answer_EM` / 官方 answer-vocabulary accuracy：MUSIC-AVQA（以实际官方脚本和协议为准）。
- `Gold NLL`：当模型答案空间和评分规则明确时记录；对 MUSIC-AVQA 自由文本不应将无对齐的字符串概率冒充标准规范的分类 NLL。
- `Rescue / Harm`：对同一 QA 比较 `D4` vs `D0`、`D4` vs `D2`、`D4` vs `D6`；记录两者都对、D4 独赢、D0 独赢、两者都错。
- `Prompt-only gain`：`D0_RGB_neutral - D0_RGB_legacy`。
- `Second-image gain`：`D4 - D2`（两图数相同）。
- `Grounding-specific gain`：`D4 - D6`（同等提示成本的空间错误对照）。
- `Order sensitivity`：`D4 - D5`。
- `Extra-token benefit vs equal64`：原生两图 `D4` vs 等 64-token `D11`，严格标注输入形式/压缩差异。

对所有对照按**同一视频来源配对**，报告独立视频数、QA 数、绝对分数、百分点差、源聚类 bootstrap 95% CI（建议固定 2000 次）及真实失败案例。相同源的多道题要保留聚类，不能把问答数当成独立视频数。

### 8.2 空间、提示和系统成本的辅助指标

- `VGG-SS`（独立诊断）：Localizer predicted map vs 官方 bbox 的面积校正 ROI / SoundMass、空间排序与定位正确性；Oracle 热图不能作为自身定位成功证据。
- 在 MUSIC 中若无 bbox：不能凭输出热图自行声明 Localization Accuracy；只能报告可视化、已有模型自身的分数与下游 QA，不凭肉眼选出好图作为定量样本。
- Pixel/Feature 层：与原 M5 诊断兼容的 SigLIP feature distance、Projector/Pooling、`R/P/Z` 变化、Teacher KL，仅作为机制分析。双图原生串联时不要伪造单 `P` 与 VGG-SS 64-query 的同定义。
- 推理成本：每条真实媒体统计 frame decoding、audio decoding、Localizer、额外视觉编码、64-token resampling、LLM prefill、TTFT、整体 latency、GPU peak alloc。缓存时延与全管线时延必须分列；两图与一图视觉 Token 真实计数必须报告。
- 鲁棒性：错位、Sham、无音频、错误热图与异常音频下能否回退；当前冻结 Gate 若参与选择，须在评估前锁定其阈值，不能依据 confirmation 反馈重训。

### 8.3 科学解释等级

| 结果 | 可以支持什么 | 不可以宣称什么 |
|---|---|---|
| D4 优于 D0，但不优于 D2 | 可能受益于多图冗余/增加输入 | 有效 Audio Grounding |
| D4 优于 D2、D6，且在新来源 CI 支持 | 预测音频提示有超出双图/随机提示的证据 | 已证实传感器专用语义进入 LLM |
| D4 原生两图提高、D11 等 64 不提高 | 额外视觉输入或跨流融合可能有用 | 同等视觉预算下路由更优 |
| 仅 GT Oracle 双图有效 | 正确空间提示具有可能的上限价值 | 已经实现 GT-free 音频推断 |
| 只有 Gold NLL 小幅改变 | 有连续指标探索性信号 | 已获得稳定 QA Accuracy 提升 |
| Visual-only 子集明显退化 | 可能过度依赖提示图或视觉扰动 | 所有下游理解能力均改善 |

---

## 9. 代码布局、建议 CLI 与工程单测

按实际仓库目录调整，不得改写旧的同名模块：

```text
utility_gate/
  grounded_dual_image/
    dataset_registry.py
    music_avqa_adapter.py
    avqa_hard_adapter.py
    valor32k_adapter.py
    source_split_audit.py
    audio_map_provider.py
    grounded_image.py
    multi_image_runtime.py
    system_prompts.py
    eval_qa.py
    analyze_results.py
    latency.py
    report.py
configs/grounded_dual_image/
  datasets.yaml
  pilot.yaml
  dev.yaml
  confirm_LOCKED.yaml
tests/
  test_grounded_dual_image.py
artifacts/utility_gate/audio_grounded_dual_image_qa_<run_id>/
  audit/
  manifests/
  inputs/
  maps/
  qa_eval/
  figures/
  reports/
  pretest_lock.json
  commands.json
  final_audit.json
```

建议提供以下**命令级入口**（仅定义意图，具体参数名由 Codex 根据仓库实现，不可把伪命令写成已执行）：

```bash
python -m utility_gate.grounded_dual_image.dataset_registry --audit
python -m utility_gate.grounded_dual_image.eval_qa --config configs/grounded_dual_image/pilot.yaml
python -m utility_gate.grounded_dual_image.eval_qa --config configs/grounded_dual_image/dev.yaml
python -m utility_gate.grounded_dual_image.eval_qa --config configs/grounded_dual_image/confirm_LOCKED.yaml
python -m utility_gate.grounded_dual_image.report --run-dir <actual_run_dir>
```

### 自动测试必须覆盖

- 原 Stage1 `B0`/原 Stage3 fixture 在新功能关闭时 `P/Z/generated_ids` 逐值一致，不修改原 checkpoint。
- 一张图 vs 两张图实际 `processor` 编码数与 LLM visual token 计数正确；双图被识别为同一时间点的两个视图，而非前后两帧。
- 新双图 prompt 不携带 gold 答案、官方类别/声源名称、GT bbox；两种图像顺序的实现一致且有独立测试。
- `alpha=0` 的等 64-token mixture 与原 Stage1 逐值一致；特征 shape `[64,d]`，避免不同 Query Bank 混合。
- 本地验证某个已采样时间点添加未来帧/音频，不改变当前 map / Z / gate 决策；两秒原始音频先按时间截断再编码。
- 错配和随机热图独立于同一题目的答案和 GT bbox，不得从 test 标签构造。
- 音频缺失、PTS 错误、样本时长不足、empty question、错误编码版本、非有限 heatmap 的处理可追溯。
- 数据角色、原 source 去重、下载版本 SHA、测试前阈值锁与旧 checkpoint SHA 完整。
- 真正调用 GPU 生成至少两个固定 fixture（单图/双图），比较重跑一致性；CPU stub 测试不能替代真实 VLM 前向。

---

## 10. 研发里程碑与交付（严格执行）

| 阶段 | 最低交付 | 完成标准 |
|---|---|---|
| **M0** 调研 + 数据/仓库审计 | `M0_dataset_research.md`、可用性与来源排重清单 | 确认 MUSIC-AVQA 等题目/媒体真值，不误用 VGG-SS proxy |
| **M1** 新 QA 基线 | V64 / V196 / A64 / A64-null 真实生成、逐题表 | 原始 VLM 下游分数、答案分布、headroom 清晰 |
| **M2** 预测提示图 | Frozen Localizer 真实时序热图、可视化和 provenance | 正式图无需 GT/Gold，满足 `[t-2,t]` 因果 |
| **M3** 双图 + Prompt | D0–D6 真实双图、prompt/顺序/重复图对照，等 64 路径 | 原生图像数真实、输出正确、无信息泄漏 |
| **M4** Dev 消融 | prompt/图像方案选择记录、训练之外的固定参数 | 所有开发尝试完整记录，明确最终锁定主方法 |
| **M5** 新确认 + 统计 | 配对 QA、NLL/EM、效率和错误案例 | 未见来源一次性确认，完整 CI 和限制 |
| **M6** 决策 | `GO_NO_GO.md`、报告、代码和审计 | 根据下游视觉 QA 而不只是 ROI/Teacher 判断是否继续 |

### 10.1 GO / NO-GO 的判断

**优先考虑 GO**：在合法新来源上，`D4` 不仅超过 `D0`，还超过 `D2`（重复图）和 `D6`（等面积错误提示），存在可核验的 Rescue/Harm 优势，且在 `Visual` 子集没有不可接受的负迁移；如等 64-token D11 也获益，则证据更强。

**可以继续探索但不能宣称已成立**：原生双图胜过单图但不能超过重复图，或者只有 Oracle 声源位置才获益；需要先检查多图冗余、Localizer 预测图质量和输入说明，而不是立即训练 Gate。

**NO-GO（本轮锁定配置）**：在足量、合理难度且来源独立的 QA 上，双图提示不能超过重复图/错误位置图，或者明显伤害视觉子任务。**仅样本不足、媒体不可用、任务仍饱和时写 INCONCLUSIVE，不得以未测足数据提前宣布方法失败。**

### 10.2 不可伪造的验收报告格式

每个阶段必须写：

```text
STATUS: completed / partial / not_run / blocked
WHY: ...
ACTUAL_VIDEO_SOURCES: ...
ACTUAL_QA_PAIRS: ...
ACTUAL_AUDIO_VALID: ...
TOKEN_COUNTS: ...
PROMPT_VERSION: ...
GT_ORACLE_INPUT: true/false
MODEL_SHA256: ...
DATA_SHA256: ...
TEST_SELECTION_LOCK: ...
COMMANDS_AND_LOGS: ...
OBSERVED_NUMBERS: ...
LIMITATIONS: ...
```

失败日志、下载失败、模型不支持双图、OOM/时间戳问题、测试脚本和环境更改都须记录；不得把没有运行的 Accuracy/CI 填成 0 或编造最佳结果。

---

## 11. 参考论文与官方资源

1. **MuseVLA** — https://arxiv.org/html/2606.17598 — 原图与 Grounded Sensor Image 共同参与后续动作模型；它进行 VLM/VLA 联合训练，**不是冻结双图的性能保证**。
2. **LLaVA-OneVision 官方文档** — https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md — 原生多图/视频输入形式和 token 数差异。
3. **MUSIC-AVQA** — https://gewu-lab.github.io/MUSIC-AVQA/ — 视觉/音频/联合空间时间 QA 和官方媒体链接。
4. **AVQA-Hard / UniMambaMia-AV** — https://huggingface.co/datasets/gwkrsrch/avqa_hard ; https://github.com/naver-ai/unimambamia-av — 官方难例索引及 AVQA 题目的关联方法。
5. **Valor32k-AVQA v2.0** — https://huggingface.co/datasets/inesriahi/valor32k-avqa-v2 — 视觉/音频/联合和空间/时间四选项任务。
6. **MUSIC-AVQA-R** — https://github.com/reml-group/MUSIC-AVQA-R — 语言改写与偏差压力测试，非新的独立视频。
7. **EgoSound** — https://groolegend.github.io/EgoSound/ — 声音相关空间/时间/因果 QA。
8. **VGG-SS** — https://www.robots.ox.ac.uk/~vgg/research/lvs/ — Sound Source Grounding GT，仅做 Oracle/定位诊断。
9. **DenseAV / EZ-VSL** — https://github.com/mhamilton723/DenseAV ; https://github.com/stonemo/ez-vsl — 有可获取预训练定位器的候选实现。
10. **AVHBench** — https://avhbench.github.io/ — 无关声音/跨模态冲突的额外鲁棒性评价。

---

## 12. 发给 Codex 的第一条任务

> 阅读 `AGENTS_AUDIO_GROUNDED_DUAL_IMAGE_QA.md`，结合当前仓库的 AGENTS.md、旧 Gate2/Dual-Head 和 VGG-SS M5 工件开展工作。**先完成 M0、M1**：审计 MUSIC-AVQA / AVQA-Hard / Valor32k 的真实标签与媒体可用性，严格排除所有已检视来源，跑通冻结 Stage1 Vision-only 与 Stage3 AudioRouter 的新下游 QA 基线。随后执行 M2/M3：在不使用 GT/答案的情况下由冻结 Localizer 生成合法两秒因果预测热图，真实实现原图 + Grounded Image 的原生双图输入，并进行 system prompt、重复图、Sham、图序和同 64-token 的必要对照。只用开发集选择配置，锁定后运行独立确认集；所有旧模型和旧报告只读。输出真实数值、全部失败和 GO/NO-GO，不允许用 VGG-SS 类别代理冒充正式 QA。
