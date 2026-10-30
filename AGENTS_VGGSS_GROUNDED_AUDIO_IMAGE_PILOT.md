# AGENTS — VGG-SS Grounded Audio-Image 轻量可行性实验

> **给 Codex / Coding Agent 的可执行任务书**  
> 版本：2026-10-28，Pilot v1  
> 适用对象：**现有 AudioRouter / Gate2 / Dual-Head 仓库**，不是新建工程。  
> 主问题：**将声源空间信息表示成 RGB 可处理的图像提示，再走冻结的 Vision-only VLM 的视觉编码与 64-token resampling，能否比不使用此提示的 Vision-only 路径保留更多与可见声源有关的证据，并在可定义的下游视觉任务上产生收益？**  
> **第一轮不训练新 Gate、Localizer、Adapter、VLM；不以动态压缩率、IMU/GUI 扩展为目标。**

---

## 0. 最重要的实验解释约束（先读完）

1. **VGG-SS 官方数据提供的是视频中间帧的可见声源 Bounding Box 与类别，而不是官方可在线部署的“真实音频热图”。** 所谓 VGG-SS *GT Heatmap* 是**利用这些标注框生成的 Oracle 空间提示**；其来源是评估标签，绝不能把 Oracle 结果称为“音频模型推断的热图”或“真正可部署的音频增益”。
2. 如果仓库中已经有 **DenseAV / EZ-VSL / 旧 AudioRouter 产生的预测 heatmap**，可以优先复用，但必须审计每张图的 `source_model`、checkpoint、输入音频时间范围、图像时刻、是否接触过 GT 框、分辨率和生成命令。**旧 `routing heatmap` 不等于声源 Grounding 真值**，也不等于独立 AV Localizer。
3. **VGG-SS 不是现成的视频 QA benchmark。** 它能直接支持声源空间保留/定位实验；若要在 VGG-SS 内比较“Vision-only QA 准确率”，只能构造**事先明确标为 VGGSound-label-derived proxy QA** 的受限任务，或使用与其真正对齐的独立 QA 标注。不得虚构 VGG-SS 官方 QA，也不能用 `bbox` 生成正确答案后将 `bbox` 叠加到输入，声称证明了正常音频推理提升。
4. 本轮必须把三种证据分栏报告：`oracle upper bound`（GT 框生成提示）、`predicted audio-derived cue`（无需任何 GT 的预测热图）和 `downstream task utility`（有合法答案/参考的结果）。如果只做到 Oracle，结论就是**“理想空间提示是否值得继续”**，不是最终方法有效。
5. 参考 **Microsoft MuseVLA** 的 *Grounded Sensor Image* 表达形式，但不加载 MuseVLA 的机器人策略或假定其训练成果可以零样本迁移。MuseVLA 本身不是“拿单通道音频转频谱图直接给冻结 VLM”的同条件实验。
6. **主系统始终为冻结的 Visual-only VLM**：禁止在 LLM 侧新增独立 Audio token、隐藏音频特征或修改文字 prompt 以传递 GT 位置信息。主比较固定为 **64 个视觉 token/标注帧**，可单列 196-token Full Vision 参考。
7. 历史结论不可更改：Vision-only stage1 与 AudioRouter stage3 来源于不同训练历史；`extension.enabled=False` 恢复原 Stage3 AudioRouter，并不等于 `always_off` 的 Stage1 视觉分支。所有实验明确列出使用哪个 checkpoint 和分支。
8. 保护所有旧模型、阈值、源码快照与实验归档：**只能在新目录写结果，不覆盖既有文件或 checkpoint。**

## 1. 本轮范围与成功定义

### 1.1 必须回答的四个问题

- **Q1：Oracle 可行性。** 将真实声源框变成空间热图（不改变 prompt），叠加到 RGB 后走冻结 Vision-only 路径，是否比纯 RGB Vision-only 更好？答案只能算 *GT-assisted diagnostic*。
- **Q2：融合方式。** 直接 RGB overlay 与仅将热图作为 routing bias，谁更好？后者继续保留原生 RGB value path `Z=P@V`。
- **Q3：是否只是画上一个醒目区域？** 与等面积随机位置、均匀热图、随机颜色/同 alpha、只画 bbox 边框等控制比，是否只有正确的空间提示起效？
- **Q4：可部署前景。** 如果已有无需 GT 的真实音频预测图（旧 AudioRouter 热图或 DenseAV/EZ-VSL 预测），能否复现与 Oracle 相同方向的 ROI 保留收益，并在有合法代理标签的下游任务上改善 Vision-only？

### 1.2 不属于第一轮目标

- 训练或调节 Dual-Head Gate、重新训练 AudioRouter；
- 训练大模型、重新训练 SigLIP；
- 在 VGG-SS 的人工 GT 框上反复优化参数后，宣称独立测试提升；
- 直接给模型视觉提示“正确声源名称”，然后用同一名称评价 QA；
- 使用音频 `[t,t+2]` 等未来窗口来生成主结果热图；
- 为没有真实答案的样本造出“VGG-SS QA accuracy”。

### 1.3 核心验收

先运行冻结模型的 `RGB clean`、`GT-grounded RGB`、`equal-area sham RGB`、`GT routing bias` 四个分支，全部有真正的**原生 VLM 前向输出**。不允许只展示热图，不跑输出。若有可追溯的预测图，再补 `predicted map` 两个分支；**若无可部署预测图，如实标记 not_run**，不要为了制造成功在测试集上训练预测器。

## 2. 研究参考（区分「要模仿」和「不能照抄」）

| 工作 | 来源 | 要复用的思想 | 本轮不照搬 |
|---|---|---|---|
| **MuseVLA**（2026，Microsoft） | https://arxiv.org/abs/2606.17598 ; https://github.com/microsoft/MuseVLA | *Grounded Sensor Image*: 将已对齐的传感器空间图像融合到 RGB 输入；先选传感器再利用 | 它用麦克风阵列/传感器校准和训练过的 VLA；这里仅做 GT/预测声源图的冻结视觉模型诊断 |
| **Localizing Visual Sounds the Hard Way**（CVPR 2021） | https://www.robots.ox.ac.uk/~vgg/research/lvs/ | VGG-SS 中间帧可见声源 bbox、类别、数据格式和评价 | bbox 不是模型生成的热图，不能在真实部署分支使用 |
| **DenseAV**（CVPR 2024） | https://github.com/mhamilton723/DenseAV | 有公开预训练音视频细粒度 Grounding 模型，可以生成**预测**热图 | 首轮不要求新下载；若预测图缺失，可作为后续可部署分支的默认候选 |
| **EZ-VSL**（ECCV 2022） | https://github.com/stonemo/ez-vsl | 公开预训练声源定位和导出预测可视化的代码 | 可选的第二个冻结 localizer 对照；不要把其默认较长音频窗口算作 2s 因果测试 |
| **原 AudioRouter** | 当前仓库、只读 | 原 `P@V` 路由、SigLIP、stage1/stage3、已有 VGG-SS 空间实验 | 不混合两个独立训练的 Query Bank；不改旧模型配置 |

> 注：所谓 **Grounded RGB** 在本 pilot 中是用 `heatmap` 作为空间辅助图形，让视觉 Encoder 重新读取；它不意味着冻结模型已经学会了真正的声学语义。若仅用 GT bbox 构造热图，更应称为 **Oracle visual spatial cue**。

## 3. 项目审计 / 数据审计（M0，必须先做）

### 3.1 查找现有资产，不假定路径真实存在

优先检查：

```text
artifacts/utility_gate/gate2_grounding_20261027/
artifacts/utility_gate/gate2_20261027/
artifacts/utility_gate/dual_head_proxy_20261028_v1/
utility_gate/grounding/
utility_gate/gate2_runtime.py
/nvme_data/pkt/streaming/dataset/
```

- 检查 VGG-SS 是否已有先前的 128 源/109 个有效音频配对和相应图片、盒子、原生 `P:[64,196]`、`V:[196,3584]` 或缓存可复用；存在则复用，不重复下载。
- 找到 VGG-SS bbox 的原生 schema、单位、具体标注帧时刻（官方为**视频中间帧**）、媒体路径、视频 ID 和原始 YouTube 来源；多 bbox 需要取并集，不自动把框外区域当负声源对象。
- 检查 `heatmap` 资产属于：`GT_BOX_RENDERED`、`LEGACY_ROUTING_PRED`、`DENSEAV_PRED`、`EZVSL_PRED`、`UNKNOWN`。为每条资产存 provenance JSONL；`UNKNOWN` 不进入主实验。
- 核对视觉 Token 映射：已知历史审计为 384x384 缩放、27x27 有效 Patch14 卷积、插值成 14x14、`align_corners=False`、row-major，右/下有边界有效域差异；**以本仓库实际实现为准重验**，不要简单以 14x14 patch 方块替代真实映射。
- 核对 Stage1 视觉分支原始输入与输出，`extension.enabled=False` 恢复的 Stage3，不是 Stage1。
- 固定原始输入、随机种子、评估代码、环境 SHA、checkpoint SHA；记录历史使用过的来源 ID，避免把已看过的结果称为新的独立验证。
- 每条媒体真正重新定位中间帧的时间戳。若视频被截取成 10 秒，默认中间帧约在片段相对 5s，但要用实际媒体时长、VGG-SS 定义和时间戳校验；不得机械假定任何源文件都从 0s 开始。

### 3.2 必须输出

`M0_asset_audit.md`、`asset_provenance.jsonl`、`manifest.jsonl`、`checkpoint_locks.json`。报告：可用源数、有效框数、可复用图片数、现有预测热图数、GT 生成热图数、可评估的类别数、实际媒体缺失数。**若预测图为 0，继续 Oracle pilot，但明确可部署分支缺失。**

## 4. 小样本设计与严格的开发/确认隔离（M1）

1. 从当前**实际可访问的 VGG-SS** 媒体中，按原视频 ID 去重。可复用之前 128 源做已检视的开发诊断；如存在额外合法数据，用固定 `seed=1234`、来源 hash 排序提前锁定新的 confirmation sources。不可用测试成绩倒推选择来源。
2. **pilot 首次只跑 24–32 个来源**，确认端到端流程、可视化、坐标、对照和答案一致性；再扩大至所有可用来源（优先已有 109 音频有效源）。不能凭 24 样本声称稳定统计优势。
3. 对融合色彩/alpha / routing bias 强度等超参数只允许使用明确的 **development source split**；新的 confirmation source split 必须在代码/配置/参数/问题模板冻结后运行一次。
4. 历史 VGG-SS 128 源已用于空间研究，不能重新把它们包装为完全未见过的独立测试。若没有新的来源，本轮只能报告 **development feasibility**。
5. 保存 `selection_manifest.jsonl`，含 `source_id,video_id,frame_pts,media_path,bbox,split,has_audio,heatmap_source,annotation_version`。多源别名/哈希重复必须排除。
6. 本轮对所有比较分支固定相同的标注帧、视频输入前缀、prompt、视觉 token 数与冻结主干；只改变 **GT/预测图提示的输入方式**。
7. 若使用预训练 localizer 生成新的预测热图，保证仅用标注时刻之前的 `[max(0,t-2),t]` 音频，明确检查其 checkpoint 的原生音频窗长/填充、因果前端与过期音频处理；若无法实现合法因果输入就标记 `not_run` 而不是偷偷使用未来声音。

## 5. VGG-SS：从 bbox 到热图（M2）

### 5.1 GT 热图的两种确定性生成法

输入：当前中心帧尺寸 `H,W`；官方 bbox `[{xmin,ymin,xmax,ymax}, ...]`，坐标归一化/像素坐标要从原官方 schema/README 验证；禁止无依据默认。

- `binary_roi`: 将多个有效框取并集，生成 `H_gt[x,y] ∈ {0,1}`。可选**轻微固定 Gaussian 平滑**得到 `soft_gt`，不允许把平滑半径在确认集上调优。
- `gaussian_bbox`: 每个框中心一个 2D Gaussian，半径按 bbox 尺寸确定并设上下限；多框用逐像素 `max` 或 capped sum（在 config 中预声明），归一到 `[0,1]`。
- **必须绘制**：原 RGB / GT bbox / GT heatmap / Overlay / Token-grid ROI 并列图，验证坐标、宽高缩放、边界和多框。

每张图独立保存：

```json
{
  "source_id": "<video-source>",
  "frame_time": 5.0,
  "map_origin": "GT_BOX_RENDERED",
  "map_kind": "soft_gt",
  "bbox_provenance": "official_vggss_annotation",
  "oracle": true,
  "uses_test_annotation_as_input": true,
  "audio_window": null,
  "gt_version": "<checked-in-commit-or-sha>"
}
```

注意：这个示例中的 `frame_time` 为占位示意，不是规定全部样本必须 5.0 秒；实际按 manifest 写入。

### 5.2 现有预测热图的审计与导入

- 若此前保存了 `P_stage3(A,V)`，可以从 `[64,196]` 平均 64 行得到 14x14 的 **原 AudioRouter 路由权重图**，再插值为 RGB 大小。其 `map_origin=LEGACY_ROUTING_PRED`，它**不是独立音视频定位器**，输入只能来自合法 `A[t-2:t]`。
- 若已有 DenseAV/EZ-VSL 预测图，优先导入**原始连续数值 map**；不要从带文字、颜色标注的 JPEG 拼图反推得分图。如果只剩渲染图，标 `visualization_only`，不可用于量化主实验。
- 每张预测图记录所使用的音频片段、预训练模型与版本、归一化策略；检查与中心帧是否对齐。
- 预测热图的 Localization 指标用官方 bbox 评估；GT 热图不能反过来作为其自己的 Localization 成绩。

### 5.3 禁止的捷径

禁止按 GT 框直接裁切目标图像，之后把“声源识别改善”说成音频融合增强。若做 `GT crop`，必须标为另一个 *Oracle crop upper bound*，不能成为主方法。禁止把 GT 框里的对象类别编码成特定颜色、文字或图例嵌入图像。

## 6. 输入融合实验矩阵（M3，核心）

**首先仅运行这些配置，不训练新模型：**

| ID | 输入/改动 | 64-token 路径 | 是否依赖 GT | 预期解释 |
|---|---|---|---|---|
| **B0** | 原 RGB（无热图） | 原冻结 Stage1 Vision-only | 否 | **必须优先跑通的基线** |
| **B1** | GT 热图叠加在 RGB 上（Oracle Overlay） | 同一 Stage1 Vision-only | **是** | 理想空间提示可行性 |
| **B2** | GT bbox 等面积随机平移/放置的热图（Sham Overlay） | 同一 Stage1 Vision-only | **面积来自 GT** | 排除仅新增显著颜色/纹理造成的效应 |
| **B3** | 恒定或全图均匀亮度热图（Neutral Overlay） | 同一 Stage1 Vision-only | 否 | 排除 overlay 颜色/对比度一般扰动 |
| **B4** | GT 热图仅作为原视觉 Query logits 的 spatial bias | Stage1 原生 RGB `V`，仅修改 `P` | **是** | 测试是否只靠 ROI 先验就足够 |
| **B5** | **已有预测图**叠加 RGB（Predicted Overlay） | 同一 Stage1 Vision-only | **否（需要审计）** | 首个可能可部署的方案 |
| **B6** | **已有预测图**作为 routing spatial bias（Predicted Bias） | Stage1 原生 RGB `V` | **否（需要审计）** | 更接近原 AudioRouter 原生视觉值机制 |
| **B7** | 原 Stage3 AudioRouter（真实音频） | 原有音频条件化 64 tokens | 否 | 旧主方法参照，不是同训练历史消融 |
| **B8** | Stage3 Null Audio | 同一个 Stage3 固定音频控制 | 否 | 控制 Stage3 音频输入影响 |

- **最小实验至少 B0–B4**；如果有现成预测图，必须尝试 B5–B6，否则清楚记作 `not_run: no_predicted_map_asset`。
- 对同一来源，B0/B1/B2/B3/B5 使用**完全相同的 Stage1 checkpoint、视觉编解码参数和输出 token budget 64**；B4/B6 使用同一个 Stage1 的 RGB value path。B7/B8 作为外部参照分列。
- 196-token Full Vision 仅可作为补充参考，不与 64-token 系统作“同等预算”比较。
- 第一轮不要尝试每帧额外的 spectrogram-as-image、第二张 image token stream、视频重新采样、动态 budget 或新音频语义 projector。

### 6.1 B1 GT Overlay 具体实现

固定使用**同尺寸 RGB 图像**，不添加第二张图：

```python
# 伪代码：实际颜色映射、dtype、空间坐标、归一化以项目实现为准
heat_rgb = fixed_colormap(heatmap)  # H x W x 3, no text/legend
alpha_map = alpha * heatmap[..., None]
image_fused = (1 - alpha_map) * rgb + alpha_map * heat_rgb
visual_tokens = frozen_siglip_and_original_projector(image_fused)
Z = original_stage1_resample_64(visual_tokens)
```

- 默认 `alpha=0.20`，作为**预声明的初始值**。若调参，只允许在开发来源上，候选可预先限定 `0.10/0.20/0.30`，并冻结后跑确认来源；不能挑确认集最佳 alpha。
- `map==0` 的区域需严格保持原 RGB 不变，`Gate OFF` 回退等同 B0。叠加前后的图像尺寸/颜色空间/预处理路径要一致。
- 提前明确是否针对框内增加视觉对比度，避免 heatmap 颜色对 VLM 类别产生意外语义暗示。

### 6.2 B4 GT Routing Bias 具体实现

保持 **原始 RGB token values `V` 不变**，只用 spatial prior 修改 Stage1 路由 logits：

```python
# 仅示意：base_logits 必须来自与 B0 相同的 Stage1 原始查询
prior_196 = project_map_to_exact_196_token_grid(heatmap)
prior_196 = prior_196 - prior_196.mean()
biased_logits = base_logits + lambda_bias * prior_196[None, :]
P = softmax_with_original_temperature(biased_logits)
Z = P @ V_original_rgb
```

- `lambda_bias` 在开发集选择并锁定；第一轮建议先跑 `lambda_bias=0` 的**逐值退化测试**，再跑一个固定正值（项目配置中明确，不能在确认集搜）。
- 禁止把不同 checkpoint 的 Query/P 硬混，禁止把 `oracle_bbox` 编码成额外文本。
- B4 是 GT Routing Oracle；若效果好，不能被称为实用音频局部匹配结果。

### 6.3 控制变量 / 可复现性

- 随机 Sham：**bbox 面积、形状、数量、heatmap 强度、alpha 与 B1 匹配**；随机位置独立于对象，仅用固定 seed；另提供全局均匀热图控制。
- 原始视觉编码输出 B0 需要和原 Stage1 逐值一致；`B4(lambda=0)` 的 `P`、`Z`、最终 generated IDs 必须逐值/逐 ID 一致。
- 代码路径绝不以 `map_origin` 名字在下游偷偷改变 prompt 或选模型。
- 生成并保存每样本 `RGB`, `oracle map`, `sham map`, `predicted map (if any)`, `overlay`, `routing heatmap` 的对照图；图像仅用于审核，**不是当作 QA 真值**。

## 7. VGG-SS 的评价：先空间、再回答问题（M4）

### 7.1 Primary：空间视觉证据保留（原生有真值）

对于每个模型 `P∈R^{64×196}`：

```text
p_j = mean_i P[i,j]                  # 196 visual positions, sum_j p_j = 1
w_j = fractional bbox-union coverage of position j, in [0,1]
ROI_mass = sum_j p_j * w_j
ROI_area_fraction = sum_j w_j / 196
mass_excess = ROI_mass - ROI_area_fraction
mass_lift = ROI_mass / max(ROI_area_fraction, eps)
```

- 将 bbox 按视觉 Encoder 原生空间几何映射到 token grid，优先用**真实双线性插值权重**实现 fractional coverage；不得偷换成简单 patch 中心落框而未说明。
- 必须输出 B0/B1/B2/B3/B4/B5/B6 每种分支的 `ROI_mass`、`mass_excess`、`mass_lift`、paired `Δ(Bk−B0)`；使用源视频级成对 bootstrap（建议 2000 次）汇总 CI，报告实际有效源数。
- GT overlay 方案由于从 GT bbox 构造输入，ROI mass 增加**本身有循环证据性质**；它只是 proof-of-concept。真正重要的是预测 map 的定位效果和下游任务影响。
- 额外报告 bbox 占图面积分层、单框/多框分层、P-row 重复/塌缩诊断，避免全部 Query 集中一个点就被误认为保留全部视觉语义。

### 7.2 Secondary：必须明确 VGG-SS 没有官方 QA

**先审计是否存在可信的独立、同源同时间视觉 QA 真值**（如仓库已经合法下载并对齐的 QA；记录 video ID、原始视频时间偏移、QA 数据集版本）。若没有，不得造出“官方 VGG-SS QA Accuracy”。

允许两个不同的**明确标记为代理**的下游任务，优先不使用任何手工新增审核：

**Proxy A：VGGSound 标签映射的可见声源类识别（有限类别）**
- VGG-SS 原标签是**声音事件类别**，并不总等于图像中的单一物体类别，禁止全部直接当 visual QA 真值。
- 先在**开发来源上、只看官方类别名**建立固定 `sound_label -> visually-identifiable source category` 映射；排除抽象/环境/多义标签，未知一律 `not_applicable`；未选择测试样本后再改映射。
- 统一 prompt（**不注入 bbox、audio 标签或额外图例**），固定选择题选项/干扰项生成规则与候选类别表；所有分支使用同一道题同一选项顺序。
- 仅报告 `VGGSS-derived source-category proxy accuracy`、Gold-letter NLL 与 `Rescue/Harm`，同时给出全样本覆盖率及排除原因；不称其为 VGG-SS 官方 QA。
- GT Oracle 输入能泄漏目标位置，因此**即使改善也仍属于空间上限实验**；只有预测热图无需任何 GT 的分支能支持较实际的音频引导推理可行性。

**Proxy B：FullVision Teacher 的回答保持率（不需要标注正确性）**
- 使用预先固定的一组视觉理解提问模板；对同样的原始 RGB，冻结 FullVision 196 作为 teacher，记录其输出（只作为参考，不视为 ground truth）。
- 各 64-token 分支与 teacher 比较 `prediction agreement` 和 teacher-distribution KL / 目标字母 NLL，明确定义评分序列和候选答案；teacher 一旦错误，agreement 并不代表真正答对。
- 若不存在可靠可评分任务或输出/输入与 196 的模型不一致，报告 `not_run`，不填造分数。

**不允许的 QA 捷径**：基于 GT bbox 答案生成“哪个区域被标亮？”或者同时让 prompt 透露正确声源类别；这样 B1 获胜并不证明音频视觉融合。

### 7.3 评价结论口径

- 只测 ROI mass、有 GT Oracle 收益：写“空间提示可引导路由”，**不写“VLM 下游提高”**。
- Oracle Proxy QA 提高、预测图 Proxy QA 未提高：写“理想定位有潜力，但音频驱动的可部署效果未证实”。
- 预测图的 Proxy QA 经过合法配对评估提高：写“在该 proxy 子任务上观察到冻结 VLM 的音频视觉化收益”，**不能泛化为通用 VideoQA SOTA**。
- 若 B1/B5 都没有超过 B0 或只等于 Sham：报告负结果，分析叠加扰动、SigLIP 分布偏移、声源与任务答案不一致等原因。

## 8. Gate：作为次级实验，不要先训练（M5，可选）

- 第一阶段全部 `always_on`，因为当前目标是检验**图像表示本身是否有效**；不要直接用旧 Gate 的低启用率过滤掉大部分样本，导致看不出新融合方式是否有帮助。
- 只有 B5 或 B6 在开发集出现可靠的正向作用，才用**冻结的既有 Gate**（例如旧 dual_combined / gate2）测试 `gate_on` / `gate_off`。旧 Gate 的 Utility Head 训练于旧音频路线，**不等于新的 picture 分支 utility**；这里只做迁移诊断。
- `gate_off` 精确回到 B0（Stage1），不是 `extension.enabled=False` 的旧 Stage3。明确维护两个开关，运行日志必须含 `gate_applied`, `branch_selected`, `extension.enabled`, `active_fraction`。
- 只在 development split 选择启用规则；新 confirmation split 用冻结配置。可以列预声明固定低启用率（例如 15%）作为诊断，但此前 15–16% 只是旧策略观察，不是新路径的已证实最优比例。
- 若 Gate ON 选择由 GT bbox 存在/大小直接决定，那是 Oracle Gate，必须单列，不能混入部署测试。

## 9. 工程目录与拟提供的 CLI（M6）

**新增文件建议**（以仓库实际结构调整，绝不覆盖旧文件）：

```text
utility_gate/grounded_image_pilot/
  __init__.py
  audit.py               # M0 assets, original checkpoint and split audit
  vggss_data.py          # manifest, center frame, bbox schema, source isolation
  heatmaps.py            # GT box->map, old predicted maps import, sham, uniform
  image_fusion.py        # overlay; no prompting or label-coded color
  routing_bias.py        # frozen Stage1 logits spatial bias
  proxy_eval.py          # optional derived source-category proxy/teacher agreement
  eval.py                # dispatch real frozen VLM / tracing / paired results
  report.py              # spatial and valid downstream metrics, bootstrap
  configs/pilot.yaml

tests/test_grounded_image_pilot.py

artifacts/utility_gate/vggss_grounded_image_pilot_20261028/
  M0_asset_audit.md
  manifest.jsonl
  asset_provenance.jsonl
  checkpoint_locks.json
  configs_locked/
  spatial/
  proxy_qa/
  examples/
  logs/
  summary.json
  STATUS.json
  final_audit.json
```

新模块运行入口**需要 Codex 实现并测试**；以下是目标命令，不是假定当前已经可运行：

```bash
# 先审计旧资产，不下载/覆盖任何已有媒体
PYTHONPATH=. .venv-gate/bin/python -m utility_gate.grounded_image_pilot.audit \
  --data-root /nvme_data/pkt/streaming/dataset \
  --output artifacts/utility_gate/vggss_grounded_image_pilot_20261028

# 运行最小 B0-B4 四/五路径的真实冻结 VLM pilot
PYTHONPATH=. .venv-gate/bin/python -m utility_gate.grounded_image_pilot.eval \
  --config utility_gate/grounded_image_pilot/configs/pilot.yaml \
  --modes rgb_clean,gt_overlay,sham_overlay,uniform_overlay,gt_routing_bias \
  --subset pilot

# 同步生成空间指标、示例图和（若有合法答案）代理任务评价
PYTHONPATH=. .venv-gate/bin/python -m utility_gate.grounded_image_pilot.report \
  --output artifacts/utility_gate/vggss_grounded_image_pilot_20261028
```

应支持 `--dry-run`、`--resume`、`--sources-manifest`、`--device`、`--max-sources`，并保留统一 `seed=1234`。所有结果按 source ID 配对存 JSONL，重复推理必须能够复现相同 generated IDs。

## 10. 必须通过的测试

1. `bbox -> heatmap -> image overlay` 输入像素范围合法，bbox 像素/归一坐标正确，多框并集合规；无效框明确拒绝、不得默默裁成假框。
2. `GT` 与 `PREDICTED` provenance 完全分开；未知/不存在热图不伪造。
3. `sham` 与 GT 的面积、框数、alpha 和 intensity 对应；随机位置可复现，且不能偷读正确源区域作筛选。
4. `overlay alpha=0` 对所有采样图与 B0 相等；`gt_routing_bias lambda=0` 对所有 `P/Z/generated_ids` 与 B0 相等。
5. 64-token 计数与值维度符合原 Stage1；没有额外音频 token、另一张图片 token 或变化的 QA prompt。
6. checkpoint SHA、以前代码路径、所有旧归档不变，测试不能修改旧 output。
7. 各变体同帧、同 prompt、同样本顺序、同样的 score parser，实际生成/真实前向，不得只缓存一个模型输出在多个策略复用。
8. 可用音频只使用 `[max(0,t-2),t]`；如果 predicted map 模型使用旧 checkpoint 默认窗口含未来，则自动禁止作为因果主结果。
9. 新 confirmation source 与旧 VGG-SS 开发、所有训练/选模来源无交集；source ID/byte hash/alias 三种维度检查。
10. VGG-SS 空间 bbox 一旦用作生成 GT map，就只能参与 Oracle 分支；不得在 predicted 分支里出现 GT 传参或残留强制 ROI。
11. 输出 `pytest`、一条真实 GPU frame & answer generation、可视化对照和所有路径的实际执行状态。

## 11. 结果表模板：真实完成后填写，不得预填数字

```text
| Strategy | Source count | Map source | VLM tokens | ROI mass excess | Proxy QA (if valid) | Delta vs B0 | Status |
|---|---:|---|---:|---:|---:|---:|---|
| B0 RGB Vision-only | | None | 64 | | | | not_run |
| B1 GT Oracle RGB Overlay | | GT bbox | 64 | | | | not_run |
| B2 Equal-area Sham Overlay | | GT area only | 64 | | | | not_run |
| B3 Uniform Overlay | | Constant | 64 | | | | not_run |
| B4 GT Oracle Routing Bias | | GT bbox | 64 | | | | not_run |
| B5 Predicted RGB Overlay | | Predicted map | 64 | | | | not_run |
| B6 Predicted Routing Bias | | Predicted map | 64 | | | | not_run |
| B7 Original Stage3 AudioRouter | | BEATs | 64 | | | | not_run |
| B8 Stage3 Null Audio | | Fixed null | 64 | | | | not_run |
```

报告 `paired Δ` 及 source-level bootstrap CI；若做 MCQ，还需四象限 `both_correct / cue_only_win / vision_only_win / both_wrong`，以及 `oracle_endpoints` 诊断上限。不可把 `bootstrap CI` 当作已校正多重比较的确认性结论；小样本标明探索性质。

## 12. 明确的 GO / NO-GO 决策

- **GO-1（值得研究视觉化融合）**：B1 在有合法视觉任务/代理任务中有超过 B0 且超过 Sham 的一致正向信号，或空间 ROI mass 与独立的视觉表示保真指标同时改善；这个 GO 只说明 *Oracle spatial cue 有潜力*。
- **GO-2（值得进入训练 Adapter / 新 Gate）**：B5 或 B6（无需 GT）在新的独立来源上优于 B0、Sham 和等资源的现有方法；实际 Audio-driven cue 才能支持升级。
- **NO-GO-1**：只有 B1 的 ROI mass 上升、没有合法下游指标；结论限制为空间 Oracle，不许宣称 QA 提分。
- **NO-GO-2**：B1 的合法 QA 没有正向趋势，或与 Sham 类似，先分析是否 RGB 颜色扰动、频谱/grounding 域偏移或 task 与声源不对应，再决定是否值得继续。
- **NO-GO-3**：无法找到任何合法 GT-free 预测图，完成 Oracle 先导实验后报告真实限制，不得用 GT 代替部署图并继续拟合 Gate。

即使 GO-2 不成立，也要交付所有有效结果和负例分析。**不要因为实验没涨分就修改验证来源、阈值或补造问答真值。**

## 13. 运行时注意事项与交付清单

- 所有新下载如不可避免必须写入 `/nvme_data/pkt/streaming/dataset/`，不得混入源码仓库和其他数据根；记录 URL、许可、revision、SHA256、下载失败原因。首轮优先复用现有 VGG-SS。VGG-SS 官方 bbox 来源：`https://www.robots.ox.ac.uk/~vgg/research/lvs/`。
- 真实推理 GPU，逻辑测试 CPU；保留原 Python 环境；不要未经必要在同一个环境升级 torch/transformers。若 DenseAV 新环境有依赖冲突，可用独立环境离线导出 map。
- 测时分别记录 `map creation/localizer`、`RGB overlay`、`visual encoder`、`resampler`、`LLM prefill`、`end-to-end`；若 Oracle GT map 生成成本极低，也不能当作音频预测成本为零的部署速度。
- 写 `STATUS.json`：每阶段 `not_run/running/completed/blocked` 和原因，异常不可记为成功。
- 交付 `M0_asset_audit.md`、`M1_manifest_and_heatmap_provenance.md`、`M2_spatial_results.md`、`M3_proxy_qa_results.md`、`M4_final_judgment.md`、`summary.json`、`results.jsonl`、`examples/`、`commands.json`、`final_audit.json`、自动测试输出。
- 成果优先回答：**GT 理想空间提示有没有帮助？模型实际预测的提示有没有帮助？视觉路由提高的是声源 ROI 还是下游回答？**

## 14. 可以直接执行的第一条 Codex 指令

> 阅读 `AGENTS_VGGSS_GROUNDED_AUDIO_IMAGE_PILOT.md`，在现有 AudioRouter 项目上实施 VGG-SS Grounded Audio Image 轻量实验。**先完成 M0 资产/热图来源审计，再实现 B0 原始视觉、B1 GT oracle overlay、B2 等面积 sham、B3 uniform、B4 GT routing bias 的真实冻结 VLM 运行。**不要训练 Gate/Encoder/LLM，不修改已有 stage1/stage3 checkpoint 或旧归档。确认 VGG-SS 官方只有 bbox 而非预测热图：如果旧资产中已经有可溯源预测图，再添加 B5/B6；否则如实记为 not_run。输出成对空间 ROI mass、可合法评分的代理视觉 QA（若可构造）、源级置信区间、视觉对照图、实际推理成本及 GO/NO-GO 结论。严格标注 Oracle 使用 GT 的标签泄漏边界。执行中不需要反复询问确认，只在文件中记录缺失/阻碍并尽可能完成合法阶段。
