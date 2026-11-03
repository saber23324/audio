# 更早的 Audio Gate 历史：单独队列，不能与 MUSIC／Valor 混排

## 2026-10-27 视觉锚定Gate2

该阶段来自原AVQA标签（不是MUSIC），训练512题／512原始YouTube源，验证128／128，确认256／256；View4、Sound10、Both242。每源一道四选一问题，严格生成ABCD匹配；首答案Token四选项归一化NLL。最多10秒观察、2秒因果窗口、每帧64视觉Token，五帧时320Token。前缀Gate在问题到来时决定整段是否用音频，逐窗口Gate另行训练，二者监督目标不同。


| 前缀策略中文名 | 正确／QA | Accuracy ↑ | 时间与决策协议 | 原始记录 |
| --- | --- | --- | --- | --- |
| 固定视觉64 | 222/256 | 86.72% | 前缀可行性代理，最多10秒5次观察；非逐帧收益Gate | [artifacts/utility_gate/gate2_20261027/labels/evaluation_assets/test.jsonl](../../artifacts/utility_gate/gate2_20261027/labels/evaluation_assets/test.jsonl) |
| 原始音频Router64 | 211/256 | 82.42% | 前缀可行性代理，最多10秒5次观察；非逐帧收益Gate | [artifacts/utility_gate/gate2_20261027/labels/evaluation_assets/test.jsonl](../../artifacts/utility_gate/gate2_20261027/labels/evaluation_assets/test.jsonl) |
| 合法可用性音频Router64 | 213/256 | 83.20% | 前缀可行性代理，最多10秒5次观察；非逐帧收益Gate | [artifacts/utility_gate/gate2_20261027/qualified_labels/evaluation_assets/test.jsonl](../../artifacts/utility_gate/gate2_20261027/qualified_labels/evaluation_assets/test.jsonl) |
| 旧视觉锚定前缀Gate | 222/256 | 86.72% | 前缀可行性代理，最多10秒5次观察；非逐帧收益Gate | [artifacts/utility_gate/gate2_20261027/integration/gated_results.jsonl](../../artifacts/utility_gate/gate2_20261027/integration/gated_results.jsonl) |

完整视觉196参考225/256来自原聚合summary.prefix.methods.full_vision，本次没有重验该端点生成文件，不把它当逐题重验通过。精确匹配启用数量的随机策略与Gate同222/256、能量阈值全关；该两端点来自原summary，仅聚合引用。Gate只开启3/256前缀，不能据此证明有益片段识别。[prefix.methods](../../artifacts/utility_gate/gate2_20261027/summary.json)。

逐时刻因果重放：另64题／64源，原AVQA四选项，2秒到达、总317观察窗，保持每时刻64Token；编码器缓存时间不能当完整Pipeline。


| 该64题策略 | 正确／QA | 准确率 ↑ |
| --- | --- | --- |
| 固定视觉64 | 56/64 | 87.50% |
| 原始音频Router64 | 50/64 | 78.12% |
| 可用性约束音频64 | 51/64 | 79.69% |
| 完整视觉196 | 55/64 | 85.94% |
| 逐时刻因果Gate64 | 54/64 | 84.38% |
| 验证预算随机Gate64 | 54/64 | 84.38% |

训练：视觉+严格过去音频摘要、两层小MLP；只训练Gate，原Router／BEATs／VLM冻结。训练配置为AdamW lr.001、wd.01、batch32、最多60轮，种子1234／2333／3407；两布局、辅助分类／权重候选在验证选择，主seed预先固定。旧前缀、可用性修正、逐窗口三个拟合阶段不得合算独立数据。每步更新未在这次报告重建，精确轮数与配置保存于[artifacts/utility_gate/gate2_20261027/qualified_model/train_config.json](../../artifacts/utility_gate/gate2_20261027/qualified_model/train_config.json)、[artifacts/utility_gate/gate2_20261027/temporal_model/train_config.json](../../artifacts/utility_gate/gate2_20261027/temporal_model/train_config.json)；原源码及完整候选选择见[reports/GATE2_EXPERIMENT_RESULTS_20261027.md](../../reports/GATE2_EXPERIMENT_RESULTS_20261027.md)。

## 冻结Gate2的全新197源确认

192 Both、1 View、4 Sound；独立197题／197源，四个2秒到达观察（8秒）、严格生成ABCD、四选项NLL。冻结Gate2／原Router，没有重新训练或重选阈值；与前面256题是不同队列。


| 完整策略名 | 正确／QA | 准确率 ↑ | Gold NLL ↓ | 相对视觉挽救／损害 | 既有源95%CI | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 固定视觉路由64 | 165/197 | 83.76% | 0.5510 | 0/0 | [0.7868, 0.8883] | E08323 |
| 原始真实音频路由64 | 161/197 | 81.73% | 0.5798 | 6/10 | [0.7563, 0.8680] | E08324 |
| 可用性约束音频路由64 | 162/197 | 82.23% | 0.5506 | 5/8 | [0.7665, 0.8731] | E08325 |
| 完整视觉196参考 | 166/197 | 84.26% | 0.5378 | 5/4 | [0.7919, 0.8934] | E08326 |
| 同Stage3零波形BEATs编码 | 161/197 | 81.73% | 0.6107 | 6/10 | [0.7614, 0.8731] | E08327 |
| 冻结前缀Gate | 165/197 | 83.76% | 0.5454 | 0/0 | [0.7868, 0.8883] | E08328 |
| 冻结逐窗口Gate | 164/197 | 83.25% | 0.5563 | 0/1 | [0.7766, 0.8832] | E08329 |
| 匹配前缀启用数随机 | 165/197 | 83.76% | 0.5511 | 0/0 | [0.7868, 0.8883] | E08330 |
| 匹配窗口预算随机 | 165/197 | 83.76% | 0.5481 | 1/1 | [0.7868, 0.8883] | E08331 |

原始文件[artifacts/utility_gate/gate2_grounding_20261027/avqa/per_qa.jsonl](../../artifacts/utility_gate/gate2_grounding_20261027/avqa/per_qa.jsonl)，汇总键`all.methods`；这轮仍未超过固定视觉。并行空间验证109合格VGG-SS源，Stage3真实减同权重零波形ROI质量+.030011，CI[.016501,.044311]；旧LLP126源／617合格2秒点AUROC .471382，CI[.390384,.553650]。这些是旧的诊断协议，不能与随后训练代理头的571点AUROC混比。仅聚合引用，未在本附录重算原空间矩阵与617点分数，保留限制。[main_table](../../artifacts/utility_gate/gate2_grounding_20261027/combined/summary.json)。
