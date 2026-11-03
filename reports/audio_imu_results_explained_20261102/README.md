# Audio 与 IMU 历史实验中文总报告

审计日期：2026-11-02。本次只读取历史源码、逐题预测、窗口行为、训练日志、锁与统计文件并重算指标；没有新训练、推理或调参，也没有改写旧结果。**请先读本页，再按链接看明细。**

## 一屏结论

- **Audio：有任务适配收益，真实音频独特收益未确证。** 固定8秒MUSIC-AVQA确认288题／200源：训练后融合Router为206/288（71.53%），固定视觉基线173/288；同一个Router改用重复RGB、空间错移图、零音频定位图也均206/288。Valor另一确认508题／254源，融合Router321/508对视觉326/508，未确认迁移改善。此结论只属于该队列，不能与冻结迁移701题60.20%的原生双图比较提升。
- **IMU：训练适配明显，但传感器特有价值未证明。** 最新HoloAssist确认55题／7录制／4保守家族：视觉16/55，旧真实IMU Router19/55，新真实IMU Router37/55；独立重复RGB训练36/55、零IMU训练38/55。学习Gate38/55，170/220窗口ON（77.27%），固定最早两窗39/55。不能声称真实IMU优于零信号，不能声称跨参与者泛化。
- **Gate：条件省算与“选得更好”分开。** IMU学习Gate平均额外编码3.091张，而Always-on为4张；相同付费协议约省18.19%计算秒数（同一策略Cost别名约20.71%）。尚未证明优于固定两窗或同预算随机。Audio固定R_A成本Gate确实少编码，但准确率非劣性及音频来源收益未确认。

关键数字原始依据：[artifacts/utility_gate/gate_first_router_second_native_ablation_v1/results/summary.json](../../artifacts/utility_gate/gate_first_router_second_native_ablation_v1/results/summary.json)，`confirmation_music/confirmation_valor.strategies`；[artifacts/utility_gate/imu_visual_scaleup_train_v1/M6_confirmation/PAIRED_SUMMARY.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M6_confirmation/PAIRED_SUMMARY.json)，`holdout metrics/paired`；[artifacts/utility_gate/imu_visual_scaleup_train_v1/M7_cost_final/PAIRED_PIPELINE_COST.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M7_cost_final/PAIRED_PIPELINE_COST.json)，`holdout`。

## 阅读导航

| 文件 | 内容 |
| --- | --- |
| [Audio明细](audio_results.md) | 每轮MUSIC／Valor独立表、训练路径、原生双图、预算与VGG-SS／LLP诊断 |
| [IMU明细](imu_results.md) | HoloAssist全部确认策略、7个Router和4个Gate训练、同Mask配对、真实计时及失败历史 |
| [完整架构，重点Gate与Router](architecture_audio_imu.md) | 当前Audio／IMU从数据输入、因果窗口，到Gate、Router与LLM接口的逐层形状／公式／参数 |
| [指标中文词典](metric_dictionary.md) | 公式、统计单位、方向、NLL协议、Bootstrap及成本边界 |
| [所有策略中文对照](strategy_name_mapping.csv) | 原始ID、完整名称、训练输入与推理干预、Gate／Router／Token区别 |
| [数值证据索引](evidence_index.csv) | E编号→原始文件、结果键／选择条件、源码位置 |
| [数据与题型解释](dataset_details.md)／[划分CSV](dataset_split_ledger.csv) | manifest逐行重数的QA／来源／家族与问题类型 |
| [自动核验](verification_report.json) | 通过／失败／缺失、原始输入SHA保护与可比性限制 |

## 不需要内部缩写的架构入门

RGB是普通彩色视频帧。VLM（Vision-Language Model，视觉语言模型）使用冻结的SigLIP视觉编码器、视觉投影器和Qwen2语言模型。视觉专用Router通过64个查询对196个视觉位置做加权重采样；不是挑64块的硬Top-K。原始音频Router用冻结BEATs音频特征控制查询，音频值不直接送入语言模型。

后续双图Fusion Router把RGB与辅助图各196个视觉特征融合，在固定视觉专用输出的基础上加低秩残差，始终每帧64视觉Token。Audio辅助图来自冻结EZ-VSL的预测声源热图；IMU辅助图来自真实同步加速度／陀螺仪曲线。Gate先看当前／过去RGB摘要与便宜的传感器特征，决定是否调用辅助绘图与第二次视觉编码；OFF严格退回V1。原生双图是另一个接口，绕过Router，让语言模型看到大量原生图像Token。

模型参数与冻结状态以原始审计为准：SigLIP397,747,744、视觉投影器16,980,992、语言模型及原生特殊嵌入7,612,631,040均冻结；原Router总1,991,680，视觉专用活跃训练933,888；后续Fusion Router696,320。不能把总参数量当本轮训练量。证据：[逐模块审计](../../artifacts/imu_visual_next_phase_20261030/M0_snapshot/model_inventory.json)；[GroundedFusionRouter](../../utility_gate/gate_first_native/fusion.py)。

## 数据与公平比较登记

以下是分队列索引，不是跨队列排行榜。MUSIC是原生短答案规范化精确匹配，Valor与HoloAssist是严格生成ABCD；Gold NLL亦不跨协议直接比较。


| 阶段 | 数据集 | 独立队列名称 | QA／来源 | 时间范围 | 答案／评分 | 原证据 |
| --- | --- | --- | --- | --- | --- | --- |
| 冻结模型迁移 | music_avqa | music_confirm | 701／500 | 按时长比例四帧 | MUSIC规范化文本EM／整串NLL | [artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/summary.json](../../artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/summary.json) |
| 冻结模型迁移 | valor32k_avqa_v2 | valor_confirm | 256／128 | 按时长比例四帧 | 原选项ABCD／四选项NLL | [artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/summary.json](../../artifacts/utility_gate/audio_grounded_dual_image_qa_20261029_v1/summary.json) |
| 新视觉路由器与单窗口收益 Gate | music_avqa | confirmation_music | 439／300 | 2秒起最多32次因果观察 | MUSIC规范化文本EM／整串NLL | [artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/qa_eval/paired_summary.json](../../artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/qa_eval/paired_summary.json) |
| 新视觉路由器与单窗口收益 Gate | valor32k_avqa_v2 | confirmation_valor | 512／256 | 2秒起最多32次因果观察 | 原选项ABCD／四选项NLL | [artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/qa_eval/paired_summary.json](../../artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/qa_eval/paired_summary.json) |
| 先训练 Gate、再训练融合 Router | music_avqa | confirmation_music | 288／200 | 已观察8秒，四次观察 | MUSIC规范化文本EM／整串NLL | [artifacts/utility_gate/gate_first_router_second_native_ablation_v1/results/summary.json](../../artifacts/utility_gate/gate_first_router_second_native_ablation_v1/results/summary.json) |
| 先训练 Gate、再训练融合 Router | valor32k_avqa_v2 | confirmation_valor | 508／254 | 已观察8秒，四次观察 | 原选项ABCD／四选项NLL | [artifacts/utility_gate/gate_first_router_second_native_ablation_v1/results/summary.json](../../artifacts/utility_gate/gate_first_router_second_native_ablation_v1/results/summary.json) |
| 固定融合 Router、重新训练整段策略 Gate | music_avqa | confirmation_music | 282／200 | 已观察8秒，四次观察 | MUSIC规范化文本EM／整串NLL | [artifacts/utility_gate/fixed_router_gate_v1/paired_summary.json](../../artifacts/utility_gate/fixed_router_gate_v1/paired_summary.json) |
| 固定融合 Router、重新训练整段策略 Gate | valor32k_avqa_v2 | confirmation_valor | 504／252 | 已观察8秒，四次观察 | 原选项ABCD／四选项NLL | [artifacts/utility_gate/fixed_router_gate_v1/paired_summary.json](../../artifacts/utility_gate/fixed_router_gate_v1/paired_summary.json) |
| 扩充后的IMU主确认 | HoloAssist标签衍生QA | holdout | 55／7录制／4家族 | 已知完成边界内8秒，四次观察 | 动作动词四选一ABCD；四选项NLL | [artifacts/utility_gate/imu_visual_scaleup_train_v1/M6_confirmation/PAIRED_SUMMARY.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M6_confirmation/PAIRED_SUMMARY.json)，`cohorts` |

训练与验证来源统计、问题类别见 [完整划分CSV](dataset_split_ledger.csv)。Audio V1/Fusion训练656 QA／192源、验证157／48；固定R_A的新Gate沿用训练源但新验证335／96。最新IMU训练500 QA／65录制／35家族，验证166／22／12，确认55／7／4。MUSIC原始YouTube血缘不可核验；Valor已知YT去重不能证明基础模型预训练未见；所有这些已检视确认来源现已是历史，不允许未来调参后再称独立测试。
## 同一队列内哪些对照回答什么问题


| 比较 | 控制住什么 | 可以解释什么 | 不能据此解释什么 |
| --- | --- | --- | --- |
| Stage3真实音频 vs 同权重零波形编码 | 同Router、Token、问题与音频编码器 | 该旧Router对真实波形是否敏感 | Stage1与Stage3训练历史的纯因果分离 |
| 原生RGB+预测图 vs 原生重复RGB／错移图 | 同图数、同提示词／顺序需明确 | 辅助图内容是否带来增益 | 与压缩256Token模型的公平同预算排名 |
| 固定R_A真实图 vs 同权重重复／错移／零音频图 | 同已训练Router，仅推理干预 | 训练后对音频提示内容的依赖 | 独立重复图训练Router的效果 |
| IMU REAL训练 vs DUP／ZERO独立训练 | 同结构、初始化类别／预算及训练来源，具体热启动不同 | 普通任务训练／额外视图可否解释收益 | 1题差异即真实IMU独特贡献 |
| Gate真实图 vs 同Mask DUP／ZERO／WRONG | 控制每题所选时间和64Token输出 | 辅助内容特异性 | 错误来源干预是一套可部署策略 |
| Gate vs Always-on／固定k／随机Gate | 同队列、Router与输出预算；额外图预算需实测 | 选择质量、预算效率与负迁移 | 少算自动等于识别了有益传感器片段 |

## 训练与可复现性



| 模型／阶段 | 训练与验证数据 | 初始化 | 实际训练模块；冻结项 | 轮数与更新 | 目标及优化设置 | 模型选择与失败记录 | 原始元数据 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原始视觉专用Stage1 | VideoMME旧训练split，720源；QA数按旧manifest另见索引 | 旧训练中间权重续训；最终adbt_epoch_3.pt | 总1991680，历史活跃933888：query_tokens与visual_key | 配置3轮；最终global_step6480；元数据epoch=2按零索引解释，不能当只训练两轮 | 首答案Token全词表CE；AdamW lr1e-5, wd.01, clip1, seed1234 | 未核验旧checkpoint最优选择／旧优化器状态缺失 | [artifacts/imu_visual_next_phase_20261030/M0_snapshot/checkpoint_training_metadata.json](../../artifacts/imu_visual_next_phase_20261030/M0_snapshot/checkpoint_training_metadata.json) |
| 原始音频Stage3 | 同一旧VideoMME split | 另一个音频训练checkpoint；非Stage1同权重音频开关 | 总1991680；历史活跃参数未核验 | 配置3轮；最终global_step6480；旧优化器状态缺失 | 旧phase4首答案Token CE；其他训练历史不外推到phase5 | 旧最优选择未核验；不混用论文摘要 | [artifacts/imu_visual_next_phase_20261030/M0_snapshot/checkpoint_training_metadata.json](../../artifacts/imu_visual_next_phase_20261030/M0_snapshot/checkpoint_training_metadata.json) |
| 新视觉专用V1 | MUSIC训练656QA／192源；验证157／48 | 旧Stage1最终权重，fresh AdamW | 933888，query_tokens+visual_key；VLM冻结 | 3轮／1968更新 | 首答案Token CE；lr1e-5, wd.01, clip1, QA batch1, seed1234 | 验证首标准答案Token NLL最低；选第3轮；不是Full-Vision KD | [artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/locks/stage1_retrained_lock.json](../../artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/locks/stage1_retrained_lock.json) |
| 自适应收益Gate两种 | 5697训练有效窗口／1436验证有效窗口；完整视频QA单窗口干预 | 一支重置收益网络；另一支冻结旧combined共同事件代理主干只换收益头 | 573953活跃；冻结主干支仅65活跃；各总574059 | 各30轮／2700更新 | 源加权Huber，训练分位尺度；lr.001, wd.01, batch64 | 验证MAE选第1／8轮；完整混合QA选阈值，均全关 | [artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/models/new_utility_gate/development_summary.json](../../artifacts/utility_gate/adaptive_grounded64_stage1_retrain_20261029_v1/models/new_utility_gate/development_summary.json) |
| 先训练Gate阶段五模型 | 训练656QA／192源、验证157／48；全部16组合13008次QA | 固定V1/F0；收益／排序／双头代理／视觉／分类候选 | 不同小Gate，主Gate576097 | 各30轮／360更新，共1800更新 | 完整mask收益与条件边际的Huber／排序／分类；不是简单相加；lr.001 | 合法验证生成EM、NLL及视觉保留约束；主dual_rank第30轮 | [artifacts/utility_gate/gate_first_router_second_native_ablation_v1/locks/gate_a_lock.json](../../artifacts/utility_gate/gate_first_router_second_native_ablation_v1/locks/gate_a_lock.json) |
| 音频融合Router R_A | 同656训练QA／192源，验证157／48 | V1查询／键截取初始化；低秩output为零；Gate先冻结 | 696320：query,key,value,output；全部主干/V1/Gate冻结 | 3轮／选定轨迹1968更新；中断续跑记录保留 | 首答案Token CE；AdamW1e-5,wd.01,clip1,seed1234；强制ON训练暴露 | 验证Always-on真实分支EM再NLL；第3轮；14零梯度样本，30未提交更新回滚 | [artifacts/utility_gate/gate_first_router_second_native_ablation_v1/locks/fusion_router_lock.json](../../artifacts/utility_gate/gate_first_router_second_native_ablation_v1/locks/fusion_router_lock.json) |
| 固定R_A的新完整策略Gate | 训练656QA／192源；新验证335QA／96源 | R_A与V1固定；五种输入家族×四lambda×二温度 | 各465409–576161；所有Router更新0 | 40个模型各30轮／360更新，总14400 | 源等权的16轨迹软目标CE：q(S)∝exp[-(NLL(S)+λk)/T]；lr.001 | 977验证候选中锁准确率优先与成本优先；不是确认集选参 | [artifacts/utility_gate/fixed_router_gate_v1/train/catalog.json](../../artifacts/utility_gate/fixed_router_gate_v1/train/catalog.json) |

旧模型的6480是checkpoint记录的global_step，不等于6480个独立视频。原训练每段视频多道QA，精确QA分布与验证记录若无日志则不补造。原始Stage1/Stage3 checkpoint文件名与元数据不同索引，表中保留原值。
IMU全部训练更新与checkpoint SHA见 [IMU明细](imu_results.md)。基础VLM与V1始终冻结；原AudioRoutercheckpoint保持。预测试锁、随机种子、旧失败与回滚记录均原样保留。历史训练／推理命令见[artifacts/utility_gate/gate_first_router_second_native_ablation_v1/commands.json](../../artifacts/utility_gate/gate_first_router_second_native_ablation_v1/commands.json)、[artifacts/utility_gate/fixed_router_gate_v1/commands.json](../../artifacts/utility_gate/fixed_router_gate_v1/commands.json)和[artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/protocol.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/protocol.json)。前两个`.json`文件实际为逐行JSON命令日志，按JSONL读取。命令只是可复现记录，本轮没有执行它们。

本次审计的可复现命令（只进行CPU读取／统计）：

```bash
PYTHONPATH=LLaVA-NeXT:lmms-eval:. PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv-gate/bin/python reports/audio_imu_results_explained_20261102/audit_tools/audit.py
PYTHONPATH=LLaVA-NeXT:lmms-eval:. PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv-gate/bin/python reports/audio_imu_results_explained_20261102/audit_tools/supplement.py
PYTHONPATH=LLaVA-NeXT:lmms-eval:. PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv-gate/bin/python reports/audio_imu_results_explained_20261102/audit_tools/render_report.py
```

## 结论的证据边界

目前可以支持：小型融合Router经过真实训练后在部分标签衍生任务上适配更好；Gate能够减少第二次视觉编码及实测计算时间。尚不足以支持：音频／IMU比重复或零传感器提供独特理解信息；学习Gate稳定优于固定预算或随机选择；跨人、跨域、通用多模态理解或SOTA。四家族HoloAssist及不同Audio队列限制统计外推，Accuracy提升同时NLL可能恶化。VGG-SS空间Oracle、LLP共同事件代理与真实下游QA分别报告，不互相替代。所有缺失值、未重验聚合数据与成本命名问题均在核验报告中明示。

## 本次审计发现的历史字段歧义

Gate-first原生分支的`actual_second_encoder_called`在源码中写成`bool(native or ese>0)`，所有原生调用都记真。动态单图C_DEPLOY有1379个OFF窗口同条记录`native_image_count=1`，因此不能把该布尔字段当物理“第二图调用”计数。源码按`ims=[frame] if C_DEPLOY and not on else [frame,second]`执行。保留旧日志，重算图数时按native_image_count，性能结论使用付费计时；不把它报告为Gate关闭仍偷偷编码第二图。自动核验保留这一字段不一致项并分类说明。

原始证据：[逐窗口记录](../../artifacts/utility_gate/gate_first_router_second_native_ablation_v1/results/per_window_gate_decisions.jsonl)，`strategy=C_DEPLOY,gate_on=false`；[代码](../../utility_gate/gate_first_native/evaluate.py:126)。另有固定R_A旧“amortized”成本字段的单位歧义，已在成本段按实际公式解释，未修改任何旧JSON。

附加可复现审计脚本：`audit_tools/architecture.py`（CPU权重形状）、`audit_tools/enrich.py`（窗口／成本核验与附表）、`audit_tools/finalize.py`（历史附录与最终SHA）。使用上文相同CPU环境运行。全候选模型日志索引：[training_model_ledger.csv](training_model_ledger.csv)。
