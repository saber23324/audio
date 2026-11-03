# IMU 历史实验结果明细

IMU（Inertial Measurement Unit，惯性测量单元）在本项目中是同步加速度和角速度。六轴曲线画成图像后，与RGB共享同一个冻结视觉编码器；IMU数值没有作为单独Token送入语言模型。

## 数据与评估边界

最新HoloAssist实际可用94个录制前缀、721道QA，训练500／验证166／封存确认55；对应65／22／7个录制，保守录制家族35／12／4。96个传感器与时钟验收记录中2个无合格QA，不能把96写成94个完整视频。R／z编号用于合并任务变体与Part1/2，不等于参与者身份；身份未知，不能声称跨人泛化。题目由官方已完成动作Verb／Noun构建，问题中可明确对象，选项是动词，不是官方动作识别排行榜、一般视觉问答或未来动作预测。

每题使用以已知完成动作边界结束的8秒观察片段，片段内2／4／6／8秒四帧；每次只看此前2秒IMU。边界已给定是任务构造的Oracle时间条件，不是在线检测完成时刻。原始RGB／IMU时钟通过原始ticks对齐；只做因果零阶保持，原始采样来源也同时受上下界约束，禁止未来插值。答案ABCD严格生成匹配，Gold NLL是ABCD首Token重新归一化；全部压缩策略256视觉Token／题。

数据证据：[artifacts/utility_gate/imu_visual_scaleup_train_v1/data/qa_manifest.jsonl](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/data/qa_manifest.jsonl)；[artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/data_split.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/data_split.json)；[官方标签与sensor_window](../../utility_gate/imu_visual/scaleup_train/data.py)。

WEAR取得15人的传感器片段和标签，但没有原始时间时钟，官方预处理存在双侧插值，严格主训练为0对；没有补造陀螺仪。EgoExo4D权限未验证／未取得；UTD-MHAD只用作工程诊断。相关获取与同步限制见[artifacts/utility_gate/imu_visual_scaleup_train_v1/M1_download_sync/ACQUISITION_REPORT.md](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M1_download_sync/ACQUISITION_REPORT.md)和[artifacts/utility_gate/imu_visual_scaleup_train_v1/M0_audit/ARCHITECTURE_AND_HISTORY.md](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M0_audit/ARCHITECTURE_AND_HISTORY.md)。
## 实际训练流水账

所有模型结构复用Audio的双图64-token低秩残差融合Router；REAL、DUP、ZERO真正分别训练。主干、视觉投影器、语言模型、BEATs及V1都不更新；IMU阶段不调用BEATs／EZ-VSL。三种主Router对应旧v3同类权重热启动，重置AdamW；训练3轮、lr1e-5、wd.01、clip1，首答案Token全词表CE。训练特征可缓存，正式成本验证不复用图表或视觉特征。验证真实贪心Accuracy优先、再Gold NLL选checkpoint。训练全ON、没有Gate参与Router训练。


| 模型的训练输入与初始化 | 训练QA／来源／家族 | 轮数／更新 | 可训练参数 | 选定轮与验证正确／QA | checkpoint及SHA |
| --- | --- | --- | --- | --- | --- |
| 独立用重复 RGB 训练的融合 Router，四时刻全部启用；matched variant v3 warm-start; fresh AdamW states, same seed/order/budget; canonical B1 training | 500/65/35 | 3／1500 | 696320 | 3；118/166 | [artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_DUP/selected.pt](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_DUP/selected.pt)<br>26a003ee9c8400905c1f2d48323a30ff2f528a2042a1e555b7afbd7d013f5638 |
| 从相同零输出残差初始化训练的重复 RGB Router，始终启用；zero-output residual scratch | 500/65/35 | 3／1500 | 696320 | 3；120/166 | [artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_DUP_scratch/selected.pt](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_DUP_scratch/selected.pt)<br>cbe6a1a933b18c4680ef6d795ad98d65db1c89bb32491801ad5ad06af783ea94 |
| 真实同步 IMU 曲线图训练的融合 Router，四时刻全部启用；matched variant v3 warm-start; fresh AdamW states, same seed/order/budget; canonical B1 training | 500/65/35 | 3／1500 | 696320 | 3；118/166 | [artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_REAL/selected.pt](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_REAL/selected.pt)<br>5c82a1118078b0bfa34459dcb99c160a4294902ae7b88371f8d22e0426c3052e |
| 较少训练来源的真实 IMU Router，始终启用；matched variant v3 warm-start; fresh AdamW states, same seed/order/budget; canonical B1 training | 91/12/9 | 3／273 | 696320 | 3；98/166 | [artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_REAL_scale025/selected.pt](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_REAL_scale025/selected.pt)<br>e411f4f0d12810638c6579256887ef4b436d2297aa31e5f345f29e7e24053ff5 |
| 中等训练来源的真实 IMU Router，始终启用；matched variant v3 warm-start; fresh AdamW states, same seed/order/budget; canonical B1 training | 200/27/18 | 3／600 | 696320 | 3；101/166 | [artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_REAL_scale050/selected.pt](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_REAL_scale050/selected.pt)<br>6286b40c0aa895674ab67dcb238c8178c03db6d5e15b7a6c49d56b7b878dc830 |
| 从零输出残差初始化训练的真实 IMU Router，始终启用；zero-output residual scratch | 500/65/35 | 3／1500 | 696320 | 3；114/166 | [artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_REAL_scratch/selected.pt](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_REAL_scratch/selected.pt)<br>2f04bec56f2e16f8d0527e273d6f20b83f729c3c468bcae90c314d65b02bd7f2 |
| 独立用零 IMU 曲线图训练的融合 Router，四时刻全部启用；matched variant v3 warm-start; fresh AdamW states, same seed/order/budget; canonical B1 training | 500/65/35 | 3／1500 | 696320 | 3；113/166 | [artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_ZERO/selected.pt](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/checkpoints/R_ZERO/selected.pt)<br>0bef8b2e32646e17ab689b64efdaa7b5590f7000a4ee44dfa2daaaa198129c7d |

7个Router合计8373更新，不是8373独立样本。9／18／35训练家族的规模实验分别91／200／500QA，273／600／1500更新，共享全训练集固定画图尺度并复用同类旧权重；数据量、更新数同时增加，不能单称纯规模效应。另有相同零输出残差初始化的REAL／DUP从头对照。原始每步梯度与冻结审计：[artifacts/utility_gate/imu_visual_scaleup_train_v1/logs/R_REAL_train_steps.jsonl](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/logs/R_REAL_train_steps.jsonl)；[artifacts/utility_gate/imu_visual_scaleup_train_v1/M3_router_train/ROUTER_GRADIENT_AUDIT.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M3_router_train/ROUTER_GRADIENT_AUDIT.json)。


| Gate训练目标 | 参数 | 轮数 | 真实更新 | 非零梯度更新 | 选择规则 |
| --- | --- | --- | --- | --- | --- |
| 完整轨迹收益排序（ranking） | 479553 | 30 | 1890 | 1890 | 验证真实QA，候选权重／阈值随后冻结 |
| 完整轨迹最优Mask分类（classification） | 479553 | 30 | 1890 | 1890 | 验证真实QA，候选权重／阈值随后冻结 |
| 每张成本惩罚0.05的软轨迹（cost_005） | 479489 | 30 | 1890 | 1890 | 验证真实QA，候选权重／阈值随后冻结 |
| 每张成本惩罚0.20的软轨迹（cost_020） | 479489 | 30 | 1890 | 1890 | 验证真实QA，候选权重／阈值随后冻结 |

Gate训练前冻结新的REAL Router，训练8000／验证2656次完整16组合QA，共10656次实际生成；这是666道QA的干预次数。训练295/500、验证98/166题至少一个组合改变答案，不再沿用旧32题全不变的结论。Accuracy目标为2×错误指示+NLL/(1+NLL)，排名／分类学完整轨迹；成本候选为NLL+λk，λ=.05/.2，温度.25。四Gate各30轮1890更新、合计7560。验证后准确率与成本名称均选ranking第30轮、阈值.5、相同SHA，不是两种不同决策。证据：[artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/utility_lock.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/utility_lock.json)；[artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/Gate_training_catalog.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/Gate_training_catalog.json)；[artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/validation_selection.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/locks/validation_selection.json)，`selected`。
## 新封存确认：55QA／7录制／4家族，完整策略表

所有行是同一HoloAssist确认队列、同样已观察8秒及ABCD评分；原生单图／双图仅作不同Token预算参考。Rescue/Harm均相对本表V1。ON比例=选中窗口数／220；额外编码为每题均数，4张基础RGB另计。原始文件在[artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost)，主结果来自付费Pipeline真实生成，eval中的复制记录不能算第二次独立实验。均值与中位耗时均为秒／题。


| 完整策略说明 | 正确／QA | 准确率 ↑ | 既有95%家族CI | Gold NLL ↓ | 挽救／损害 | ON率 | 额外编码图／题 | 视觉Token／题 | Pipeline均秒 ↓ | Pipeline中位秒 | 证据 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 准确率 Gate 的因果辅助图上限 0 张（`BUDGET_accuracy_cap0`） | 16/55 | 29.09% | 未存该项 CI | 1.3562 | 0/0 | 0.00% | 0.000 | 256 | 0.1437 | 0.1366 | E07066、E07067、E07068、E07072、E07073、E07069、E07074 |
| 准确率 Gate 的因果辅助图上限 1 张（`BUDGET_accuracy_cap1`） | 36/55 | 65.45% | 未存该项 CI | 1.1446 | 24/4 | 25.00% | 1.000 | 256 | 0.3096 | 0.2897 | E07078、E07079、E07080、E07084、E07085、E07081、E07086 |
| 准确率 Gate 的因果辅助图上限 2 张（`BUDGET_accuracy_cap2`） | 39/55 | 70.91% | 未存该项 CI | 1.4059 | 27/4 | 50.00% | 2.000 | 256 | 0.4677 | 0.4363 | E07090、E07091、E07092、E07096、E07097、E07093、E07098 |
| 准确率 Gate 的因果辅助图上限 3 张（`BUDGET_accuracy_cap3`） | 38/55 | 69.09% | 未存该项 CI | 1.5708 | 27/5 | 74.55% | 2.982 | 256 | 0.6357 | 0.5930 | E07102、E07103、E07104、E07108、E07109、E07105、E07110 |
| 成本 Gate 的因果辅助图上限 0 张（`BUDGET_cost_cap0`） | 16/55 | 29.09% | 未存该项 CI | 1.3562 | 0/0 | 0.00% | 0.000 | 256 | 0.1376 | 0.1367 | E07114、E07115、E07116、E07120、E07121、E07117、E07122 |
| 成本 Gate 的因果辅助图上限 1 张（`BUDGET_cost_cap1`） | 36/55 | 65.45% | 未存该项 CI | 1.1446 | 24/4 | 25.00% | 1.000 | 256 | 0.3110 | 0.2888 | E07126、E07127、E07128、E07132、E07133、E07129、E07134 |
| 成本 Gate 的因果辅助图上限 2 张（`BUDGET_cost_cap2`） | 39/55 | 70.91% | 未存该项 CI | 1.4059 | 27/4 | 50.00% | 2.000 | 256 | 0.4724 | 0.4355 | E07138、E07139、E07140、E07144、E07145、E07141、E07146 |
| 成本 Gate 的因果辅助图上限 3 张（`BUDGET_cost_cap3`） | 38/55 | 69.09% | 未存该项 CI | 1.5708 | 27/5 | 74.55% | 2.982 | 256 | 0.6233 | 0.5879 | E07150、E07151、E07152、E07156、E07157、E07153、E07158 |
| 陀螺仪标准差阈值 0.20 的固定规则选择真实 IMU Router（`Cheap_gyro_std_0p20`） | 30/55 | 54.55% | [0.4194, 0.7083] | 1.5649 | 18/4 | 50.91% | 2.036 | 256 | 0.4834 | 0.4506 | E07162、E07163、E07164、E07168、E07169、E07165、E07170 |
| 固定最早 0 个时刻启用真实 IMU Router（其余用视觉路由器）（`Fixed_k_0`） | 16/55 | 29.09% | [0.2174, 0.3571] | 1.3562 | 0/0 | 0.00% | 0.000 | 256 | 0.1380 | 0.1376 | E07176、E07177、E07178、E07182、E07183、E07179、E07184 |
| 固定最早 1 个时刻启用真实 IMU Router（其余用视觉路由器）（`Fixed_k_1`） | 36/55 | 65.45% | [0.5161, 0.8333] | 1.1555 | 24/4 | 25.00% | 1.000 | 256 | 0.3023 | 0.2867 | E07190、E07191、E07192、E07196、E07197、E07193、E07198 |
| 固定最早 2 个时刻启用真实 IMU Router（其余用视觉路由器）（`Fixed_k_2`） | 39/55 | 70.91% | [0.5806, 0.8750] | 1.4154 | 27/4 | 50.00% | 2.000 | 256 | 0.4514 | 0.4330 | E07204、E07205、E07206、E07210、E07211、E07207、E07212 |
| 固定最早 3 个时刻启用真实 IMU Router（其余用视觉路由器）（`Fixed_k_3`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.5208 | 27/5 | 75.00% | 3.000 | 256 | 0.6208 | 0.5811 | E07218、E07219、E07220、E07224、E07225、E07221、E07226 |
| 固定最早 4 个时刻启用真实 IMU Router（其余用视觉路由器）（`Fixed_k_4`） | 37/55 | 67.27% | [0.5161, 0.8750] | 1.7791 | 26/5 | 100.00% | 4.000 | 256 | 0.8117 | 0.7345 | E07232、E07233、E07234、E07238、E07239、E07235、E07240 |
| 原生 RGB 单图输入，完全绕过路由器（`N0_RGB`） | 23/55 | 41.82% | 未存该项 CI | 1.5653 | 9/2 | 无Gate | 0.000 | 2920 | 0.2018 | 0.1998 | E07246、E07247、E07248、E07252、E07253、E07249、E07254 |
| 原生 RGB 与真实 IMU 曲线双图，完全绕过路由器（`N1_RGB_IMU`） | 25/55 | 45.45% | 未存该项 CI | 1.6564 | 11/2 | 无Gate | 4.000 | 5840 | 0.9759 | 0.8889 | E07258、E07259、E07260、E07264、E07265、E07261、E07266 |
| 旧小样本重复 RGB 训练 Router，在本轮新队列始终启用（`Old_R_DUP_always`） | 20/55 | 36.36% | [0.2609, 0.4375] | 1.3303 | 4/0 | 100.00% | 4.000 | 256 | 0.2248 | 0.2242 | E07270、E07271、E07272、E07276、E07277、E07273、E07278 |
| 旧小样本真实 IMU Router，在本轮新队列始终启用（`Old_R_REAL_always`） | 19/55 | 34.55% | [0.2459, 0.4219] | 1.3293 | 3/0 | 100.00% | 4.000 | 256 | 0.7631 | 0.7298 | E07284、E07285、E07286、E07290、E07291、E07287、E07292 |
| 旧小样本零 IMU 训练 Router，在本轮新队列始终启用（`Old_R_ZERO_always`） | 19/55 | 34.55% | [0.2459, 0.4219] | 1.3309 | 3/0 | 100.00% | 4.000 | 256 | 0.8074 | 0.7264 | E07298、E07299、E07300、E07304、E07305、E07301、E07306 |
| 旧小样本准确率 Gate 迁移，配旧真实 IMU Router（`Old_v3_gate_accuracy`） | 37/55 | 67.27% | [0.5161, 0.8750] | 1.7791 | 26/5 | 100.00% | 4.000 | 256 | 0.8056 | 0.7335 | E07312、E07313、E07314、E07318、E07319、E07315、E07320 |
| 旧小样本成本 Gate 迁移，配旧真实 IMU Router（`Old_v3_gate_cost`） | 16/55 | 29.09% | [0.2174, 0.3571] | 1.3562 | 0/0 | 0.00% | 0.000 | 256 | 0.1385 | 0.1377 | E07326、E07327、E07328、E07332、E07333、E07329、E07334 |
| 从相同零输出残差初始化训练的重复 RGB Router，始终启用（`R_DUP_scratch_always`） | 34/55 | 61.82% | [0.4194, 0.8750] | 2.1243 | 23/5 | 100.00% | 4.000 | 256 | 0.2256 | 0.2247 | E07340、E07341、E07342、E07346、E07347、E07343、E07348 |
| 较少训练来源的真实 IMU Router，始终启用（`R_REAL_scale025_always`） | 31/55 | 56.36% | [0.3968, 0.6786] | 0.9885 | 18/3 | 100.00% | 4.000 | 256 | 0.7832 | 0.7302 | E07354、E07355、E07356、E07360、E07361、E07357、E07362 |
| 中等训练来源的真实 IMU Router，始终启用（`R_REAL_scale050_always`） | 34/55 | 61.82% | [0.4464, 0.7949] | 1.3425 | 22/4 | 100.00% | 4.000 | 256 | 0.8014 | 0.7326 | E07368、E07369、E07370、E07374、E07375、E07371、E07376 |
| 从零输出残差初始化训练的真实 IMU Router，始终启用（`R_REAL_scratch_always`） | 36/55 | 65.45% | [0.4839, 0.8750] | 2.0408 | 25/5 | 100.00% | 4.000 | 256 | 0.7905 | 0.7337 | E07382、E07383、E07384、E07388、E07389、E07385、E07390 |
| 固定视觉专用路由器，所有时刻只用 RGB（`T0_V1`） | 16/55 | 29.09% | [0.2174, 0.3571] | 1.3562 | 0/0 | 0.00% | 0.000 | 256 | 0.1381 | 0.1370 | E07396、E07397、E07398、E07402、E07403、E07399、E07404 |
| 固定真实 IMU 训练 Router，始终换成错误来源 IMU 图（`T10_REAL_wrong_always`） | 36/55 | 65.45% | [0.4839, 0.8750] | 1.7777 | 25/5 | 100.00% | 4.000 | 256 | 0.8077 | 0.7563 | E07410、E07411、E07412、E07416、E07417、E07413、E07418 |
| 真实同步 IMU 曲线图训练的融合 Router，四时刻全部启用（`T1_REAL_always`） | 37/55 | 67.27% | [0.5161, 0.8750] | 1.7791 | 26/5 | 100.00% | 4.000 | 256 | 0.8176 | 0.7345 | E07424、E07425、E07426、E07430、E07431、E07427、E07432 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器（`T2_accuracy`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.5704 | 27/5 | 77.27% | 3.091 | 256 | 0.6689 | 0.5990 | E07438、E07439、E07440、E07444、E07445、E07441、E07446 |
| 准确率 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成重复 RGB（`T2_accuracy_same_mask_DUP`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.6059 | 27/5 | 77.27% | 3.091 | 256 | 0.2158 | 0.2124 | E07452、E07453、E07454、E07458、E07459、E07455、E07460 |
| 准确率 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成错误来源 IMU 曲线图（`T2_accuracy_same_mask_WRONG`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.5681 | 27/5 | 77.27% | 3.091 | 256 | 0.6727 | 0.6159 | E07466、E07467、E07468、E07472、E07473、E07469、E07474 |
| 准确率 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成零 IMU 曲线图（`T2_accuracy_same_mask_ZERO`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.5591 | 27/5 | 77.27% | 3.091 | 256 | 0.6662 | 0.5985 | E07480、E07481、E07482、E07486、E07487、E07483、E07488 |
| 成本优先选择名义分支；最新实验实际与准确率 Gate 相同（`T2_cost`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.5704 | 27/5 | 77.27% | 3.091 | 256 | 0.6483 | 0.5946 | E07494、E07495、E07496、E07500、E07501、E07497、E07502 |
| 成本 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成重复 RGB（`T2_cost_same_mask_DUP`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.6059 | 27/5 | 77.27% | 3.091 | 256 | 0.2151 | 0.2127 | E07508、E07509、E07510、E07514、E07515、E07511、E07516 |
| 成本 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成错误来源 IMU 曲线图（`T2_cost_same_mask_WRONG`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.5681 | 27/5 | 77.27% | 3.091 | 256 | 0.6848 | 0.6102 | E07522、E07523、E07524、E07528、E07529、E07525、E07530 |
| 成本 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成零 IMU 曲线图（`T2_cost_same_mask_ZERO`） | 38/55 | 69.09% | [0.5484, 0.8750] | 1.5591 | 27/5 | 77.27% | 3.091 | 256 | 0.6484 | 0.5912 | E07536、E07537、E07538、E07542、E07543、E07539、E07544 |
| 准确率验证预算的随机 Gate，种子 1234（`T4_random_accuracy_1234`） | 38/55 | 69.09% | [0.5283, 0.8750] | 1.6255 | 26/4 | 75.91% | 3.036 | 256 | 0.6272 | 0.5882 | E07550、E07551、E07552、E07556、E07557、E07553、E07558 |
| 准确率验证预算的随机 Gate，种子 2234（`T4_random_accuracy_2234`） | 37/55 | 67.27% | [0.5161, 0.8750] | 1.6916 | 25/4 | 75.91% | 3.036 | 256 | 0.6245 | 0.5919 | E07564、E07565、E07566、E07570、E07571、E07567、E07572 |
| 准确率验证预算的随机 Gate，种子 3234（`T4_random_accuracy_3234`） | 36/55 | 65.45% | [0.4839, 0.8750] | 1.7068 | 25/5 | 75.91% | 3.036 | 256 | 0.6399 | 0.5961 | E07578、E07579、E07580、E07584、E07585、E07581、E07586 |
| 成本验证预算的随机 Gate，种子 1234（`T4_random_cost_1234`） | 38/55 | 69.09% | [0.5283, 0.8750] | 1.6255 | 26/4 | 75.91% | 3.036 | 256 | 0.6203 | 0.5864 | E07592、E07593、E07594、E07598、E07599、E07595、E07600 |
| 成本验证预算的随机 Gate，种子 2234（`T4_random_cost_2234`） | 37/55 | 67.27% | [0.5161, 0.8750] | 1.6916 | 25/4 | 75.91% | 3.036 | 256 | 0.6066 | 0.5922 | E07606、E07607、E07608、E07612、E07613、E07609、E07614 |
| 成本验证预算的随机 Gate，种子 3234（`T4_random_cost_3234`） | 36/55 | 65.45% | [0.4839, 0.8750] | 1.7068 | 25/5 | 75.91% | 3.036 | 256 | 0.6097 | 0.5852 | E07620、E07621、E07622、E07626、E07627、E07623、E07628 |
| 独立用重复 RGB 训练的融合 Router，四时刻全部启用（`T8_DUP_trained_always`） | 36/55 | 65.45% | [0.4839, 0.8750] | 1.5801 | 24/4 | 100.00% | 4.000 | 256 | 0.2248 | 0.2243 | E07634、E07635、E07636、E07640、E07641、E07637、E07642 |
| 独立用零 IMU 曲线图训练的融合 Router，四时刻全部启用（`T9_ZERO_trained_always`） | 38/55 | 69.09% | [0.5161, 0.9167] | 1.7118 | 27/5 | 100.00% | 4.000 | 256 | 0.7884 | 0.7261 | E07648、E07649、E07650、E07654、E07655、E07651、E07656 |

原生参考的k若记录为辅助图计数只说明4张额外图，不意味着使用Gate；N0为2920 Token，N1为5840 Token。DUP训练分支在计时中仍第二次编码重复RGB，但无需绘制IMU图，其更短耗时不能全归因Gate。T2_accuracy／T2_cost同一Mask的不同耗时来自测量波动。

### 同队列关键配对

以下仍为55题／4家族、8秒、ABCD。原有CI是按保守录制家族10000次bootstrap，seed1234；4家族仅16种符号置换，双侧最小p=.125。没有预存CI的对照标明未存，不补造。


| 配对比较 | Rescue | Harm | 准确率差 pp | 既有95%CI | NLL改善 ↑ |
| --- | --- | --- | --- | --- | --- |
| 真实同步 IMU 曲线图训练的融合 Router，四时刻全部启用 − 固定视觉专用路由器，所有时刻只用 RGB | 26 | 5 | +38.182 | [19.64, 58.33] pp | -0.4230 |
| 真实同步 IMU 曲线图训练的融合 Router，四时刻全部启用 − 独立用重复 RGB 训练的融合 Router，四时刻全部启用 | 3 | 2 | +1.818 | [-3.57, 8.70] pp | -0.1990 |
| 真实同步 IMU 曲线图训练的融合 Router，四时刻全部启用 − 独立用零 IMU 曲线图训练的融合 Router，四时刻全部启用 | 1 | 2 | -1.818 | [-6.25, 4.35] pp | -0.0673 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 固定视觉专用路由器，所有时刻只用 RGB | 27 | 5 | +40.000 | [19.64, 58.33] pp | -0.2142 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 真实同步 IMU 曲线图训练的融合 Router，四时刻全部启用 | 1 | 0 | +1.818 | [0.00, 4.92] pp | 0.2088 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 固定最早 2 个时刻启用真实 IMU Router（其余用视觉路由器） | 1 | 2 | -1.818 | 未存该项 CI | -0.1550 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 准确率 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成重复 RGB | 0 | 0 | +0.000 | [0.00, 0.00] pp | 0.0355 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 准确率 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成零 IMU 曲线图 | 0 | 0 | +0.000 | [0.00, 0.00] pp | -0.0113 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 准确率 Gate 相同 Mask，固定真实 IMU 训练 Router，推理换成错误来源 IMU 曲线图 | 0 | 0 | +0.000 | [0.00, 0.00] pp | -0.0023 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 准确率验证预算的随机 Gate，种子 1234 | 2 | 2 | +0.000 | [-9.38, 9.84] pp | 0.0551 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 准确率验证预算的随机 Gate，种子 2234 | 2 | 1 | +1.818 | [0.00, 4.92] pp | 0.1212 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 − 准确率验证预算的随机 Gate，种子 3234 | 2 | 0 | +3.636 | [0.00, 9.84] pp | 0.1364 |

配对数值来源：[artifacts/utility_gate/imu_visual_scaleup_train_v1/M6_confirmation/PAIRED_SUMMARY.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M6_confirmation/PAIRED_SUMMARY.json)，`paired`及本轮按原始付费记录重算的 [imu_paid_pairs](supplement.json)。随机Gate用验证ON概率生成，本确认每种种子167次ON，新Gate170次ON，是期望预算匹配，不是逐队列精确等额。Fixed-k2的39/55是预先列出的固定策略本次点估计，不能事后宣称验证选定或作为新的最优模型。
### Accuracy更好、Gold NLL却更差

REAL为37/55、NLL1.779150；V1为16/55、NLL1.356172。Accuracy只统计最终对错，NLL还惩罚把标准答案概率压得极低的错误。不能单凭这组变化判定实现Bug，也不能声称概率校准改善。逐题尾部证据（同一55题）：


| QA与家族 | 标准／REAL输出／V1输出 | REAL NLL | V1 NLL | NLL变化 | 原始记录 |
| --- | --- | --- | --- | --- | --- |
| holo:R197-25Oct-RAM:scaleup:6；holo-family:R197 | C/B/B | 9.6565 | 2.5664 | 7.0901 | [REAL](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost/holo_R197-25Oct-RAM_scaleup_6_T1_REAL_always.json) [V1](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost/holo_R197-25Oct-RAM_scaleup_6_T0_V1.json) |
| holo:R204-11Nov-ATV:scaleup:5；holo-family:R204 | B/A/B | 7.7508 | 1.1158 | 6.6351 | [REAL](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost/holo_R204-11Nov-ATV_scaleup_5_T1_REAL_always.json) [V1](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost/holo_R204-11Nov-ATV_scaleup_5_T0_V1.json) |
| holo:R209-16Nov-RAM:scaleup:7；holo-family:R209 | D/B/D | 6.8921 | 0.6653 | 6.2268 | [REAL](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost/holo_R209-16Nov-RAM_scaleup_7_T1_REAL_always.json) [V1](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost/holo_R209-16Nov-RAM_scaleup_7_T0_V1.json) |
| holo:R204-11Nov-Belt:scaleup:5；holo-family:R204 | B/C/A | 7.1817 | 1.0738 | 6.1078 | [REAL](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost/holo_R204-11Nov-Belt_scaleup_5_T1_REAL_always.json) [V1](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/results/cost/holo_R204-11Nov-Belt_scaleup_5_T0_V1.json) |

## 预算曲线与真实计时

付费计时含重新读取压缩视频、四帧解码、单帧B=1视觉编码、因果IMU裁剪统计、Gate、ON时绘图与第二次视觉编码、融合Router和贪心生成；不含网络获取、模型冷加载、物理等待8秒与Gold打分。媒体字节可在本地范围缓存，图像特征／图表不缓存。H800共享主机多GPU随机顺序，显存为PyTorch峰值allocated，未测能耗。四次采样是观察协议，不应把计算TTFT称实时答案等待时延。[artifacts/utility_gate/imu_visual_scaleup_train_v1/M7_cost_final/MEASUREMENT_SCOPE.md](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M7_cost_final/MEASUREMENT_SCOPE.md)。
[确认预算与准确率曲线](figures/imu_budget_curve.png)；[逐策略完整数表](imu_paid_results.csv)。


| 同55题策略 | 正确／QA | NLL ↓ | 额外编码 | Pipeline均秒 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 固定视觉专用路由器，所有时刻只用 RGB | 16/55 | 1.3562 | 0.000 | 0.1381 | E07396、E07397、E07398、E07402、E07403、E07399、E07404 |
| 固定最早 1 个时刻启用真实 IMU Router（其余用视觉路由器） | 36/55 | 1.1555 | 1.000 | 0.3023 | E07190、E07191、E07192、E07196、E07197、E07193、E07198 |
| 固定最早 2 个时刻启用真实 IMU Router（其余用视觉路由器） | 39/55 | 1.4154 | 2.000 | 0.4514 | E07204、E07205、E07206、E07210、E07211、E07207、E07212 |
| 固定最早 3 个时刻启用真实 IMU Router（其余用视觉路由器） | 38/55 | 1.5208 | 3.000 | 0.6208 | E07218、E07219、E07220、E07224、E07225、E07221、E07226 |
| 固定最早 4 个时刻启用真实 IMU Router（其余用视觉路由器） | 37/55 | 1.7791 | 4.000 | 0.8117 | E07232、E07233、E07234、E07238、E07239、E07235、E07240 |
| 准确率优先学习 Gate，开启用真实 IMU 融合 Router、关闭用视觉路由器 | 38/55 | 1.5704 | 3.091 | 0.6689 | E07438、E07439、E07440、E07444、E07445、E07441、E07446 |
| 成本优先选择名义分支；最新实验实际与准确率 Gate 相同 | 38/55 | 1.5704 | 3.091 | 0.6483 | E07494、E07495、E07496、E07500、E07501、E07497、E07502 |



| 与REAL Always-on配对的同策略计时 | 均值差秒 | 既有家族95%CI | 节省比例 | 减少辅助编码图／题 |
| --- | --- | --- | --- | --- |
| T2_accuracy | -0.1487 | [-0.1882, -0.0753] | 18.19% | 0.909 |
| T2_cost | -0.1693 | [-0.2129, -0.1087] | 20.71% | 0.909 |

来源：[artifacts/utility_gate/imu_visual_scaleup_train_v1/M7_cost_final/PAIRED_PIPELINE_COST.json](../../artifacts/utility_gate/imu_visual_scaleup_train_v1/M7_cost_final/PAIRED_PIPELINE_COST.json)。Gate少编码0.9091张／题，Accuracy与Cost两别名测得约18.19%／20.71%省时；这支持条件计算节省，但未证明Gate优于Fixed-k2或同预算随机。
## 小样本与失败历史：不改写为成功

最初冻结IMU曲线诊断没有训练Router或Gate。UTD108工程样本，原生RGB66/108、曲线单图28/108；Holo64开发题，原生RGB24/64、曲线单图25/64、RGB+单张汇总曲线24/64，属于旧的不同视图／时钟协议。M3–M5当时not_run，不能写成完成。原始[artifacts/imu_visual_next_phase_20261030/M2_frozen_imu_plots/utd_statistics_hierarchical.json](../../artifacts/imu_visual_next_phase_20261030/M2_frozen_imu_plots/utd_statistics_hierarchical.json)、[artifacts/imu_visual_next_phase_20261030/M2_frozen_imu_plots/holo_statistics_hierarchical.json](../../artifacts/imu_visual_next_phase_20261030/M2_frozen_imu_plots/holo_statistics_hierarchical.json)，本轮仅聚合引用，未把它们当严格确认。

v1直接迁移确实训练过，但原始采样来源下界检查失败，保留失败；v2修复后REAL／DUP／ZERO各96次更新，曾发现多帧批量与B=1细微数值差异，影响17个fp16位置及NLL。v3复用v2的真实训练权重、用B=1重新验证各epoch，重新生成1024次完整16组合QA，再训练4个Gate各120次更新；不能称v3重新训练了Router。

v3仅32训练QA／2来源、32验证QA／2来源，无新独立确认：V1及三个训练后Router都是15/32，REAL NLL由1.226434变1.189613但生成答案不变。准确率Gate全ON，成本Gate全OFF，省时来自视觉退回，不证明学到片段收益。原生RGB+IMU17/32、原生RGB12/32是5840对2920 Token不同预算。[artifacts/utility_gate/imu_gate_router_transfer_20261030_v3/results/paired_summary.json](../../artifacts/utility_gate/imu_gate_router_transfer_20261030_v3/results/paired_summary.json)；[utility_gate/imu_gate_router_transfer/revalidate_router.py](../../utility_gate/imu_gate_router_transfer/revalidate_router.py)。

最新扩充实验已实际训练并独立评估。REAL37、DUP36、ZERO38，Gate38、Fixed-k2为39，错误来源36，整体说明任务适配与条件省算；没有确认真实IMU独特理解贡献、强Gate选择能力或跨参与者泛化。
