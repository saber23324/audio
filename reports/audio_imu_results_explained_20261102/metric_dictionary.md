# 指标解释词典

统一记号：题目 i 的标准答案 y_i，生成文本 a_i，二值正确性 c_i；方法 m、配对基线 b；总题数 N；每题／每来源合格窗口 T；窗口开关 g_t∈{0,1}。↓越小越好，↑越大越好；ON率和预算本身没有固定“越高越好”。所有公式先在完全相同的队列、问题和评分规则内计算。


| 指标中文定义 | 计算公式 | 统计单位／方向 | 不可比情况及注意事项 | 原始计算位置 |
| --- | --- | --- | --- | --- |
| QA Accuracy／EM，问答准确率／精确匹配率 | Acc=Σ_i 1[norm(a_i)=norm(y_i)]/N | 题；比例或% ↑ | MUSIC全短答案规范化、数字0–11映射词；严格规范化EM不做数字别名。Valor/Holo必须生成完整ABCD单字母，不能用logit最大值替代真实生成。 | utility_gate/grounded_dual_image/data.py:normalize；runtime.py:answer |
| Gold NLL，正确答案负对数似然 | 整串：−Σ_j log p(y_j／x,y_<j)；四选项：−log[exp l_y/Σ_{z∈ABCD}exp l_z] | 题均值；自然对数单位nat ↓ | MUSIC主表整串全词表，不含EOS；Valor/Holo是首Token四选项重归一化，另存full_vocab。整串、首Token全词表和四选项三者数值不能混比。 | utility_gate/grounded_dual_image/runtime.py:answer |
| Training Loss，训练损失 | Router：−log p_vocab(首标准答案Token／视觉与问题)；Gate各目标另见训练流水账 | 优化样本均值；nat或自定义损失 ↓ | 训练首Token CE不是QA整串NLL，不是FullVision知识蒸馏；一次更新不是一个独立视频。 | streamingtom/train_adbt_videomme.py:qa_loss；utility_gate/imu_visual/scaleup_train/train_gate.py:objective |
| Rescue／Harm，挽救／损害 | R=Σ1[c_m=1,c_b=0]；H=Σ1[c_m=0,c_b=1] | 同一道题配对；题数，R↑／H↓ | 必须注明基线；没有同QA配对不能相减。随机三种子的逐题平均正确率是分数，R/H为期望计数、可非整数。 | utility_gate/adaptive_grounded64/report.py:paired |
| Paired Delta，配对准确率差 | ΔAcc=(R−H)/N=mean(c_m−c_b) | 百分点pp ↑ | 0.01=1pp，不是相对提升1%；不同队列不能计算配对提升。NLL改善在本报告定义mean(NLL_b−NLL_m)，正值更好；原IMU NLL_delta通常方法减基线，符号相反。 | utility_gate/fixed_router_gate/paired_stats.py:paired |
| 95% CI，置信区间 | 按来源／家族抽取有放回cluster，保留cluster内全部QA；重复统计取2.5%和97.5%分位 | 准确率或差值的原单位；区间而非方向分数 | Audio大多数源聚类2000次seed1234；最新Holo4家族10000次seed1234。不是按220窗口独立抽样。4家族区间不稳，零翻转[0,0]不是等效性证明；多比较多为探索性无校正。 | utility_gate/imu_visual/scaleup_train/report.py；M6_confirmation/EXACT_FAMILY_SIGN_PERMUTATION.json |
| Gate ON Rate，辅助分支启用比例 | Σ g_t / 合格窗口总数；Holo每题4窗，55题分母220；Audio先按来源去重 | 合格决策窗口；% 无固定优劣 | Gate决定开启、热图有效、实际第二编码是不同事件。Always-on是策略规则，不表示Gate判断正确。相同QA来源重复记录不可重复数Audio窗口。 | utility_gate/gate_first_native/report.py:summarise；per_window输出 |
| Added Image Count，额外编码图像数 | k=Σ_t 实际第二视觉编码调用；平均k=Σ_i k_i/N；基础图为4张RGB | 张／题或张／来源 ↓（给定精度） | Holo compressed编码数4+k，Token始终256；Audio无效图可回退。原生OFF仍重复双图时不省第二图。DUP编码是否复用按实现，不能仅按视图数猜。 | utility_gate/imu_visual/scaleup_train/cost.py:counts |
| Visual Tokens，进入LLM的视觉Token | 压缩：64×实际观察数；四观察为256；原生图729+image_newline=730 | Token／题 ↓（给定精度） | 原生四RGB为2920、四对图5840；旧全视觉视频4×196+newline=785。729→196是视觉池化中间层，64是压缩输出，文字／选项Token另计。不同接口不混算。 | utility_gate/grounded_dual_image/native.py；runtime.py:answer |
| Pipeline Runtime／TTFT，实际计算耗时／首Token时间 | wall_end−wall_start；TTFT=首次生成Token时刻−相同start | 秒／题；均值／中位／p95 ↓ | 明确是否含解码、音频、BEATs、Gate、定位、画图、第二编码、Router、prefill、decode。主预测共享前端的LLM时间不能叫Pipeline。最新付费计时不含加载、网络、物理等待、Gold评分；不能跨阶段硬件／缓存／顺序比速度。 | M7_cost_final/MEASUREMENT_SCOPE.md；utility_gate/fixed_router_gate/paired_stats.py:summarize |
| Cost Saving，耗时节省比例 | saving=1−mean(time_gate)/mean(time_always) | 相同队列／计时协议；% ↑ | 应同时给秒数、差值和配对CI；Gate少算不等于选择更有益传感器。两相同Gate别名不同耗时是测量变动。原字段amortized_video_all_QA_pipeline_s是平均每源回答全部QA总秒，不是秒/题。 | M7_cost_final/PAIRED_PIPELINE_COST.json |
| AUROC／AUPRC，排序判别指标 | AUROC=P(s_pos>s_neg)+0.5P(tie)；AP=Σ_n (Recall_n−Recall_(n−1)) Precision_n | 带已知正负标签的诊断点；[0,1] ↑ | LLP正负为共同事件／非共同代理，不是true visibility；AP受正例占比影响，必须报正例数。一个源有多个秒点，CI按源聚类。utility-only诊断用u分数，其他对应头用r。 | utility_gate/dual_head/proxy/evaluation.py:methods |
| Spatial ROI Mass／Excess／Lift，区域路由质量 | q_j=mean_k P_kj；ROI r_j为token与GT框重叠占比；Mass=Σ q_j r_j；面积A=mean r_j；Excess=Mass−A；Lift=Mass/A | 源／标注帧；无量纲 ↑ | ROI是Region of Interest；均匀路由Lift=1。框面积变化要做配对。GT叠加和直接logit偏置均为Oracle；无标注不能计算真实定位质量。质量高不自动等于QA好。 | utility_gate/grounding/spatial_mapping.py:routing_metrics |
| Source Family，来源／录制聚类 | 把同一可审计来源的相关问题合并抽样 | 来源数、录制数、家族数 | MUSIC仅确认本地官方视频ID，缺原始YT血缘；Valor使用已知YT源；Holo家族合并R/z号及任务变体，不能推断人。基础模型预训练重叠未知。 | 数据manifest与locks/data_split.json |
| Oracle Upper Bound，事后上限 | 根据标准答案／GT／完整16mask结果选max正确或min NLL | 同题事后诊断；不作部署排名 | 测试答案不能参与Gate选择；真实部署只能当前与过去输入。16mask训练/验证标签可用，不能把测试16mask最优结果当学习Gate。 | utility_gate/fixed_router_gate/enumerate_masks.py |


Accuracy改善而NLL恶化不矛盾：多救回小优势题，同时少数错误把标准答案概率压到极低，即可让总NLL变差。NLL是正确答案概率的适当评分，但仅凭均值不能完整描述校准，需要进一步可靠性图／Brier等；本轮只展示原有NLL和逐题尾部，不补造未做的校准实验。证据见 [IMU逐题尾部](imu_gold_nll_tail_examples.csv)。
