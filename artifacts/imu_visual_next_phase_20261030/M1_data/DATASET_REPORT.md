# 数据集与时钟报告

所有数据来自官方公开端点，访问、失败、Range 响应、条目 CRC 和文件 SHA 保存在本目录 JSON/JSONL。按范围取得的归档没有整包 SHA，不冒充完整下载。

|数据集|实际取得/检查|传感器与单位|严格因果可用性|本轮角色|
|---|---|---|---|---|
|UTD-MHAD|完整 Inertial.zip / Sample_Code.zip；861个MAT、8人；108个RGB AVI完整条目通过CRC并实际推理|50Hz，右腕(动作1–21)/右大腿(22–27)；acc g→m/s²，gyro deg/s|分发MAT仅d_iner[N,6]，无原始时间戳；名义时长2.14–6.52s，0段达到8s|已完成记录级开发 smoke，不能作为严格8s流式同步主数据|
|WEAR|官方目录、一个50Hz CSV头/片段、完整官方代码commit65cb6c4fda912dbce9b243e67d8a9bc73f4d87ba；未下载164GB整库|四腕/踝加速度计、12轴、50Hz、g；没有gyro|CSV没有时间列；官方预处理用首尾同步跳变建立时钟、先插值再20ms重采样，导出时丢掉时间。未取得插值前原始采样时钟，不能认证原始采集时间因果性|可下载≠已满足本协议；未运行QA/训练|
|HoloAssist|10段完整acc/gyro文本；覆盖完整视频时间线的稀疏RGB检查与前缀解码；12段IMU索引，40段视频索引；243条完整标注对象从12MB部分文件合法恢复，非完整117MB标注下载|HoloLens头部acc m/s² / gyro rad/s；原始.NET 100ns UTC ticks；已发布同步流约30Hz，非原始高频采样|10段中6段去重相机tick数与实际MP4帧数完全一致；4段差1帧，排除；可用标注与精确时钟交集4段，64道开发QA|2段单独校准固定量程；4段开发；无独立确认|

HoloAssist 不能直接用压缩MP4播放PTS当采集时间。以首段为例，完整时间线缩略图播放203.07s对应相机采集198.12s。采用 VideoMp4Timing 起点，在整数域减去.NET原点后再换秒；对 Pose_sync 的单调原始相机ticks去重，要求与MP4帧数精确一一对应；解码时逐帧验证CFR，按采集时钟选不晚于t的RGB。同步IMU行的原始传感器时刻可能晚于其第一列的RGB名义时间，必须按第二列原始ticks筛选。相差1帧不通过猜测偏移修复。

acc/gyro不同步时，在原始时刻并集上做只取过去值的zero-order hold；保存每个值的原始时刻，断言原始时刻≤并集时刻≤决策t。不做双向滤波或未来插值。发布数据的30Hz同步过程已经丢失的高频样本不能恢复；本轮不声称重建了原始高频IMU。量程只由另外2段校准来源估计，不用测试/当前窗口min-max。

HoloAssist派生QA来自官方 Fine grained action 的已完成Verb/Noun区间，不使用未来纠错结果。任务构建给定标注事件边界：把标注媒体时间映射到不早于结束边界的媒体帧，再取其原始相机时刻作为观察终点，四次观测为该终点前6/4/2/0秒。此为给定查询到达/裁剪边界的派生QA，不是官方QA，也不是在线自主事件检测或提前预测。候选动词词表由其它标注来源固定，选项位置由固定哈希随机化；各方法共享相同问题、候选、RGB、时间。身体位置/物理单位之外的标签不进入图表。受试者身份未提供，不把R编号当参与者编号，不声称跨参与者隔离。

UTD所有试验均为development，非官方27类排行榜。量程校准来源与测试take分离，但部分受试者重合，故不作跨人确认。所有已检视记录登记为seen；历史AudioRouter来源账本仍只读保留。未证明跨库人员重合或基础模型预训练无重合。

10段HoloAssist完整传感器文本和全时段RGB缩略图已检查；没有下载/逐帧解码10段完整视频，也没有独立标定残余同步误差。时间戳计数一致及定性运动检查不能被称为硬件同步误差为零。4段开发不足以建立大型主训练/验证/确认队列。

来源与许可：[UTD官方](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)（本次未找到单独明确的SPDX数据许可，不擅自称CC0）；[WEAR官方](https://mariusbock.github.io/wear/) / [预处理源码](https://github.com/mariusbock/wear/blob/65cb6c4fda912dbce9b243e67d8a9bc73f4d87ba/data_creation/utils.py)（CC BY-NC-SA4.0）；[HoloAssist官方](https://holoassist.github.io/)（CDLA Permissive2.0）；[Microsoft PSI导出格式](https://github.com/microsoft/psi/blob/master/Sources/MixedReality/HoloLensCapture/HoloLensCaptureExporter/Readme.md)（原始时钟及IMU单位）。WEAR论文22人，当前目录含24个subject条目，数据发布差异未混称同一版本。


逐文件实际字节数/SHA见`acquired_and_derived_file_sha256.json`；`acquisition_size_summary.json`区分完整RGB条目字节、已校验网络Range负载、部分文件、派生图/NPZ及缓存存储，组合缓存不重复计为网络流量。没有UTD RGB/Holo tar整包SHA。
