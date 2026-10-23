# v79 四循环过程对齐与参考梯度：冻结只读协议

前置v78四组固定step420已完整本地核验，173项manifest，归档SHA256 c080833e43b9eca15f37aad4db062bed096d8b8fac299a859271562dbba1a9f4。四组均未过原容量门。新目录process_four_loop_alignment_v79，0optimizer、0拟合、0新图或评分；原源码、模型、检查点和门限保持。

同六严格TRAIN教师和全部35真正确态，同v77冻结目标/尺度/RMS，重新读取各v78固定检查点。逐态复核164个物理MSE及hidden loss，rtol1e-4/atol1e-6，不预设bitwise预测。各路线记录条件视觉token的外/内规范化hidden误差、原状态至预测与目标的方向余弦/RMS。三原几何和三原生教师分别统计，保留全部失败。

主anchor_trainable4在41态同一次前向读取固定1/2/4前缀，评估(1,1)、(2,2)、(4,4)、(4,2)、(2,4)的物理预测及hidden；只作固定head下的表征响应诊断，不重新拟合、不挑预算、不把头敏感解释为架构收益。四组各六教师完整14速度续跑，共24个终点latent；仅对齐v50/v69目标的规范化terminal MSE，无VAE、图或质量评分。物理变化后禁止旧KV/prefix。

主anchor_trainable4六教师另各做一次完整14终点前向和反传，分别读取physical/terminal/hidden参数梯度，报告公共与共享低秩参数范数及两两余弦；局部梯度不能单独证明失败原因。targets detach，不更新参数。

独立恢复v77实际初始operator/零feedback，固定首教师00006的原状态及实际additive hidden目标。外/内各测1/2/4预算，比较既有同前向缓存且连接的参考与每次重算且连接的参考。前向必须严格等原hidden。记录真实BF16/FP32 dtype、每参数梯度和范数；CPU FP64同式零反馈取消检查。差异只能支持计算图/数值归因，不直接作为v78语义失败因果，也不改部署实现。所有探针无参数修改，基础/原core/adapter/head哈希和无基础梯度全部复查。

协议/源码、四固定检查点、41 state、目标manifest/RMS与来源哈希训练前冻结。按物理编号核对空闲GPU及UUID后启动一个有限只读worker。实际速度/共享调用、反向重算、walltime/峰显存分列，FLOPs null。完整source/config/原始逐态JSON/终点latent/数值梯度/manifest归档后本地字节和标准库统计核验，服务器Torch范围单列。DEV1和dev/val/test禁训练；controller0，不自动终图/val/下一轮训练。
