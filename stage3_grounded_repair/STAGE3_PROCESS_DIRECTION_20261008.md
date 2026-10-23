> 最新接续：v52–v55均完成并本地核验，当前无运行任务；v53仍0语义修复，v55证明两例TRAIN目标具有可用线性读出。下一步v56配对终点试验尚未启动。见[STAGE3_READOUT_CAPACITY_20261008.md](STAGE3_READOUT_CAPACITY_20261008.md)。

# 第三阶段：物理更新方向与读出可达性（2026-10-08）

v52/v53已完成并完整本地核验，过程监督未带来语义修复。v54只读诊断也已完成：48个终点严格复现，基础/适配器及检查点均保持，无optimizer更新或新评分图。v55线性读出可达性探针已在GPU0完成，原PID93369已结束，结果见文首最新接续。

## 1. 方向偏差是已测观测

同六TRAIN状态、四组、循环预算1/2。下表仅列三教师、两次循环，MSE以原状态为1：

| 组 | TRAIN教师 | 更新/教师RMS | 方向余弦 | 物理相对MSE |
|---|---|---:|---:|---:|
| joint_control | train_extra_00021_seed101_i35 | 0.2417 | 0.2239 | 0.9502 |
| joint_control | train_extra_00039_seed101_i35 | 1.5801 | 0.3059 | 2.5302 |
| joint_control | train_extra_00046_seed101_i35 | 1.6096 | 0.3499 | 2.4643 |
| outer_process | train_extra_00021_seed101_i35 | 0.0527 | 0.1092 | 0.9913 |
| outer_process | train_extra_00039_seed101_i35 | 0.5429 | 0.1828 | 1.0963 |
| outer_process | train_extra_00046_seed101_i35 | 0.6102 | 0.2162 | 1.1085 |
| inner_process | train_extra_00021_seed101_i35 | 0.2013 | 0.3286 | 0.9082 |
| inner_process | train_extra_00039_seed101_i35 | 0.9736 | 0.4167 | 1.1364 |
| inner_process | train_extra_00046_seed101_i35 | 0.9410 | 0.4543 | 1.0305 |
| joint_process | train_extra_00021_seed101_i35 | 0.1984 | 0.3347 | 0.9065 |
| joint_process | train_extra_00039_seed101_i35 | 0.9779 | 0.4658 | 1.0452 |
| joint_process | train_extra_00046_seed101_i35 | 0.9540 | 0.5007 | 0.9548 |

联动组horse/frisbee与frisbee/couch的更新RMS已分别达到教师的97.79%和95.40%，方向余弦仍仅0.466与0.501。cow/fridge更新RMS只有19.84%，但方向余弦也仅0.335。即使事后选择最优单一缩放，三例物理相对MSE的下界仍约0.888、0.783、0.749；这仅是欧氏方向的解析诊断，不是生成质量评测。继续扫强度不能解决该方向差异。

## 2. 梯度有局部冲突，不能当统一原因

全参数state/hidden梯度余弦在三个过程组教师上接近0（绝对值最大约0.0047），其中一部分来自读出与核心使用的参数子空间不同。低秩核心内部有正有负的冲突，例如联动cow/fridge内层state/hidden余弦约−0.514，horse外层约−0.239。

但是在已训练检查点上，三个过程组、三教师的实际总损失负梯度，对terminal和state的局部方向导数均为负。不能把所有失败直接归因为梯度互相抵消，也不能凭某一局部夹角删除监督项。没有执行任何参数更新来测试因果效应。

原始逐组梯度见[v54_gradient_geometry.json](v54_gradient_geometry.json)，物理方向与最优缩放下界见[v54_physical_alignment.json](v54_physical_alignment.json)。

## 3. 已实际启动v55：固定特征的线性读出可达性

在同一六TRAIN状态上，只读v52固定joint_process检查点与匹配的初始operator，提取原循环head实际输入。五个预定探针为initial_no_loop、initial_loop2、trained_no_loop、trained_loop2、trained_nested。无循环对照保留同一反馈/证据/坐标构造，仅跳过共享Transformer重访；nested使用一组读出系数同时拟合一次和两次循环。

用同一固定正则1e−4的加权ridge解析拟合已有线性读出空间，逐列做加权L2归一化。目标为三个合格教师的物理增量和三个真实正确态的零增量；等状态权重、沿用过程损失分母。目标patch顺序通过官方unpatchify逐元素往返检查。仅TRAIN，无val/dev/test，没有阈值或正则扫描。

这是读出可行性探针：会拟合并保存离线系数，原模型和检查点不变，0optimizer、0新生成评分图。小训练误差可证明这些特征存在可用的线性读出；大误差不能单独证明不可达，因为还有正则、条件数与样本约束。nested同时约束两预算，比单预算多约束，差异不能只归于Transformer循环。任何误差下降都不是质量或泛化提升。

优先检查process_readout_probe_v55/pipeline_status.json、results.json和transfers/process_readout_probe_v55_launcher.log。现已全部完成并归档，结果见文首最新接续。探针完成后，依据无循环/循环、初始/已训练及共享多预算的拟合差异，决定优化或表征接口的下一步，不重跑已完成v25–v54。

## 4. 原始归档与范围

v54归档stage3_process_objective_v54_results.tgz SHA256：f41e22a012038affb9e4cc5a593ef87fb468f7d842b2deaac37b862be8cc9450。29文件manifest、source/config/检查点对应和本地梯度统计恒等式均核验通过，见[v54_portable_verification.json](v54_portable_verification.json)。48次模型tensor复现由服务器执行，本地核对原始归档及统计，未重新运行模型。

累计仍为2,050已完成真实optimizer更新、4,156评分记录（342已记录的基线别名）、77动作翻正含重复、21独立纠正均离线、DEV1/TRAIN3、166分轮单测，controller0。v54不增加更新或评分；v55解析拟合另记，不计作optimizer。

阶段3未完成；外层/内层/联动循环继续保留，独立val前不训练controller。FID/FVD、人评、多样性、广域GenEval、理解反向提升、共享core替代独立深度及总参数/FLOPs下降仍待完成，FLOPs为null。10分钟接续继续ACTIVE。
