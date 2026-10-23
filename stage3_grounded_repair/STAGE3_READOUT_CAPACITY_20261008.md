> 最新接续：v25–v66全部完成并按范围核验，当前无运行任务；v67本地TRAIN原生教师元数据审计完成，6个新候选尚非成功修复教师。下一项v68原生教师接口按新冻结协议实际实现／验证，不重跑旧任务。见[STAGE3_NATIVE_TEACHER_20261008.md](STAGE3_NATIVE_TEACHER_20261008.md)。

# 第三阶段：读出可达性结果与固定终点试验接续（2026-10-08）

v52、v53、v54和v55均已完成并完整本地归档核验。当前没有运行中的这些任务，不得重启。v53仍0语义翻正；v55仅证明某些TRAIN物理目标存在较好线性拟合，没有生成图或质量评分。下一步v56尚未实现或启动。

## 1. v55结果：不能把两例失败归因于读出表征完全不可用

同六TRAIN状态、五独立提示；三严格教师、三真实正确态。原始/已训练特征、共享循环/同参数无循环使用相同预定ridge1e−4、加权L2列归一化和目标。下表是物理相对MSE，错误教师的原状态为1；正确态归一化到原状态能量：

| 固定特征/读出约束 | cow/fridge | horse/frisbee | frisbee/couch | 正确三态均值 |
|---|---:|---:|---:|---:|
| initial_no_loop/no_loop | 0.430908 | 0.005797 | 0.004341 | 0.028303 |
| initial_loop2/loop2 | 0.433008 | 0.005724 | 0.004351 | 0.026235 |
| trained_no_loop/no_loop | 0.436771 | 0.005965 | 0.004471 | 0.028238 |
| trained_loop2/loop2 | 0.429224 | 0.005653 | 0.004322 | 0.026462 |
| trained_nested/loop1 | 0.491046 | 0.020560 | 0.020347 | 0.027825 |
| trained_nested/loop2 | 0.543642 | 0.020663 | 0.021492 | 0.019625 |

v54已训练联动组两次循环horse/frisbee、frisbee/couch的物理相对MSE为1.045221、0.954770。v55固定同一循环特征后的线性拟合分别为0.005653、0.004322。这是可用读出存在的具体证据，支持下一步将物理监督拟合出的读出实际推进到终点；不能再仅用低秩舍入或缺少有效方向解释全部问题。

无循环对照也达到0.005965、0.004471，说明该诊断尚未支持循环特异优势。cow/fridge仍约0.43，且正确态非零误差约0.026；不能预设新读出的生成质量或保真通过。共享一次/两次预算的单一读出误差较高，还叠加了多前缀一致性约束，不能只归于循环表征损坏。

模型和检查点始终未修改。五个解析拟合系数保存为离线probe.pt，FP64 solver实际运行于服务器；本地核验字节/source/config和有限结果，未重复Torch solver。0optimizer、0新评分图；解析拟合另记，不包装成dry-run或optimizer训练。

## 2. 下一步v56：实际残差终点与同参数无循环对照（尚未启动）

先实现新的prepare/build/run脚本和目录，禁止修改v52/v53/v55冻结源码/检查点。固定使用v55 trained_loop2与trained_no_loop两个预定解析系数，不按最终图、val或测试挑系数。原始teacher仍仅三个，不转入DEV/val/test。

准备阶段将FP64系数转为现有FP32线性readout权重：按outer/inner实际readout输入维度拆分最后一维，公共bias放入outer、inner bias置零，记录转换误差及与解析预测的一致性。只有现有readout权重改变；其余v52 step100适配器与共享core保持。派生检查点必须明确derived_from_optimizer_step100+analytical_fit，不假称新的100步训练。浮点转换后也要实际验证，不以FP64拟合误差代替部署结果。

生成阶段只读原中途MMU/视觉token/latent/坐标与冻结派生参数，不能读取外部mask、像素教师或最终标签。no_loop保留同一反馈、证据和线性head，只跳过共享Transformer重访并按实际0共享调用计费；loop2执行真实共享Qwen26/27和flow8/9。每次物理状态改变后完整14次速度，禁用旧KV/prefix。专门新建未训练zero读出，严格复现原基线。

预定loop2组六TRAIN态×六动作=36图，含baseline/fresh_zero/正确/乱序/反向/一次循环；no_loop组六态×五公共动作=30图，不重复生成无意义的另一次零循环。共66条官方/HPS/Pick/SigLIP评分。固定同一cohort、seed和中途时间，完整记录全部状态与成本。先冻结此协议再运行，不随首批分数改变动作或教师。

这是TRAIN容量/质量实验，未授权自动val/controller。两组若都学会修复，则只能宣称学习的读出映射可行，不能把收益归于循环；必须检验loop额外收益、证据方向和保真。若循环组出现严格语义及HPS/Pick非负修复，仍须独立val和真实core对照后才能推广结论。若失败，按部署FP32误差、终点可塑性和保真诊断，不扫幅度/阈值/时间、不降低质量门。

## 3. 当前完整归档与状态

| 归档 | SHA256 |
|---|---|
| v52 stage3_process_v52_complete_results.tgz | 32837fbf1863f79664e3388f95c0060f01642f07e87049deabec6c20fcfddfa8 |
| v53 stage3_process_trainfit_v53_results.tgz | 1eb5794da19fd25e575ae8445a8702b4e6e98c6b9a7fdc7fe9ddf446d219bccb |
| v54 stage3_process_objective_v54_results.tgz | f41e22a012038affb9e4cc5a593ef87fb468f7d842b2deaac37b862be8cc9450 |
| v55 stage3_process_readout_v55_results.tgz | 44a883b19f8cc0f95fc871e12965b593f842e816f4f5335a39c79aa8b1d2408b |

分别有v52_complete_portable_verification.json、v53_complete_portable_verification.json、v54_portable_verification.json、v55_portable_verification.json。v52四组各100真实更新；v53全部144图、原始四评分便携逐字段严格一致；v54全部48终点严格tensor复现且无更新；v55五解析系数/source/config及17文件manifest核验。实际模型/solver操作在服务器，本地字节与统计核验范围明确区分。

累计2,050真实optimizer更新、4,156评分记录（342已记录基线别名，不是独立图数）、77动作翻正含重复、21独立纠正均离线、DEV1/TRAIN3、166分轮接口单测、controller0。当前loop质量/语义提升仍未成功，阶段3继续，10分钟轮询ACTIVE。

外层/内层/联动循环保留；总参数/FLOPs降低仍未实现。MMU/AR共同监督、循环开启理解保真及反向提升、共享core替代独立深度、FID/FVD、人评、多样性和广域GenEval未完成，不预填。
