> 最新接续：v25–v66全部完成并按范围核验，当前无运行任务；v67本地TRAIN原生教师元数据审计完成，6个新候选尚非成功修复教师。下一项v68原生教师接口按新冻结协议实际实现／验证，不重跑旧任务。见[STAGE3_NATIVE_TEACHER_20261008.md](STAGE3_NATIVE_TEACHER_20261008.md)。

# 第三阶段：非线性token读出容量结果（2026-10-08）

v60 nonlinear_readout_capacity_v60已完成，两组各1000次真实head optimizer更新；原始归档已下载、本地逐文件和逐字段统计核验通过。两组均未通过预定容量门，不自动进行终点评测、独立val或controller。

## 已证实的进展

相对同38态的v58线性读出，非线性读出降低全部35正确态的物理误差，两组35/35改善。循环组均值0.041204→0.002083，约下降94.9%；无循环0.041317→0.002186，约下降94.7%。三教师平均误差也降低，但只有cow/fridge改善，其余两个教师误差升高。

| 特征 | 同一配对组 | 物理相对MSE均值 v58→v60 | 最大值 v58→v60 | 改善状态 |
|---|---|---:|---:|---:|
| trained_loop2 | 三教师 | 0.187493→0.137871 | 0.552290→0.342153 | 1/3 |
| trained_loop2 | 35正确态 | 0.041204→0.002083 | 0.164461→0.049542 | 35/35 |
| trained_no_loop | 三教师 | 0.189865→0.096419 | 0.559161→0.236877 | 1/3 |
| trained_no_loop | 35正确态 | 0.041317→0.002186 | 0.155279→0.044365 | 35/35 |

| 特征 | 教师 | v60物理相对MSE |
|---|---|---:|
| trained_loop2 | train_extra_00021_seed101_i35 | 0.342153 |
| trained_loop2 | train_extra_00039_seed101_i35 | 0.035662 |
| trained_loop2 | train_extra_00046_seed101_i35 | 0.035797 |
| trained_no_loop | train_extra_00021_seed101_i35 | 0.236877 |
| trained_no_loop | train_extra_00039_seed101_i35 | 0.027392 |
| trained_no_loop | train_extra_00046_seed101_i35 | 0.024988 |

两组最大的保真误差均来自train_extra_00035_seed101_i35，提示为“a photo of a cow left of a refrigerator”。其循环／无循环误差0.049542／0.044365。同对象、同seed的教师00021要求cow right of refrigerator。这提示应核对相反关系的条件区分；尚不能认定碰撞或失败机制。第二对同对象同seed反向关系为教师00046 frisbee below couch与正确态00029 frisbee above couch，也要纳入审计，避免只挑最大的误差。

## 固定协议与验证

协议STAGE3_NONLINEAR_CAPACITY_PROTOCOL_20261008.md在运行前冻结。三严格TRAIN教师＋35真实正确TRAIN态；v52 joint_process step100的base、低秩共享core和旧adapter均冻结。真实loop2访问Qwen26/27与flow8/9各四次共享block，无循环保留相同反馈与证据、额外共享调用为零。76个实际输入／目标hash与v59逐一相等。

新head为3716→128→64、SiLU、FP32、最终层零初始化，484032个trainable参数。两组trainable初始化tensor hash相同；各自无标签TRAIN输入列RMS归一化。full-batch Adam lr1e-3，固定1000步，教师／保真组总权重各0.5，使用原物理目标归一化。无早停或checkpoint选择。

四项CPU接口测试全部通过；两组2000条实际更新日志均有限且梯度非零。模型及旧adapter全tensor hash、原checkpoint字节保持，无梯度。峰值6.71GiB；这是head训练成本，不是完整终点训练成本。固定特征缓存不代表部署省去基础模型调用。

预定容量门为每个教师物理MSE<0.01且35正确态最大MSE<0.001，两组均未达到。该门只决定是否值得做新终图验证，原官方错→对且HPS/Pick有限非负的质量门保持。固定1000步失败不证明非线性映射不可达，也不能排除优化／条件表达问题。

## 原始归档与本地核验

stage3_nonlinear_capacity_v60_results.tgz，4258147字节，30文件manifest：

`16e237d7f72a05ecc0b70d93fb3498b404c5a81a5cc8d81d7c908376719a286e`

v60_portable_verification.json核对source/config/冻结协议、固定head checkpoint、状态来源及76项特征／目标fingerprint，复核匹配初始化、实际2000条日志和38态分组结果。服务器Torch训练和tensor核验、本地标准库字节／统计核验分别记录；本地没有GPU重跑。

## 当前结论与下一步

v59显示固定线性头的严格保真约束，v60显示非线性token头能显著降低多数保真误差，但尚未兼顾教师与最大保真误差。无循环的教师误差更低；循环额外优势仍未成立。没有新增终图、官方分数或HPS/Pick，不能把物理拟合改善写成生成质量提升。

以下为v60完成时的后续计划，v61现已完成并核验：按元数据固定全部三教师、所有同对象同seed反向关系的正确伙伴，以及同提示不同seed正确控制，共六态；检查原latent、原始／压缩MMU条件、外／内hidden、head输入及物理更新的距离。仅在相同原latent和原提示下交换target/opposite MMU证据，量化读出敏感性；这不等于重新以反向提示生成，不续跑flow、不读终点选动作。先核查运行状态，防止重复启动。

累计真实optimizer更新4050，其中此前完整终点训练2050＋本轮固定特征head-only 2000；本轮共享core新增更新0。评分仍4222记录含342已记录基线别名、95动作翻正含重复、原教师DEV1/TRAIN3、178分轮接口测试，controller0。当前所有原独立层保留，部署参数账尚未改变；新head未部署。FID/FVD、人评、多样性、循环开启理解保真及反向提升、广域GenEval、共享core替换独立深度及总参数/FLOPs下降未完成，FLOPs null。第三阶段继续，10分钟轮询ACTIVE。
