> 最新接续：v52–v55均完成并本地核验，当前无运行任务；v53仍0语义修复，v55证明两例TRAIN目标具有可用线性读出。下一步v56配对终点试验尚未启动。见[STAGE3_READOUT_CAPACITY_20261008.md](STAGE3_READOUT_CAPACITY_20261008.md)。

# 第三阶段：过程监督实验结果与下一步诊断（2026-10-08）

v52四组各100步、共400次真实更新已完成；v53的144张TRAIN终图及官方/HPS/Pick/SigLIP评分全部完成。本地原始配对评分逐字段复算一致。所有非零动作仍0错→对，尚未实现共享循环的语义修复或总体生成提升。v54/v55后续诊断现也已完成，见文首最新接续。

## 1. 完成范围与公平对照

joint_control、outer_process、inner_process、joint_process全部固定最终step100。两joint前向、参数、初始化、教师目标和100步抽样完全一致，仅后者增加物理state/hidden辅助损失。实际初始化与v46匹配，全部抽样序列与v46相同。每组67次教师、33次真实正确态抽样，实际覆盖35保真池中的24个状态；不能将35写成每次都使用。

三教师仅TRAIN的00021/39/46_seed101，质量守门严格对原canonical基线。过程目标采用v50实际残差接口终点，隐藏目标来自同一additive状态的冻结基础表征。没有增加部署参数或循环调用。三教师+三正确态构成六状态、五独立提示；这是容量诊断，不是独立val。

| 组 | 真实更新 | 保真抽样 | 计时更新耗时/秒 | 峰值/GiB |
|---|---:|---:|---:|---:|
| joint_control | 100 | 33 | 2314.2 | 16.68 |
| outer_process | 100 | 33 | 2321.3 | 15.25 |
| inner_process | 100 | 33 | 2398.4 | 15.64 |
| joint_process | 100 | 33 | 2325.2 | 16.67 |

计时仅为记录的训练更新区间，未覆盖模型加载、首次目标/尺度捕获、教师构建或评测。不同辅助反传计算量，FLOPs未测，不称等FLOPs。

## 2. 正确证据、两次循环

| 组 | 错→对 | 对→错 | Pick差 | HPS差 | SigLIP差 |
|---|---:|---:|---:|---:|---:|
| joint_control | 0 | 1 | -0.284929 | -0.008341 | -0.001112 |
| outer_process | 0 | 0 | -0.041546 | -0.001506 | -0.001580 |
| inner_process | 0 | 1 | -0.245188 | -0.006063 | +0.004081 |
| joint_process | 0 | 1 | -0.281438 | -0.006185 | +0.004004 |

四组基线均3/6正确，质量守门离线oracle仍3/6。外层保住全部三正确态；其他组各损伤一个，均没有教师翻正。

教师三态：

| 组 | 错→对 | 对→错 | Pick差 | HPS差 | SigLIP差 |
|---|---:|---:|---:|---:|---:|
| joint_control | 0 | 0 | -0.349420 | -0.009033 | -0.010208 |
| outer_process | 0 | 0 | -0.007552 | -0.000488 | -0.001945 |
| inner_process | 0 | 0 | -0.285757 | -0.006429 | -0.004473 |
| joint_process | 0 | 0 | -0.266152 | -0.005941 | -0.000216 |

真实正确三态：

| 组 | 错→对 | 对→错 | Pick差 | HPS差 | SigLIP差 |
|---|---:|---:|---:|---:|---:|
| joint_control | 0 | 1 | -0.220439 | -0.007650 | +0.007984 |
| outer_process | 0 | 0 | -0.075540 | -0.002523 | -0.001214 |
| inner_process | 0 | 1 | -0.204619 | -0.005697 | +0.012635 |
| joint_process | 0 | 1 | -0.296724 | -0.006429 | +0.008223 |

## 3. 一次循环与证据方向

| 组 | 错→对 | 对→错 | Pick差 | HPS差 | SigLIP差 |
|---|---:|---:|---:|---:|---:|
| joint_control | 0 | 1 | -0.365427 | -0.008748 | -0.003289 |
| outer_process | 0 | 0 | -0.039499 | -0.001709 | -0.001982 |
| inner_process | 0 | 1 | -0.316856 | -0.007284 | +0.001938 |
| joint_process | 0 | 1 | -0.258061 | -0.006836 | +0.001266 |

乱序、两次循环：

| 组 | 错→对 | 对→错 | Pick差 | HPS差 | SigLIP差 |
|---|---:|---:|---:|---:|---:|
| joint_control | 0 | 1 | -0.227194 | -0.007894 | -0.001538 |
| outer_process | 0 | 0 | -0.048477 | -0.001668 | -0.002347 |
| inner_process | 0 | 1 | -0.244537 | -0.007039 | +0.004941 |
| joint_process | 0 | 1 | -0.250277 | -0.005778 | +0.004375 |

反向、两次循环：

| 组 | 错→对 | 对→错 | Pick差 | HPS差 | SigLIP差 |
|---|---:|---:|---:|---:|---:|
| joint_control | 0 | 1 | -0.241267 | -0.008586 | +0.000676 |
| outer_process | 0 | 0 | -0.065101 | -0.001872 | -0.003153 |
| inner_process | 0 | 1 | -0.250236 | -0.006226 | +0.004941 |
| joint_process | 0 | 1 | -0.259887 | -0.006510 | +0.004887 |

相比匹配joint_control，joint_process两次循环全六态平均Pick差额+0.003491、HPS+0.002157、SigLIP+0.005116。这是相对较少损伤，尚未超过原基线或完成语义修复。joint乱序Pick/HPS仍优于正确证据，证据特异性未成立。

局部质量非负观测完整保存在[v53_quality_only_states.json](v53_quality_only_states.json)。两次循环中joint_control的horse、inner_process的horse/frisbee两个教师、joint_process的horse出现局部HPS/Pick非负，语义全部仍错；不挑局部替代整体结果。

## 4. 实际残差教师的终点距离

下表相对MSE对v50同状态additive终点，基线为1，越小越接近教师。与原v36历史终点距离分开记录：

| 组 | cow/fridge | horse/frisbee | frisbee/couch |
|---|---:|---:|---:|
| joint_control | 0.373472 | 0.761830 | 0.628279 |
| outer_process | 0.922123 | 0.992973 | 0.979012 |
| inner_process | 0.411406 | 1.065016 | 0.865867 |
| joint_process | 0.407610 | 0.777852 | 0.664381 |

联合过程组的三教师终点距离均未优于匹配终点控制。训练日志最后一个教师的state损失仍约1，说明加入过程项尚未充分逼近物理教师更新；此时不能仅凭hidden损失下降判定生成核心有效。对齐MSE在服务器Torch CPU计算，本地核对其文件、冻结源码及原始latent hash；本地未用Torch重复该算术。

## 5. 已完成的v54只读诊断

process_objective_audit_v54，原launcher78653。四worker已结束：joint_control78665/GPU0、outer_process78664/GPU4、inner_process78662/GPU6、joint_process78663/GPU7。启动前逐物理编号核对UUID与空闲，不中止他人。

固定六TRAIN状态、两预算、四组，严格复现48个v53终点。逐态测量预测/教师物理增量RMS与方向、两层循环hidden对齐误差，以及终点/state/hidden梯度的夹角和沿实际总损失下降方向的导数，分别记录核心、读出、反馈、证据与token/位置投影。只用autograd.grad，无optimizer、无新评分图、无自动训练/val/controller。检查基础和适配器hash、检查点字节均保持，且不填充参数.grad。

现已完成48次严格终点复现且原始数据本地核验。物理更新方向、局部梯度冲突及后续v55线性拟合见文首最新接续；这些诊断尚非质量失败的因果证明。继续保留真实共享外层/内层/联动循环。

## 6. 原始归档、计数与尚未完成的目标

| 归档 | SHA256 |
|---|---|
| stage3_process_v52_complete_results.tgz | 32837fbf1863f79664e3388f95c0060f01642f07e87049deabec6c20fcfddfa8 |
| stage3_process_trainfit_v53_results.tgz | 1eb5794da19fd25e575ae8445a8702b4e6e98c6b9a7fdc7fe9ddf446d219bccb |

v52归档含完整400步、固定四检查点、source/config、服务器CPU tensor审计，本地44文件manifest及日志核验。v53归档含144图及latent、四项原始评分、source/config，685文件manifest、144官方布局别名与便携配对报告严格复核。详见[v52_complete_portable_verification.json](v52_complete_portable_verification.json)、[v53_complete_portable_verification.json](v53_complete_portable_verification.json)。服务器tensor检查与本地字节/统计检查明确分开。

累计2,050次已完成真实更新；4,156评分记录含342已记录的基线别名，不是独立图数；77动作级翻正含重复、21独立纠正均离线，DEV1/TRAIN3；166接口单测分轮通过，controller0。v54不增加评分或更新。

现有新增参数outer681608、inner808584、joint1185736；全部原独立层保留，尚非ELT压缩骨干。完整名义速度50对照49；两循环的Qwen/flow/semantic总调用外层1404/500/1300，内层1400/504/1300，联动1404/504/1300，对照1372/490/1274。MMU/AR检查、VAE、读出、低秩构造、教师采集和端到端成本另计，FLOPs为null。当前推理walltime只包含已捕获状态后的延续，不是端到端生成加理解检查耗时。

独立val、广域GenEval、FID/FVD、人评、多样性、理解反向提升、MMU/AR共同监督及总参数/FLOPs下降仍未完成。阶段3继续，10分钟接续保持ACTIVE。
