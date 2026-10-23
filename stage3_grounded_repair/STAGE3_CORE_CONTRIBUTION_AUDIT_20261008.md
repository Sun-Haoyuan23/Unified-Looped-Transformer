# 共享循环核心实际贡献审计（v48，2026-10-08）

v48已完成：24次原有v47终点严格复现，以及24次关闭学得低秩更新的反事实完整flow延续。更新0，新评分图0；基础权重与适配器hash不变，没有梯度。本地归档逐文件hash、冻结source/config/checkpoint及便携拟合统计已核验。

最关键的新证据是：低秩更新在BF16下确实生效，但对当前修复器的物理更新作用很小。不能再用“低秩完全被舍入吞掉”解释失败；也不能仅凭梯度非零证明循环核心带来了生成收益。

## 审计方法

固定v46最终step100、同六个TRAIN状态、正确MMU证据、两次共享循环和同一训练读出。学习核心开启时必须与既有v47 latent逐元素相等，全部24次通过。随后只将各共享核心的有效权重W+BA切回W，保留循环次数、反馈、证据、读出和14次完整剩余速度。该对照测量固定检查点对学得低秩更新的推理依赖，不能代表重新训练后的核心贡献或等FLOPs优势。

“有效权重改变”统计q/k/v/o投影中BF16材质化后(W+BA)与W不相等的元素，按元素数加权。“物理更新变化比例”为两种delta差的RMS除以学得核心开启时delta的RMS，表内只列三教师。

| 组 | BF16有效权重改变比例 | 三教师物理更新变化比例范围 |
|---|---:|---:|
| outer_ilsd | 47.33% | 0.464%–1.671% |
| inner_ilsd | 38.72% | 0.534%–1.080% |
| joint_ilsd | 36.24% | 0.837%–0.940% |
| joint_terminal | 36.85% | 0.685%–0.834% |

有效权重改变约36%–47%，对应物理更新只变化约0.46%–1.67%。这些比例的分母和含义不同，不可相除推断贡献率；也未证明readout绕过了全部冻结循环的价值。

## 关闭低秩后，终点拟合几乎仍在

以下终点残差MSE以教师−基线MSE归一化，基线为1。无低秩列仍执行同一冻结共享核心；不是零循环或仅MLP：

| 组 | 教师 | 开启低秩 | 关闭低秩 |
|---|---|---:|---:|
| outer_ilsd | train_extra_00021_seed101_i35 | 0.883899 | 0.883781 |
| outer_ilsd | train_extra_00039_seed101_i35 | 0.917157 | 0.917875 |
| outer_ilsd | train_extra_00046_seed101_i35 | 0.926913 | 0.928404 |
| inner_ilsd | train_extra_00021_seed101_i35 | 0.400592 | 0.403342 |
| inner_ilsd | train_extra_00039_seed101_i35 | 1.093021 | 1.095809 |
| inner_ilsd | train_extra_00046_seed101_i35 | 0.907897 | 0.910687 |
| joint_ilsd | train_extra_00021_seed101_i35 | 0.375107 | 0.377013 |
| joint_ilsd | train_extra_00039_seed101_i35 | 0.805602 | 0.814170 |
| joint_ilsd | train_extra_00046_seed101_i35 | 0.632408 | 0.637830 |
| joint_terminal | train_extra_00021_seed101_i35 | 0.368317 | 0.369753 |
| joint_terminal | train_extra_00039_seed101_i35 | 0.761402 | 0.763425 |
| joint_terminal | train_extra_00046_seed101_i35 | 0.653083 | 0.655589 |

例如joint_ilsd的cow/fridge误差0.375107，关闭低秩后仍为0.377013；主要拟合进展可以在未学得低秩更新的当前推理路径中保留。LoRA-off没有官方/HPS/Pick评分，不能判断它改善或损害语义与质量。

## 下一实验：先对齐成功教师的中途状态和循环表征

v47没有语义修复，v48显示学得核心的作用有限。下一步先建立可核验的过程监督接口，保持真正共享Transformer循环：

1. 只从三个严格TRAIN成功教师的原始几何/VAE文件与已冻结bridge重建index35修复输入。cow/fridge用原inferred_correlated镜像残差接口，另两例用原inferred Linear bridge。输入tensor hash必须与v23/v35记录一致；完整14步终点必须严格复现canonical教师。35个真实正确保真态目标为原状态，DEV/val/test禁止使用。
2. 冻结三条教师中途state与完整flow轨迹，并读取相同时间、相同冻结基础模型的Qwen27/flow9表征。区分直接赋值teacher state与“原状态+delta”浮点重建；若不严格一致，记录差异并核验终点，不假设残差接口完全可达。
3. 为现有同一可训练共享core增加TRAIN过程监督：循环hidden对齐修复状态的基础模型hidden，物理状态读出对齐修复输入，保留完整终点及真实正确态保真。教师只在训练监督中出现，推理仍只读中途MMU/visual token/latent/坐标。
4. 新损失先做真实核心/证据梯度、base无梯度、corehash不变和未训练零干预终点恒等检查，再冻结过程监督与终点监督的同初始化对照。先验证三教师容量与质量，不扩大旧负结果的val，不训练controller。

这里的推断是全局终点MSE对循环核心的约束可能太间接；v48不证明它是唯一原因。过程监督仍需实际检验，不能预写生成提升。若三教师都不能可靠拟合，应继续检查teacher状态到共享表征的可达性；若能拟合才扩充独立TRAIN教师和独立val。后续还须实现共享core替代原独立深度，才能支持总参数/FLOPs降低。

## 原始证据

归档stage3_recurrent_core_audit_v48_results.tgz，SHA256：e53343aebecdbfda002eca07f98b08c2b9f6681e998644ef905a928dfb216f98。

本地results/recurrent_core_audit_v48包含48个终点/physical_delta文件、权重统计、逐状态日志、配置与源码快照。核验记录[v48_portable_verification.json](v48_portable_verification.json)明确区分服务器实际tensor恒等检查与本地字节hash/统计复核，本地没有重复运行Torch模型。

第三阶段仍未完成；评分记录4,006、真实optimizer更新1,650、接口测试161分轮、controller0的计数均不变。
