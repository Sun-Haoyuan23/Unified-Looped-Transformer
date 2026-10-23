> 最新接续：v25–v66全部完成并按范围核验，当前无运行任务；v67本地TRAIN原生教师元数据审计完成，6个新候选尚非成功修复教师。下一项v68原生教师接口按新冻结协议实际实现／验证，不重跑旧任务。见[STAGE3_NATIVE_TEACHER_20261008.md](STAGE3_NATIVE_TEACHER_20261008.md)。

# 第三阶段：相反关系与MMU条件区分审计（2026-10-08）

v61 condition_separation_audit_v61已经完成并完整本地核验。v25–v61当前均无运行任务，禁止重复启动。v61是只读诊断：0optimizer、0解析拟合、0新图或评分。

## 结论

当前残差不能简单解释为“相反提示条件完全重合”或“读出完全忽略MMU”。两对同对象、同seed、相反关系的状态，其原latent、MMU条件和head token输入均可区分；只交换同一原状态的target/opposite MMU证据，也会明显改变预测物理更新。这个结果只排除了当前六TRAIN态中的完全碰撞／完全不敏感解释，没有证明正确使用证据或正确空间修复。

## 预定选择与实际距离

按元数据选择全部三严格TRAIN教师，再加入所有同对象同seed反向关系正确伙伴、同提示不同seed正确控制，得到六状态、两对反向关系和一对seed控制。没有按v61终点图选择状态：

- cow right / cow left refrigerator，seed101：教师00021与正确00035。
- frisbee below / above couch，seed101：教师00046与正确00029。
- 同cow right refrigerator提示、seed101 / 202：教师00021与正确00021。
- horse above frisbee seed101教师保留作单态证据敏感性诊断。

下表距离为差值RMS除以两输入平均能量的平方根。token列使用各自v60冻结无标签RMS归一化，跨模式值仅作描述，不能据此判断循环抹掉语义。

| TRAIN状态对 | 配对类型 | latent相对RMS | 原MMU条件 | 压缩条件 | head输入 循环／无循环 |
|---|---|---:|---:|---:|---:|
| train_extra_00021 / train_extra_00035 | 同seed反向关系 | 0.115201 | 0.460100 | 0.130309 | 0.397691 / 0.456082 |
| train_extra_00046 / train_extra_00029 | 同seed反向关系 | 0.296315 | 0.732185 | 0.230407 | 0.623461 / 0.708668 |
| train_extra_00021 / train_extra_00021 | 同提示不同seed | 1.360549 | 0.479429 | 0.147072 | 1.079957 / 1.163633 |

cow同seed反向关系的原latent相对差0.115201，MMU条件差0.460100，head输入差0.397691／0.456082；它们没有严格重合。frisbee反向关系对应0.296315、0.732185、0.623461／0.708668，也可区分。这里的距离不是可分类准确率，不足以断言条件表示已经足够。

## 同状态证据扰动

原latent和原提示保持，仅交换target/opposite的margin、hidden、question，将evidence取负并交换关系字符串。两次交换回到原条件tensor，实际检查通过。重新执行原预算的共享重访与冻结head，不改变物理state、不继续flow。

在12个状态×模式观测中，交换证据使预测物理更新的相对RMS变化约0.453–1.412，token变化约0.062–0.147。读出对证据有反应，但方向是否正确、是否改善终图仍未验证。这不是将prompt改成反向关系后的完整生成，不能当作该实验。

三教师target/opposite MMU margins分别为cow 1.000/2.875、horse 0.625/1.500、frisbee -0.375/2.875；contrastive evidence为target减opposite。不能只看target margin正负代替原contrastive证据规则。

## 验证和归档

12项冻结v60物理MSE复现通过，固定容差rtol1e-4、atol1e-6。逐状态与full-batch FP32 head的GEMM形状不同，未声称这些预测bitwise相等。base、旧adapter和两head全tensor hash保持、无梯度；未执行终点rollout或新官方／偏好评分。

stage3_condition_separation_v61_results.tgz，28783字节，17文件manifest：

`eef64c1772c2f157c60cd2d7495305ed0a98326970af03e7707f2091487fc744`

v61_portable_verification.json已核对source/config/checkpoint/head/state来源、固定cohort、12项参考误差和实际共享调用。服务器Torch运算与本地字节／标量复核分开记录；本地未GPU重跑。

## 下一步与当前进度

以下为v61完成时的后续计划，v62现已完成并核验：在相同38TRAIN状态、相同v60固定head结构和step1000参数、相同归一化与损失下，用预先固定的full-batch L-BFGS强Wolfe过程检验残差是否主要受优化限制。每组max_iter100、max_eval125、history_size20、lr1、tolerance_grad1e-7、tolerance_change1e-9；两组同上限，实际函数调用、内部迭代、optimizer.step调用和walltime分别记录，不称等FLOPs。不选最优图、checkpoint或超参，不改变容量／质量门。

先核查服务器Torch LBFGS接口和现有目录，禁止重复v60或v61。若有界优化仍未过门，再依据误差与跨token空间映射需求设计更有表达力的共享循环修复；不因本次1000步拟合失败便宣称非线性不可达。终点教师只用于TRAIN监督和事后分析；推理无teacher／像素／mask／终标签。TRAIN总体质量与保真未过不自动val/controller。

v60已经使同35正确态物理误差均值约下降95%，但残留最大误差和三教师未过容量门；无循环教师拟合更好，没有循环额外收益结论。v56已有两个严格TRAIN学习读出修复，但无循环也有、整体HPS负且损伤一正确态；广域生成质量与GenEval目标仍未实现。

累计4050真实optimizer更新＝此前2050完整终点训练＋2000固定特征head-only；v61新增0。评分4222条含342已记录基线别名，95动作翻正含重复，DEV1/TRAIN3原教师池不变，178分轮接口测试，controller0。FID/FVD、人评、多样性、循环开启理解保真／反向提升、ELT共享core替换独立深度、广域GenEval、总参数/FLOPs下降仍待完成，FLOPs null。第三阶段继续，10分钟轮询ACTIVE。
