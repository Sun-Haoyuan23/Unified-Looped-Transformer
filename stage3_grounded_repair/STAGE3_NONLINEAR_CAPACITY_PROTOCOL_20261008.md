# 第三阶段：非线性读出容量实验固定协议（v60）

本协议在运行前冻结。目标是检验v59固定线性特征的修复／保真取舍能否由非线性读出缓解。它是TRAIN容量诊断，共享core本轮不更新，不承担生成质量或循环架构收益结论。

## 数据、对照与目标

保持v59同38个TRAIN状态：三严格成功教师00021/39/46 seed101，以及全部35真实baseline-correct态。ProcessCorpus保留原始官方与HPS/Pick质量门、身份及hash检查。dev/val/test和DEV成功态不入训练。

两组为v52 joint_process固定step100的真实两次共享循环特征、相同反馈但跳过共享core的无循环特征。外层Qwen26/27与内层flow8/9各重访四次block调用；无循环额外共享调用为零。两组输入逐态与v59 fingerprint严格相等，目标patch使用官方unpatchify严格往返。全部原base、core及旧adapter冻结，不新增共享core optimizer更新。

教师目标为v50同状态质量过门additive物理delta，正确态目标零。损失保持教师／保真总权重各0.5、组内状态均匀，分别用gold delta／原latent均方归一化。

## 读出与训练

将两个分支的token读出输入拼接，移除旧线性常数列，实际3716维。采用Linear(3716,128)→SiLU→Linear(128,64)，FP32、最后一层严格零初始化。每列除以38个TRAIN状态的无标签RMS，最小1e-4；两组normalizer各由自身特征计算，trainable tensor初始hash必须完全相同。没有状态ID、教师像素、mask、终图标签或终标签路由。

每组固定1000个full-batch Adam更新，lr1e-3、weight_decay0、seed20261008。固定head_step1000.pt，完整逐步日志；不早停、不挑checkpoint、不扫学习率、幅度、阈值、时机。不生成终图、不自动val、不训练controller。

运行前四项CPU测试覆盖零初始化与梯度、匹配初始化、真实循环／无循环路由和cache拒绝、物理CFG复制及梯度。实际特征必须通过76项v59 tensorhash一致检查。监测有限head梯度，结束后完整base／旧adapter／checkpoint hash保持、无梯度。

## 报告与后续门控

分别报告三教师、原三正确态、全部35正确态的逐态物理相对MSE和两组匹配差。预定容量门为每个教师MSE<0.01且35正确态最大MSE<0.001，仅用于决定是否值得做终点验证；它不替代官方错→对且HPS/Pick均有限非负的质量门。

head真实optimizer次数和历史端到端／共享core更新分别记录。对照即使都拟合成功，也只证明非线性读出表达能力。后续部署需验证未训练零完整14步终点严格相等、真实完整终点梯度、teacher-free前向、实际参数／成本，并做终图质量和35态保真；共享core可训练贡献、证据特异性、独立val和ELT式压缩仍要分别检验。

状态：源码准备完成，尚未启动；以pipeline_status.json和真实train.jsonl为准。
