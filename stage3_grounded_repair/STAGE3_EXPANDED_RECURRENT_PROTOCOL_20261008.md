# 六教师共享循环匹配实验：冻结协议 v70–v72

目标是在相同起点、相同 TRAIN 数据和损失下，检验共享 Transformer 核心是否提供额外修复能力。所有原独立层保留，本实验是容量原型，不声称压缩或等 FLOPs。

v70 独立组合旧 ProcessCorpus 的三个 v50 加法教师与 v69 的三个加法教师；35 个真实 baseline-correct 状态全部保留，共41态。来源为 canonical112态／56提示，全部 TRAIN。原文件不覆盖。使用原 FP32 physical_delta；原状态加 delta 必须对应已经官方错到对且对原 canonical HPS／Pick 有限非负的加法终点。DEV／val／test 不入训练。

v71 在真实模型上复现六个加法终点，并检查41态新零输出循环终点与原 baseline 严格一致，六教师另检查无循环零输出。完整14次剩余速度重算，无旧 KV／inner prefix。重新读取实际加法状态的 Qwen27／flow9 表征，仅导出条件视觉 tokens；旧 v49 direct hidden 不当作加法 hidden。

三组为 no_loop、frozen_loop、trainable_loop。三组 operator 均始于 v52 joint_process 固定 step100，head 均为相同随机种子的3716→128→64 SiLU，输出层严格零初始化。所有41态原状态上，loop2 与 no_loop 特征等权计算共同逐维 RMS（FP64 累积、RMS 最小1e-4、保存 FP32），不读取目标计算归一化，三组使用完全相同 buffer。加载的低秩前向保留；frozen_loop 仅冻结低秩参数梯度。no_loop 跳过共享核心，其余反馈、证据、坐标与 head 保留。三组所有 adapter／head 初始 tensor hash 必须完全相同。

梯度证明分别记录真正零输出下的终点 head 梯度，以及独立随机 std1e-4 非零输出权重探针的完整14步终点梯度。探针后恢复原参数，optimizer0，不作训练起点。trainable_loop 外／内低秩核心与证据梯度均须非零；另外两组核心无梯度；基础权重无梯度且 hash 不变。

v72 每组固定420次 AdamW 更新，lr1e-4、weight_decay0、clip1、seed20261008。教师与保真交替各210次：6教师各35次、35正确态各6次。三组同序列、同损失、同目标、同 step420 checkpoint，不按终图／val／loss选 checkpoint。损失为0.5规范化物理状态 MSE＋0.5完整终点 MSE，沿用已有规范化公式；hidden 辅助权重0，以隔离新教师与匹配起点下的共享核心贡献。过程表征保存供后续诊断，不预称过程监督收益。

实际一次修复 loop2 重访 Qwen26／27 两次及 flow8／9 两次，外／内各4块调用；no_loop 为0／0。更新后完整14次速度，每次原26 semantic／28 Qwen／10 flow；base feature 采集、反向 checkpoint 重算、MMU／VAE、教师准备、端到端时间另计。FLOPs 未测为 null，调用更多不宣称等计算优势。

固定 step420 后只审计41态物理容量。门槛不变：每个六教师规范化 MSE<.01，35正确态最大值<.001。未过不自动生成终图／扩 val／训练 controller。容量通过后另冻结正式 TRAIN 终点评测，必须包括全部35态保真、官方／HPS／Pick／SigLIP、正确／乱序／反向证据和预算；TRAIN 总体质量与保真过门后才独立 val。最终推理只读中途 MMU／视觉 token／latent／坐标，不读父图、teacher、mask、像素、终标签或 teacher ID 来选动作。
