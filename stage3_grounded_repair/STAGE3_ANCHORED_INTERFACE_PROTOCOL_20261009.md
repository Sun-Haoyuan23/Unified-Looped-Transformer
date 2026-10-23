# v75 原表征锚定的共享反馈循环：冻结接口与诊断协议

v74三组从同一v72可训练循环step420参数出发，均完成100内部接受更新但容量门失败。可训练循环对冻结循环六教师物理误差均更高，34/35正确态更低；无循环对可训练循环六教师均更低。当前plain循环把已经经过Qwen26/27、flow8/9的输出再次送入同组，即使反馈为零也会重复改变表征。此事实提出待验证的表达假设；尚不认定它是训练失败原因。

v75新目录anchored_recurrent_interface_v75，0optimizer、0解析拟合、0新评分图。旧源码、权重、协议及结果不覆盖。先复核v74完整本地便携证明、固定最终接受checkpoint及六教师／35正确态来源。所有诊断使用v74 trainable_loop最终接受operator、现有共同RMS；原非线性head仅用于来源校验，新接口head重新以seed20261008零输出初始化，不部署旧head跨路由预测。

对同一共享Transformer组G、原始输出h0与反馈u，先计算同前向reference=G(h0)，循环h1=h0+[G(h0+u)-G(h0)]，h2=h0+[G(h1)-G(h0)]。计算顺序必须先FP32减法形成差分，再加h0并回原dtype，保证u=0时严格恒等。reference不detach，梯度进入同一组低秩A/B并在零反馈处抵消；reference只在一次调用内复用，不跨物理状态／prompt／index缓存。外Qwen26/27、内flow8/9共享参数及A/B跨预算复用。一次／两次预算实际额外外内块各4/6，plain两次各4，计算增加且FLOPs未测；部署参数不增加，仍保留全部原独立层，不能声称压缩。

4项CPU检查涵盖零反馈严格恒等及参考梯度抵消、BF16恒等、嵌套预算与调用数、拒绝KV。真实模型对41TRAIN态记录plain零反馈、plain真实反馈、anchored真实反馈的条件视觉hidden位移／目标方向。六教师target读取v71对v50/v69实际加法输入采集的Qwen27/flow9特征，逐hash校验；35正确target为原表征。hidden比较使用无仿射FP32 layernorm。报告全部6教师与35保真，不根据诊断挑教师或调幅度／时机。

新零head对六教师预算1/2、35正确预算2均完整14速度严格等baseline；辅助无循环对六教师同查。选元数据顺序第一教师做零head完整终点梯度，再分别冻结／可训练核心、预算1/2做独立非零readout探针完整14反传，探针恢复而非checkpoint。基础全hash／原core hash不变且无梯度；head、证据梯度非零，可训练核心梯度非零，冻结核心无梯度。记录真实调用、峰值、walltime和853名义速度；反传checkpoint重算另计。

推理只读原中途MMU／视觉token／latent／坐标。教师只参与本次TRAIN事后表征诊断和接口loss，不读父图、mask、像素、终标签或teacherID选择动作。实际物理更新后禁用旧KV／inner prefix，完整14速度重算。接口通过只说明可训练与零操作保真，尚未训练／评分新方法。后续必须先依据完整诊断冻结匹配的plain／anchored和冻结／可训练对照，六教师物理<.01且35正确最大<.001原门不变；TRAIN总体终图质量和保真通过才独立val，controller0。
