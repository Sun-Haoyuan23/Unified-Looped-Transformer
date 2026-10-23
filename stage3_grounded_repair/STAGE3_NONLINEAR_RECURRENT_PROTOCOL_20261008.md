# 第三阶段：可训练共享循环与非线性读出固定协议（v63/v64）

v62在固定特征上进一步降低误差，但原容量门仍失败；不能据此认定优化收敛或结构不可达。新协议将同一个非线性head接入可训练的共享注意力核心，以检验可学习的跨token映射。原外层Qwen26/27、内层flow8/9各执行两次共享循环，低秩A/B跨循环复用。

三个组：no_loop、frozen_loop、trainable_loop。两个循环组均从同一v52 joint_process step100 operator和同一v62 loop2最终接受head开始，全部初始化tensor严格匹配；仅低秩核心参数是否更新不同。冻结核心保持已加载LoRA前向贡献，不调用set_core_learning(False)。无循环组从自己v62 no_loop最终head开始，属于辅助映射对照，不能说三组起点完全匹配或等FLOPs。

原基础模型全部冻结。三个组都训练非线性head、共享MMU证据编码器、视觉／坐标投影和feedback；原线性readout不使用且冻结。仅trainable_loop训练共享核心的rank4 q/k/v/o低秩参数；两组实际共享调用各外4／内4，无循环0／0。输入仅中途MMU、visual token、latent与坐标；无外部mask、像素教师或最终标签决策。

v63先执行四项CPU接口测试，再在预先固定第二个严格TRAIN教师上验证：新未训练零输出head在loop2/no_loop下完整14次剩余flow终点严格等于基线；三个warm-start组均执行完整14步终点反传，head与证据梯度非零，只有trainable_loop有真实外／内核心梯度，基础权重无梯度和hash不变。所有状态更新后禁用旧KV/inner prefix，速度完整重算。此阶段optimizer更新0，不作为训练结果。

只有v63通过后才启动v64。固定3个原严格TRAIN教师+全部35真实baseline-correct态，目标和来源逐hash继承v60/v62。固定210步：105教师步（每教师35次），105保真步（每正确态3次），交替抽样、seed20261008；两个循环组使用同抽样／损失／lr／head／归一化buffer。AdamW lr1e-4、weight_decay0，梯度裁剪1。目标为物理delta规范化MSE与完整终点规范化MSE各0.5；物理分母为教师delta能量或保真原latent能量，终点分母为教师与基线差能量或保真基线能量，clamp1e-5。目标detach。每步一次完整14速度rollout反传，不用终图挑checkpoint，固定最终step210。

训练缓存只保存对应原始state fingerprint／提示／index的原模型预反馈特征，不能在物理state修改后当旧prefix继续。共享循环特征每步重新计算且可微。所有35保真态都被覆盖；不能把更低均值当作所有状态改善。

训练后先在全部38TRAIN态做物理容量审计，每教师MSE<.01且35正确态最大MSE<.001的原门不变。未过不自动生成／val／controller。即使过门，仍需正式终点评分和35态保真、正确／乱序／反向证据与预算对照，才能讨论质量与证据作用。TRAIN成功也不代表独立val泛化。

所有原独立层仍保留，尚非ELT压缩；新增非线性head484032参数，旧线性readout冻结而未删除。参数、实际共享调用、完整速度、MMU、VAE、教师采集和端到端成本分别记录，FLOPs未测null。外层／内层独立对照、理解MMU/AR共同监督与循环开启保真、共享core替换独立深度、FID/FVD、人评、多样性及广域GenEval仍待完成。

协议准备时v63/v64尚未启动，以新目录pipeline_status与原始日志为准。源文件／配置／固定源checkpoint／最终checkpoint归档，不修改既有v25–v62结果。
