# v81 逐循环连接参考重算：有限匹配训练

本轮仅新训练recomputed_trainable4/recomputed_frozen4两组。先要求v80真实接口完整归档、本地逐文件与来源核验passed，禁止以运行中或启动前缀替代完成证明。两组同一G、同A/B跨四循环复用，参考不detach，逐循环重新计算G(h0)。数学前向保持h0+[G(h)-G(h0)]。仅LoRA是否requires_grad更新不同，冻结组保留已加载低秩前向。每次外/内各16真实共享块，分别计数；不增加部署参数，当前接口保留重算，FLOPs null。

两组共同v74最后接受operator、feedback清零、fresh3716→128→64 SiLU零输出head；实际初始化tensor须同时等v77/v80。复用v77原41态×四路线(包括plain)目标无关RMS与actual-additive hidden targets/scales，明确不按本轮路线或目标重估。与已完成v78固定420抽样/初始化/损失/目标一致。每组420 AdamW，lr1e-4/weight_decay0/clip1，seed20261008；六教师各35次、35正确各6次，教师/保真交替各210。loss=.5规范化physical+.5完整14terminal+1规范化hidden，外/内hidden等权、FP32非仿射layernorm eps1e-5。固定step420不按终图/val选checkpoint。

已完成v78 anchor_trainable4(cached连接参考)与no_loop作为已有对照复用，锁定各自step420 checkpoint哈希，不重新训练。已有对照来源/抽样/初始化/目标匹配，但调用数与walltime不同：cached外/内各10、重算各16、no_loop0；不称等计算或本轮新执行四组。比较应报告新匹配trainable/frozen主对照、重算与既有cached/no_loop辅助配对及全部35保真。

每次更新记录三损失、分组有限梯度、裁剪前范数、真实共享块计数、完整14前向速度、反向checkpoint额外重算、无更新后旧KV/prefix、walltime/峰值显存。clip频率不直接解释因果。base/原core/readout冻结与hash保持，冻结LoRA不变，有效BF16权重变化单列。420后只计算41TRAIN态物理/hidden容量；六教师每个规范化physical MSE<.01、35正确max<.001原门不降低。完整源码/config/初始化/420日志/checkpoint/容量/成本归档本地核验。

v79数值诊断在零反馈前向严格恒等时发现缓存参考四循环core残余梯度，重算大幅减小但外层仍非严格零。这是有限匹配重训依据，不预言修复容量、生成或语义学习。若门失败，依据最终残差/过程对齐/有效核心定位下一接口，不盲扫幅度/阈值/时机。无自动终图、评分、val或controller；容量通过后需另冻结完整TRAIN质量与35保真、证据正确/乱序/反向及独立外/内/联动预算协议，TRAIN总体质量与保真通过才固定方法独立val。

严格TRAIN六教师00006/10/21/39/46/54 seed101与35真正确。旧三v49原FP32 delta→v50已评分additive终点、新三v69实际delta/input/final，hidden v71 actual-additive，禁止替为direct终点或hidden。DEV1/dev/val/test不得训练。teacher/父图/mask/pixel/终标签/teacherID不进入推理动作。物理更新后完整14速度重算，旧特征cache只绑定原state fingerprint/prompt/index，禁旧KV/inner prefix继续flow。外/内/联动保留，ELT替换独立深度、共同理解监督、广域GenEval及总参数/FLOPs下降仍待完成。
