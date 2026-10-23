# v80 逐循环重算连接参考接口与 v81 有限匹配训练计划

v79已完整本地核验，74项manifest，归档0f0bb06b873ea6f06b884e4411df0febc369c66006b055ece045f970a8d10654。四循环零反馈严格前向恒等，但共享同一参考图的BF16反传残余外/内core梯度绝对和1.533170881/.075567226；逐循环重算连接参考为.000064761/3.365e-11。FP64 toy取消通过，单循环残余小。此差异支持计算图/有限精度归因，尚未证明v78质量失败原因，外层重算后仍非严格零。

新ReferenceRecomputedRepair保持同一G、A/B跨四循环复用，h0+[G(h)-G(h0)]前向数学式不变，参考不detach。每次循环重新计算G(h0)，recomputed训练外/内各16块，cached各10块，no_loop0。无新增部署参数。当前实现保留重算调用，不预填部署缓存优化或FLOPs下降。

v80新目录reference_recomputed_interface_v80。四路线cached_trainable4/recomputed_trainable4/recomputed_frozen4/no_loop，共同v74最后接受operator且feedback清零、fresh零head；严格复用v77已冻结的共同原状态RMS、41态hidden targets/scales。此RMS来自原四路线，包含plain，明确是固定已有归一化，不随本轮路线或target重估。主cached/recomputed初始化/前向/数据/目标/loss完全相同，差别是参考计算图与额外重算成本。frozen仅关闭LoRA梯度、保留加载前向。

先两CPU检查：非零feedback前向及FP64梯度一致、实际调用与参数共享。真实模型重新加载v78 anchor_trainable4固定step420，41态cached/recomputed物理delta与hidden须严格相等，六教师recomputed完整14终点须严格复现v79固定cached终点。零head四路线41态物理delta严格0；recomputed_trainable4全部41态完整14零终点严格baseline，其他路线各六教师，共59。四组实际初始化tensor匹配v77，基础/原core无梯度/hash保持。

完整14梯度：recomputed真实零head到head、零feedback hidden到feedback，残余core梯度如实记录；四路线独立相同非零head/feedback探针各完整14总loss反传，重算trainable的core/证据非零，frozen/no_loop core为0，探针全部恢复。总计5次完整14反传。0optimizer/拟合/新评分，source/config/targets/固定参数/梯度/调用/终点hash完整归档本地核验后才允许v81。

v81仅新训练recomputed_trainable4和recomputed_frozen4两组，各固定420 AdamW，lr1e-4/weight_decay0/clip1，seed20261008；六教师各35次、35正确各6次，与v78完全同抽样/RMS/目标/初始tensor/loss=.5physical+.5完整14terminal+1hidden。两新组全部前向和初始化匹配，只LoRA是否训练不同。v78 cached anchor_trainable4及no_loop完成结果作为来源严格匹配的已有对照复用，禁止重跑旧训练；明确既有对照非本轮新运行，不称相同walltime/调用/FLOPs。固定step420后仅41态容量，六教师各<.01且35正确max<.001原门保持。此计划尚不等于实现或启动v81。

TRAIN六教师/35正确、旧v49 delta→v50 additive终点/新v69实际delta/input/final、v71 actual-additive hidden不变，DEV1及dev/val/test禁训练。teacher仅准备/loss/事后统计，推理不能读取teacher/父图/mask/pixel/终标签/teacherID选动作。物理更新后禁旧KV/prefix并完整14速度重算。无自动终图/val/controller；TRAIN总体质量及全部保真通过后才固定方法独立val。外/内/联动保留，总参数/FLOPs下降和ELT替换独立层仍未完成。
