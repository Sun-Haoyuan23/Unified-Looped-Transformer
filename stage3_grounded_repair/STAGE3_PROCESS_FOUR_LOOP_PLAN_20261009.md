> 当前运行状态已由[STAGE3_PROCESS_FOUR_LOOP_STATUS_20261009.md](STAGE3_PROCESS_FOUR_LOOP_STATUS_20261009.md)覆盖：v77已完成且完整本地核验，v78四组420步有限训练已启动。下文旧状态/计划为历史。

# 四循环共享映射的匹配过程监督：下一项实施计划

此文件为下一项实施计划，尚未实现v77训练接口或启动v78训练。v76四循环已在真实模型通过零终点与梯度检查，新增内外调用得到用户明确授权。不能将本计划当optimizer日志、完成训练或生成收益。

v74无逐态裁剪的有界优化仍未过容量门；可训练核心对冻结核心六教师全部略差。v75证明plain零反馈重访本身会改变原表征，锚定可严格保持零反馈身份，但带旧plain反馈的外层教师方向仍较差。因此下一步同时学习锚定反馈与实际加法教师过程方向，先验证新loss在真正零head／零反馈起点的梯度，随后做匹配训练；不能只沿用旧反馈跨路由推理便声称改进。

预定四组anchor_trainable4、anchor_frozen4、plain_trainable4、no_loop，主对照为前两组。共同加载v74 trainable_loop固定最后接受operator，所有分支feedback权重清零，新非线性head以seed20261008重新零输出初始化；原encoder/visual/position/rank4参数保留同一来源。四组全部实际初始tensor完全一致，只有共享路由与LoRA是否可训练不同。冻结核心仅requires_grad=False，前向保留已加载低秩，不用set_core_learning(False)移除前向。旧线性readout冻结不用。

共同输入RMS在41原状态、四条预定路线的原始输入特征上等权计算，目标不参与；全部四组使用同buffer。锚定四次预算外／内各10额外共享块（含参考），plain四次各8，无循环0；同参数但初始表征可能不同，不能称三种路由初始性能相同或等FLOPs。所有循环同组A/B跨四次共享。预算1/2与独立外4内2、外2内4作为冻结推理对照，不按终图选择训练预算。

六教师00006/10/21/39/46/54 seed101加全部35真正确TRAIN不变。物理及终点目标只使用v50/v69实际过门加法目标。hidden目标使用v71实际加法输入的Qwen27/flow9条件视觉token；35正确目标为原状态base同层hidden。所有targets detach且身份/hash必须匹配，不能使用v49 direct教师hidden替代实际加法hidden。hidden在FP32无仿射layernorm后计算；教师尺度为原状态与实际教师hidden差的MSE（min1e-5），正确尺度为原状态规范化hidden均方（min1e-5），两者都在训练前冻结，不能由训练中operator首次出现时决定。外／内hidden loss等权平均。

固定总loss=.5规范化physical+.5完整14步terminal+1规范化hidden。四组同目标、抽样与loss；无循环同样监督其反馈后的hidden，不能删除辅助loss制造不匹配对照。此新loss与v74物理-only、v72完整终点loss均不同，不将跨版本差异归因循环。过程hook或临时记录只用于TRAIN损失，flow延续前移除；teacher不进入部署forward或动作选择。

v77先实现独立新模块/目录并做CPU检查，以及真实六教师/35正确目标读入hash、head和feedback真正清零的全14零终点恒等、主两组初始化/共同RMS严格一致。真正零feedback时core梯度可能因参考抵消为0，必须如实记录；过程hidden梯度应直接到feedback，零head完整终点梯度应到head。另用独立非零反馈/head探针检查核心与证据完整14梯度，probe全部恢复，base无梯度/全hash及原corehash保持。v75/v76没有本loss，不能代替这次实际验证。

v77通过且完整归档后，v78预定每组固定420 AdamW更新，lr1e-4、weight_decay0、clip1、seed20261008；六教师各35次、35正确各6次，教师/保真交替各210，与v72相同抽样序列。不扫参数/幅度/阈值/时间，不根据终图/val挑checkpoint。记录实际梯度、裁剪比例、shared调用、有效BF16核心改变、walltime与显存；裁剪比例仅描述不当失败因果。

固定step420只审计41态物理容量：六教师每个MSE<.01且35正确max<.001原门保持。0自动终图、val、controller。完成后导出完整source/config/固定checkpoint/420条实际日志/容量/hash，本地逐字段核验主两组与辅助对照。若过门，再冻结六教师及全部35正确保真、正确/乱序/反向证据、1/2/4预算、外/内/联动及四评分的完整TRAIN终点评测协议。TRAIN总体终图质量与保真通过才固定方法独立val，controller0。

此计划增加共享调用但不增加部署参数；全部原独立层仍保留。ELT共享core替代独立深度、MMU/AR共同监督、循环理解保真/反向提升与最终总体参数/FLOPs下降仍是后续目标，不能预填为本轮结果。
