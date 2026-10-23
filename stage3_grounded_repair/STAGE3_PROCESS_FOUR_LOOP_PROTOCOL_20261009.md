# v77接口与v78匹配四循环过程监督：冻结协议

四组anchor_trainable4、anchor_frozen4、plain_trainable4、no_loop；主对照前两组。共同v74 trainable_loop固定最后接受operator，反馈权重与bias清零；fresh 3716→128→64 SiLU零输出head，seed20261008。四组初始全部tensor完全相同；冻结仅关闭LoRA梯度，保留加载前向。共同RMS由41原状态×四路线等权计算，目标不参与。额外外/内共享块anchor10/10、plain8/8、no_loop0/0，同A/B跨四循环复用，参考保持梯度连接。

六严格TRAIN教师00006/10/21/39/46/54 seed101及35真实正确态。物理/终点目标严格继承v50/v69已评分additive。过程目标为v71实际加法输入Qwen27/flow9条件视觉token；正确态为原source hidden。FP32无仿射layernorm eps1e-5，teacher尺度source-target规范化MSE，正确尺度source规范化均方，min1e-5；全部训练前冻结并hash。targets detach，外/内loss各半。教师只用于TRAIN准备、损失和事后分析，推理不读教师、父图、mask、像素、终标签或teacherID。

总loss=.5 normalizedphysical+.5完整14步terminal+1 normalizedhidden。四组同loss/数据/抽样，辅助no_loop也监督反馈hidden。过程hidden由forward临时返回，flow前不注册hook；物理修改后禁旧KV/prefix，完整14剩余速度重算。

v77四CPU测试；41态每路线零head物理delta严格为零，全部41 anchor_trainable4完整14零终点严格baseline，其他三路线各六教师完整14零终点严格baseline。检查四组全部初始tensor/RMS相同、真实零head终点梯度到head、零反馈过程梯度到feedback；不强制零反馈core梯度非零。独立非零feedback/head探针验证四组完整14总loss梯度，另anchor终点单独loss验证core/证据，探针全部恢复；基础权重无梯度/hash与原corehash保持。0optimizer/0评分。原始归档下载本地核验后才启动v78。

v78每组固定420 AdamW，lr1e-4/weight_decay0/clip1。seed20261008，教师/保真交替各210：六教师各35次、35正确各6次，与v72相同序列；固定step420，禁止终图/val挑选。日志逐次记录实际optimizer更新、三loss、head/feedback/core/证据梯度、裁剪前norm、真实shared调用、14flow、walltime与峰显存。裁剪比例只作描述。记录有效BF16低秩核心变化，基础及原独立核心不得改动。

训练后只做41态物理容量：每教师MSE<.01且35正确max<.001。不得降低门、不扫描强度/时机/阈值，不自动生成、val或controller。完整归档source/config/固定checkpoint/全部日志/容量/hash，本地便携核验。只有容量过门才另冻结正式TRAIN六教师+35正确的完整终点评测四评分及正确/乱序/反向证据、1/2/4预算、外/内/联动对照；TRAIN总体质量/保真过门后才独立val。

调用增加，无新增部署参数；全部原独立层仍保留，联合额外1669768存储参数。26semantic/28Qwen/10flow、额外共享块、MMU/VAE/head/低秩/教师/端到端及反传重算分列，FLOPs null，不能称等计算或总体压缩。ELT替代独立深度、MMU/AR共同监督及最终总参数/FLOPs下降仍未完成。
