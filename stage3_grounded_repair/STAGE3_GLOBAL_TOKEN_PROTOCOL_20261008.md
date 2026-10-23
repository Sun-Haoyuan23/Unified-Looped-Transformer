# v66 显式跨token状态传递容量协议

在v65完成且本地核验后冻结。v64／v65未取得生成质量成功；有效BF16核心有变化，目标方向／跨态关系并不支持单一梯度冲突解释。下一步检验当前token-local读出是否缺少对原始latent的显式跨位置传递。本协议是容量假设，尚未证明失败原因或方法优势。

使用v64 trainable_loop固定step210 operator，所有共享core／证据／visual／position／feedback冻结。两组分别读取同operator的loop2／no_loop特征；后者跳过共享块，真实调用0／0，前者外／内4／4。该operator原经循环训练，两组仅用于此固定表征下的容量比较，不作通用架构因果结论。

新head用3716维特征的FP32 TRAIN RMS归一化：局部分支3716→64→64 SiLU；全局分支Q／K各3716→32，按每个独立状态的729个token计算softmax(QK^T/√32)，value为官方patch顺序的原始中途latent。输出为局部分支加W(A V−V)。最后局部输出和W零初始化，严格零物理更新；每个state的注意力独立，禁止不同提示混合。

head参数483968，比原width128 token-local head的484032少64。只匹配读出参数预算，不声称总模型参数或FLOPs下降；跨token注意力新增运算需实测。推理前向只读中途证据／视觉特征／原latent，教师、mask、像素编辑和终标签不进入前向。

固定原38TRAIN：3严格教师+35真实正确态，原v60目标patch、原ProcessCorpus与v64来源逐hash检查。全batch物理损失为教师总权重0.5、保真总权重0.5，组内均匀，原归一化。两组trainable初始tensor相同，RMS buffer由各自38态特征确定并冻结；Adam lr1e-3、1000步、固定最终checkpoint，不选最佳loss／终图／val。四CPU测试先通过再启动。

原容量门每教师MSE<.01且35态最大MSE<.001不变。只有容量通过后再做新head的真实零终点／完整梯度接口和正式TRAIN终点评分，不能自动部署、扩val或controller。该实验只更新head、共享core0更新、0生成图与评分；无论门是否通过，都先归档并本地核验。

运行前状态NOT_STARTED；实际运行以新目录global_token_capacity_v66的status／日志为准，本协议不回写。
