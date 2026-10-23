# 第三阶段：有界优化收敛诊断固定协议（v62）

在v60固定结构、固定38TRAIN状态、相同物理目标及损失下，从每组自己的Adam step1000 head开始，运行一次full-batch L-BFGS strong_wolfe。两组的已训练起点不同；不声称本轮起点tensor相同。

固定lr1、max_iter100、max_eval125、history_size20、tolerance_grad1e-7、tolerance_change1e-9。Torch源码和版本实际保存。Torch原实现在线搜索后才检查max_eval；新增包装在closure前严格限制125次，预算中断时恢复该线搜索开始的已接受参数，不保留未经评估试探状态。不恢复此时的优化器内部状态，不从最佳图或最佳loss挑checkpoint。

分别保存optimizer.step外部调用、attempted内部迭代、实际被接受且改变参数的更新、真实closure评估及Torch内部函数计数。预算中断时Torch内部计数可能未记入该未完成线搜索，真实closure日志为实际成本依据。另有两次诊断目标forward，其中最终一次反传测量最终参数的梯度，单独计费。四项CPU测试先验证收敛／计数、预算中断严格恢复、异常中断恢复和非有限损失失败。

只优化现有484032参数head，归一化buffer、v52完整base／共享core／旧adapter、源checkpoint不变；新head参数最终单独保存。保持76项输入／目标fingerprint与v59一致、76项v60起点物理MSE复现，教师由原严格TRAIN门控验证，dev/val/test不进入。

每教师物理MSE<0.01且35真实正确态最大MSE<0.001的容量门保持；未通过不自动终图、val或controller。此实验诊断优化，不证明全局最优、结构不可能或生成质量成功。后续部署仍需完整14步梯度／零操作接口和官方、HPS、Pick终点评分。实际成本分别报告，不称等FLOPs。

状态：协议和源码准备完成，尚未启动；以pipeline_status.json和原始评估／更新日志为准。
