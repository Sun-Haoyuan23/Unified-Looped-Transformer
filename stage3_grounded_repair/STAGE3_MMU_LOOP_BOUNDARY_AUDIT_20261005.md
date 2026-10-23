# 外层循环的理解任务边界核对（2026-10-05）

已核对服务器官方Show-o/show-o2/inference_mmu.py第80–86、133–157行和modeling_showo2_qwen2_5.py的模态位置处理，当前EvidenceReader._question的MMU输入顺序、时间embedding与modality_positions写法遵循官方MMU实现。

实际MMU输入为：文字前缀、boi、time、视觉token、eoi、问题、assistant前缀。官方MMU位置为text_tokens_a长度+2，num_mmu_image_tokens在add_time_embeds时加1。T2I外层修复器目前按模态位置起点是time、start+1开始视觉token的约定工作。两种任务不能直接共用这段mask构造；否则可能遗漏首个视觉token并把eoi包含进修复区域。

后续MMU/AR联合监督须传入独立且明确的visual_start/visual_count，按真正嵌入序列构造修复mask；保留官方任务的注意力mask、问题和文字token完整。理解输入为单分支，不能假设T2I的双CFG分支。用官方MMU路径为基线，循环未训练零操作的logits/hidden必须严格一致，禁用不合法KV后完整重算，再验证循环开启时的真实理解保真及提升。

这次核对没有修改EvidenceReader、已捕获MMU证据、训练数据或冻结源码。当前v41/v42仍是生成端校准实验；不声称理解能力已经提高，也不把官方MMU位置约定自行改写为已证实的bug。理解端后续试验需新目录、冻结来源，并与当前生成端结果分开核验。

## v43实模型验证完成

新mmu_outer_context.py明确实际visual_start/count，只调整适配器的视觉修复范围，官方输入/注意力mask与旧EvidenceReader保持。3接口测试通过；两真实TRAIN问题的原reader margin/hidden严格等价，未训练零循环完整logits/hidden严格相等。非优化器的非零头梯度探针证据梯度0.015406585、总适配器梯度0.343279959，基础权重无梯度、共享core hash不变，峰值7.279GiB。更新0、没有理解准确率/保真或反向提升结论。原始归档与源码已本地核验；后续MMU/AR共同任务监督与独立理解评测仍待执行。
