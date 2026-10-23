# v81 有限匹配训练启动核验

v80完成49项完整归档本地核验后，实际启动v81。launcher104788；GPU0 UUID GPU-4a177c46-c73f-d963-8d61-b7fd90196038，recomputed_trainable4 worker104791；GPU4 UUID GPU-7c36f685-23c2-b1e5-dab9-b9a815b045da，recomputed_frozen4 worker104792。启动前物理0/4/6/7均实查5MiB并核对既有UUID，没有停止他人进程。

启动快照各7次真实optimizer更新，共14。94项manifest完整本地核验通过，归档SHA a27b68e1ab8651e59a18487a4fda583362c5095027e5597c167802af31eaa7a2，136,898字节。两组实际初始化tensor、RMS、actual-additive targets/scales、全部420抽样逐字段匹配v78已有cached/no_loop；前缀loss加权、有限分组梯度、冻结core零梯度和每次外/内各16真实共享块计数通过。证明见v81_launch_portable_verification.json；本地核验字节/来源/记录标量，服务器执行Torch训练，本地没有Torch重跑。

快照可训练/冻结组训练累计wall83.066283/82.468438秒、峰20.992902/17.736275GiB；各98名义终点速度、共享外内各112次，反向checkpoint重算另计。时间是启动前缀，不能外推总训练成本或FLOPs。新增14更新对应累计10134参数更新/9639外部step，最新进度另读运行中train.jsonl，计划840尚未完成。

每组固定420 AdamW lr1e-4/wd0/clip1/seed20261008，新主组只LoRA是否训练不同，参考保留连接并逐循环重算。v78已完成cached/no_loop为来源匹配的已有对照，不重跑旧训练，不称调用/时间相等。420后只做41TRAIN物理容量，原六教师各<.01与35正确max<.001门保持。无最终checkpoint、容量结果、生成质量/GenEval、val或controller结果。

10分钟接续保持ACTIVE，优先只读transfers/monitor_reference_training_v81.py；完成后按冻结协议完整归档与本地核验。状态见STAGE3_PROCESS_ALIGNMENT_STATUS_20261009.md，协议见STAGE3_REFERENCE_RECOMPUTE_TRAINING_PROTOCOL_20261009.md。
