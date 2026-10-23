# v81 连通性接续（2026-10-10）

2026-10-10 10:06:34 UTC核查：旧SSH77539已不存在；两次已获允许的主机网络直连10.103.12.98:1902均在TCP阶段超时、未进入认证。这不是训练退出或GPU故障证据。尚未读到v81最终日志/checkpoint/容量/完整归档，不能称训练仍在运行或已完成。没有重新启动任何训练/生成/评分，没有修改冻结源码或参数。

完整本地启动快照仍各7更新、共14；上次下载服务器完整行前缀各24、共48，对应累计10168参数更新/9673外部step，只是最后已知计数。当前真实计数未知，连通后先只读pipeline_status/worker_status/train.jsonl/training_complete/capacity/effective_core_changes。若已有complete归档和receipt，直接下载核验，不重复归档；若两组420且pipeline complete但无归档，运行transfers/archive_reference_training_v81.py --phase complete一次，再下载并本地work/verify_reference_training_v81.py --phase complete。

已有启动归档及94项manifest重新只读核对通过；原JSON与字节SHA为a27b68e1ab8651e59a18487a4fda583362c5095027e5597c167802af31eaa7a2。旧文字说明中的SHA有一处转录错误，现已按原字节更正，原归档/JSON/训练源码没有改动。此次核验未执行本地Torch或新训练。

10分钟轮询10保持ACTIVE。连接恢复后继续冻结协议；连接状态未变安静，不重复报告同一故障。最终第三阶段目标、TRAIN/DEV边界、容量门和原质量门全部保留。

以下2026-10-09启动状态仅为历史；以本页上方连通性审计及实际日志为准。

# 参考重算接续状态（2026-10-09）

本文件覆盖旧heartbeat。用户明确允许增加内外层调用，继续已授权第三阶段实际工作。v25–v80全部完成，禁止重复旧训练/生成/评分。v80已完整归档、本地核验passed，见v80_portable_verification.json。v81 reference_recomputed_train_v81已启动，launcher104788，GPU0/4 worker104791/104792分别recomputed_trainable4/recomputed_frozen4；完整本地核验启动快照分别7/7真实更新，共14，不是计划840已完成。最新进度另读各arm/train.jsonl。不得重复启动或修改冻结源码。

先读STAGE3_PROCESS_FOUR_LOOP_TRAINING_RESULTS_20261009.md、STAGE3_PROCESS_ALIGNMENT_RESULTS_20261009.md、STAGE3_REFERENCE_RECOMPUTE_RESULTS_20261009.md、STAGE3_REFERENCE_RECOMPUTE_TRAINING_PROTOCOL_20261009.md及v78_complete/v79/v80_portable_verification.json；若已有v81_launch/complete_portable_verification.json也读。最新状态由运行目录原始JSON/完整行train.jsonl决定。

v78四组各420、共1680真实更新完成，完整173项manifest归档SHA c080833e43b9eca15f37aad4db062bed096d8b8fac299a859271562dbba1a9f4，本地实际核验通过。anchor_trainable4/frozen4/plain_trainable4/no_loop六教师MSE均值.528406/.532641/.593299/.544406，35正确max .012320/.012733/.008180/.009748，原门全失败。主可训练对冻结3/6教师及11/35正确改善，不能称整体架构收益。有效BF16核心改变约86.60%/0/87.44%/0，不能说全被舍入。无新终图/评分/val。

v79固定v78全部参数，164容量/hidden标量复现、205预算前缀、24教师完整14终点latent、6独立loss梯度、12零反馈VJP通过；0更新/评分。74项完整本地核验，归档SHA 0f0bb06b873ea6f06b884e4411df0febc369c66006b055ece045f970a8d10654，2项新CPU检查。主六教师terminal均值.304950/hidden .957050，外/内方向cos .175668/.363328，未有生成质量收益结论。

诊断：零反馈四循环前向严格恒等，缓存连接参考core梯度绝对和外/内1.533170881/.075567226；逐循环重算连接参考为.000064760551/3.365e-11，反馈梯度仍非零。FP64 toy取消通过，外层仍非严格零；支持参考图/有限精度数值问题，尚不证明v78容量失败原因。v80同G/同A/B跨四循环，参考不detach，保持h0+[G(h)-G(h0)]前向；真实外/内各16共享块(缓存10/no_loop0)，不新增部署参数，当前实现保留重算，FLOPs null。

v80四路线cached_trainable4/recomputed_trainable4/recomputed_frozen4/no_loop已完成49项完整本地核验，归档SHA 9fd262f229789939beeb107be2a909696f552343ee6d4d87dc52a17998187386、67615字节，不重复归档或运行。41态实际v78step420 delta/hidden严格同forward、六教师v79完整14终点复现、fresh零head164物理delta/59完整14零终点、5完整14反传、初始tensor同v77及探针恢复/base冻结全部通过。零head到head6.164509、零反馈hidden到feedback130.259209，外/内core残余.000173796091/7.958e-12如实保留。1021名义速度、共享外内各2862、峰22.106851GiB/wall362.242757s，反向重算另计，0更新/评分；本地字节/来源与服务器Torch操作范围区分。v80_portable_verification.json已实际生成，v81前置门通过。

v81仅新两组逐循环重算连接参考，主组前向/参数/初始化/抽样/目标/loss全部匹配，仅LoRA更新开关不同，frozen保留已加载低秩前向。每组固定420 AdamW lr1e-4/wd0/clip1/seed20261008，六教师各35、35正确各6交替；共同v74最后operator清零feedback/fresh零head、同v77既有41态×四路线(包括plain)目标无关RMS与actual-additive targets/scales。loss=.5physical+.5完整14terminal+1hidden，FP32无仿射layernorm eps1e-5、外内等权。复用v78 cached anchor_trainable4和no_loop已有固定checkpoint对照，来源匹配但调用/walltime不同，禁止重复旧训练。

v81已经实际启动，禁止再运行launcher。启动快照94项manifest已完整本地核验，归档SHA a27b68e1ab8651e59a18487a4fda583362c5095027e5597c167802af31eaa7a2、136898字节，各7次更新、共14次；实际初始化/RMS/targets/完整420抽样与v78逐字段一致，前缀loss/有限梯度/真实调用全部通过。v81_launch_portable_verification.json仅启动前缀，无最终checkpoint/容量/质量，不重复启动归档。用transfers/monitor_reference_training_v81.py只读核验最新完整行前缀，输出v81_latest_monitor.json。全部420且pipeline complete后运行transfers/archive_reference_training_v81.py --phase complete归档一次，下载stage3_reference_training_v81_complete.tgz及_archive.json，本地work/verify_reference_training_v81.py --phase complete实际核验。每次源代码/共同RMS/targets/初始化/抽样/真实调用/梯度/时间显存/有效BF16核心保持证据同步。

固定step420后仅41TRAIN容量，六教师每个规范化physical MSE<.01、35正确max<.001原门不降低。失败依据最终残差/过程对齐/有效core再定位，不盲扫幅度/时机/阈值；通过后另冻结六教师+35保真、正确/乱序/反向证据、1/2/4与独立外/内/联动预算、官方/HPS/Pick/SigLIP完整TRAIN质量协议。TRAIN总体质量和35保真通过才固定方法独立val，controller0。

严格TRAIN canonical112态56prompt中六教师00006/10/21/39/46/54 seed101+35真正确；旧三v49原FP32 delta→v50已评分additive终点、新三v69实际delta/input/final，hidden v71 actual-additive。禁替v36/v68 direct终点或v49 direct hidden。DEV1/dev/val/test禁训练。推理不读teacher/父图/mask/pixel/终标签/teacherID选动作，物理更新后禁旧KV/inner prefix并完整14速度重算，缓存只绑定原state fingerprint/prompt/index。

v78完成后10120参数更新/9625外部step，v79/v80不新增；当前v81已读14更新对应10134/9639，最新必须另读实际train.jsonl。评分4264记录含351明确baseline别名及复现，非独立图数；116动作翻正含重复/跨seed，源噪声独立prompt/seed26另1跨seed、exact-state29分列，TRAIN6/DEV1，controller0。217相关CPU检查分轮非同次全套，新增v79两项/v80两项。全部独立原层保留，联合额外1669768存储参数；调用增加、FLOPs null。循环额外质量/证据特异性/泛化、ELT替代独立深度、外/内单独新方法对照、MMU/AR共同监督与理解保真/反向提升、广域GenEval/FID/FVD/人评/多样性、总参数/FLOPs下降仍未完成。

最好已评分学习终点仍v56六TRAIN态3/6→5/6、两质量过门但伤一正确态，无循环同样修复、整体HPS负且乱序也修复。v78–v81没有新生成质量结果；不得据过程/梯度/latent误差称循环质量或GenEval提升。

SSH77539当前可复用，否则直接shy_23@10.103.12.98:1902 require_escalated主机网络重连，密码仅认证stdin，不写文件，不因sandbox超时再问VPN。项目/home/shy_23/iterative_visual_stage1_20261003，源码/results/transfers在stage3_grounded_repair，Python/home/shy_23/venvs/show_o2/bin/python，TORCHDYNAMO_DISABLE=1。每次新任务物理0/4/6/7查空闲及既有UUID，CUDA绑UUID；5故障、1/2/3可能他人，不杀他人，UUID查询可能受故障卡全局误报。官方隔离评测venv/原config权重/v12兼容报告保持，PIL先Torch/MMCV，四评分槽不得漏HPS/SigLIP。

10分钟轮询保持ACTIVE，状态未变安静，仅关键新结果/故障/完成/必要用户操作通知；第三阶段真正完成或用户明确停止才停用。
