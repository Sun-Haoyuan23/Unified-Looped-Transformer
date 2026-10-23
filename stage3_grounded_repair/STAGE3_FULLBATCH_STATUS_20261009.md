> 最新状态见[STAGE3_FOUR_LOOP_STATUS_20261009.md](STAGE3_FOUR_LOOP_STATUS_20261009.md)：v74/v75/v76均已完成并完整本地核验；下一项v77/v78尚未实现或启动。

# 第三阶段接续：有界全批物理容量诊断（2026-10-09）

最新运行状态由STAGE3_ANCHORED_STATUS_20261009.md覆盖：v74三组均完成且完整本地核验，v75新接口已启动；以下保留启动历史，不作为当前状态。

v74启动快照40项文件已完整本地核验，归档SHA256 `3cb38263cd1390ab0804d74cbee456555ea9d0825983819efeab60b9ba796c9b`，见v74_launch_portable_verification.json。三组实际初始化tensor完全一致，2项新CPU测试、每组前2次完整41态closure及记录的2次接受更新通过核验；这仅为有界启动前缀，不是最终checkpoint或完成容量。两loop初始教师／保真均值严格同为.545183／.003721；同参数无循环起点为.716385／.009931，确认路由改变后的初始拟合差异。最新运行数另读v74_latest_monitor.json和远程日志。

v72／v73均完成并完整本地核验，不重跑。先读STAGE3_EXPANDED_RESULTS_20261009.md、STAGE3_EXPANDED_MAPPING_RESULTS_20261009.md及v72_complete_portable_verification.json／v73_portable_verification.json。v72三组各420更新，共1260；累计8140参数更新／7942外部step。v73只读0更新／0新评分。

当前expanded_fullbatch_capacity_v74已实际启动，launcher40551。GPU0 no_loop worker40831、GPU4 frozen_loop40832、GPU6 trainable_loop40833，源码及协议冻结，不重复启动。新2项CPU加权梯度累积／target detach检查已实际通过，接口测试累计202项分轮，非同次全套。

三组都从同一个v72 trainable_loop固定step420的adapter／非线性head出发，相同RMS与41TRAIN目标；两个loop初始前向相同，无循环路由不同，且共同head曾用循环训练，因此无循环起点拟合可能不同。冻结组保留已加载低秩前向，只冻结其梯度。新诊断只优化组间各.5的全批物理目标，不含terminal loss、不执行终图延续，不能当完整终点训练。共享循环真实参与物理目标梯度。

复用既有BoundedLBFGS，固定strong_wolfe／lr1／100内部迭代／125 closure硬上限／history20／1外部step每组；预算耗尽恢复最后接受参数，不保留未评估trial，不恢复优化器。按实际accepted_updates.jsonl／evaluations.jsonl／training_complete.json分别计数，不将closure或外部step混记参数更新。原容量门六教师每个<.01、35正确最大<.001保持，无自动评分／val／controller。

最近服务器快照：无循环已完成100接受更新／106 closure／1外部step，容量门失败；冻结与可训练循环仍运行，分别14／13接受更新、16／15 closure。该快照仅为服务器读取，最终三组归档尚未完成；不能预填全部300更新或固定最终checkpoint。先检查pipeline_status、各arm/worker_status、evaluations.jsonl、accepted_updates.jsonl、initial_capacity.json、training_complete和capacity_report。

两循环组每closure41态实际外／内共享调用各164，无循环0；每组原状态特征采集41次名义速度。当前物理优化不续flow，部署终点评测仍须更新后禁旧KV/prefix并完整14速度。完整终点接口沿用v71／v73证明，教师只入TRAIN损失和事后分析，推理不读父图／teacher／mask／像素／终标签／teacherID。全部独立层仍保留，新增联合存储1669768，FLOPs null；不声称生成、理解、广域GenEval或参数／计算下降。

六严格教师仍00006／10／21／39／46／54 seed101，35真实正确池，canonical112态56提示。旧三目标取v49原FP32 delta及v50加法终点，新三取v69实际加法目标；DEV1及dev/val/test不得训练。评分仍4264条含351明确基线别名及其他复现，动作翻正116含重复／跨seed，controller0。

v74完成后必须归档原始source/config、Torch LBFGS源码、初始／最终tensor hash、全部closure／接受更新、最终接受checkpoint及41态容量，本地逐文件和独立标量核验。任何本轮收益只代表物理容量；过门后先冻结正式TRAIN完整14终图质量、全部35保真、证据与预算协议，再决定独立val。若未过，依据最终残差／优化终止／核心变化推进表达方式，不盲扫幅度／阈值／时间。10分钟轮询ACTIVE。
