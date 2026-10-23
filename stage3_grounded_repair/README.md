> 最新状态以[STAGE3_PROCESS_ALIGNMENT_STATUS_20261009.md](STAGE3_PROCESS_ALIGNMENT_STATUS_20261009.md)为准：v78/v79/v80已完成并完整本地核验，v81两组逐循环重算连接参考匹配训练已启动。v78容量门全失败；暂无新生成质量收益。下文运行状态仅为历史。

> 最新状态见[STAGE3_PROCESS_FOUR_LOOP_STATUS_20261009.md](STAGE3_PROCESS_FOUR_LOOP_STATUS_20261009.md)：v77已完成且完整本地核验；v78四组匹配过程监督训练已启动，固定420步，不自动扩终图/val/controller。下文旧状态为历史。

> 最新状态见[STAGE3_FOUR_LOOP_STATUS_20261009.md](STAGE3_FOUR_LOOP_STATUS_20261009.md)：v74/v75/v76均已完成并完整本地核验；下一项v77/v78尚未实现或启动。

> 最新运行状态：见[STAGE3_ANCHORED_STATUS_20261009.md](STAGE3_ANCHORED_STATUS_20261009.md)。v74三组已完成并完整本地核验，v75已启动；用户已允许增加内外层共享调用。下方旧快照仅为历史。

> 最新接续：v72／v73已完成并完整本地核验；v74有界全批物理容量优化已实际启动，launcher40551，GPU0／4／6，非完整终点训练。先读 [STAGE3_FULLBATCH_STATUS_20261009.md](STAGE3_FULLBATCH_STATUS_20261009.md) 和冻结协议。下文旧运行状态仅为历史记录。

> 最新接续：v68／v69均已完成且完整本地核验，新增3个原生加法教师；先读 STAGE3_NATIVE_RESULTS_20261008.md。下文旧运行状态保留为历史记录。

> 最新接续：v68 原生教师接口已实际启动，先读 STAGE3_NATIVE_INTERFACE_V68_20261008.md；下文旧状态按历史记录保留。

> 最新接续：v25–v66全部完成并按范围核验，当前无运行任务；v67本地TRAIN原生教师元数据审计完成，6个新候选尚非成功修复教师。下一项v68原生教师接口按新冻结协议实际实现／验证，不重跑旧任务。见[STAGE3_NATIVE_TEACHER_20261008.md](STAGE3_NATIVE_TEACHER_20261008.md)。

> 最新状态（2026-10-08）：[第三阶段接续与完整归档复核](STAGE3_CONTINUATION_20261008.md)、[正向结果](STAGE3_POSITIVE_RESULTS_20261008.md)。v46/v47完成且原始数据本地逐字段核验，v48只读审计已完成并本地核验。旧章节为历史记录。

> 当前综述见[2026-10-05进展总结](../LOOPED_UNIFIED_VISUAL_GENERATION_PROGRESS_20261005.md)。本次冻结日志快照：v46两joint组各100步完成，outer75步、inner34步；四组实际GPU初始化hash匹配；v47尚未开始。第三阶段真实更新1,559（其中完成固定组1,450、运行中109）；终图生成提升尚未成立。

> 最新以[STAGE3_PROGRESS_20261005.md](STAGE3_PROGRESS_20261005.md)及[STAGE3_RECURRENT_CORE_20261005.md](STAGE3_RECURRENT_CORE_20261005.md)为准：v44四组100更新、v45的120终图全部完成，0教师翻正且各损伤1正确态；未有循环生成提升。v46开始训练共享低秩核心和嵌套循环终点蒸馏，四组固定step100，全部成功后自动v47的144图。资源接续32461运行，6/7执行、其余等待空闲，不杀他人；固定质量门，不自动扩val/controller。累计3862评分记录；v44后1250已完成更新，v46实际日志另计。10分钟轮询ACTIVE。

> 历史快照（v23–v25启动时，运行状态已失效）：v23首次取得严格过门TRAIN教师train_extra_00021_seed101_i35/checked_inferred_correlated_target，Pick+0.019726086744、HPS+0.000732421875、SigLIP+0.004985675216。原始归档b570fcc1cd17d5f17728e33a2b545a141362c42ea89370f8503c50b085782831已本地同步，配对重分析逐字段严格一致；training_pairs复核1个成功教师+18个真实baseline-correct保真态。v25已进入真实token级共享修复器训练：core8/core2各固定50步，最近核对分别11/10步真实optimizer更新，有限梯度、峰值14.73GiB。launcher60856，训练PID60868（GPU4）/60865（GPU6）；GPU0/7捕获已冻结的12条独立val提示×2seed。训练完成后自动生成每core120个独立val终图（基线/零/正确证据/乱序/反向）并四项评分。controller不训练；单教师pilot和代理分数微增不等于泛化、人评或总体质量提升。详见[STAGE3_TRAIN_TEACHER_20261005.md](STAGE3_TRAIN_TEACHER_20261005.md)。10分钟轮询保持ACTIVE。

以下保留历史记录。

> 最新（v13/train pilot）：v13的144终图及四类评分已完成，首次1个开发态官方错→对且HPS/Pick均非负：00412_seed202_i35，Pick+0.315066、HPS+0.001465，SigLIP−0.002882。全24原seed目标4翻正、总体质量均值仍下降；质量守门离线oracle仅5/24→6/24，不能当共享循环或整体质量成功。累计1968已评分终图、18动作级翻正、5独立纠正提示/seed、1合格开发态；train成功教师和optimizer更新仍0。24条平衡train提示×2seed的index35捕获已在空闲GPU0/4/6/7启动（PID19987），开发态不入训练。官方评测恢复和全144旧图兼容性已通过，所有v11/v12/v13原始归档本地同步并核验。详见[STAGE3_V13_TRAIN_PILOT_20261004.md](STAGE3_V13_TRAIN_PILOT_20261004.md)。10分钟轮询ACTIVE；下面保留历史。

# 第三阶段：理解证据驱动的共享循环修复

本阶段检验“画到一半，先检查，再多想一次”能否改善最终图像。按技术方案先证明同状态可修复性，再训练固定修复器，最后训练收益与成本控制器。

## 实现

- `runtime.py`：官方 Show-o2 432×432、50时间节点 / 49次 Euler 速度预测、CFG=5；完全相同 float32 中途状态延续。
- `capture_repair_states.py`：保存 t≈0.345 / 0.455、原块7后的 prefix、RNG、时间网格和 CFG。基线与零适配器均要求 latent 完全相等、像素差为0。
- `evidence_encoder.py`：同一模型理解通路的目标/相反关系回答 margin 和 hidden states。L0标量、L1双 margin、L2语义隐藏表征。
- `evidence_conditioned_revisit.py`：复用原块2的权重，在原块7之后重访。只有条件分支视觉 token 被修复；首条10层路径冻结。
- `build_repair_candidates.py`：基线、普通循环、额外一次 ODE、零修复、目标/反向语义注入、目标/反向提示强化。
- `evaluate_oracle_pipeline.py`：官方 GenEval、PickScore、HPSv2、SigLIP，全部用原始提示评分。
- `train_repair_operator.py`：从训练集同状态终图正向延续蒸馏修复器，完整剩余轨迹反传，穿插基线正确态保真。
- `check_actual_rollout_gradient.py`：实模型完整终点反传dry-run，不做参数更新。官方生成入口的no_grad装饰在训练时必须绕过，函数正文和推理计算保持一致。

## 零干预与共享计算

零 FiLM 不保证额外 Transformer 为恒等映射。实际公式：

`h_new = h + visual_cond_mask * [F(h + gamma * LN(h) + beta) - F(h)]`

F 两次调用共享原始权重，gamma/beta 初始为0。冻结 F 参数仍允许梯度进入证据调制器。额外两次 F 须计入成本；当前没有降低总参数或 FLOPs 的结果。

## 数据隔离与训练门槛

结构开发12提示×2种子×2中途状态，48状态、8动作、384图。bootstrap 按12提示聚类。训练120条新增平衡空间提示、验证40官方提示、封存测试36官方提示；manifest在服务器 `splits_v1`。旧12/14提示只用于开发。新增提示与官方数据共享对象词汇及规则，须披露。

教师必须来自 train split、相同 state fingerprint，并满足官方语义错误→正确、HPS和PickScore均不下降；同时保留实际基线正确态。提示强化仅作为离线教师接口；学习后的条件共享循环须在独立验证集产生收益。终图择优的 Oracle 是上界。

训练损失是成功教师的终点 latent 蒸馏、终点距离排序与正确态保真，不是可微 GenEval 或人类偏好评价。固定修复器验证通过后才进入 controller。

## 复现、成本与证据范围

新实验冻结代码/配置、checkpoint、图像和终点文件 hash；变化必须另开目录。打乱证据来自同 split、同 flow step、另一提示。梯度检查点绑定每一步的时间、层列表及证据。

记录速度调用、视觉块调用、重访次数、剩余延续耗时与参数。FLOPs 未测为 null。prefix 和 MMU 在 Oracle 预计算，剩余延续成本不能当作端到端成本。额外 ODE 对照只在首个区间增加一次预测，尚未严格匹配循环 FLOPs。

空间修复是机制验证入口。FID/FVD、广域 GenEval、真实人类偏好、多样性及理解能力反向提升仍需后续实验。冻结理解权重尚不能证明理解能力改善。

## 2026-10-04 续跑

v3提前修复仍无语义翻正；v4对象定位/镜像诊断已完成；v5对象token空间路由进入共享层重访正在运行。详细记录见 STAGE3_CONTINUATION_20261004.md。

新增：diagnose_repair_actuation.py（终点实际幅度）、diagnose_grounding.py / evaluate_grounding_diagnostic.py（冻结MMU及官方对象诊断）、routed_shared_revisit.py / build_routed_candidates.py（同一原始block的空间注意力路由）、rollback_evidence.py（只导入较晚中途证据、保留早期状态和显式成本）。这些接口本身不代表修复成功。

最新：v5的144图已完成，仍无错→对；当前推进v6实例掩码位置修复，分别检验离线latent可塑性和共享block8重访，目标/反向均评测。object_transport.py 与 build_transport_candidates.py 均为离线机制诊断，外部检测器定位不能当作最终MMU自身能力。

v6结果：首次官方错→对1个来自latent_transport，Pick提高但HPS下降，不能作为成功教师；共享重访无翻正。v7（oracle_checked_rewind_dev12x2_v7）已启动120图，检验index35检查后回溯index20、导入中途clean及目标/反向搬移，保留噪声残差并完整续跑。该实验是离线教师可塑性诊断，不能当作最终loop成功。

v7已完成120图，目标方向0翻正；反向动作1翻正仍为同一个00412_seed202，HPS下降，无合格教师。当前v8（transport_geometry_dev6_v8，PID110755）核验6个既有状态的原始/目标/反向clean中途几何，共18图；区分搬移动作失败与后续flow回退。不能将中途分数当终图收益。

v8已完成：目标/反向搬移clean都0/6正确，多个案例丢失对象；尚未进入后续flow就没有正确几何。停止在无效latent/hidden搬移前提上继续扫强度或时机。下一步优先核验实例mask、源区域清除、目标几何及表征重建，保住对象身份后再检查终点质量与共享循环。


## v9/v10：实际修正对象搬移接口

已核对官方WanVAE确定性编码和归一化接口；原实例像素搬移、清除源区域后再编码，目标像素及目标往返均4/6关系正确，原始和反向均0/6。新增加3+2项测试通过，累计103项分轮通过。v10六个worker正在生成120张终图，并等待完整四类评分；不把中途关系正确计为成功教师。完整接口、对象审计、对照和成本解释见 [STAGE3_PIXEL_TEACHER_20261004.md](STAGE3_PIXEL_TEACHER_20261004.md)。


## v10完成及v11残差衔接对照

v10全部120终图评分完成：目标2个错→对、反向0；00412_seed202的Pick +0.476110、HPS −0.003174，00414_seed202的Pick −2.183771、HPS −0.031738，均不合格。原始往返对照没有语义翻正。全部24、原6可定位、实际4触发均已统计；零/20不触发态严格相等，原始官方便携导出和本地重分析与服务器report严格一致。累计1680已评分终图、4个动作级翻正、2独立案例、合格教师0。v11的144终图已经生成，正在评分：固定同index35及同原seed，用完整首步重算比较局部反推残差与原始seed噪声Linear bridge。新增3项测试通过，累计106项分轮通过。详细记录及原始文件见 [STAGE3_PIXEL_TEACHER_20261004.md](STAGE3_PIXEL_TEACHER_20261004.md)。


## v11完成与v12评测恢复

v11原始seed目标4个错→对，但四例HPS/Pick均下降，合格教师0。v12最小缩放/大对象锚定接口已生成36中途图，原外部GenEval评测环境路径失效；随后SSH超时。当前只恢复评分，不重生成图，v13未启动。本地已实现带全144图官方语义兼容性检查的恢复脚本。详见 [STAGE3_MINIMUM_GEOMETRY_20261004.md](STAGE3_MINIMUM_GEOMETRY_20261004.md)。原10分钟轮询已恢复ACTIVE。


## 最新接续：SSH已恢复，v12原图同步，v13条件队列

主机网络SSH已恢复；四个现有候选venv均缺MMCV，项目专用官方GenEval环境正在安装。v11原始归档已本地同步、SHA256核验且便携重分析与原report完全一致。v12的36中途图已同步并检查：保住更大实例，但填补/悬浮/截断仍在，未补填官方分数。v13使用新脚本和原中途实例身份审计（拒绝车厢环境当train），必须全144图评分兼容性及目标像素/目标往返几何通过才条件启动。详情见STAGE3_MINIMUM_GEOMETRY_20261004.md。累计1824已评分终图、10动作级翻正、4独立案例、合格教师0；训练尚未开始。10分钟轮询ACTIVE。
