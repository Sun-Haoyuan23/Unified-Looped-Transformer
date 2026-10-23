# 第三阶段：最小外观破坏与评测环境恢复

日期：2026-10-04。最新状态：主机网络SSH恢复，隔离官方GenEval安装及全144旧图兼容性验证通过。v12补评分目标像素/目标往返均5/6正确；v13全部144终图评分完成，首次1个开发态通过官方错→对及HPS/Pick守门，整体均值仍负。独立train_pilot_minimal_v14的24平衡提示×2seed捕获已启动PID19987，尚未训练。详细最终数值见STAGE3_V13_TRAIN_PILOT_20261004.md，下面保留先前过程。10分钟轮询ACTIVE。

## v11 已取得的结果

固定index35、同原始seed、同v9像素/VAE教师，局部反推残差与原始seed噪声两路都在新输入上重算完整首个速度。

|动作（全24状态）|错→对|对→错|Pick变化|HPS变化|SigLIP变化|
|---|---:|---:|---:|---:|---:|
|反推残差+目标教师|2|0|−0.263459|−0.003855|−0.002707|
|原始seed+目标教师|4|0|−0.289881|−0.003693|−0.004297|
|原始seed+原始编码图|0|0|−0.022231|−0.000102|−0.001027|
|原始seed+反向教师|0|0|−0.236635|−0.003265|−0.002635|
|检查后零操作|0|0|0|0|0|

原始seed目标动作使4个实际触发状态都实现官方错→对，分别是00369_seed101、00375_seed101、00412_seed202、00414_seed202；全24原始语义由5/24升至9/24。但四例Pick和HPS均下降，质量守门仍为5/24，合格教师0。反推残差目标动作的两例仍为banana/suitcase和cat/backpack。

反推噪声与原seed噪声的relative L2为0.057417–0.086839，cosine为0.996282–0.998352。小的残差差异仍影响终点正确性；不能据此断言残差存有完整对象语义。4个干预样本名义完整flow50次速度/500次视觉块，对照49/490；额外MMU、VAE、检测、像素合成及端到端时间另计。

累计1824张已评分终图，10个动作级语义翻正观测、4个独立被纠正提示/seed。独立开发提示仍12，合格教师0。v11原始JSON、噪声诊断、分组报告、源代码及配置快照已导出在服务器 `stage3_grounded_repair/transfers/stage3_fm_v11_results.tgz`，SHA256为 `5ce1ad24ddfadbd350c530d842d254549054aa9d0dd4484c36c82faace67da73`；便携语义重分析在服务器与原report严格一致。该归档已下载到本地work/stage3_fm_v11_results.tgz并解包至results/oracle_fm_rebridge_dev12x2_v11；哈希一致，本地evaluate_repair_pairwise重分析与服务器paired_report逐项严格相等。

## v12 最小几何改动

新文件 `minimal_pixel_teacher.py` 使用闭式几何投影：优先锚定mask面积较大的对象；只有两个原始对象轴宽加12像素间距超过432像素画幅时，才共同等比例缩小。较大对象原位置不可行时，仅移动到最近可行位置。另一轴中心尽量保留。源区域只清除实际移动对象，合成仍为固定Telea5/膨胀5及premultiplied alpha，未扫描时机、强度或阈值。

例如banana/suitcase在画幅可容纳时，suitcase保留原框和原尺寸，只把banana搬到右侧；boat/suitcase合计轴宽492像素，共享缩放420/492，而非先前将大箱体缩至约一半。该设计针对已观察到的固定缩小和源区域破坏，尚不表示终点质量改善。

3项最小几何测试通过，累计109项分轮通过；不是同次全套重跑。v12目录 `stage3_grounded_repair/minimal_teacher_geometry_dev6_v12`，PID96701，使用健康空闲GPU0。既有6个可定位状态×原始/原始VAE往返/目标像素/目标往返/反向像素/反向往返，共36张图已经生成。没有optimizer更新。

## 当前故障和恢复检查

评分失败的明确原因：`/home/shy_23/calm_geneval_20260930/eval_venv/bin/python` 不存在。读取确认外部calm目录及其detector目录均不存在，DPO环境也未找到mmdet/mmcv。项目自身还保有 `stage2b/geneval_models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth`；不能未经hash及兼容性验证就声称完成同官方评分。

后续SSH连接超时，尚未读取其他venv的可用性。候选环境包括 `venvs/geneval2_official`、`corebench_official`、`compbench_full_official`。已有图、教师latent、原失败status和run.log必须保留，不重新生成。

已在本地实现 `recover_geometry_evaluation.py`：显式指定评测Python、官方配置和本项目checkpoint，先核对v6冻结的detector hash；以v11全部144既有终图重新评分，要求正确性和失败原因逐图严格一致，才补评v12的36图。恢复配置、版本及源代码hash独立冻结，保留原失败日志，不将缺失分数填0。该脚本的语义一致/缺失/重复/错prompt拒绝测试通过1项；它尚未上传或在服务器运行，分轮测试累计110项。

## 接续顺序

1. 恢复SSH，定位/恢复本项目可用的官方GenEval环境。
2. 核对权重hash及v11全144图兼容性，仅补v12评分；同时查看源/目标图，不能把车厢环境检测框当成可移动train实例。
3. 中途目标像素图及目标VAE往返都保住对象身份、满足目标关系后，在新目录 `oracle_minimal_teacher_dev12x2_v13` 固定同index35和原seed，运行与v11相同六动作；全部24状态保留，并按原6定位/实际触发/全24分别统计。
4. 同时修复终点评分pipeline的失效外部路径，完整验证官方语义、HPS/Pick/SigLIP与方向/原始编码对照。
5. 仅在出现终图官方错→对且HPS/Pick均非负的案例后，收集train split的24条平衡提示×2seed成功教师和真实正确态，学习共享Transformer重访，再做独立val与证据消融。

外部检测和像素操作仍只用于离线教师；论文核心必须由可训练共享Transformer重访实现。修复训练/controller均未开始；FID/FVD、人评、多样性、理解反向提升、参数/FLOPs降低未完成。


## 主机网络恢复及隔离环境安装

用户确认服务器和内网正常后，主机网络SSH完成认证；普通沙盒SSH通道此前超时。后续SSH/SCP使用已授权的require_escalated主机网络，密码仅在认证提示中输入。现有会话48237有效时可以复用；会话断开后必须重连，不凭旧ID假定连通。

四个候选venv均缺少mmcv。新建项目自己的evaluator_runtime_v1/venv，setup_official_evaluator.py固定torch1.13.1+cu117、torchvision0.14.1+cu117、mmcv-full1.7.1、mmdet2.28.2及相应依赖；官方权重仍用本项目stage2b/geneval_models，冻结SHA256743b7d99015f1224c6d57fd4b14d04b15cc8ec72ae7eee7831e7c71d8873b7a54。安装成功本身不代表评分兼容，必须全部144个v11原终图的correct/prompt/reason严格一致。恢复脚本已修正读取冻结config的configuration层级，保留初次失败status/log。

queue_evaluator_recovery.py等待安装、核验PIL先导入及MMCV compiled ops，再等GPU0空闲；仅补评原36图。通过后可条件启动固定v13：index35、同原seed、六动作与v11相同，144终图；六张健康空闲GPU0/2/3/4/6/7，GPU1仍被他人任务占用，5故障。

v12未评分原始归档work/stage3_minimal_v12_diagnostic_unscored.tgz的SHA256为e10f9afb31ddc61a77bcd00ab0361409d766d4b5c9700348140d0c767fc42a70，已解包36PNG及记录到results/minimal_teacher_geometry_dev6_v12。只是原图同步，不新增生成计数或补填官方分数。

## 中途身份与外观审计

查看原始/目标/反向及VAE往返六列contact图：suitcase保住更大尺寸，bed也保留了较完整的形状；boat原位置仍有黑色填补痕迹，banana悬浮及源位置残影仍在，cat/backpack掩码重叠造成头部和背包截断，oven切片仍缺乏自然支撑。外观改善只是观察，不能替代终点HPS/Pick。

minimal_teacher_identity_audit_v13.json只读取原始中途图、冻结mask和状态fingerprint，明确拒绝00379_seed202的车厢环境作为独立train实例；其他可见实例仍交给官方目标/往返几何门控及终点质量检验。该人工离线身份审计不是最终MMU定位器。build_minimal_fm_candidates.py和launch_minimal_fm_oracle.py用新脚本实现该检查；全部24状态×六动作仍保留。没有终图或终标签进入动作决定。

evaluate_oracle_pipeline.py已增加显式的恢复Python/官方配置/权重路径与兼容性证明检查，并冻结新的evaluation_config，不再让新实验依赖已消失的calm目录。旧实验源代码快照保留。新增脚本语法编译通过，原恢复一致性测试再次通过；仍为110项分轮测试，不将重跑重复计数。

## 共享修复器容量审查

现有ConditionedRevisit将L2理解证据经全局FiLM广播到所有视觉token，并在block7后重访原block2；空间细粒度搬移可表达性尚未证实。合格教师出现后，训练前需明确与教师动作匹配的token级证据映射，并比较重访block8等位置匹配的共享core。两次共享core调用的零初始化差分形式有严格零操作保证，但尚无训练成功或参数/FLOPs下降结果。

累计已评分终图仍1824、动作级翻正10、独立纠正提示/seed4、合格教师0、optimizer更新0；v13尚未运行。FID/FVD、人评、多样性、理解反向提升、总参数/FLOPs降低均未完成。


## v13最新结果

详见[STAGE3_V13_TRAIN_PILOT_20261004.md](STAGE3_V13_TRAIN_PILOT_20261004.md)，其中记录唯一过HPS/Pick守门的开发态、整体负均值、独立train采集与原始归档。此前“v13尚未运行/合格教师0”的过程记录由该结果替代；train教师仍0，optimizer更新0。
