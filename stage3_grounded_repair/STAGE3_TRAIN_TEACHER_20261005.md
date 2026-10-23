> 当前综述见[2026-10-05进展总结](../LOOPED_UNIFIED_VISUAL_GENERATION_PROGRESS_20261005.md)。本次冻结日志快照：v46两joint组各100步完成，outer75步、inner34步；四组实际GPU初始化hash匹配；v47尚未开始。第三阶段真实更新1,559（其中完成固定组1,450、运行中109）；终图生成提升尚未成立。

> 最新以[STAGE3_PROGRESS_20261005.md](STAGE3_PROGRESS_20261005.md)及[STAGE3_RECURRENT_CORE_20261005.md](STAGE3_RECURRENT_CORE_20261005.md)为准：v44四组100更新、v45的120终图全部完成，0教师翻正且各损伤1正确态；未有循环生成提升。v46开始训练共享低秩核心和嵌套循环终点蒸馏，四组固定step100，全部成功后自动v47的144图。资源接续32461运行，6/7执行、其余等待空闲，不杀他人；固定质量门，不自动扩val/controller。累计3862评分记录；v44后1250已完成更新，v46实际日志另计。10分钟轮询ACTIVE。

# 第三阶段：独立 train 教师实验（2026-10-05）

## 原始材料与验证范围

本记录来自 train_pilot_minimal_v14 的24条冻结train提示×seed101/202共48状态，index35。原始 JSON、288张终图对应的四类评分、源代码/config hash 和配对结果已归档。此阶段不包括optimizer更新、人评或独立val；离线教师不是共享循环方法。

v17归档SHA256：53b75e78f986d8a6dbad837744b547f7fdd86cf46477500ecbcb379a08739cd8。服务器export_results.py严格核对官方便携导出与paired_report，本地evaluate_repair_pairwise.py重分析也逐字段严格一致。原始路径：results/oracle_minimal_teacher_train24x2_v17。

## v17终点结果

48状态中原中途双对象可定位10个，实际几何及身份门控触发8个，来自7条提示。两种目标动作各8个官方错→对、0个对→错；反向和原始编码均0个翻正。基线18/48正确，目标26/48正确。所有8个实际触发状态都翻正，但HPS/Pick质量守门成功教师为0。40个未触发状态严格沿基线生成，不能用于说明修复器保真能力。

| 范围/动作 | 错→对 | Pick均值差 | HPS均值差 | SigLIP均值差 |
|---|---:|---:|---:|---:|
| 全48，目标原seed | 8 | −0.112199 | −0.001602 | +0.000443 |
| 全48，目标inferred | 8 | −0.091186 | −0.001531 | +0.000245 |
| 实触发8，目标原seed | 8 | −0.673195 | −0.009613 | +0.002660 |
| 实触发8，目标inferred | 8 | −0.547118 | −0.009186 | +0.001472 |
| 实触发8，反向原seed | 0 | −0.218419 | −0.003967 | +0.001864 |
| 实触发8，原始编码原seed | 0 | −0.138705 | −0.000732 | −0.001500 |

全48目标语义配对增益+0.166667，提示聚类探索性95%区间[0.0625,0.291667]。该结果限定于已选择的训练子集及离线教师，未经多重探索调整，不能称最终GenEval提升。质量守门offline oracle仍18/48，增益0。真实baseline-correct train状态18个已经存在，但成功教师缺失，training_readiness.json明确NO-GO。

实触发编辑50次完整速度/500视觉块，基线49/490。表中的端点walltime不含预计算理解检查、检测器、教师VAE和构图，不是端到端成本；FLOPs尚未实测。

## 损伤定位与下一实验

contact图显示：spoon源位置残影、surfboard/bicycle背景填补和实例重叠、zebra/umbrella垂直搬移造成悬浮、cat/traffic-light源残余及背景痕迹。原始编码自身平均造成小幅质量下降，但目标剪贴损伤明显更大，不能仅归因VAE。

v18/v19验证场景完整性：左右关系的目标中途图采用整幅水平镜像，保留对象、背景、支撑面和遮挡一起变化，不剪贴、缩放或填补。垂直关系保持零操作且不触发；纵向镜像作为明确的wrong-axis控制，不称反向证据。先在全部10个既有定位train状态生成/官方评测中途原始、目标、wrong-axis及其确定性VAE往返，身份审计沿用原中途来源；只有目标像素与目标往返均正确的状态触发终点延续。所有48状态仍保留配对。

此接口只测试能否获得场景一致且质量过门的离线训练目标。最终核心仍是理解证据驱动可训练共享Transformer重访；整幅镜像本身不作为最终方法，也不证明共享修复器能学习该空间变换。训练前仍须审查token级证据映射及共享core深度匹配。

## v18/v19场景镜像结果

v18的60中途图已完成官方评分：目标像素/往返均5/10，原始/原始往返/wrong-axis均0/10。实际触发5个状态、5条提示：train_extra_00001_seed101、00008_seed202、00011_seed202、00014_seed202、00021_seed101。v19的288终图及四项评分完成，原始归档SHA25622c29ce8293ee3892a71936a24463bfa654264101b4023134252f41e0104f11b已同步，本地配对重分析逐字段严格一致。

| 同一5个实际触发状态 | 错→对 | Pick均值差 | HPS均值差 |
|---|---:|---:|---:|
| v17剪贴，原seed | 5 | −0.375757 | −0.006738 |
| v19场景镜像，原seed | 4 | +0.012712 | −0.003906 |
| v17剪贴，inferred | 5 | −0.322900 | −0.006934 |
| v19场景镜像，inferred | 5 | −0.021412 | −0.004297 |

这是相同状态的接口对照，镜像降低了平均质量损失，但没有翻正样本的HPS非负。原seed平均Pick为正包含一个未翻正状态，不能把平均Pick正值当成功教师。全48原seed4个翻正、inferred5个翻正、0个退化、wrong-axis0个翻正；质量守门oracle仍18/48，train教师0。

新增60中途图HPS/Pick/SigLIP评分已经完成。五个实际触发的target_pixels的HPS均较原中途图低：−0.000488、−0.000732、−0.001221、−0.002930、−0.001465。后续终点HPS损失通常更大；因此既有镜像方向本身的代理分数影响，又有后续生成的额外损失，不能全部归于VAE或flow。中途评分不代替终点质量守门。

## v20共享修复器实现与容量审查

TokenConditionedRevisit已实现：MMU目标/反向隐藏状态、边际、flow时间，经证据编码；每个视觉token结合自身内容和固定二维坐标，预测不同的gamma/beta，再重复调用同一冻结Transformer。文本与无条件CFG分支保持原样，零初始化严格恒等。外部实例mask和像素操作不进入修复器推理。

共享core默认8：插入点在原block7后，core8是原路径紧接的深度，作为原core2的深度匹配对照。train_token_repair_operator.py支持固定core2/8比较，完整剩余flow终点蒸馏、ranking和真实baseline-correct保真；质量门未通过前不做optimizer更新。四项新单测以及原四项重访不变量测试通过。

真实模型dry-run在train_extra_00001_seed101_i35完成：完整14次剩余flow终点可反传，零操作latent严格一致，基础参数无梯度且共享core hash不变；证据路径梯度绝对和5.07872655e−5，约10.94秒/峰值14.72GiB，optimizer0。新增可训练参数1,390,472个，因此目前没有证明总参数降低。四次干预窗口执行8次共享core调用，增加计算；参数/FLOPs压缩目标须后续单独验证。

token级映射为位置不同的修复提供表达能力，但整幅镜像涉及全局空间重排，现有有界FiLM仍可能学不动。后续拿到合格train教师后，必须先做终点拟合及core2/8对照，再做独立val与证据乱序/反向消融；不能因梯度通过就声称修复成功。

## 当前有依据的下一实验

v21/v22检查噪声/视觉结构一致性：此前镜像clean时，seed或inferred噪声残差仍保留原空间朝向。新对照固定同提示、原seed、index35：原生当前latent横向镜像；镜像原seed噪声与镜像教师clean一起rebridge；镜像inferred残差与教师clean一起rebridge；原生latent纵向wrong-axis控制。不搜索新seed、不扫强度或时机。先用原生predicted-clean latent镜像的VAE解码做官方中途几何门，再从同状态重算完整首速度并终点评分。仅为离线可塑性上限，最终仍需共享循环学习。

累计已评分终图2544，动作级翻正43，独立被纠正提示/seed仍13；合格dev态1、合格train态0、optimizer更新0。FID/FVD、人评、多样性、理解反向提升、总参数/FLOPs下降均未完成。阶段3继续，10分钟轮询保留。

## v21–v25最新结果与真实训练

v21的60中途图目标pixels5/10正确，原生predicted-clean latent镜像解码0/10。contact图有spoon/bicycle/cow等形状损伤，不能把官方关系失败简单说成坐标未移动。v22中途门控停止，没有终图。v20真实梯度/60中途质量评分/v21原始归档hashddd1672be4b1666eae8d2b9b2ab841349300d146c64848f84c503fc4acbb0365已本地同步核验。

v23排除原生latent接口，只用v18已验证的像素镜像→确定性VAE教师，比较seed/inferred残差同向镜像与原朝向inferred。288终图及四项评分完成，三种目标各5个翻正，wrong-axis0、对→错0。未镜像inferred对照所有同状态原始HPS/Pick/SigLIP与v19严格相同。

首个合格train教师是cow right of refrigerator、train_extra_00021_seed101_i35/checked_inferred_correlated_target：官方错→对，Pick+0.019726086744、HPS+0.000732421875、SigLIP+0.004985675216。未镜像残差版本Pick−0.443037/HPS−0.007324；seed同向版本Pick−0.197611/HPS+0.000732仍不合格。仅inferred同向版本入教师。单提示单seed，代理分数微增不等于人评提升。

原始归档hashb570fcc1cd17d5f17728e33a2b545a141362c42ea89370f8503c50b085782831已本地同步；本地配对分析逐字段严格一致。v23_portable_verification.json记录paired_report hashb1d440a15d67ae6889d864c25a54332d28dcc96667f1668a64e9e3c012cfbaee及training_pairs复核：1教师、18实际baseline-correct保真态。全48目标动作平均HPS/Pick仍下降，质量守门offline oracle18/48→19/48，探索性提示聚类区间[0,0.0625]，不代表controller或整体质量成功。

run_token_capacity_and_val.py已启动launcher60856，并核对真实optimizer更新：core8至少11步，core2至少10步，有限梯度，峰值约14.73GiB。两组各固定50步、L2/width128/window4/lr1e−4/相同随机顺序，每第三步用真实正确态保真；GPU4/6训练，GPU0/7并行捕获12条val提示×2seed。容量pilot的单个成功教师不作为足够训练覆盖。

val提示启动前按每关系排序首次3条冻结，不看图/评分；checkpoint固定最终step50，不按val选择。流水线自动从两组step50生成基线、专门未训练zero_op、正确/乱序/反向证据，共每core120图、240图，并官方/HPS/Pick/SigLIP评分。val推理不读外部mask和像素编辑，controller不训练。先读token_capacity_and_val_v25/pipeline_status.json及core8/2/train.jsonl，再token_repair_val_v26/core8/2的paired_report。

总已评分终图2832、动作级翻正58（含同状态重复对照），独立纠正提示/seed仍13，合格dev1/train1。optimizer数动态读取日志。未使用train扩展32提示代码已预备但未启动，2项隔离测试通过，累计126项相关单测分轮通过。FID/FVD、人评、多样性、理解反向提升、参数/FLOPs降低未完成，阶段3继续。
