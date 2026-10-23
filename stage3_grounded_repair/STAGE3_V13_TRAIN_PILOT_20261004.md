# 第三阶段：v13终点验证与独立train采集

日期：2026-10-04。状态：v13全部144终图和官方GenEval/HPS/Pick/SigLIP完成；首次有1个开发态通过严格HPS/Pick守门。train_pilot_minimal_v14已经启动24条平衡train提示×2seed的index35状态采集，尚无train成功教师或optimizer更新。10分钟轮询ACTIVE。

## Material Passport

- 类型：代码实验执行及结果核验；源数据为真实模型生成和原始评分。
- 验证：v11全部144旧图官方correct/prompt/reason严格相同；v12和v13原始归档哈希核验，v13本地便携配对重分析与服务器report逐项完全一致。
- 推断范围：12个重复使用的开发提示，24个提示/seed；离线像素/VAE教师接口，尚非共享Transformer循环成功。开发态不会进入train。

## 官方评测恢复和中途几何

已恢复主机网络SSH。项目专用evaluator_runtime_v1固定torch1.13.1+cu117、torchvision0.14.1+cu117、mmcv-full1.7.1、mmdet2.28.2；原GenEval源代码和冻结detector权重不变。安装成功后全144旧终图逐图兼容性验证通过，再补评v12原36中途图，没有重生成。

v12最小改动目标像素图、目标VAE往返都5/6正确；原始clean、原始往返及反向两组均0/6。通过的五个状态为00369_seed101、00375_seed101、00412_seed202、00414_seed202、00444_seed202。00379的车厢环境不构成独立train实例，离线身份审计明确拒绝；没有终标签决定动作。

## v13终点结果

固定index35、原seed、与v11相同六动作，只有最小几何教师及中途身份审计改变。全部24状态保留，5个实际触发。新输入重算完整首个速度。

|动作（全24）|错→对|对→错|严格合格翻正|Pick变化|HPS变化|SigLIP变化|
|---|---:|---:|---:|---:|---:|---:|
|原seed+目标|4|0|1|−0.207355|−0.003042|−0.000826|
|反推残差+目标|4|0|0|−0.217645|−0.002950|+0.001221|
|原seed+原始往返|0|0|0|−0.035867|−0.000275|−0.000827|
|原seed+反向|0|0|0|−0.233385|−0.003255|−0.000446|
|检查后零操作|0|0|0|0|0|0|

原seed目标的4个翻正为boat/suitcase、banana/suitcase、cat/backpack、bed/frisbee。00369的table/oven此次未翻正。两个目标动作共8个动作级翻正、4个独立提示/seed；反向0。原seed目标的实际5触发平均Pick−0.995302、HPS−0.014600，原6定位平均−0.829418/−0.012166。全24负值被19个不触发零变化稀释，不能声称总体生成质量提升。

唯一严格通过官方错→对且HPS/Pick均非负的开发态为00412_seed202_i35、checked_seed_target：Pick+0.3150664128、HPS+0.00146484375、SigLIP−0.0028821528。反推残差同例Pick−0.245377、HPS−0.004883，不能合格。该开发图仍有悬浮banana和原位置残余形状，HPS/Pick代理守门不等于真实人类偏好改善；在训练教师收集时保留这类外观审计。

全24原始语义5/24→9/24；严格质量守门离线oracle只5/24→6/24（+1/24），prompt cluster探索性95%区间[0,0.125]。oracle使用终标签，不是部署controller；多轮开发选择也未作确认性校正。

## 成本与资源调整

实际干预50次完整速度/500次视觉块，对照49/490；全24目标动作均值49.208333/492.083333。剩余flow及解码walltime不含MMU、检测器、教师构图/VAE或预计算，这些开销另计，FLOPs未实测。这里没有新增可训练参数，也没有共享core重访成功或总参数降低证据。

起初使用空闲0/2/3/4/6/7。四分片完成96图后，GPU2/3被另一用户任务占用；核对本项目启动器身份、所有子任务及等待状态后，仅停止两个本项目等待中的启动器，将未开始的shard1/2分别移到空闲0/6。四项评分用0/4/6/7。resource_reschedule.json保留原/新进程、GPU及源码hash，提示/seed/六动作/模型计算不变；未中止其他用户任务。

## 独立train采集已启动

launch_train_capture_pilot.py调用prepare_train_pilot.py，从冻结splits_v1/train.jsonl中按四关系分别选最前6条prompt_id，共24提示×seed101/202。选择不看图或终点分数，冻结train_prompts.json及配置；所有行必须split=train。启动PID19987，GPU0/4/6/7四个捕获worker，index35，保留原MMU证据、prefix、RNG、baseline终latent及真实零操作验证。它是48个新train状态采集，不是训练，也还不是48个成功教师。

接下来在新train目录实现泛化的中途clean导出、官方实例定位与最小几何/确定性VAE诊断；不能复用硬编码六dev状态的旧脚本。只根据中途证据决定候选，随后同状态完整终点官方/HPS/Pick/SigLIP评分。必须存在train中合格成功教师及真实baseline-correct保真态，才能进入L2修复器终点训练。

训练前审查全局FiLM是否能表达空间动作、token级理解证据映射和共享core位置匹配（现有原block2在block7后重访，需比较block8）。之后用独立val终图、乱序和反向证据检验；固定修复器验证前不训练controller。离线像素编辑只提供教师，最终方法仍为可训练共享Transformer重访。

## 原始材料

- v11归档SHA256：5ce1ad24ddfadbd350c530d842d254549054aa9d0dd4484c36c82faace67da73。
- v12未评分36PNG归档：e10f9afb31ddc61a77bcd00ab0361409d766d4b5c9700348140d0c767fc42a70。
- v12补评/教师latent/兼容性/环境与源码归档：0630b122d4f84692a0c9783af5b5d7e054186eb19492ed8ade179db0f0117a80。
- v13原始JSON、源码配置、资源调整和contact图归档：6b2a93b347086094ce2168d5c3828689bf406aa27a60d4d54caf7c9418aa4c6f。
- 上述归档均在本地work/及results/解包目录，v13便携重分析work/v13_portable_reanalysis.json与原paired_report严格相等。

累计1968张已评分终图、18个动作级翻正观测、5个独立纠正提示/seed、1个过HPS/Pick守门的开发态、train教师0。110项相关测试分轮通过，未重复计算本轮重跑，也未声称新增脚本已做完整训练测试。optimizer更新仍0。FID/FVD、人评、多样性、理解反向提升、总参数/FLOPs降低未完成。
