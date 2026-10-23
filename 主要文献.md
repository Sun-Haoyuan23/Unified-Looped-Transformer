# 共享循环与统一多模态视觉生成：论文导读及文献交接

整理与原始来源核验日期：**2026-10-23**。

本资料围绕本项目的完整研究问题组织：**在统一多模态模型中，用共享Transformer核心增加或替代执行深度，让理解、生成和中途反馈发生有效互动，并在语义、画质、保真、多样性和计算成本上形成可信收益。**

它与[技术报告及实验交接](./技术报告-交接文档.md)配套：技术报告回答实际做了什么、结果如何；本文回答相关研究已经做了什么、应如何阅读，以及哪些论文需要作为后续方法的对照。

## 1. 收录范围、历史来源与阅读标记

本次从91份不含大体积实验结果目录的Markdown记录中提取历史来源，再以全域名复查补上作者项目站和正式工具来源。先逐项找回历史来源，再补充基础方法、直接替代路线和评价论文；另外保留历史只出现题名的线索。覆盖数量见第4节及结构化索引。

**这是覆盖本项目历史参考并扩展邻近研究的导读，不是声称穷尽整个领域的系统综述。** 没有执行数据库全量检索、PRISMA筛选或所有参考文献的递归追踪。新增论文的选择标准是：能解释本项目实现、构成直接比较方法，或支撑尚未完成的评测。

所有本地路径均以`outputs`文件夹为根，写作`.\...`。本文及其结构化目录不依赖可删除的大体积实验结果目录。

### 1.1 “参考过”具体指什么

| 标记 | 含义 | 如何使用 |
|---|---|---|
| **H：历史明确参考** | 旧文档中已有该论文、正式出版页或作者项目站来源 | 可追溯为当时的设计参考；不表示曾完整复现或逐页读过 |
| **B：历史方法／工具背景** | 旧项目已用到或计划使用该方法、组件、指标；正式原论文题录本次补齐 | 可解释实现或评测来源；不能追认为早期已读原论文 |
| **H?：历史题名线索** | 旧记录只写题名，本次才找到精确编号 | 保留不确定性，不能倒填早期阅读版本 |
| **N：本次补充** | 为完整导读新增的相关论文或特定扩展版本 | 是后续阅读与比较候选，不表示已用于旧实验 |

**P0/P1/P2都是阅读优先级。** P0是理解本项目决策最优先的论文；P1是重要对照、信号或评价来源；P2是扩展背景。P0不等于每篇都是共享循环的直接竞争方法。

每条均注明本次实际核验范围。多数条目核验的是原始题录和摘要；ELT、Looped-DiT、部分反馈方法等另核对了选定方法章节。**未将摘要核验写成全文精读，未把作者论文成绩写成本项目成绩。** 旧记录若记载曾核对正文或实现，也不能据此推定本次重新完整审阅了所有版本。

作者列采用“首作者等”，完整作者表、BibTeX和最新PDF请以原始链接为准。年份沿用对应来源的首稿年或正式发表年，发表信息另列；判断先后关系时应查看明确的首版日期，而非仅比较年份标签。新增会议状态仅在原始来源已确认时填写。

### 1.2 先分清六种容易混淆的“多做计算”

| 机制 | 改变的计算 | 代表文献／本项目定位 |
|---|---|---|
| 采样时间步 | 沿扩散／flow时间积分更多次 | DDIM、Flow Matching、SiT；与层循环分别计数 |
| 同一步内共享深度 | 同一组权重重复更新hidden | ELT、Looped-DiT；本项目最直接目标 |
| 跨采样步状态递归 | 把上一步hidden等信息传到下一步 | RIN；不能仅因“recurrent”就归入ELT式压缩 |
| 自适应计算 | 选深度、宽度、token、缓存或停止时间 | AdaDiff、DyDiT、E-DiT、ACT |
| 候选搜索／梯度优化 | 更多候选、分支、验证或latent更新 | PreviewDiff、ReNO、D-Flow、VHS |
| 语义反思与再生成 | 分析图像或preview后提出下一次修改 | UniRect-CoT、AD-Loop、UniEdit-I、Self-Refine |

因此，一个方法可以增加推理预算但不共享层；也可以共享层而增加FLOPs或wall time。参数存储、激活参数、NFE、共享块调用、验证器调用和总耗时应分别比较。

## 2. 最先阅读的路线

建议先按下面的顺序建立共同语言，再进入第5节的完整目录。这里的“要回答的问题”是阅读任务，不是对原论文效果的替代性结论。

| 顺序 | 阅读组合 | 要回答的问题 |
|---|---|---|
| 1 | [Show-o2](#p-L014)、[Transfusion](#p-L015)、[JanusFlow](#p-L021) | 实际底座在哪个latent、哪个Transformer及哪种损失上生成？理解和生成共享到什么程度？ |
| 2 | [ELT](#p-L026)、[Looped-DiT](#p-L027) | 循环怎样替代独立深度？不同出口怎样监督？质量—存储—执行计算如何分开？ |
| 3 | [Depth through recurrence / Looped flow-TTS](#p-L028)、[LoopVAE](#p-L029)、[RIN](#p-L030) | 循环顺序、所在位置及跨步记忆分别有什么作用？哪些结论不能跨模态移植？ |
| 4 | [LaTtE-Flow](#p-L032)、[AdaDiff](#p-L031)、[E-DiT](#p-L035) | 同样预算投给时间专家、早退或路由，会不会比额外循环更合适？ |
| 5 | [Dreaming in Flow / GGF](#p-L045)、[UniRect-CoT](#p-L060)、[PreviewDiff](#p-L047) | 反馈究竟改变条件、latent、速度方向、搜索分支，还是模型权重？ |
| 6 | [LatentUM](#p-L046)、[RecA / RECA](#p-L073)、[Understanding–Generation Transfer](#p-L062) | 怎样让视觉理解表示真正可供生成使用？两侧能力如何验证迁移？ |
| 7 | [REPA](#p-L065)、[iREPA](#p-L067)、[RARE](#p-L071)、[Self-Flow](#p-L070) | 过程目标应该传什么信息？教师可恢复性、空间结构和目标归一化如何区分？ |
| 8 | [Diffusion Inference Scaling](#p-L076)、[ReNO](#p-L080)、[D-Flow](#p-L081)、[VHS](#p-L078) | 若多给相同计算，搜索、噪声优化或latent验证器是否已能取得收益？ |
| 9 | [Flow-GRPO](#p-L100)、[Diffusion-DPO](#p-L094)、[VADER / AlignProp后续](#p-L096) | 对齐目标如何穿过生成过程？确定性flow需要哪些额外推导？ |
| 10 | [GenEval](#p-L117)、[GenEval 2](#p-L114)、[HPSv2](#p-L108)、[Pick-a-Pic / PickScore](#p-L107) | 语义正确和偏好改善如何用独立指标交叉验证？ |
| 11 | [DRIFT](#p-L097)、[Vendi Score](#p-L110)、[DreamSim](#p-L111)、[CMMD / Rethinking FID](#p-L112) | 提升偏好后，是否仍保有有效多样性和分布覆盖？ |
| 12 | [RealUnify](#p-L003)、[Unison](#p-L002)、[MME-Unify](#p-L105) | 统一是否产生理解→生成及生成→理解的双向收益？怎样排除仅底座更强？ |

## 3. 从已有文献可以怎样定位本项目

**共享深度已有直接工作。** ELT与Looped-DiT使“重复一组Transformer提高有效深度”本身已有明确先例。本项目需要证明在统一理解／生成架构及其特定反馈接口上，循环相对冻结核心、无循环、额外非共享深度和预算替代路线有什么独立收益；仅增加循环次数不够。

**理解促进生成、生成反向促进理解、反思后再生成也都有直接先例。** Dreaming in Flow、RecA、UniRect-CoT、AD-Loop和双向迁移研究意味着相关性与区别需要落实到反馈对象、训练信号、部署信息、核心共享方式和成本。它们是后续正文逐篇讨论的邻近工作。

**过程对齐与终图收益是两层问题。** REPA系列和Self-Flow为表示监督提供动机，但本项目冻结的actual-additive教师及其物理／hidden／完整终点loss是具体实现选择。即使过程残差、梯度或latent误差改善，也仍需完整生成质量与保真评价。

**质量、正确率、多样性和理解保留应并行审查。** GenEval提供对象关系评价，偏好模型提供人类偏好代理，DreamSim／Vendi提供不同的感知与多样性视角；RealUnify／Unison提供协同问题框架。它们的测量对象不同，不适合合成一个未经验证的“总能力分”。

以上是本次对文献与项目关系的判断。本文没有开展新的模型实验，也不更新技术交接中的服务器训练状态；文献核验不能证明第三阶段已完成。

## 4. 文献统计与目录

本版收录**133条文献／版本资料**。其中包含T2I-CompBench原版及扩展版两条、1条明确撤回的历史论文和1条题名不完整的学位论文存档；因此不把条目总数称为完全独立的正式论文数。历史来源共146条出现记录、72个规范化来源URL，全部已映射，缺失为0。相同论文的arXiv与作者项目站可能分别形成来源URL，来源URL数不等于论文数。

| 类别 | 条目数 |
|---|---:|
| [综述与研究地图](#cat-overview) | 3 |
| [扩散、flow matching与视觉Transformer基础](#cat-foundation) | 9 |
| [统一多模态理解与生成：本项目的模型底座](#cat-unified) | 13 |
| [共享循环、递归深度、自适应计算与压缩](#cat-loops) | 19 |
| [内部反馈、反思修复、交错推理与双向促进](#cat-feedback) | 20 |
| [视觉表示、语义latent与过程监督](#cat-representation) | 9 |
| [推理时扩展、搜索、latent优化与计算预算](#cat-budget) | 10 |
| [空间接地、组合控制与图像编辑](#cat-guidance) | 10 |
| [偏好对齐、强化学习及质量—多样性权衡](#cat-reward) | 11 |
| [语义、偏好、保真、多样性、分布及理解评测](#cat-evaluation) | 25 |
| [低秩适配、条件调制与蒸馏基础](#cat-engineering) | 3 |
| [历史存档与未完整确认的研究线索](#cat-archive) | 1 |

| 来源类别 | 条目数 |
|---|---:|
| H | 69 |
| B | 9 |
| H? | 1 |
| N | 54 |


条目的**方法摘要**是原始文献的意译；**与本项目的关系、适用边界**为本次研究判断。未统一转录论文性能数字，以免混用修订版、数据集、分辨率和计算协议。感兴趣的具体成绩应沿原始链接核对对应表格与版本。

## 5. 分类论文导读

<a id="cat-overview"></a>

### 综述与研究地图

用于建立术语和任务地图。综述中的方法判断应继续追溯到原论文；本节的统一评测论文同时为“理解和生成是否真正互促”提供问题框架。

<a id="p-L001"></a>

#### L001 · Unified MLLM Survey · P1 · N

**[Towards Unified Multimodal Large Language Models: A survey](https://aclanthology.org/2026.findings-acl.1853/)**  
Xu Ma等，2026。Findings of ACL 2026。

- **方法摘要：**从架构、损失、表示与对齐整理统一MLLM研究，并讨论挑战。
- **与本项目的关系：**作为进入主题与继续滚雪球检索的入口。
- **适用边界／版本注意：**综述用于地图，具体方法与数值引用应回到原论文。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L002"></a>

#### L002 · Unison · P0 · H

**[Unison: Benchmarking Unified Multimodal Models via Synergistic Understanding and Generation](https://proceedings.mlr.press/v306/liu26dv.html)**  
Jinyu Liu等，2026。ICML2026/PMLR306正式页面已核验。

- **方法摘要：**以内部一致性、理解引导生成、生成引导理解和互相增强四维度，用统一及解耦track诊断协同，并训练Unison-Judge。
- **与本项目的关系：**适合作为双向性验收基准；不能只给两个孤立benchmark分数便认定协同。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**PMLR正式书目信息、摘要与BibTeX。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2_loop\NEXT_EXPERIMENT.md](./stage2_loop/NEXT_EXPERIMENT.md)；[.\stage2_loop\RESEARCH_DECISIONS.md](./stage2_loop/RESEARCH_DECISIONS.md)。

<a id="p-L003"></a>

#### L003 · RealUnify · P0 · H

**[RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark](https://arxiv.org/abs/2509.24897)**  
Yang Shi等，2025。CVPR2026正式原站搜索索引与arXiv交叉核验。

- **方法摘要：**围绕Understanding Enhances Generation和Generation Enhances Understanding，组合端到端与分步诊断评价，以分辨基本能力不足与整合失败。
- **与本项目的关系：**本项目后续MMU/AR理解保真、反向收益及协同归因的直接基准，不把冻结原权重当理解能力保留证明。
- **适用边界／版本注意：**CVPR2026正式题名与作者可与arXiv交叉匹配；未以403推断条目无效。
- **本次阅读范围：**arXiv完整元数据/摘要；CVF原始页面搜索索引含书目信息/摘要，直接抓取403。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\METHOD_AND_EXPERIMENT_PLAN.md](./stage2b/METHOD_AND_EXPERIMENT_PLAN.md)；[.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md)；另有2份记录，见结构化目录。
- **补充原始来源：**[来源1](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_RealUnify_Do_Unified_Models_Truly_Benefit_from_Unification_A_Comprehensive_CVPR_2026_paper.html)。

<a id="cat-foundation"></a>

### 扩散、flow matching与视觉Transformer基础

先分清训练的向量场、数值求解时间步和网络执行深度。局部速度或latent误差下降，并不等于完整轨迹后的终图更好。

<a id="p-L004"></a>

#### L004 · Flow Matching · P0 · N

**[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)**  
Yaron Lipman等，2022。ICLR 2023；arXiv首版2022。

- **方法摘要：**以条件概率路径上的向量场回归训练连续归一化流，支持Gaussian与OT路径及ODE采样。
- **与本项目的关系：**理解flow速度目标、完整轨迹与采样成本的基础。
- **适用边界／版本注意：**局部向量场回归损失和终图质量须分别评价。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://openreview.net/pdf?id=PqvMRDCJT9t)。

<a id="p-L005"></a>

#### L005 · Rectified Flow · P1 · N

**[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)**  
Xingchao Liu等，2022。ICLR 2023；arXiv首版2022。

- **方法摘要：**学习连接两个分布的近直路径，使用rectification使ODE更便于粗步长模拟。
- **与本项目的关系：**理解噪声-数据插值、重桥接、减少ODE步数的背景。
- **适用边界／版本注意：**rectification训练不是在任意中途改latent后保证语义不变。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://openreview.net/pdf?id=XVjTT1nw5z)。

<a id="p-L006"></a>

#### L006 · DiT · P0 · N

**[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)**  
William Peebles等，2022。ICCV 2023；arXiv首版2022。

- **方法摘要：**用latent patch上的Transformer替代U-Net，并按前向计算复杂度研究扩展。
- **与本项目的关系：**共享视觉块、深度/宽度/计算预算实验的基本架构。
- **适用边界／版本注意：**生成头GFLOPs不等于统一模型全系统FLOPs。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)。

<a id="p-L007"></a>

#### L007 · SiT · P1 · N

**[SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers](https://arxiv.org/abs/2401.08740)**  
Nanye Ma等，2024。ECCV 2024。

- **方法摘要：**在同一DiT结构下系统分离插值、时间、目标与确定/随机采样。
- **与本项目的关系：**为控制模型结构后研究flow目标及solver提供范例。
- **适用边界／版本注意：**其ImageNet条件协议不能与本项目小T2I子集直接比数值。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L008"></a>

#### L008 · Rectified Flow Transformers · P1 · N

**[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)**  
Patrick Esser等，2024。ICML 2024。

- **方法摘要：**研究rectified-flow训练噪声采样和文本/图像双向信息交互的Transformer扩展。
- **与本项目的关系：**与高分辨率T2I flow建模、质量与数据尺度有关。
- **适用边界／版本注意：**不是Show-o2已采用全部相同实现，也不是内部循环论文。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://proceedings.mlr.press/v235/esser24a.html)。

<a id="p-L009"></a>

#### L009 · Latent Diffusion · P1 · N

**[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)**  
Robin Rombach等，2021。CVPR 2022；首版2021。

- **方法摘要：**在预训练autoencoder潜空间生成，以cross-attention支持文本等条件，降低像素空间费用。
- **与本项目的关系：**解释VAE重编码教师、latent干预与感知保真的取舍。
- **适用边界／版本注意：**VAE往返可改变图像，不能预设像素和latent几何操作等价。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L010"></a>

#### L010 · DDPM · P2 · N

**[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)**  
Jonathan Ho等，2020。arXiv原论文。

- **方法摘要：**逐步去噪的概率生成建模及噪声预测训练。
- **与本项目的关系：**扩散与flow背景、采样与训练目标的入门。
- **适用边界／版本注意：**无需将DDPM全部细节照搬到噪声到数据的flow时间约定。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L011"></a>

#### L011 · DDIM · P2 · N

**[Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)**  
Jiaming Song等，2020。ICLR 2021；首版2020。

- **方法摘要：**保持相关训练目标，构造可更快采样的非Markov隐式过程，研究速度质量权衡。
- **与本项目的关系：**普通减少采样步骤是循环模型必须面对的预算对照。
- **适用边界／版本注意：**更少ODE/NFE不等于更少层调用，二者分别计费。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L012"></a>

#### L012 · CFG · P1 · N

**[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)**  
Jonathan Ho等，2022。arXiv原论文。

- **方法摘要：**组合条件/无条件预测，在样本保真与覆盖/多样性间调节。
- **与本项目的关系：**所有候选须保持CFG约定，物理delta的两分支处理要一致。
- **适用边界／版本注意：**调高guidance可能改变偏好与多样性，不能混作循环独立收益。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="cat-unified"></a>

### 统一多模态理解与生成：本项目的模型底座

这些工作回答图像表示、文本建模和视觉生成如何共存。Show-o2是本项目实际底座；其他模型主要用于界定设计空间和后续迁移候选。

<a id="p-L013"></a>

#### L013 · Show-o · P0 · H

**[Show-o: One Single Transformer to Unify Multimodal Understanding and Generation](https://arxiv.org/abs/2408.12528)**  
Jinheng Xie等，2024。ICLR 2025；首版2024。

- **方法摘要：**统一自回归文字建模与离散扩散视觉生成，支持理解、生成和混合模态任务。
- **与本项目的关系：**理解本项目统一模型背景及离散生成与flow路线的差别。
- **适用边界／版本注意：**统一框架不自动带来任务协同；Show-o与Show-o2不是同一视觉生成机制。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2_loop\NEXT_EXPERIMENT.md](./stage2_loop/NEXT_EXPERIMENT.md)；另有2份记录，见结构化目录。

<a id="p-L014"></a>

#### L014 · Show-o2 · P0 · H

**[Show-o2: Improved Native Unified Multimodal Models](https://arxiv.org/abs/2506.15564)**  
Jinheng Xie等，2025。NeurIPS 2025；arXiv v3。

- **方法摘要：**在3D因果VAE空间构建双路径视觉融合，语言头用AR、视觉头用flow matching，面向图像及视频理解生成。
- **与本项目的关系：**实际实验底座；必须先读其forward、视觉表示与训练配方，再讨论外Qwen/内flow循环。
- **适用边界／版本注意：**版本名1.5B不等于整个系统参数；其原始结果不能替代本项目432px子集结果。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；另有5份记录，见结构化目录。
- **补充原始来源：**[来源1](https://github.com/showlab/Show-o/tree/main/show-o2)。

<a id="p-L015"></a>

#### L015 · Transfusion · P0 · H

**[Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039)**  
Chunting Zhou等，2024。arXiv预印本。

- **方法摘要：**在同一个Transformer和混合模态序列中结合文本next-token损失与连续图像扩散损失。
- **与本项目的关系：**检验共享循环是否适用于AR与连续视觉生成的另一种统一架构。
- **适用边界／版本注意：**本项目没有完成该架构迁移；模态接口与Show-o2分头实现不同。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；另有4份记录，见结构化目录。

<a id="p-L016"></a>

#### L016 · BAGEL · P1 · H

**[Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683)**  
Chaorui Deng等，2025。arXiv v3。

- **方法摘要：**在大规模交错图文视频数据上预训练统一decoder-only模型，观察理解、生成和复杂操作能力。
- **与本项目的关系：**统一模型、数据规模及复杂视觉操作的强背景，也可作为后续迁移对象。
- **适用边界／版本注意：**官方完整题名不是BAGEL；大规模预训练能力不等于小数据后训练可直接获得。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L017"></a>

#### L017 · BLIP3-o · P1 · H

**[BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset](https://arxiv.org/abs/2505.09568)**  
Jiuhai Chen等，2025。arXiv预印本。

- **方法摘要：**用diffusion transformer生成CLIP语义特征，并研究先理解后生成的顺序训练和开放数据配方。
- **与本项目的关系：**比较VAE潜空间与语义特征空间，参考理解保真与生成适配的训练顺序。
- **适用边界／版本注意：**换表示、换数据和循环深度是不同因素，不宜同时变化后归因循环。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L018"></a>

#### L018 · MetaQueries · P1 · H

**[Transfer between Modalities with MetaQueries](https://arxiv.org/abs/2504.06256)**  
Xichen Pan等，2025。arXiv预印本。

- **方法摘要：**以可学习query连接多模态LLM隐藏状态与diffusion decoder，冻结MLLM也可转移理解知识。
- **与本项目的关系：**为轻量证据接口和冻结骨干生成适配提供对照。
- **适用边界／版本注意：**属于跨模块接口，不是同一Transformer权重的反复重访。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L019"></a>

#### L019 · Chameleon · P2 · N

**[Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818v1)**  
Chameleon Team等，2024。arXiv v1元数据/摘要；另有v2。

- **方法摘要：**将图文token早期融合，支持任意次序的混合模态理解和生成。
- **与本项目的关系：**理解全离散AR统一方案，与AR加flow混合路线形成背景对照。
- **适用边界／版本注意：**不应把图像token的AR生成等同于连续潜变量flow循环。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://arxiv.org/abs/2405.09818)。

<a id="p-L020"></a>

#### L020 · Janus · P1 · N

**[Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848)**  
Chengyue Wu等，2024。技术报告。

- **方法摘要：**将理解和生成视觉编码解耦，同时保留统一Transformer。
- **与本项目的关系：**解释理解所需语义与生成细节可能存在不同接口需求。
- **适用边界／版本注意：**编码解耦与循环机制不同，不能用编码器选择掩盖循环增益。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L021"></a>

#### L021 · JanusFlow · P1 · N

**[JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2411.07975)**  
Yiyang Ma等，2024。CVPR 2025；首版2024。

- **方法摘要：**结合AR与rectified flow，解耦视觉编码并对齐理解/生成表示。
- **与本项目的关系：**与Show-o2最接近的混合建模补充，支持跨骨干及共同表示对照。
- **适用边界／版本注意：**尚非本项目已运行模型；其对齐与反馈循环不可混称。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L022"></a>

#### L022 · Janus-Pro · P2 · N

**[Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://arxiv.org/abs/2501.17811)**  
Xiaokang Chen等，2025。研究报告。

- **方法摘要：**通过训练策略、数据与规模扩展提高Janus的理解及生成稳定性。
- **与本项目的关系：**排除方法改善只是增加数据/模型规模的解释。
- **适用边界／版本注意：**它不是共享循环或flow预算分配论文。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L023"></a>

#### L023 · Emu3 · P2 · N

**[Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869)**  
Xinlong Wang等，2024。arXiv预印本。

- **方法摘要：**统一离散化图文视频并仅用next-token prediction训练。
- **与本项目的关系：**界定混合AR/flow方案与纯AR方案的区别。
- **适用边界／版本注意：**不同生成范式的步数、token数和FLOPs不能直接互换。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L024"></a>

#### L024 · SEED-X · P2 · N

**[SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation](https://arxiv.org/abs/2404.14396)**  
Yuying Ge等，2024。arXiv v2更新2025。

- **方法摘要：**支持任意尺寸比例视觉理解与多粒度图像生成，重视指令适配。
- **与本项目的关系：**了解统一任务与更丰富条件接口，辅助设计理解/生成任务集合。
- **适用边界／版本注意：**不是内部循环深度贡献的直接证据。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L025"></a>

#### L025 · UniAR · P2 · N

**[Unified Multimodal Autoregressive Modeling with Shared Context-Visual Tokenizer is Key to Unification](https://arxiv.org/abs/2606.18249)**  
Wujian Peng等，2026。ICML 2026（arXiv作者注记）。

- **方法摘要：**共享离散视觉tokenizer与上下文，让模型直接理解自己生成的视觉token，并用并行bit预测降低序列长度。
- **与本项目的关系：**直接潜空间自读的另一种设计，可对照解码再编码的反馈开销。
- **适用边界／版本注意：**离散共享token方案不等同于对当前VAE latent直接运行MMU。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="cat-loops"></a>

### 共享循环、递归深度、自适应计算与压缩

ELT、Looped-DiT是共享深度的直接近邻。早退出、稀疏路由、feature caching、tokenizer循环和文本循环具有不同的成本与任务边界，需要分别定位。

<a id="p-L026"></a>

#### L026 · ELT · P0 · H

**[ELT: Elastic Looped Transformers for Visual Generation](https://arxiv.org/abs/2604.09168)**  
Sahil Goyal等，2026。ECCV 2026；arXiv首版2026-04-10，所查v3为2026-07-23；Springer正式章节2026-09-16发表。

- **方法摘要：**在每次采样内部重复同一组Transformer，分离唯一参数量与执行深度。ILSD将最大循环作为在线教师，随机抽取其严格前缀出口；浅出口同时接受真实目标和stop-gradient深出口监督，并逐渐由真实监督切向自蒸馏，得到同一模型的弹性多出口。
- **与本项目的关系：**项目最直接的方法来源：真实共享重访、1/2/4循环预算、循环内蒸馏。项目v46等只是Show-o2后训练适配，未复现原文完整训练体系。
- **适用边界／版本注意：**不能把参数共享自动等同于FLOPs下降，也不能把原文类别条件图像/视频指标与本项目432px文本生图小样本相比较；本记录不复制旧报告性能数字。
- **本次阅读范围：**选定章节：arXiv题录/摘要；HTML第4节循环与ILSD公式、第5.4节及算法摘录；Springer章节元数据和Google作者机构会议目录。未逐页精读全篇。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；另有9份记录，见结构化目录。
- **补充原始来源：**[来源1](https://arxiv.org/html/2604.09168v3)、[来源2](https://doi.org/10.1007/978-3-032-37324-3_17)、[来源3](https://research.google/conferences-and-events/google-at-eccv-2026/)。

<a id="p-L027"></a>

#### L027 · Looped-DiT · P0 · H

**[Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305)**  
Yong Xien Chng等，2026。arXiv预印本，2026-09-30 v1；所查原始页面未列正式会议接收。

- **方法摘要：**在单个去噪步内复用Transformer以增加有效深度。针对中间循环监督弱、重复attention更新侵蚀局部信息，结合多循环出口深监督与self-modulating attention；实现包括attention-head门控及去除自身value方向分量的投影式更新。
- **与本项目的关系：**直接约束项目的创新边界：普通图像生成循环及多出口监督已有工作。也为项目中的表征漂移、循环不单调改善和反馈幅度之外的更新稳定性提供相邻设计。
- **适用边界／版本注意：**文中的latent reasoning是行为观察，不是本项目已证实的理解推理能力；其匹配计算结论不能继承到保留原全部层的Show-o2修复原型。
- **本次阅读范围：**选定章节：题录/摘要，HTML第2节方法、深监督及第2.2节self-modulating attention摘录。未逐表核验性能数字。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；另有3份记录，见结构化目录。
- **补充原始来源：**[来源1](https://arxiv.org/html/2609.40305v1)。

<a id="p-L028"></a>

#### L028 · Depth through recurrence / Looped flow-TTS · P1 · H

**[Depth through recurrence: Looped transformers for flow-matching TTS](https://arxiv.org/abs/2609.29768)**  
Jiabao Ai等，2026。arXiv预印本；首版2026-09-24，所查最新v2为2026-10-21；原始题录未确认会议接收。

- **方法摘要：**在flow-matching语音合成中以相同18块执行深度比较七种参数复用布局：整组CYCLE、相邻SEQUENCE及Prefix/Middle/Suffix等。分别考察可懂度、说话人相似度、预测音质与采样预算，显示复用顺序和位置影响不同质量维度，且与求解步数交互。
- **与本项目的关系：**历史上直接启发项目实现CYCLE、SEQUENCE、MIDDLE和位置对照；提醒相同参数量与块调用并不保证同质量。
- **适用边界／版本注意：**证据为英语TTS与客观语音指标，不能外推为图像收益或人类听评。当前v2晚于10月3日项目试验，不能把v2全部新分析写成当时已经参考过。
- **本次阅读范围：**选定章节：题录/摘要，v2布局定义、实验表及第4节主要分析和第5节边界；没有将全文各表数字逐项复算。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\METHOD_AND_EXPERIMENT_PLAN.md](./stage2b/METHOD_AND_EXPERIMENT_PLAN.md)；另有3份记录，见结构化目录。
- **补充原始来源：**[来源1](https://arxiv.org/html/2609.29768v2)、[来源2](https://jiabaoai67.github.io/loop-f5/)。

<a id="p-L029"></a>

#### L029 · LoopVAE · P1 · H

**[LoopVAE: Recurrent Depth Across Scales for Visual Tokenization](https://arxiv.org/abs/2609.11516)**  
Zhiying Lu等，2026。arXiv预印本，2026-09-10 v1；所查原始页未列正式会议接收。

- **方法摘要：**视觉tokenizer在尺度内及跨尺度复用由尺度/循环索引调节的核心，分辨率变换仍独立。研究卷积和Transformer操作、不同latent分辨率接口，并通过删减循环检验实际递归作用；较少存储权重在所测实现中伴随更多算术和更长运行时间。
- **与本项目的关系：**帮助区分“参数共享”“完成训练过的循环有用”和“中间出口可靠”。项目保持Wan VAE，未把它实现为新的tokenizer实验。
- **适用边界／版本注意：**rFID属于重建指标，不是新图生成FID；tokenizer重建收益不证明Show-o2生成质量收益。不能从参数减少推断加速。
- **本次阅读范围：**选定章节：题录/摘要，HTML第3.1节共享处理/独立尺度变换及多接口方法摘录；未逐项复核实验表。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\METHOD_AND_EXPERIMENT_PLAN.md](./stage2b/METHOD_AND_EXPERIMENT_PLAN.md)；另有3份记录，见结构化目录。
- **补充原始来源：**[来源1](https://arxiv.org/html/2609.11516v1)。

<a id="p-L030"></a>

#### L030 · RIN · P1 · H

**[Scalable Adaptive Computation for Iterative Generation](https://proceedings.mlr.press/v202/jabri23a.html)**  
Allan Jabri等，2023。ICML 2023，PMLR202；arXiv首版2022-12-22，v2为2023-06-14。

- **方法摘要：**Recurrent Interface Network用cross-attention在高维数据token与紧凑latent token间读写，把主要全局self-attention放在latent中。latent self-conditioning将上一去噪步hidden作为当前步上下文，训练用stop-gradient自举估计模拟该上下文，分摊重复路由成本。
- **与本项目的关系：**是项目跨步状态/紧凑记忆的早期参考，说明利用已有hidden和跨步自条件本身不是新颖点。
- **适用边界／版本注意：**跨采样步的latent递归不同于同一flow时间内用少量共享块替换独立深度；不应把RIN名称直接解读成ELT式全层权重绑定。
- **本次阅读范围：**选定章节：PMLR会议元数据和摘要；arXiv第2.2节read-compute-write、第2.3节latent self-conditioning及训练/推理算法摘录。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。
- **补充原始来源：**[来源1](https://arxiv.org/abs/2212.11972)、[来源2](https://arxiv.org/html/2212.11972v2)。

<a id="p-L031"></a>

#### L031 · AdaDiff · P1 · H

**[AdaDiff: Accelerating Diffusion Models through Step-Wise Adaptive Computation](https://arxiv.org/abs/2309.17074)**  
Shengkun Tang等，2024。ECCV 2024；arXiv首版2023-09-29，所查v3为2024-08-16。

- **方法摘要：**每个采样步按预测不确定性选择网络早退出深度。timestep-aware uncertainty estimation module结合中间层状态与时间评估不确定性；uncertainty-aware layer-wise loss与原任务共同训练，恢复浅层出口相对完整模型的预测能力。
- **与本项目的关系：**项目预算分配与提前退出必须对照的先例：仅预测当前不确定性或按时间选择深度，不能当作新的理解条件循环。
- **适用边界／版本注意：**早退出不等同参数共享，也不等同额外循环修复。当前不确定性与“同状态再计算一次的最终因果收益”是不同目标。
- **本次阅读范围：**选定章节：arXiv元数据/摘要、方法第3.2–3.4节和参数共享消融摘录；ECVA官方PDF搜索结果确认题名/作者/会议，直接打开超时，未声称读完该PDF。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。
- **补充原始来源：**[来源1](https://arxiv.org/html/2309.17074v3)、[来源2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10120.pdf)、[来源3](https://github.com/Tangshengku/AdaDiff)。

<a id="p-L032"></a>

#### L032 · LaTtE-Flow · P0 · H

**[LaTtE-Flow: Layerwise Timestep-Expert Flow-based Transformer](https://proceedings.mlr.press/v306/shen26w.html)**  
Ying Shen（与Zhiyang Xu共同一作）等，2026。ICML 2026，PMLR306；arXiv首版2025-06-08，v2为2026-06-17。

- **方法摘要：**在预训练VLM上扩展flow生成，将Transformer层划为负责不同时间区间的层组，每个采样时间仅激活对应组；timestep-conditioned residual attention通过逐head时间门控复用前层图像attention。它把统一模型内按时间分配层计算作为架构设计。
- **与本项目的关系：**是项目统一理解/生成及时间计算调度的直接近邻，需区分层组时间专门化与证据驱动的同状态共享重访。
- **适用边界／版本注意：**不是对原Show-o2直接删层的证明；其专门训练的时间专家不能靠后处理截层替代。本次仅核验文献方法，没有复现其模型或效率。
- **本次阅读范围：**选定章节：PMLR和作者项目页元数据/共同一作；arXiv摘要、第4.2节时间专家及第4.3节残差attention公式/边界。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；另有1份记录，见结构化目录。
- **补充原始来源：**[来源1](https://arxiv.org/abs/2506.06952)、[来源2](https://arxiv.org/html/2506.06952v2)、[来源3](https://plan-lab.github.io/projects/latte-flow/)。

<a id="p-L033"></a>

#### L033 · DyDiT++ · P1 · H

**[DyDiT++: Diffusion Transformers with Timestep and Spatial Dynamics for Efficient Visual Generation](https://arxiv.org/abs/2504.06803)**  
Wangbo Zhao等，2026。IEEE TPAMI，arXiv作者注记2026-01-09接收；首版2025-04-09，所查v4为2026-01-14。

- **方法摘要：**继承按时间动态attention/MLP宽度和按空间动态token计算，将动态DiT扩展到flow matching、视频和文本生图。TD-LoRA按时间混合低秩专家来减少适配训练成本，避免把所有时间共享同一静态低秩更新。
- **与本项目的关系：**项目应面对的时间/空间计算分配与PEFT效率对照；能帮助区分更少可训练参数、更少存储参数和更少推理运算。
- **适用边界／版本注意：**动态激活保留完整网络容量，不是永久移除权重，也不是循环共享深度。不能把router少算与实际总参数压缩混用。
- **本次阅读范围：**选定章节：arXiv元数据/摘要，HTML第3节TDW/SDT和第7节TD-LoRA方法摘录、作者主页与官方repo版本信息。未通读长篇全部实验附录。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\RESEARCH_DIRECTION.md](./stage2b/RESEARCH_DIRECTION.md)。
- **补充原始来源：**[来源1](https://arxiv.org/html/2504.06803v4)、[来源2](https://github.com/alibaba-damo-academy/DyDiT)、[来源3](https://wangbo-zhao.github.io/)。

<a id="p-L034"></a>

#### L034 · PPCL · P1 · H

**[Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](https://arxiv.org/abs/2511.16156)**  
Jian Ma等，2026。CVPR 2026，pp.18640–18650；arXiv首版2025-11-20，v2为2026-02-24。

- **方法摘要：**先用逐层线性probe及CKA相似度一阶变化定位连续冗余区间，再做深度剪枝和宽度剪枝，以可插拔师生交替蒸馏支持不同裁剪率，最后短程微调。重点是先测可替代区间，再恢复压缩路径，而非假定相邻预训练层可互换。
- **与本项目的关系：**历史直接支持项目从任意绑层转向逐位置/区间诊断和局部教师恢复；它是压缩训练近邻，不是项目已复现的算法。
- **适用边界／版本注意：**结构剪枝、蒸馏与共享递归是不同手段；其性能/压缩比不能移植为本项目结果。
- **本次阅读范围：**选定章节：arXiv题录/摘要及第3节方法摘录；CVF官方搜索结果确认正式作者/会议/页码，直接页面打开403，因此不声称读取该页面正文或官方PDF。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md](./stage2b/MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md)。
- **补充原始来源：**[来源1](https://arxiv.org/html/2511.16156v2)、[来源2](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Pluggable_Pruning_with_Contiguous_Layer_Distillation_for_Diffusion_Transformers_CVPR_2026_paper.html)。

<a id="p-L035"></a>

#### L035 · E-DiT · P1 · H

**[Elastic Diffusion Transformer](https://proceedings.mlr.press/v306/wang26dd.html)**  
Jiangshan Wang等，2026。ICML 2026，PMLR306:127492–127508。

- **方法摘要：**为DiT块加轻量router，从输入latent估计样本相关稀疏性：先选择是否跳块，保留时再选MLP宽度比例；推理再结合router决定块级feature caching。它把深度、宽度及缓存组合为弹性加速框架。
- **与本项目的关系：**项目早期已引用的动态预算基线；说明只做样本相关层跳过/宽度/缓存不足以构成新的共享循环贡献。
- **适用边界／版本注意：**E-DiT与ELT是两篇不同论文；前者自适应稀疏与缓存，不是同权重循环自蒸馏。由摘要不能推定统一理解能力或证据修复收益。
- **本次阅读范围：**abstract_and_metadata：已打开PMLR正式论文页，核对作者、会议、页码、摘要和BibTeX；未读取正式PDF方法/实验全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2_loop\RESEARCH_DECISIONS.md](./stage2_loop/RESEARCH_DECISIONS.md)。

<a id="p-L036"></a>

#### L036 · DyDiT（原版） · P1 · N

**[Dynamic Diffusion Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a44a70acd5d0abc1a252ada9719dd06d-Abstract-Conference.html)**  
Wangbo Zhao等，2025。ICLR 2025；arXiv首版2024-10-04，v2为2024-10-09。

- **方法摘要：**按去噪时间动态选择attention头/MLP通道组宽度，按空间token选择是否执行部分计算，减少静态DiT在容易时间和区域上的冗余。
- **与本项目的关系：**本次补充的DyDiT++基础版本，避免把期刊扩展与会议原版混成一篇或重复计算同一方法贡献。
- **适用边界／版本注意：**不是权重共享循环；原版与++的任务范围、训练方式和发表年份不同，应分别注明。
- **本次阅读范围：**abstract_and_metadata：arXiv题录/摘要、ICLR正式会议页及作者官方repo；未读完整PDF。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://arxiv.org/abs/2410.03456)、[来源2](https://github.com/alibaba-damo-academy/DyDiT)。

<a id="p-L037"></a>

#### L037 · Universal Transformer · P2 · N

**[Universal Transformers](https://arxiv.org/abs/1807.03819)**  
Mostafa Dehghani（与Stephan Gouws共同一作）等，2019。ICLR 2019；arXiv首版2018-07-10，v3为2019-03-05。

- **方法摘要：**把self-attention的并行token处理与深度方向递归结合，同一变换反复更新各位置表征；再加入逐位置动态停止，使计算深度随输入变化。它把共享权重、多步内部计算和自适应退出放在同一序列模型中。
- **与本项目的关系：**本次补充的循环Transformer基础，说明“重复同一Transformer”和“动态停止”并非视觉生成论文才首次提出。
- **适用边界／版本注意：**主要为算法/语言序列任务，不能由其外推生成质量、MMU互促或Show-o2后训练可压缩。
- **本次阅读范围：**选定章节：arXiv元数据/摘要、HTML方法及动态halting章节入口；ICLR官方PDF搜索可见封面/摘要核对发表。非全文精读。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://arxiv.org/html/1807.03819v3)、[来源2](https://openreview.net/pdf/6ee41939003eaa38439a2607d081864b4ba5fea4.pdf)。

<a id="p-L038"></a>

#### L038 · ALBERT · P2 · N

**[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)**  
Zhenzhong Lan等，2020。ICLR 2020；arXiv首版2019-09-26，所查v6为2020-02-09。

- **方法摘要：**以词嵌入因子分解和跨层参数共享降低BERT存储参数；默认共享跨层全部参数，并比较仅共享FFN/attention等策略；另用sentence-order prediction学习句间一致性。核心是分离存储参数与网络展开深度。
- **与本项目的关系：**补充权重绑定基础，可帮助读者理解项目为何分别计唯一参数、训练参数和实际块调用；共享降低存储不自动减少每次前向运算。
- **适用边界／版本注意：**不是弹性视觉生成器，也没有证明已有预训练不同层能无需充分训练直接绑层。
- **本次阅读范围：**选定章节：arXiv元数据/摘要，HTML第3.1节cross-layer sharing及第4.5节部分共享消融；ICLR官方PDF题录可见。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://arxiv.org/html/1909.11942v6)、[来源2](https://openreview.net/pdf?id=H1eA7AEtvS)。

<a id="p-L039"></a>

#### L039 · DEQ · P2 · N

**[Deep Equilibrium Models](https://papers.nips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html)**  
Shaojie Bai等，2019。NeurIPS 2019，Advances in Neural Information Processing Systems32；arXiv首版2019-09-03。

- **方法摘要：**把无限深权重绑定网络的极限表示为平衡点，前向用求根方法直接找状态，反向用隐式微分穿过平衡点，避免存储全部展开层激活。
- **与本项目的关系：**补充递归深度与数值稳定性的基础，便于说明项目显式1/2/4循环、ILSD和参考重算并不属于DEQ隐式求解。
- **适用边界／版本注意：**固定点求解需要额外数值迭代及稳定性条件，常数激活内存不等同推理加速；本项目未实现DEQ。
- **本次阅读范围：**abstract_and_intro：NeurIPS正式元数据/摘要、arXiv题录、正式PDF引言和权重绑定动机摘录；未核验求解器全推导。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://arxiv.org/abs/1909.01377)、[来源2](https://papers.nips.cc/paper/2019/file/01386bd6d8e091c2ab4c7c7de644d37b-Paper.pdf)。

<a id="p-L040"></a>

#### L040 · ACT · P2 · N

**[Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)**  
Alex Graves等，2016。arXiv预印本；首版2016-03-29，所查v6为2017-02-21；所查原始页未确认正式会议/期刊。

- **方法摘要：**在收到输入与输出之间，用共享RNN状态变换执行可变数量内部更新。可微halting节点决定累积停止质量，以加权状态/输出和ponder成本联合学习准确率与计算量折中，最大步数作为成本保护。
- **与本项目的关系：**本次补充动态循环预算的基本来源；项目若训练controller，需区分这种端到端停止学习与当前离线oracle或固定阈值路由。
- **适用边界／版本注意：**计算惩罚权重敏感；原任务是合成问题和字符语言建模，不提供图像修复证据。不能未经新监督直接移用停止规则。
- **本次阅读范围：**选定章节：元数据/摘要，HTML方法第2节halting、mean-field更新与ponder penalty及其敏感性摘录。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://arxiv.org/html/1603.08983v6)。

<a id="p-L041"></a>

#### L041 · PonderNet · P2 · N

**[PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407)**  
Andrea Banino（与Jan Balaguer共同一作）等，2021。8th ICML Workshop on Automated Machine Learning，2021；arXiv首版2021-07-12，v2为2021-09-02；不是ICML主会论文。

- **方法摘要：**把每步停止概率写成在此前未停条件下的hazard，形成停止步数分布；训练优化各出口预测损失的期望及相对几何先验的KL，推理按Bernoulli停止采样或受最大预算限制。
- **与本项目的关系：**为未来循环预算学习提供比“有错就加深”更明确的概率模型；当前项目controller尚未训练，不能把该参考写成已经采用。
- **适用边界／版本注意：**主要验证推理/问答等任务；概率停止框架本身不保证证据特异修复，也不解决多步生成中误差与终图质量错配。
- **本次阅读范围：**选定章节：题录/摘要和workshop注记；HTML第2.2–2.5节停止分布、期望损失/KL与推理；OpenReview官方旧版附件搜索可见，论坛直开为浏览器验证页。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://arxiv.org/html/2107.05407v2)、[来源2](https://openreview.net/attachment?id=1EuxRTe0WN&name=original_version)。

<a id="p-L042"></a>

#### L042 · Looped Diffusion Language Models / LoopMDM · P2 · H?

**[Looped Diffusion Language Models](https://arxiv.org/abs/2605.26106)**  
Sanghyun Lee等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**选择性重复masked diffusion Transformer早中层，在不增加参数时增加有效深度；推理改变loop数，并研究随采样过程调整loop预算。
- **与本项目的关系：**旧记录只有题名未锁定，本次已找到唯一精确匹配；这是文本masked diffusion，不能直接把其训练效率结论移植到Show-o2视觉flow。
- **适用边界／版本注意：**历史仅出现题名、未锁定版本；本次才找到精确匹配2605.26106，不能声称早期已经明确阅读/实现了这篇。文本masked diffusion结论不能直接移植视觉flow。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史只出现题名；本次才锁定此编号，不追认当时读过该版本。

<a id="p-L043"></a>

#### L043 · ALoDLM · P2 · N

**[ALoDLM: Adaptively Looped Diffusion Language Models](https://arxiv.org/abs/2610.04198)**  
Liancheng Fang等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**逐token自适应latent recurrence；已可提交token变离散上下文，未解token继续保留latent迭代；token计算日程为潜变量，用条件NELBO学习。
- **与本项目的关系：**补充计算难度匹配和token级预算分配的最新文本参考，区别于本项目固定loop预算；未复现。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L044"></a>

#### L044 · Looped Uniform DLM · P2 · N

**[Looped Uniform Diffusion Language Models](https://arxiv.org/abs/2610.30066)**  
Saiteja Utpala等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**在uniform diffusion LM中保留原训练目标和sampler，只引入早层选择性loop，研究训练loop分布与匹配NFE推理。
- **与本项目的关系：**2026-10-21新作，仅本次补充；强调iso-FLOP/同NFE及目标不变对照，不能冒称早期实验已据此设计。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="cat-feedback"></a>

### 内部反馈、反思修复、交错推理与双向促进

这一组与“理解帮助生成、自我批评、生成反向帮助理解”最接近。多数方法已使用反馈或latent推理，因此本项目需要用共享核心的独立收益、证据特异性和受控预算说明区别。

<a id="p-L045"></a>

#### L045 · Dreaming in Flow / GGF · P0 · H

**[Dreaming in Flow: Generative Grounding Feedback for Self-Evolving Unified Multimodal Models](https://arxiv.org/abs/2609.08282)**  
Ke Hao等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**用自生成图像作为共享经验；同一带噪latent分别做文本、图像、图文及图文加修复条件速度预测，将图像接地/修复方向蒸馏给文本条件；caption→re-imagination回放的一致性和无关样本分离反向训练理解。
- **与本项目的关系：**强邻近：双向反馈和flow内部方向监督已经存在。本项目须区分共享深度循环是否有额外质量/成本收益，不能把理解促进生成或双向性本身当新颖点。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv元数据、摘要、Introduction与Method（flow反馈、dream replay）；未重算其结果。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\METHOD_AND_EXPERIMENT_PLAN.md](./stage2b/METHOD_AND_EXPERIMENT_PLAN.md)；另有6份记录，见结构化目录。
- **补充原始来源：**[来源1](https://arxiv.org/html/2609.08282v1)。

<a id="p-L046"></a>

#### L046 · LatentUM · P0 · H

**[LatentUM: Unleashing the Potential of Interleaved Cross-Modal Reasoning via a Latent-Space Unified Model](https://arxiv.org/abs/2604.02097)**  
Jiachun Jin等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**在共享语义latent空间中统一模态，使理解与生成交错推理不再必须经像素解码桥接。
- **与本项目的关系：**对应本项目理解/生成表示接口不匹配及VAE桥接代价；其原生共同latent设计不同于在既有Show-o2上加共享重访。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L047"></a>

#### L047 · PreviewDiff · P0 · H

**[PreviewDiff: Multimodal Critic-Guided Search over Diffusion Latents](https://arxiv.org/abs/2609.36199)**  
Vighnesh Subramaniam等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**在选定去噪点解码preview，multimodal critic评分/批评，分支尝试语义prompt编辑和局部重新加噪续跑，再择分支推进。
- **与本项目的关系：**中途检查、回溯/重噪及预算匹配Best-of-N的重要对照；这是training-free搜索，不能直接等同学得共享Transformer计算。
- **适用边界／版本注意：**arXiv HTML全文端点本次404；摘要页正常，未宣称通读PDF。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；另有1份记录，见结构化目录。

<a id="p-L048"></a>

#### L048 · R3-Bench / R3-Refiner · P1 · H

**[Benchmarking and Evolving Reason-Reflect-Rectify for Reflective Visual Generation](https://arxiv.org/abs/2605.19639)**  
Junjie Wang等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**提出Reason–Reflect–Rectify多轮视觉生成框架与R3-Bench；R3-Refiner用GRPO及层次奖励训练可执行的反思/修正。
- **与本项目的关系：**直接关联“能识错却不能给出有效修复动作”；可用于把判断质量与动作质量分开衡量。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L049"></a>

#### L049 · UniGen / CoT-V · P1 · H

**[UniGen: Enhanced Training & Test-Time Strategies for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2505.14682)**  
Rui Tian等，2025。arXiv预印本；具体会议注释见limit。

- **方法摘要：**研究统一模型的多阶段训练与DPO；CoT-V让同模型作为生成器和逐步语义核验者，以Best-of-N推理提高生成。
- **与本项目的关系：**需要对照自验证最终样本选择与中途可执行修复；Best-of-N收益不证明单轨迹或共享循环收益。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L050"></a>

#### L050 · FoX / MCoT · P1 · H

**[Towards Enhanced Image Generation Via Multi-modal Chain of Thought in Unified Generative Models](https://arxiv.org/abs/2503.01298)**  
Yi Wang等，2025。arXiv预印本；具体会议注释见limit。

- **方法摘要：**FoXperts按功能组织专家；MCoT把规划、执行、反思、修正串成图文工作流，并用多任务训练分解各步能力。
- **与本项目的关系：**明确先例：规划—生成—反思—修正及多任务能力训练；专家结构不同于重复同一共享块。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L051"></a>

#### L051 · MILR · P1 · H

**[MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning](https://arxiv.org/abs/2509.22761)**  
Yapeng Mi等，2025。arXiv预印本；具体会议注释见limit。

- **方法摘要：**在统一向量空间中联合搜索图像/文本离散token的向量表示，用质量critic引导policy gradient，完全在test time执行。
- **与本项目的关系：**latent联合推理与critic引导的对照；本项目需单列其test-time优化费用，不能将latent优化和模型参数训练混算。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L052"></a>

#### L052 · SRUM · P1 · H

**[SRUM: Fine-Grained Self-Rewarding for Unified Multimodal Models](https://arxiv.org/abs/2510.12784)**  
Weiyang Jin等，2025。arXiv预印本；具体会议注释见limit。

- **方法摘要：**统一模型理解模块作为内部evaluator，global/local双奖励训练生成，分别约束总体语义/布局及对象细节。
- **与本项目的关系：**理解奖励生成已经是明确先例；本项目HPS/Pick外部守门与内部reward需分别披露，并查局部误修复和整体保真。
- **适用边界／版本注意：**arXiv注释称Accepted to ECCV2026；本次仅核验该声明，未另查会议页。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md](./stage2b/MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md)。

<a id="p-L053"></a>

#### L053 · Ask, Solve, Generate · P1 · H

**[Ask, Solve, Generate: Self-Evolving Unified Multimodal Understanding and Generation via Self-Consistency Rewards](https://arxiv.org/abs/2606.27376)**  
Ritesh Thawkar等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**未标注图像出发，Proposer提问、Solver回答评价、Generator生成；用Solver Token Entropy及问答/循环caption一致性形成内部奖励。
- **与本项目的关系：**双向自改进强邻近；需区分使用外部未标注真实图像和仅文本自生成经验，也需评测真实理解任务。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md](./stage2b/MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md)；另有1份记录，见结构化目录。

<a id="p-L054"></a>

#### L054 · UniCorn · P1 · H

**[UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision](https://arxiv.org/abs/2601.03193)**  
Ruiyan Han等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**Proposer/Solver/Judge三角色self-play及认知模式重建，将潜在理解转为生成监督；以Text→Image→Text循环基准UniCycle检验一致性。
- **与本项目的关系：**不能将自生成数据、自评判或循环一致性训练本身当作本项目独有；需额外证明共享深度机制。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md](./stage2b/MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md)；另有1份记录，见结构化目录。

<a id="p-L055"></a>

#### L055 · SRRL · P1 · H

**[Self-Reflective Reinforcement Learning for Diffusion-based Image Reasoning Generation](https://arxiv.org/abs/2505.22407)**  
Jiadong Pan等，2025。arXiv预印本；具体会议注释见limit。

- **方法摘要：**把完整去噪轨迹作为一段CoT，跨多轮轨迹反思，借条件化forward过程连接反思轮并做RL训练。
- **与本项目的关系：**有助于避免对带噪中途图不可靠奖励的混淆；完整轨迹多轮不同于同一步Transformer共享深度。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md](./stage2b/MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md)。

<a id="p-L056"></a>

#### L056 · UniEdit-I · P1 · H

**[UniEdit-I: Training-free Image Editing for Unified VLM via Iterative Understanding, Editing and Verifying](https://arxiv.org/abs/2508.03142)**  
Chengyu Bai等，2025。arXiv预印本；具体会议注释见limit。

- **方法摘要：**在统一VLM语义latent中执行Understanding–Editing–Verifying闭环，免训练地利用语义理解动态控制编辑。
- **与本项目的关系：**直接关联表示差距和闭环编辑；其编辑任务与从噪声T2I、strict same-state教师需区别。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md](./stage2b/MIDCOURSE_GROUNDED_LOOP_STATUS_20261004.md)。

<a id="p-L057"></a>

#### L057 · AD-Loop · P1 · H

**[Synergizing Understanding and Generation with Interleaved Analyzing-Drafting Thinking](https://arxiv.org/abs/2602.21435)**  
Shengqiong Wu等，2026。ICLR2026正式页面已核验。

- **方法摘要：**交替文本分析与视觉草拟；先监督训练交错thought数据，再RL学习自适应控制，服务理解和生成。
- **与本项目的关系：**外部/多轮分析草拟循环已有先例，本项目只能把同次速度内部的共享计算和边际成本作为待证区别。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv元数据/摘要/历史与ICLR2026正式题名作者页面。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md)；[.\stage2_loop\NEXT_EXPERIMENT.md](./stage2_loop/NEXT_EXPERIMENT.md)；另有1份记录，见结构化目录。
- **补充原始来源：**[来源1](https://proceedings.iclr.cc/paper_files/paper/2026/hash/741ff9e19bc28050b4d815726ec792c9-Abstract-Conference.html)、[来源2](https://www.microsoft.com/en-us/research/publication/synergizing-understanding-and-generation-with-interleaved-analyzing-drafting-thinking/)。

<a id="p-L058"></a>

#### L058 · Turning Internal Gap · P1 · H

**[Turning Internal Gap into Self-Improvement: Promoting the Generation-Understanding Unification in MLLMs](https://arxiv.org/abs/2507.16663)**  
Yujin Han等，2025。ICLR2026正式页面已核验。

- **方法摘要：**用较强理解能力给生成样本评分构建SFT/DPO数据；研究两侧共同改善、共享经验NTK解释及课程式扩充自训练样本。
- **与本项目的关系：**理解筛选生成数据与SFT/DPO自改进并非空白；共同提高仍须对照只训练一侧和共享/解耦结构。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**arXiv元数据/摘要/历史与ICLR2026正式页面。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md)。
- **补充原始来源：**[来源1](https://proceedings.iclr.cc/paper_files/paper/2026/hash/dceda5883d3223231bef121cc8da2d61-Abstract-Conference.html)。

<a id="p-L059"></a>

#### L059 · Reversing the Flow · P1 · H

**[Reversing the Flow: Generation-to-Understanding Synergy in Large Multimodal Models](https://arxiv.org/abs/2605.15792)**  
Yujun Tong等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**以可控生成动作（细节增强、扩展上下文、结构可视化）产生visual thoughts，再反馈改善理解，免重训；同时指出自主想象的任务对齐仍不稳定。
- **与本项目的关系：**生成→理解是独立验收方向，不应从生成分数提升推断；需评测自选动作是否任务相关。
- **适用边界／版本注意：**摘要页注释为CVPR2026 Findings；不要误写为CVPR main。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md)。

<a id="p-L060"></a>

#### L060 · UniRect-CoT · P0 · H

**[UniRect-CoT: Enhancing Generation in Unified Multimodal Models via Reflective Rectification with Inherent Understanding](https://arxiv.org/abs/2604.13540)**  
Yibo Jiang等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**同一flow轨迹中估计clean preview，自模型生成状态条件reflection；teacher-force反思目标的AR NLL形成latent梯度，GITO反复修正并按原指令选择候选。
- **与本项目的关系：**与本项目前身MMU梯度/中途修复强相关；不能声称首次画到一半检查。其参数冻结latent优化与学得共享核心深度须分开。
- **适用边界／版本注意：**本次未独立复现实验；不导入未核实历史性能数字。
- **本次阅读范围：**元数据/摘要及正文§3–§4（ISR、CSA、GITO），未重算结果。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2_loop\NEXT_EXPERIMENT.md](./stage2_loop/NEXT_EXPERIMENT.md)；[.\stage2_loop\RESEARCH_DECISIONS.md](./stage2_loop/RESEARCH_DECISIONS.md)。
- **补充原始来源：**[来源1](https://arxiv.org/html/2604.13540v2)。

<a id="p-L061"></a>

#### L061 · UniT · P1 · H

**[UniT: Unified Multimodal Chain-of-Thought Test-time Scaling](https://arxiv.org/abs/2602.12279)**  
Leon Liangyu Chen等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**agentic合成交错轨迹数据训练统一模型多轮推理、核验、分目标和内容记忆；推理时可扩展更长链。
- **与本项目的关系：**对照顺序CoT与并行采样的质量—预算曲线，以及理解/生成跨任务迁移；其多轮链并非权重共享深度本身。
- **适用边界／版本注意：**arXiv注释为CVPR2026；同名UniT可能另有早期多任务模型，此处编号锁定为2026论文。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2_loop\RESEARCH_DECISIONS.md](./stage2_loop/RESEARCH_DECISIONS.md)。

<a id="p-L062"></a>

#### L062 · Understanding–Generation Transfer · P0 · H

**[Transferability Between Understanding and Generation in Unified Multimodal Models](https://arxiv.org/abs/2607.04423)**  
Jiwon Kang等，2026。arXiv预印本；具体会议注释见limit。

- **方法摘要：**控制实验比较跨任务同能力迁移，发现全共享backbone与统一视觉encoder更易迁移；用理解训练计数/空间/OCR能力，减少直接生成微调造成的分布偏移。
- **与本项目的关系：**双向迁移依赖架构而非自动成立；启发真实MMU/AR共同监督和共享/解耦对照。
- **适用边界／版本注意：**arXiv注释为ECCV2026；本次未另核会议正式页。
- **本次阅读范围：**arXiv题名/作者/摘要/提交历史，未通读正文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\METHOD_AND_EXPERIMENT_PLAN.md](./stage2b/METHOD_AND_EXPERIMENT_PLAN.md)；[.\stage2b\RECIPROCAL_LOOP_HYPOTHESIS.md](./stage2b/RECIPROCAL_LOOP_HYPOTHESIS.md)。

<a id="p-L063"></a>

#### L063 · Self-Refine · P2 · N

**[Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)**  
Aman Madaan等，2023。arXiv原始题录核验。

- **方法摘要：**同一语言模型先生成，再给自己的输出反馈，并依据反馈反复修改，无需额外训练。
- **与本项目的关系：**提供生成—评议—修改这一外部自反馈范式的早期背景；有助于澄清本项目内部共享深度应独立承担的贡献。
- **适用边界／版本注意：**以语言等任务为主要证据；提示工作流的自修正不等于连续图像latent中的因果修复。
- **本次阅读范围：**本次核验arXiv元数据和摘要；未通读全文或复现实验。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L064"></a>

#### L064 · Reflexion · P2 · N

**[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)**  
Noah Shinn等，2023。arXiv原始题录核验。

- **方法摘要：**将任务反馈转为文字反思，写入情节记忆，影响语言agent后续尝试，无需更新模型权重。
- **与本项目的关系：**是外部记忆和反思驱动改进的背景对照，避免把所有反复尝试都统称为网络循环学习。
- **适用边界／版本注意：**文字记忆、环境重试与同一图像状态的共享块重访是不同计算机制；其编码任务结果不迁移为生成质量证据。
- **本次阅读范围：**本次核验arXiv元数据和摘要；未通读全文或复现实验。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="cat-representation"></a>

### 视觉表示、语义latent与过程监督

理解特征进入生成过程已有大量先例。需要区分外部视觉教师、重建对齐、自监督表示、成功续跑教师和本项目actual-additive过程目标。

<a id="p-L065"></a>

#### L065 · REPA · P0 · H

**[Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://arxiv.org/abs/2410.06940)**  
Sihyun Yu等，2024。ICLR 2025 Oral；arXiv v4。

- **方法摘要：**把去噪网络投影后的中间表示对齐到冻结视觉编码器的干净图特征，改善生成训练。
- **与本项目的关系：**过程hidden监督和教师特征选择的重要方法背景。
- **适用边界／版本注意：**本项目actual-additive教师来自成功延续，不是原文的外部视觉encoder，不能说已复现REPA。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；另有1份记录，见结构化目录。

<a id="p-L066"></a>

#### L066 · REPA-E · P1 · H

**[REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](https://arxiv.org/abs/2504.10483)**  
Xingjian Leng等，2025。arXiv v3。

- **方法摘要：**通过表示对齐使VAE与diffusion transformer端到端联合调整；普通扩散损失直接联合可能退化。
- **与本项目的关系：**提醒潜空间、重建与生成目标不能只由单步MSE判断。
- **适用边界／版本注意：**本项目保持原VAE；该论文不是当前VAE已改好的证据。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L067"></a>

#### L067 · iREPA · P0 · H

**[What matters for Representation Alignment: Global Information or Spatial Structure?](https://arxiv.org/abs/2512.10794)**  
Jaskirat Singh等，2025。arXiv预印本。

- **方法摘要：**比较多类视觉教师，强调patch间空间结构，使用卷积投影和空间归一化增强传递。
- **与本项目的关系：**直接关联空间关系修复、token-local读出与全局语义特征的局限。
- **适用边界／版本注意：**结论针对其对齐训练协议；不能由本项目几何失败反推其方法无效。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\stage2_loop\RESEARCH_DECISIONS.md](./stage2_loop/RESEARCH_DECISIONS.md)。

<a id="p-L068"></a>

#### L068 · RAE · P1 · H

**[Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690)**  
Boyang Zheng等，2025。技术报告。

- **方法摘要：**用预训练表示编码器加训练decoder构建高维语义latent，并分析DiT在其中训练的适配问题。
- **与本项目的关系：**后续研究可比较语义latent自读，但应单列tokenizer变化。
- **适用边界／版本注意：**ImageNet结果和高维latent不自动迁移到现有Show-o2/VAE。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)。

<a id="p-L069"></a>

#### L069 · T2I RAE · P1 · H

**[Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2601.16208)**  
Shengbang Tong等，2026。arXiv预印本，2026-01。

- **方法摘要：**扩展RAE到大规模自由文本生图，对比VAE，分析维度相关噪声调度和解码器数据组成。
- **与本项目的关系：**潜空间理解与生成共享、从类别条件走向T2I的主要参考。
- **适用边界／版本注意：**原摘要支持直接latent推理潜力；不能仅凭该摘要断言已实现某个具体Best-of-N验证器。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L070"></a>

#### L070 · Self-Flow · P1 · H

**[Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://arxiv.org/abs/2603.06507)**  
Hila Chefer等，2026。arXiv预印本，2026-03。

- **方法摘要：**Dual-Timestep Scheduling给不同token不同噪声，制造信息不对称，推动生成过程自行学习语义表示。
- **与本项目的关系：**对照外部教师依赖；研究过程监督是否可改为模型内自监督。
- **适用边界／版本注意：**不等于将同一块简单重复；训练任务与本项目有限修复器不同。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L071"></a>

#### L071 · RARE · P0 · H

**[What Visual Generators Need from Teachers: Rethinking Representation Alignment](https://arxiv.org/abs/2609.34732)**  
Yongcong Wang等，2026。arXiv预印本，2026-09-28。

- **方法摘要：**以学生线性可恢复的教师增量缺口选择层和对齐权重，并随训练进展调整监督。
- **与本项目的关系：**关联v55/v59/v82的读出可达性诊断，避免凭特征相似度选择教师。
- **适用边界／版本注意：**本项目的最小二乘残差仅是固定子空间诊断，不是已实现完整RARE策略。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L072"></a>

#### L072 · UniTok · P2 · N

**[UniTok: A Unified Tokenizer for Visual Generation and Understanding](https://arxiv.org/abs/2502.20321)**  
Chuofan Ma等，2025。arXiv v3。

- **方法摘要：**采用多codebook量化增加离散token表示容量，研究生成与理解目标如何共存。
- **与本项目的关系：**统一表示容量与目标冲突的替代解释。
- **适用边界／版本注意：**rFID为重建质量，不是完整生成FID。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L073"></a>

#### L073 · RecA / RECA · P0 · H

**[Reconstruction Alignment Improves Unified Multimodal Models](https://arxiv.org/abs/2509.07295)**  
Ji Xie等，2025。ICLR2026（作者项目页/arXiv）。

- **方法摘要：**把模型自己的视觉理解embedding当dense prompt，直接重建原图做自监督后训练，将理解表示对齐到生成，保留原推理接口。
- **与本项目的关系：**理解表征对生成可用性的关键对照；与本项目过程hidden对齐/输出头探针相关，但不是共享Transformer替代独立深度。
- **适用边界／版本注意：**采用v4书目信息；历史摘要/项目页曾有不同GEdit数值，本汇总不混用版本定量。
- **本次阅读范围：**arXiv元数据/摘要/提交历史和作者项目页方法概述。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\METHOD_AND_EXPERIMENT_PLAN.md](./stage2b/METHOD_AND_EXPERIMENT_PLAN.md)；[.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md)；另有1份记录，见结构化目录。
- **补充原始来源：**[来源1](https://reconstruction-alignment.github.io/)、[来源2](https://github.com/HorizonWind2004/reconstruction-alignment)。

<a id="cat-budget"></a>

### 推理时扩展、搜索、latent优化与计算预算

增加计算可以投给采样步、网络深度、候选数、验证器、梯度优化或搜索。它们是循环方案必须面对的替代路线，比较应使用完整系统成本。

<a id="p-L074"></a>

#### L074 · Unified Sparsity · P1 · H

**[Understanding and Harnessing Sparsity in Unified Multimodal Models](https://arxiv.org/abs/2512.02351)**  
Shwai He等，2025。arXiv预印本。

- **方法摘要：**用深度/宽度裁剪分析统一模型组件，再用MoE稀疏激活适配恢复生成。
- **与本项目的关系：**支持先剖析外Qwen/内生成头敏感性，再决定共享或压缩位置。
- **适用边界／版本注意：**动态激活参数少不等于总存储参数少；并非层权重循环。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L075"></a>

#### L075 · Consistency Models · P2 · N

**[Consistency Models](https://arxiv.org/abs/2303.01469)**  
Yang Song等，2023。ICML 2023。

- **方法摘要：**直接学习沿扩散轨迹的一致映射，可蒸馏或独立训练以实现少步生成。
- **与本项目的关系：**更少采样预算的强替代路线，避免只与原49步baseline比较。
- **适用边界／版本注意：**不是同一模型末层的无训练早退；需要独立训练目标。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L076"></a>

#### L076 · Diffusion Inference Scaling · P0 · N

**[Scaling Inference Time Compute for Diffusion Models](https://arxiv.org/abs/2501.09732)**  
Nanye Ma等，2025。CVPR 2025；arXiv曾题为Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps。

- **方法摘要：**把推理扩展拆成verifier与噪声/路径搜索算法，研究超越增加去噪步的收益。
- **与本项目的关系：**必须与同预算Best-of-N、噪声搜索及轨迹搜索比较，验证器也应计费。
- **适用边界／版本注意：**搜索选择收益不是共享循环收益，verifier过拟合需独立评价。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.html)。

<a id="p-L077"></a>

#### L077 · Classical Search Diffusion · P1 · N

**[Inference-time Scaling of Diffusion Models through Classical Search](https://arxiv.org/abs/2505.23614)**  
Xiangcheng Zhang等，2025。arXiv v2；ICLR2026会议PDF另可核。

- **方法摘要：**结合局部Langevin MCMC与全局树搜索，控制扩散生成以满足测试时目标。
- **与本项目的关系：**为分叉/回溯/预算分配提供算法对照。
- **适用边界／版本注意：**规划、离线RL和图像协议不同，不能跨任务比较原始分数。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L078"></a>

#### L078 · VHS · P0 · N

**[Tiny Inference-Time Scaling with Latent Verifiers](https://arxiv.org/abs/2603.22492)**  
Davide Bucciarelli等，2026。Findings of CVPR 2026，原站comments。

- **方法摘要：**直接从单步DiT中间hidden验证候选，省去像素解码和MLLM重编码。
- **与本项目的关系：**潜空间低成本观察器的直接近邻，影响中途MMU成本主张。
- **适用边界／版本注意：**原文聚焦单步生成器和候选筛选，不是多步flow原状态自修复。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L079"></a>

#### L079 · VFScale · P2 · N

**[VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](https://arxiv.org/abs/2502.01989)**  
Tao Zhang等，2025。ICLR 2026；当前v5于2026-02。

- **方法摘要：**通过训练目标塑造内在能量验证器，并用hybrid MCTS扩展推理计算。
- **与本项目的关系：**无外部verifier的对照思路，提醒自检也需要训练可靠性。
- **适用边界／版本注意：**主要Maze/Sudoku推理任务，不能视为图像偏好已证明。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L080"></a>

#### L080 · ReNO · P1 · N

**[ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization](https://arxiv.org/abs/2406.04312)**  
Luca Eyring等，2024。NeurIPS 2024（arXiv作者注记）。

- **方法摘要：**冻结单步生成器，依照一个或多个偏好模型的信号优化初始噪声，在推理阶段提高提示遵循与偏好。
- **与本项目的关系：**是可微动作搜索和额外推理计算的直接替代路线；有助于区分优化噪声带来的收益与共享循环带来的收益。
- **适用边界／版本注意：**主要针对单步模型；噪声优化的奖励调用与反传成本必须计入，不能将其效果归因于模型学会内部修复。
- **本次阅读范围：**本次核验arXiv元数据和摘要；未通读全文或复现实验。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L081"></a>

#### L081 · D-Flow · P1 · N

**[D-Flow: Differentiating through Flows for Controlled Generation](https://arxiv.org/abs/2402.14017)**  
Heli Ben-Hamu等，2024。ICML 2024（arXiv作者注记）。

- **方法摘要：**穿过整个flow生成过程反传目标梯度，优化源噪声点，实现受控生成与逆问题求解。
- **与本项目的关系：**是完整终点反传、输入空间动作优化的重要先例，帮助解释为什么局部替代损失需要终点验证。
- **适用边界／版本注意：**其高斯路径及流形投影分析具有建模条件，不构成任意中途加性修复或Show-o2语义改善的保证。
- **本次阅读范围：**本次核验arXiv元数据和摘要；未通读全文或复现实验。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L082"></a>

#### L082 · DOODL · P1 · N

**[End-to-End Diffusion Latent Optimization Improves Classifier Guidance](https://arxiv.org/abs/2303.13703)**  
Bram Wallace等，2023。ICCV 2023（CVF正式论文已核验）。

- **方法摘要：**通过可逆扩散过程对真实生成像素上的分类/视觉目标反传，直接优化扩散latent，减少单步终点近似带来的梯度偏差。
- **与本项目的关系：**与本项目检查完整14步终点而非仅检查局部latent误差直接相关，也是latent优化类比较对象。
- **适用边界／版本注意：**它依赖相应可逆过程与外部目标，不是同一Transformer权重循环，也不能直接套用其内存结论到当前flow实现。
- **本次阅读范围：**本次核验arXiv元数据和摘要；未通读全文或复现实验。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L083"></a>

#### L083 · Progressive Seed Pruning / PSP · P1 · H

**[Inference-Time Scaling of Diffusion Models via Progressive Seed Pruning](https://arxiv.org/abs/2607.21591)**  
Rogerio Guimaraes等，2026。arXiv首版2026-07-23；作者项目页已核验。

- **方法摘要：**一开始探索较多随机种子，按中间去噪估计评分逐步剪去低分轨迹，使预算集中于少数有希望的候选。
- **与本项目的关系：**是历史已参考的中途评价与预算分配路线；需要比较内部修复与在相同生成预算下更有效地选择候选。
- **适用边界／版本注意：**相同生成器NFE不等于相同全系统成本；中间VAE解码和reward评分仍有开销，也不证明共享循环有额外收益。
- **本次阅读范围：**arXiv元数据和摘要、作者项目页方法与成本说明；未通读论文全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md)。
- **补充原始来源：**[来源1](https://www.vision.caltech.edu/psp/)。

<a id="cat-guidance"></a>

### 空间接地、组合控制与图像编辑

为位置错误、主体遗漏和属性绑定提供成熟替代方案，也帮助区分外部布局/框/mask与模型原生证据。不同信息条件下的结果应分开评价。

<a id="p-L084"></a>

#### L084 · Attend-and-Excite · P0 · H

**[Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models](https://arxiv.org/abs/2301.13826)**  
Hila Chefer等，2023。SIGGRAPH 2023。

- **方法摘要：**推理时增强被忽略主体token的cross-attention，缓解主体遗漏和属性绑定问题。
- **与本项目的关系：**语义修复和注意力干预的直接基线，影响早期动作设计。
- **适用边界／版本注意：**为特定扩散架构设计，迁到统一flow须重验；不是循环特异性证明。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L085"></a>

#### L085 · Structured Diffusion · P1 · H

**[Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis](https://arxiv.org/abs/2212.05032)**  
Weixi Feng等，2022。ICLR 2023 Camera Ready；首版2022。

- **方法摘要：**利用语言结构和cross-attention的key/value语义，改进属性绑定与组合布局。
- **与本项目的关系：**对照复杂证据修复器是否超过简单结构化注意力控制。
- **适用边界／版本注意：**语言解析/额外条件的费用及信息量要与部署方案对齐。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage3_grounded_repair\STAGE3_CONTINUATION_20261004.md](./stage3_grounded_repair/STAGE3_CONTINUATION_20261004.md)。
- **补充原始来源：**[来源1](https://research.google/pubs/training-free-structured-diffusion-guidance-for-compositional-text-to-image-synthesis/)。

<a id="p-L086"></a>

#### L086 · Attention Refocusing · P1 · H

**[Grounded Text-to-Image Synthesis with Attention Refocusing](https://openaccess.thecvf.com/content/CVPR2024/html/Phung_Grounded_Text-to-Image_Synthesis_with_Attention_Refocusing_CVPR_2024_paper.html)**  
Quynh Phung等，2024。CVPR 2024。

- **方法摘要：**按给定空间布局对cross/self-attention施加refocusing约束，布局可由LLM产生。
- **与本项目的关系：**对象定位、空间mask和证据可执行性的直接参考。
- **适用边界／版本注意：**使用显式layout/mask属于额外条件，不能混作模型原生MMU能力。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage3_grounded_repair\STAGE3_CONTINUATION_20261004.md](./stage3_grounded_repair/STAGE3_CONTINUATION_20261004.md)。
- **补充原始来源：**[来源1](https://attention-refocusing.github.io/)。

<a id="p-L087"></a>

#### L087 · Prompt-to-Prompt · P2 · N

**[Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)**  
Amir Hertz等，2022。arXiv原论文。

- **方法摘要：**控制cross-attention以文本编辑图像，同时尽量保持原布局与内容。
- **与本项目的关系：**认识prompt小修改可能带来大构图变化，需同状态配对。
- **适用边界／版本注意：**编辑流程不等于原提示不变时的内部语义修复。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L088"></a>

#### L088 · GLIGEN · P1 · N

**[GLIGEN: Open-Set Grounded Text-to-Image Generation](https://arxiv.org/abs/2301.07093)**  
Yuheng Li等，2023。CVPR 2023。

- **方法摘要：**冻结预训练生成器，通过新门控层注入box等grounding输入。
- **与本项目的关系：**为冻结原路径、逐渐开启修复条件的设计提供强基线。
- **适用边界／版本注意：**额外门控层不共享原块权重，不能称同一种循环。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://github.com/gligen/GLIGEN)。

<a id="p-L089"></a>

#### L089 · ControlNet · P1 · N

**[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)**  
Lvmin Zhang等，2023。arXiv原论文。

- **方法摘要：**冻结原模型，以零卷积连接可训练控制支路，支持边缘/深度/分割等条件。
- **与本项目的关系：**零初始化保留基线功能的工程背景。
- **适用边界／版本注意：**复制支路通常增加存储参数；零输出不保证开启控制后的质量。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://github.com/lllyasviel/ControlNet)。

<a id="p-L090"></a>

#### L090 · SDEdit · P1 · N

**[SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)**  
Chenlin Meng等，2021。arXiv首版2021。

- **方法摘要：**先对输入加噪，再用生成先验去噪以权衡输入忠实度与真实感。
- **与本项目的关系：**像素教师重新桥接/继续采样的相关基线。
- **适用边界／版本注意：**重新加噪编辑与严格同噪声同状态修复是不同实验。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L091"></a>

#### L091 · InstructPix2Pix · P2 · N

**[InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800)**  
Tim Brooks等，2022。arXiv首版2022。

- **方法摘要：**用指令编辑配对训练条件扩散模型执行自然语言修改。
- **与本项目的关系：**比较显式编辑能力与统一模型内轻量修复的差别。
- **适用边界／版本注意：**不能用额外专门编辑数据的结果当作无训练修复上限。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L092"></a>

#### L092 · Grounding DINO · P2 · N

**[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)**  
Shilong Liu等，2023。arXiv v5更新2024。

- **方法摘要：**融合语言和检测器，按类别名/指代短语做开放集目标定位。
- **与本项目的关系：**离线几何教师、定位身份审计和检测误差的相关工具背景。
- **适用边界／版本注意：**定位器给出的框是外部监督，不能算原生MMU定位；不声称本项目全部检测由此模型完成。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L093"></a>

#### L093 · SAM · P2 · N

**[Segment Anything](https://arxiv.org/abs/2304.02643)**  
Alexander Kirillov等，2023。arXiv原论文。

- **方法摘要：**可提示分割模型与大规模mask数据，支持不同提示下的区域分割。
- **与本项目的关系：**实例mask可用于离线可塑性分析与局部外观保护。
- **适用边界／版本注意：**mask本身不能保证对象身份、语义正确或编辑质量，部署信息边界须说明。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="cat-reward"></a>

### 偏好对齐、强化学习及质量—多样性权衡

这一组给出训练或推理目标的来源。优化过的reward应与独立评价分开；确定性flow、随机扩散政策和单图latent优化具有不同数学假设。

<a id="p-L094"></a>

#### L094 · Diffusion-DPO · P1 · H

**[Diffusion Model Alignment Using Direct Preference Optimization](https://arxiv.org/abs/2311.12908)**  
Bram Wallace等，2023。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**将DPO改写为适合扩散模型似然/ELBO的可微比较目标，在Pick-a-Pic成对偏好上微调SDXL；另研究AI反馈变体。
- **与本项目的关系：**普通离线偏好微调强对照；必须把共享循环路由的增益与一般偏好监督分离。
- **适用边界／版本注意：**扩散ELBO推导不能直接套到本项目确定性flow ODE及中途物理修复；需匹配训练参数化与合法似然。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文或推导。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L095"></a>

#### L095 · AlignProp · P1 · H

**[Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2310.03739)**  
Mihir Prabhudesai等，2023。WITHDRAWN；2024-11-07 v5，作者转接2407.08737。

- **方法摘要：**通过去噪过程端到端反传可微reward，结合LoRA与gradient checkpointing控制显存，覆盖语义/审美等目标。
- **与本项目的关系：**完整轨迹reward反传是本项目终点可微优化的历史先例，不能只因采用反传或checkpoint就称方法新。
- **适用边界／版本注意：**必须显式标注撤稿和后续论文；不能将摘要中的步数/奖励优势当成本项目Show-o2共享循环的可迁移保证。
- **本次阅读范围：**当前arXiv元数据、摘要、withdrawn状态与作者替代说明；未读旧版本全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L096"></a>

#### L096 · VADER / AlignProp后续 · P1 · N

**[Video Diffusion Alignment via Reward Gradients](https://arxiv.org/abs/2407.08737)**  
Mihir Prabhudesai等，2024。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**利用预训练视觉reward模型对RGB像素的密集梯度，反传到视频扩散生成器进行计算和样本较高效的对齐。
- **与本项目的关系：**因AlignProp作者明确转接而补入；若以后扩视频，可参考可微reward对齐，而非把图像实验当视频已完成。
- **适用边界／版本注意：**针对视频生成和reward优化，不证明共享层压缩、MMU协同或本项目T2I物理修复收益。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L097"></a>

#### L097 · DRIFT · P1 · H

**[Beyond the Dirac Delta: Mitigating Diversity Collapse in Reinforcement Fine-Tuning for Versatile Image Generation](https://arxiv.org/abs/2601.12401)**  
Jinmei Liu等，2026。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**面向diffusion/flow在线RL微调多样性塌缩，结合过滤reward离群值的子集抽样、随机prompt变化及组内多样性的potential-based reward shaping。
- **与本项目的关系：**直接支持同时审计task alignment与多样性；保留源噪声、加KL或仅增加seed不自动构成多样性保证。
- **适用边界／版本注意：**摘要宣称的Pareto改进限其协议；尚未核验全文假设、特征或统计。不能将其结论当本项目已验证结果。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L098"></a>

#### L098 · ParetoSlider · P1 · H

**[ParetoSlider: Diffusion Models Post-Training for Continuous Reward Control](https://arxiv.org/abs/2604.20816)**  
Shelly Golan等，2026。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**用连续变化的偏好权重作为条件，训练单个多目标RL生成模型逼近Pareto前沿，在SD3.5、FluxKontext和LTX-2上研究推理时连续权衡。
- **与本项目的关系：**限制“加权多个reward/偏好条件”作为单独新贡献；本项目应明确循环计算分配带来的额外价值。
- **适用边界／版本注意：**是偏好条件化的后训练方法，不是容量门失败的直接解法；当前仅摘要核验，未复核Pareto最优性或对照细节。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L099"></a>

#### L099 · MSDDA · P1 · H

**[Step-level Denoising-time Diffusion Alignment with Multiple Objectives](https://arxiv.org/abs/2604.14379)**  
Qi Zhang等，2026。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**建立step-level RL表述，提出无需重训的多目标去噪期融合；最优reverse distribution的均值/方差由单目标基础模型给出，摘要声称与所定义step-RL目标严格等价。
- **与本项目的关系：**按时间步处理多目标已有直接先例；拟议贡献须区别于策略融合、普通reward调权及去噪期对齐。
- **适用边界／版本注意：**闭式等价属于其模型/目标假设，未读定理不能外推到Show-o2确定性ODE、共享重访或MMU反馈。
- **本次阅读范围：**arXiv元数据和摘要；未核验闭式推导/证明。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L100"></a>

#### L100 · Flow-GRPO · P0 · H

**[Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)**  
Jie Liu等，2025。NeurIPS2025官方出版页核验。

- **方法摘要：**把确定性flow ODE转成匹配边缘分布的SDE以便online policy-gradient探索，并减少训练去噪步而保留推理步数；在SD3.5-M上报告语义与偏好改善。
- **与本项目的关系：**本项目flow reward训练的关键普通RL基线；新颖性须来自共享循环、理解反馈与计算分配，而非首次给flow加RL。
- **适用边界／版本注意：**其SD3.5-M与GenEval提升不证明Show-o2上的转移；ODE-SDE、采样和训练预算必须匹配，摘要的低reward hacking不能外推一般多样性保证。
- **本次阅读范围：**arXiv元数据/摘要及NeurIPS官方摘要，未通读正文/补充。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\RESEARCH_DIRECTION.md](./stage2b/RESEARCH_DIRECTION.md)。
- **补充原始来源：**[来源1](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a10c46572628d58cb44fb705f25cbbf-Abstract-Conference.html)。

<a id="p-L101"></a>

#### L101 · ImageReFL · P1 · H

**[ImageReFL: Balancing Quality and Diversity in Human-Aligned Diffusion Models](https://arxiv.org/abs/2505.22569)**  
Dmitrii Sorokin等，2025。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**combined generation在后期才切换reward-tuned模型、前期保留base；ImageReFL在真实图上结合diffusion/ReFL等正则微调以平衡质量和多样性。
- **与本项目的关系：**后期切换与多样性保护已有先例，限制本项目仅按时间分配reward或loop的朴素新颖性。
- **适用边界／版本注意：**模型切换与本项目同权重共享重访不同；摘要人评不等于本项目已人评，需独立测量多样性和所有额外成本。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2_loop\RESEARCH_DECISIONS.md](./stage2_loop/RESEARCH_DECISIONS.md)。

<a id="p-L102"></a>

#### L102 · Reward-diversity / AIG · P1 · H

**[Elucidating Optimal Reward-Diversity Tradeoffs in Text-to-Image Diffusion Models](https://arxiv.org/abs/2409.06493)**  
Rohit Jena等，2024。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**研究reward hacking与多样性损失及KL/LoRA scaling的限制；提出推理期Annealed Importance Guidance（AIG）以平衡base多样性和reward。
- **与本项目的关系：**支持将reward、质量和组内多样性分别报告；解释为何同噪声、KL或LoRA小幅更新都不直接保证覆盖。
- **适用边界／版本注意：**摘要的理论不可避免/最优性均有其建模假设；本次未读证明，不能用普遍措辞套到所有循环/flow方法。
- **本次阅读范围：**arXiv元数据和摘要，未核验定理或全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2_loop\RESEARCH_DECISIONS.md](./stage2_loop/RESEARCH_DECISIONS.md)。

<a id="p-L103"></a>

#### L103 · DDPO · P1 · N

**[Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)**  
Kevin Black等，2023。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**把去噪序列当多步决策过程，以policy-gradient形式DDPO直接优化压缩率、审美及VLM图文对齐reward，比较reward-weighted likelihood。
- **与本项目的关系：**普通online RL偏好训练强基线，和可微reward反传、离线DPO区分；有助于清楚描述flow适配需求。
- **适用边界／版本注意：**扩散随机转移政策不能未经推导直接用于确定性flow；模型/采样/rollout与reward预算都需匹配，训练reward不当独立评价。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文/算法推导。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L104"></a>

#### L104 · Stepwise-Flow-GRPO · P1 · H

**[Stepwise Credit Assignment for GRPO on Flow-Matching Models](https://arxiv.org/abs/2603.28718)**  
Yash Savani等，2026。CVPR 2026（arXiv作者注记和作者项目页核验）。

- **方法摘要：**从中间干净图像估计获得reward，以相邻步骤reward增量形成分步优势，另提出DDIM启发的SDE来改善训练奖励质量。
- **与本项目的关系：**是历史Stepwise Reward Assignment线索对应的正式论文；说明中间评价和更细粒度奖励分配已有先例。
- **适用边界／版本注意：**额外中间估计、解码与评分需要计费；其随机flow RL目标不是本项目固定教师的hidden回归，不能直接替换解释。
- **本次阅读范围：**arXiv元数据和摘要、作者项目页及CMU作者技术说明的选定方法段落；未通读论文全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md)。
- **补充原始来源：**[来源1](https://stepwiseflowgrpo.com/)、[来源2](https://www.cs.cmu.edu/~csd-phd-blog/2026/stepwise-flow-grpo/)。

<a id="cat-evaluation"></a>

### 语义、偏好、保真、多样性、分布及理解评测

不存在一个分数能同时覆盖对象关系、审美、条件正确、保真、分布覆盖和理解保留。文献中指标的用途须与本项目真正运行的评价分开。

<a id="p-L105"></a>

#### L105 · MME-Unify · P1 · N

**[MME-Unify: A Comprehensive Benchmark for Unified Multimodal Understanding and Generation Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8308c2c45cf95bfebb736be377f0982-Abstract-Conference.html)**  
Wulin Xie等，2026。ICLR 2026。

- **方法摘要：**整合理解、生成和混合模态任务，设计双向能力促进的统一评价。
- **与本项目的关系：**防止只看孤立理解/生成分数就声称相互促进。
- **适用边界／版本注意：**本项目尚未完成该基准；不能把论文榜单当本项目成绩。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L106"></a>

#### L106 · ImageReward · P1 · H

**[ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977)**  
Jiazheng Xu等，2023。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**构建137k专家比较的偏好数据，学习ImageReward评分器；提出Reward Feedback Learning（ReFL），对可评分生成模型直接微调。
- **与本项目的关系：**偏好代理及可微奖励训练的直接先例；说明添加reward loss本身不足以构成共享循环贡献。
- **适用边界／版本注意：**偏好预测器不是人类总体偏好真值；使用同一reward训练并评价会有自洽偏差，需独立偏好模型与人评，也需保留正确图与多样性。
- **本次阅读范围：**arXiv元数据和摘要，未通读32页全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L107"></a>

#### L107 · Pick-a-Pic / PickScore · P1 · H

**[Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://arxiv.org/abs/2305.01569)**  
Yuval Kirstain等，2023。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**通过开放网页应用收集真实T2I用户成对选择，训练CLIP基础的PickScore，研究模型评价与候选排序。
- **与本项目的关系：**本项目训练/评价所用PickScore的文献来源；支持将成对偏好与prompt隔离作为实验基础。
- **适用边界／版本注意：**分数改善只是该用户偏好代理下的改善；排名筛选、训练reward与独立评价是不同用途，提示重用及奖励投机需另行防止。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L108"></a>

#### L108 · HPSv2 · P0 · H

**[Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis](https://arxiv.org/abs/2306.09341)**  
Xiaoshi Wu等，2023。arXiv2023；本轮未证实会议出版归属，不猜ICCV/NeurIPS。

- **方法摘要：**HPDv2含433760对图像、798090个人类选择，以该数据微调CLIP得到HPSv2，并设计较稳定的模型评价提示集。
- **与本项目的关系：**本项目HPS偏好守门与交叉评价来源；与PickScore一起避免单个代理决定成功。
- **适用边界／版本注意：**训练分布和偏好标注仍有限；极小分数差不等价真实人类可感知改进，须固定checkpoint/预处理并报告逐态及区间。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L109"></a>

#### L109 · HPSv3 / CoHP · P1 · H

**[HPSv3: Towards Wide-Spectrum Human Preference Score](https://arxiv.org/abs/2508.03789)**  
Yuhang Ma等，2025。arXiv comments标注ICCV2025。

- **方法摘要：**HPDv3覆盖108万文图对和117万成对比较；VLM评分器使用不确定性感知排序损失。CoHP在每轮用HPSv3选最优候选进行迭代图像改善。
- **与本项目的关系：**直接限制“迭代改善偏好”作为新颖性主张；可作为独立较新偏好审计器，而非本项目已运行评价。
- **适用边界／版本注意：**评分筛选式迭代不等于同模型共享Transformer循环；高分候选也不能自动证明保真、独立泛化或同提示有效多样性。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L110"></a>

#### L110 · Vendi Score · P1 · H

**[The Vendi Score: A Diversity Evaluation Metric for Machine Learning](https://arxiv.org/abs/2210.02410)**  
Dan Friedman等，2022。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**由用户定义样本相似度矩阵，以谱的Shannon熵指数衡量有效多样性；不需要参考样本分布或类别标签。
- **与本项目的关系：**可在固定同提示/固定组大小下评价输出多样性，明确特征和核，并与质量门同时报告。
- **适用边界／版本注意：**高度依赖相似度定义；不同特征、核和样本组不可直接数值横比。噪声/坏图造成距离变大也非有效多样性。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文/实现。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L111"></a>

#### L111 · DreamSim · P1 · H

**[DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data](https://arxiv.org/abs/2306.09344)**  
Stephanie Fu等，2023。arXiv journal reference为NeurIPS2023。

- **方法摘要：**通过合成扰动图像与人类相似性判断，学习对前景、语义、颜色和布局更敏感的整体感知距离，并测试真实图检索/重建。
- **与本项目的关系：**补充同提示构图/外观变化和修复保真诊断，避免只用像素或单一语义余弦。
- **适用边界／版本注意：**感知距离不是审美偏好、指令正确率或分布覆盖；需要质量守门，且训练源合成图域与本项目域未必一致。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L112"></a>

#### L112 · CMMD / Rethinking FID · P1 · H

**[Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603)**  
Sadeep Jayasumana等，2024。CVPR2024官方出版页核验；arXiv首稿2023-11-30。

- **方法摘要：**指出FID的Inception表示、正态假设和样本复杂度问题；提出CLIP特征+Gaussian RBF maximum mean discrepancy的CMMD。
- **与本项目的关系：**适合与历史FID同时做足量参考分布审计，帮助解释少样本分布分数和局部MSE与终图不一致。
- **适用边界／版本注意：**不替代prompt约束、人评或同state保真；仍需匹配参考集、特征提取、预处理和样本规模，不能用72图小pilot得通用分布质量结论。
- **本次阅读范围：**arXiv元数据/摘要及CVPR2024官方摘要，未通读全文或核验实现。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。
- **补充原始来源：**[来源1](https://openaccess.thecvf.com/content/CVPR2024/html/Jayasumana_Rethinking_FID_Towards_a_Better_Evaluation_Metric_for_Image_Generation_CVPR_2024_paper.html)。

<a id="p-L113"></a>

#### L113 · Improved Precision and Recall · P1 · H

**[Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991)**  
Tuomas Kynkäänniemi等，2019。arXiv comments为NeurIPS2019 final。

- **方法摘要：**以非参数方式表示真实/生成样本流形，将生成样本质量和分布覆盖分开衡量；分析StyleGAN/BigGAN与truncation。
- **与本项目的关系：**区分偏好优化后更高保真与覆盖收缩，避免单一总体分数隐藏mode collapse。
- **适用边界／版本注意：**基于特征空间和样本流形估计，受表示/邻域/样本数影响；不直接检查空间关系或条件遵循。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L114"></a>

#### L114 · GenEval 2 · P0 · H

**[GenEval 2: Addressing Benchmark Drift in Text-to-Image Evaluation](https://arxiv.org/abs/2512.16853)**  
Amita Kamath等，2025。arXiv2025-12-18；本轮未证实后续会议归属。

- **方法摘要：**研究静态GenEval评审器随生成模型发展而产生的benchmark drift；扩充视觉原语和组合度，提出组合原语判断的Soft-TIFA。
- **与本项目的关系：**为本项目历史GenEval标签增加独立审计与更新基准理由；须保留原协议和版本，不能无缝改评测器后混算。
- **适用边界／版本注意：**作者明确新基准也不保证永不漂移；不能用对旧基准的批评撤回不利结果，更不能把小开发子集换名成独立GenEval2成绩。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文/附录人评。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)；[.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md)；另有1份记录，见结构化目录。
- **补充原始来源：**[来源1](https://github.com/facebookresearch/GenEval2)。

<a id="p-L115"></a>

#### L115 · T2I-CompBench (conference version) · P1 · H

**[T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350v2)**  
Kaiyi Huang等，2023。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**6000提示、3主类/6子类，覆盖属性绑定、空间/非空间关系和复杂组合；提出组合评价并探索MLLM限制及reward-driven sample selection的GORS。
- **与本项目的关系：**拓展本项目单一空间关系教师的组合语义覆盖；提供普通reward筛选/微调对照。
- **适用边界／版本注意：**不同子类依赖不同自动评审器；不能把其中某一项分数当整体画质。必须固定v2，不能混用v3的8000提示规模。
- **本次阅读范围：**版本固定的arXiv元数据和摘要，未通读全文。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L116"></a>

#### L116 · T2I-CompBench++ · P1 · N

**[T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-image Generation](https://arxiv.org/abs/2307.06350v3)**  
Kaiyi Huang等，2025。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**2025期刊扩展版覆盖8000提示、4主类/8子类，加入3D空间关系和数量，增强检测与MLLM评价。
- **与本项目的关系：**作为新增扩展审计选择；说明历史裸arXivURL如今会指向不同题名和规模。
- **适用边界／版本注意：**与2023会议版不能无说明横比；尚未读全文核验每种judge的可靠性与提示采样。
- **本次阅读范围：**arXiv v3元数据、摘要和版本说明，未通读全文。
- **来源归属：**历史T2I-CompBench同编号的扩展版本；本次按v3单列。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L117"></a>

#### L117 · GenEval · P0 · B

**[GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513)**  
Dhruba Ghosh等，2023。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**以检测器及辅助判别模型检查对象共现、位置、数量和颜色，提供可解释的对象级T2I组合评价。
- **与本项目的关系：**本项目官方语义标签与细分关系诊断的来源；需固定评测代码、阈值和环境，并报告失败/破坏。
- **适用边界／版本注意：**对象检测器不是终极ground truth；需注意GenEval2发现的漂移，小样本空间提示不等于完整基准，语义分数也非画质偏好。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文；历史官方GitHub另有本项目使用记录。
- **来源归属：**历史实际评价工具；正式论文题录本次补齐。历史记录：[.\stage2b\RESEARCH_DIRECTION.md](./stage2b/RESEARCH_DIRECTION.md)；[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)。
- **补充原始来源：**[来源1](https://github.com/djghosh13/geneval)。

<a id="p-L118"></a>

#### L118 · FID / TTUR · P1 · B

**[GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)**  
Martin Heusel等，2017。NeurIPS2017（arXiv journal reference）。

- **方法摘要：**提出GAN双时间尺度更新TTUR，并引入Fréchet Inception Distance用于生成与真实图像分布比较。
- **与本项目的关系：**历史分布质量评价来源；与同条件、同数据/分辨率/样本数的原模型比较，不能拿432px T2I小集横比ImageNet256类别条件FID。
- **适用边界／版本注意：**不是单图偏好或条件语义指标；实现/特征/样本数及分布假设影响显著，可结合CMMD和precision/recall交叉审计。
- **本次阅读范围：**arXiv元数据和摘要，未通读TTUR证明或FID正文。
- **来源归属：**历史评价计划；FID原论文题录本次补齐，非本项目已完成FID。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)。

<a id="p-L119"></a>

#### L119 · FVD · P1 · B

**[Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717)**  
Thomas Unterthiner等，2018。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**为兼顾视频视觉质量、时序连贯与样本变化提出Fréchet Video Distance，并给出StarCraft2视频基准及人评验证。
- **与本项目的关系：**若未来真的扩视频才适用；当前图像结果和图像FID不能充当视频FVD。
- **适用边界／版本注意：**必须匹配视频长度、帧率、分辨率、样本数和参考集；本项目没有视频实现/评价，不能填分或直接横比UCF101类别条件结果。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史评价计划；FVD原论文题录本次补齐，非本项目已完成视频评测。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)。

<a id="p-L120"></a>

#### L120 · LPIPS · P1 · N

**[The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924)**  
Richard Zhang等，2018。CVPR2018（arXiv comments）。

- **方法摘要：**收集人类感知相似判断，系统比较不同深层视觉特征与传统像素指标，建立学习式感知相似度评价基础。
- **与本项目的关系：**可量化修复前后感知保真及同prompt输出距离，与DreamSim/语义特征互补。
- **适用边界／版本注意：**感知差异不是偏好、语义正确率或多样性质量；坏图/噪声也可造成距离大，不能只优化距离来宣称有效多样性。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文/具体LPIPS校准。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L121"></a>

#### L121 · CLIP · P1 · B

**[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)**  
Alec Radford等，2021。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**从4亿网页文图对学习图文匹配，利用自然语言概念实现零样本视觉迁移。
- **与本项目的关系：**PickScore/HPS等偏好器与CMMD特征的技术背景，图文相似度可作对齐代理。
- **适用边界／版本注意：**预训练表示不是人类审美真值，也不保证数量/位置/属性绑定；不能将单余弦代理当全部生成质量。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**历史表示家族概念；原始论文题录本次补齐。历史记录：[.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md)。

<a id="p-L122"></a>

#### L122 · SigLIP · P1 · B

**[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)**  
Xiaohua Zhai等，2023。ICCV2023 oral（arXiv comments）。

- **方法摘要：**采用独立文图对的sigmoid匹配损失，不需softmax全局归一化，研究小/大batch效率并发布包括SO400M的模型。
- **与本项目的关系：**本项目视觉编码及文图匹配代理的基础；需区分作为模型组件和作为评测器两种角色。
- **适用边界／版本注意：**训练目标是文图匹配，不是审美、人类偏好或精细空间关系；与生成模型同源视觉组件可能不是完全独立评价。
- **本次阅读范围：**arXiv元数据和摘要/版本注释，未通读全文。
- **来源归属：**历史实际视觉组件与评分；正式论文题录本次补齐。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)。

<a id="p-L123"></a>

#### L123 · DINOv2 · P1 · N

**[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)**  
Maxime Oquab等，2023。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**通过大规模精选图像、自监督训练、扩展/稳定化和蒸馏产生跨任务、图像级与像素级通用特征。
- **与本项目的关系：**可补充无文本的视觉相似度、对象/构图覆盖与多样性特征，减少所有指标都用同一图文表征。
- **适用边界／版本注意：**本身是表示学习方法而非偏好/质量指标；选哪层/池化/核会改变结论，不能把特征距离直接当语义修复或审美。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L124"></a>

#### L124 · COCO · P2 · N

**[Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)**  
Tsung-Yi Lin等，2014。本次以arXiv题录为依据；正式发表信息未另核。

- **方法摘要：**以日常复杂场景和对象实例分割标注推动场景理解，提供大规模图像/对象实例和数据集划分。
- **与本项目的关系：**本项目真实图flow训练及潜在参考分布的数据来源背景；需记录实际使用的split、caption与512/64小子集。
- **适用边界／版本注意：**是数据集论文不是生成质量指标；小子集flow验证改善不保证终图改善，COCO参考分布与开放风格T2I用户分布不等价。
- **本次阅读范围：**arXiv元数据和摘要，未重新核验许可/数据文件。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L125"></a>

#### L125 · CLIPScore · P2 · N

**[CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://arxiv.org/abs/2104.08718)**  
Jack Hessel等，2021。EMNLP2021（arXiv journal reference）。

- **方法摘要：**将CLIP用于无需人工参考caption的图像描述兼容性评价，并提出结合参考的RefCLIPScore，分析上下文较复杂任务的弱点。
- **与本项目的关系：**说明图文兼容代理与偏好、细粒度组合判断各有范围，补充CLIP相似度评价的正式来源。
- **适用边界／版本注意：**原任务是image captioning，不应无说明当T2I整体生成质量指标；空间/计数/复杂上下文须另行评估。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L126"></a>

#### L126 · VQA v2 · P1 · N

**[Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://arxiv.org/abs/1612.00837)**  
Yash Goyal等，2017。CVPR2017官方出版页；arXiv首稿2016-12-02。

- **方法摘要：**用同一问题对应视觉相似但答案不同的互补图像平衡VQA，减少语言先验捷径，建立VQAv2.0。
- **与本项目的关系：**后续检测循环开启后真实图理解能力是否保真，而不是只看生成或Yes/No证据；固定QA模板、正常视觉输入与独立split。
- **适用边界／版本注意：**平衡数据减少但未消除捷径；不能由VQA成绩单独证明生成反向提升；训练时若使用COCO必须检查与VQA评价图像重叠。
- **本次阅读范围：**arXiv元数据/摘要与CVPR官方摘要，未通读全文。
- **来源归属：**本次补充的相关论文或明确版本。
- **补充原始来源：**[来源1](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html)。

<a id="p-L127"></a>

#### L127 · MME · P1 · N

**[MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://arxiv.org/abs/2306.13394)**  
Chaoyou Fu等，2023。arXiv2023首发；当前v5 comments标注NeurIPS DB2025 Spotlight，未额外独立核验出版页。

- **方法摘要：**人工设计指令/答案对，以简洁问题评价14个感知与认知子任务，减少prompt工程对不同MLLM比较的影响。
- **与本项目的关系：**评估外层循环在感知与认知多任务上的理解保真，而非只用少量空间反事实提示；应逐子任务/paired QA报告。
- **适用边界／版本注意：**基准提示/图像规模有限且有数据污染风险；其综合分不能当所有理解能力或真实偏好。后期生成教师池不可转为此独立评价集。
- **本次阅读范围：**arXiv元数据、摘要与版本comments，未通读全文/实际运行。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L128"></a>

#### L128 · POPE · P1 · N

**[Evaluating Object Hallucination in Large Vision-Language Models](https://arxiv.org/abs/2305.10355)**  
Yifan Li等，2023。EMNLP2023（arXiv comments核验）。

- **方法摘要：**研究LVLM对象幻觉及指令/共现偏置，提出polling-based对象查询POPE，以较稳定方式检查不存在对象被回答为存在。
- **与本项目的关系：**直接对应本项目MMU的Yes偏置与幻觉风险；循环前后用对象存在/不存在问题配对，单列false positives和正负问题表现。
- **适用边界／版本注意：**主要测对象幻觉而非空间方向、材质、全部认知或画质；应固定负例采样方式，不把POPE单项改善当统一模型双向协同。
- **本次阅读范围：**arXiv元数据和摘要，未通读全文/代码。
- **来源归属：**本次补充的相关论文或明确版本。

<a id="p-L129"></a>

#### L129 · Mask2Former · P1 · B

**[Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527)**  
Bowen Cheng等，2021。CVPR 2022；arXiv首版2021-12-02。

- **方法摘要：**用预测mask区域约束cross-attention，以统一架构处理全景、实例和语义分割。
- **与本项目的关系：**本项目官方GenEval检测链的重要组件背景；交接时应理解其检测/分割误差会影响对象级语义判定。
- **适用边界／版本注意：**检测器输出是自动评价证据，不是所有生成语义的最终真值；不能把其分数当审美或人类偏好。
- **本次阅读范围：**本次核验arXiv元数据和摘要；未通读全文或复现实验。
- **来源归属：**历史官方评价检测链组件；原论文题录本次补齐。历史记录：[.\技术报告-交接文档.md](./技术报告-交接文档.md)。

<a id="cat-engineering"></a>

### 低秩适配、条件调制与蒸馏基础

用于理解实际实现中的LoRA、FiLM和教师监督。它们提供工程方法背景，不单独构成共享循环生成的创新性或效果证据。

<a id="p-L130"></a>

#### L130 · LoRA · P1 · B

**[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**  
Edward J. Hu等，2021。arXiv原论文。

- **方法摘要：**冻结基础矩阵并训练低秩分解更新，减少可训练参数与优化状态。
- **与本项目的关系：**本项目q/k/v/o rank4/16适配与核心冻结对照的方法基础。
- **适用边界／版本注意：**训练参数少不等于原存储参数被删除；冻结LoRA参数须保留加载后的前向。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史实际适配方法；LoRA原论文题录本次补齐。历史记录：[.\技术报告-交接文档.md](./技术报告-交接文档.md)。

<a id="p-L131"></a>

#### L131 · FiLM · P1 · B

**[FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871)**  
Ethan Perez等，2017。AAAI 2018；首版2017。

- **方法摘要：**用条件生成逐特征仿射缩放和平移，调制视觉推理网络。
- **与本项目的关系：**早期MMU证据到token/通道gamma、beta调制的基础。
- **适用边界／版本注意：**插入共享Transformer后零FiLM不自动使整个额外块恒等。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史实际条件调制方法；FiLM原论文题录本次补齐。历史记录：[.\技术报告-交接文档.md](./技术报告-交接文档.md)。

<a id="p-L132"></a>

#### L132 · Knowledge Distillation · P2 · B

**[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)**  
Geoffrey Hinton等，2015。NIPS2014 Deep Learning Workshop；arXiv2015。

- **方法摘要：**把教师/集成的预测知识迁移到较小学生网络。
- **与本项目的关系：**理解ILSD、深预算教师与浅预算学生的公共蒸馏概念。
- **适用边界／版本注意：**普通蒸馏背景不等于本项目已复现ELT完整ILSD。
- **本次阅读范围：**原始题录与摘要。
- **来源归属：**历史使用教师/蒸馏概念；经典原论文题录本次补齐。历史记录：[.\技术报告-交接文档.md](./技术报告-交接文档.md)。

<a id="cat-archive"></a>

### 历史存档与未完整确认的研究线索

保留历史出现过但不能作为完整正式论文使用的资料，避免在后续交接中丢失线索或把不完整信息补写成确定事实。

<a id="p-L133"></a>

#### L133 · Fractal-Guided thesis · P2 · H

**[Fractal-Guided Approach for Image Generation with Diffus](https://ekmair.ukma.edu.ua/items/929a99dd-a89e-47d1-a7d8-1c7a068ae2f4)**  
Andrii Shalimov等，2026。2026硕士论文机构存档；题名不完整。

- **方法摘要：**机构摘要提出以条件DiT取代分形生成的AR模块，递归并行去噪patch；时间/分形深度条件小hypernetwork调制共享权重，保留不共享readout防表征坍缩。
- **与本项目的关系：**可作为递归访问条件与跨尺度设计线索，不能声称已经复现，也不能导入效率/质量数值。
- **适用边界／版本注意：**存档dc.title本身止于“Diffus”；不可擅补为Diffusion/Models。元数据subject含master’s thesis、type为Other。非同行评审会议论文。
- **本次阅读范围：**原机构页面搜索索引及dc元数据/摘要；直接网页429，未取到论文PDF。
- **来源归属：**历史文档有明确论文或作者项目来源。历史记录：[.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md)；[.\stage2b\METHOD_AND_EXPERIMENT_PLAN.md](./stage2b/METHOD_AND_EXPERIMENT_PLAN.md)；另有2份记录，见结构化目录。
- **补充原始来源：**[来源1](https://ekmair.ukma.edu.ua/items/929a99dd-a89e-47d1-a7d8-1c7a068ae2f4/full)、[来源2](https://ekmair.ukma.edu.ua/handle/123456789/40945)。


## 6. 如何将文献用于后续实验和报告

### 6.1 研究问题—论文—必要对照的对应

| 要解决的问题 | 最相关论文 | 对本项目应形成的检查 |
|---|---|---|
| 将独立深度替换为共享核心 | [ELT](#p-L026)、[Looped-DiT](#p-L027)、[PPCL](#p-L034) | 明确真正移除或保留哪些独立层；报告唯一存储参数、展开深度与完整调用，不能用可训练参数替代总参数。 |
| 让额外计算对困难状态有用 | [AdaDiff](#p-L031)、[E-DiT](#p-L035)、[PonderNet](#p-L041) | 若后续学习controller，应监督同状态额外计算的最终收益，并与固定预算比较；不能只预测当前错误或不确定性。 |
| 让理解证据改变flow过程 | [Dreaming in Flow / GGF](#p-L045)、[UniRect-CoT](#p-L060)、[LatentUM](#p-L046) | 明确证据进入条件、hidden、速度还是物理latent；使用正确、乱序和反向证据检验特异性。 |
| 让过程监督帮助最终生成 | [REPA](#p-L065)、[iREPA](#p-L067)、[RARE](#p-L071)、[RecA / RECA](#p-L073) | 分离教师来源、尺度和表征可恢复性；比较局部残差与完整终点，检查是否损伤原正确状态。 |
| 判断额外计算的最好用法 | [PreviewDiff](#p-L047)、[Diffusion Inference Scaling](#p-L076)、[ReNO](#p-L080)、[VHS](#p-L078) | 与更多候选、同样验证器调用和额外去噪／反传比较，计入preview解码、critic及搜索成本。 |
| 利用中间评价分配预算或训练信用 | [Progressive Seed Pruning / PSP](#p-L083)、[Stepwise-Flow-GRPO](#p-L104) | 与种子剪枝、分步reward增量等历史近邻比较；计入中间干净估计、解码、评分和额外子步。 |
| 优化偏好而保留多样性 | [Flow-GRPO](#p-L100)、[DRIFT](#p-L097)、[ParetoSlider](#p-L098)、[Reward-diversity / AIG](#p-L102) | 训练reward和独立judge分开；按固定提示多seed测有效多样性，并同时检查语义、画质与覆盖。 |
| 证明统一而非单任务变好 | [RealUnify](#p-L003)、[Unison](#p-L002)、[Understanding–Generation Transfer](#p-L062)、[MME-Unify](#p-L105) | 分别测理解→生成、生成→理解与原能力保留；不能以单一生成分数替代双向协同。 |
| 解释局部位置修复替代方案 | [Attend-and-Excite](#p-L084)、[Structured Diffusion](#p-L085)、[Attention Refocusing](#p-L086)、[GLIGEN](#p-L088) | 列明外部框、布局、mask和语言解析等信息，避免与原生MMU无额外标注的部署条件混比。 |
| 建立可靠生成评测 | [GenEval](#p-L117)、[GenEval 2](#p-L114)、[HPSv2](#p-L108)、[SigLIP](#p-L122)、[CMMD / Rethinking FID](#p-L112) | 固定judge、预处理、提示集和checkpoint；同时报告逐态结果、正确图保真与多指标差异。 |


### 6.2 哪些结论不能从“引用了这篇论文”直接推出

1. **引用ELT，不等于已经实现ELT式压缩。** 如果所有原独立层仍保留，额外共享块只是附加计算；要用实际参数清单和执行图说明。
2. **引用REPA或RARE，不等于任何hidden loss都应有效。** 外部图像教师、学生可恢复信息、状态尺度和教师动作语义需要各自验证。
3. **引用Flow-GRPO或MSDDA，不等于已经具有合法的flow RL目标。** 随机政策、转移分布和确定性ODE的差异需要推导或实现核对。
4. **引用GenEval，不等于完整运行了官方基准。** 本项目的开发提示子集、重复图、baseline别名、跨seed与独立测试应各自记录。
5. **引用HPS/PickScore，不等于人类偏好得到全面改善。** 被优化过的评分器、独立偏好代理和人评应分别报告。
6. **引用Vendi、LPIPS或DreamSim，不等于距离变大就是有效多样性。** 应先保证语义与质量，并固定特征、核、采样数和组内条件。
7. **引用RealUnify或RecA，不等于已证明双向提升。** 仍需独立理解评测、生成评测和交叉能力对照。

### 6.3 可供后续写作使用的Related Work组织

建议按研究问题组织，而非按年份堆论文：

- **统一表示与架构：** Transfusion、Show-o／Show-o2、BAGEL、JanusFlow、MetaQueries、UniTok／UniAR，说明理解与生成共享什么、分离什么。
- **共享深度与弹性计算：** Universal Transformer／ALBERT作为背景，ELT／Looped-DiT作为直接近邻，AdaDiff／DyDiT／E-DiT／LaTtE-Flow作为预算替代路线。
- **反馈、latent推理和协同：** Dreaming in Flow、LatentUM、UniRect-CoT、PreviewDiff、MILR、RecA及迁移研究，指出反馈作用点和部署信息的差异。
- **监督与对齐：** REPA系列、Self-Flow、奖励反传和flow RL，界定训练信号与终点目标。
- **评价与成本：** 语义、偏好、保真、多样性、分布、理解保留及完整系统成本，明确每类指标回答什么问题。

“区别”应写成可以被实验推翻的具体主张，例如同前向、同初始化、同数据及同目标时，仅允许共享核心更新是否有额外收益；不要只写“更紧密地融合”“更深层协同”。

## 7. 版本、撤回与旧记录名称的核验说明

| 条目／旧称 | 本次核验与处理 |
|---|---|
| AlignProp | [原编号2310.03739](https://arxiv.org/abs/2310.03739)明确为WITHDRAWN；v5日期2024-11-07。作者指向[2407.08737，Video Diffusion Alignment via Reward Gradients](https://arxiv.org/abs/2407.08737)。保留历史参考并醒目标记撤回；VADER另列，不能简单改写成同一篇未撤回论文。 |
| T2I-CompBench／CompBench++ | 同一[2307.06350](https://arxiv.org/abs/2307.06350)的题名和规模已变化；原2023版固定[v2](https://arxiv.org/abs/2307.06350v2)，扩展版固定[v3](https://arxiv.org/abs/2307.06350v3)。保留两个版本条目，不当作两个完全独立的首发。 |
| Looped Diffusion Language Models | 旧记录只有题名；本次找到[2605.26106](https://arxiv.org/abs/2605.26106)精确匹配。标为H?，不追认旧实验已参考其精确版本。ALoDLM和Looped Uniform DLM是本次补充。 |
| ELT／E-DiT | 前者为共享循环与弹性出口，后者为动态深度／宽度及缓存；名称相近但不是同一论文。 |
| Depth through recurrence | 当前已核验的v2日期为2026-10-21，晚于本项目10月3日的早期尝试；v2新增分析不能倒填为当时设计依据。 |
| DyDiT++ | 当前正式题名是“Diffusion Transformers with Timestep and Spatial Dynamics for Efficient Visual Generation”；旧仓库或早期名称可能不同。与原版DyDiT分别列出。 |
| BAGEL | 模型简称；论文正式题名为“Emerging Properties in Unified Multimodal Pretraining”。 |
| iREPA | 本目录使用正式题名“What matters for Representation Alignment: Global Information or Spatial Structure?”，而非把iREPA当完整题名。 |
| RARE | 对应“What Visual Generators Need from Teachers: Rethinking Representation Alignment”。本项目的固定子空间残差诊断不等于已实现该论文。 |
| RecA／RECA | 编号2509.07295；本次题录采用所查v4。历史作者项目摘要与修订版的数字不同，本目录不混用定量成绩。 |
| Diffusion Inference Scaling | 2501.09732的早期题名“Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps”与当前“Scaling Inference Time Compute for Diffusion Models”归为同一条，不重复计数。 |
| Progressive Seed Pruning | 历史链接是Caltech作者项目站，本次沿原站落实正式编号2607.21591。最初仅按论文域名提取会漏掉这类来源，现已补入。 |
| Stepwise Reward Assignment | 历史链接是CMU作者技术说明，本次落实正式论文“Stepwise Credit Assignment for GRPO on Flow-Matching Models”，编号2603.28718；不把博客标题当正式论文题名。 |
| Reversing the Flow | 原始注记为CVPR2026 Findings，不能写为CVPR主会。 |
| PonderNet | 所核对状态为ICML相关workshop，不写为ICML主会。 |
| Fractal-Guided存档 | 机构条目的dc.title本身只到“Fractal-Guided Approach for Image Generation with Diffus”；不能自行补齐尾词。为2026硕士论文存档，非已确认的同行评审会议论文。 |
| SigLIP、GenEval、FID/FVD、CLIP | 历史使用组件、评价工具或计划，不足以证明旧阶段已阅读正式原论文；以B标识。 |
| DINOv2、Grounding DINO | 旧记录的DINO家族概念不锁定DINOv2版本；本次列为具体补充。Grounding DINO也不替代实际官方检测链Mask2Former的记载。 |

部分正式页可能访问受限。只要作者原站、arXiv或正式出版索引能交叉确认，就不因一次403/429认定论文不存在；可访问范围与未读全文的限制已保留在条目中。没有用二手聚合摘要替代核心题录来源。

## 8. 交接后的阅读和核验顺序

### 8.1 接手实现的人

先读Show-o2、Flow Matching／Rectified Flow、ELT、Looped-DiT和LoRA，画出当前模型中“独立层、共享块、时间步、循环出口”的对应。再读LaTtE-Flow／RIN理解替代设计，最后看过程表示与反馈论文。阅读时同时记录：循环状态是什么、权重哪些共享、零输出如何建立、参考分支是否参与梯度、教师从哪里来、推理能访问哪些信息。

### 8.2 接手实验设计的人

先读Dreaming in Flow、UniRect-CoT、PreviewDiff、RecA和Understanding–Generation Transfer，列出所有可能解释收益的因素。然后依据第6节补足无循环／冻结核心／预算替代与证据扰动对照。原容量门和正确图保真门应在任何新生成评价之前固定；不因读到某篇正面论文就调整已有门槛。

### 8.3 接手评测的人

先读GenEval与GenEval2的测量边界，再读HPSv2／HPSv3、PickScore、SigLIP，确认每个槽位的模型与预处理。多样性按Vendi／DreamSim／LPIPS的定义分别设计；广域分布质量再考虑FID／CMMD和precision–recall。理解保留可参考VQA v2、MME、POPE和统一协同基准，但必须与训练图像／提示来源检查重叠。

### 8.4 接手论文写作的人

优先精读所有P0论文的方法与实验设置，补充逐页阅读笔记；对拟写入正文的每个具体性能数字、理论结论和最优性表述回到原文核验。对2026年新论文固定版本和日期，对同名缩写保留完整题名。本文的摘要级导读不能替代投稿前的全文精读。

可沿下列关键词继续检索：`looped diffusion transformer`、`elastic depth`、`weight tying visual generation`、`adaptive computation diffusion`、`unified multimodal understanding generation synergy`、`latent reasoning`、`reflective rectification`、`flow reward gradients`、`representation alignment teacher information`、`test-time compute diffusion latent verifier`。新增条目也应保留来源、版本、核验范围和与本项目的关系。

## 9. 本地来源与可持续使用的文件

以下文件路径均相对`outputs`。它们用于追溯研究意图和历史引用，不要求保留大体积原始实验结果目录。

| 文件 | 用途 |
|---|---|
| [.\hybrid_visual_generation_research_plan.md](./hybrid_visual_generation_research_plan.md) | 初始研究目标、文献地图和评价计划 |
| [.\LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md](./LOOPED_UNIFIED_VISUAL_GENERATION_FULL_PROGRESS_20261004.md) | 早期完整流程、循环压缩尝试和方向转换 |
| [.\stage2b\PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md](./stage2b/PRIOR_ART_AND_DECISIVE_EXPERIMENTS_20261003.md) | 强邻近工作与决定性对照设计 |
| [.\stage2b\METHOD_AND_EXPERIMENT_PLAN.md](./stage2b/METHOD_AND_EXPERIMENT_PLAN.md) | 方法及实验协议 |
| [.\stage2b\RESEARCH_DIRECTION.md](./stage2b/RESEARCH_DIRECTION.md) | 阶段性研究定位 |
| [.\stage2b\RECIPROCAL_LOOP_HYPOTHESIS.md](./stage2b/RECIPROCAL_LOOP_HYPOTHESIS.md) | 生成与理解互促假设 |
| [.\stage3_grounded_repair\STAGE3_RECURRENT_CORE_20261005.md](./stage3_grounded_repair/STAGE3_RECURRENT_CORE_20261005.md) | 第三阶段共享核心设计与边界 |
| [.\技术报告-交接文档.md](./技术报告-交接文档.md) | 实际尝试、结果与后续实验交接 |
| [.\TOPIC_LITERATURE_CATALOG_20261023.json](./TOPIC_LITERATURE_CATALOG_20261023.json) | 本文全部条目、原始URL、历史归属与出现位置的结构化索引 |
| [.\TOPIC_LITERATURE_VERIFICATION_20261023.json](./TOPIC_LITERATURE_VERIFICATION_20261023.json) | 本版条目数、覆盖核验、文件校验值及已知版本关系 |

## 10. 按简称检索

- [ACT](#p-L040)：L040，P2，N。
- [AD-Loop](#p-L057)：L057，P1，H。
- [AdaDiff](#p-L031)：L031，P1，H。
- [ALBERT](#p-L038)：L038，P2，N。
- [AlignProp](#p-L095)：L095，P1，H。
- [ALoDLM](#p-L043)：L043，P2，N。
- [Ask, Solve, Generate](#p-L053)：L053，P1，H。
- [Attend-and-Excite](#p-L084)：L084，P0，H。
- [Attention Refocusing](#p-L086)：L086，P1，H。
- [BAGEL](#p-L016)：L016，P1，H。
- [BLIP3-o](#p-L017)：L017，P1，H。
- [CFG](#p-L012)：L012，P1，N。
- [Chameleon](#p-L019)：L019，P2，N。
- [Classical Search Diffusion](#p-L077)：L077，P1，N。
- [CLIP](#p-L121)：L121，P1，B。
- [CLIPScore](#p-L125)：L125，P2，N。
- [CMMD / Rethinking FID](#p-L112)：L112，P1，H。
- [COCO](#p-L124)：L124，P2，N。
- [Consistency Models](#p-L075)：L075，P2，N。
- [ControlNet](#p-L089)：L089，P1，N。
- [D-Flow](#p-L081)：L081，P1，N。
- [DDIM](#p-L011)：L011，P2，N。
- [DDPM](#p-L010)：L010，P2，N。
- [DDPO](#p-L103)：L103，P1，N。
- [Depth through recurrence / Looped flow-TTS](#p-L028)：L028，P1，H。
- [DEQ](#p-L039)：L039，P2，N。
- [Diffusion Inference Scaling](#p-L076)：L076，P0，N。
- [Diffusion-DPO](#p-L094)：L094，P1，H。
- [DINOv2](#p-L123)：L123，P1，N。
- [DiT](#p-L006)：L006，P0，N。
- [DOODL](#p-L082)：L082，P1，N。
- [Dreaming in Flow / GGF](#p-L045)：L045，P0，H。
- [DreamSim](#p-L111)：L111，P1，H。
- [DRIFT](#p-L097)：L097，P1，H。
- [DyDiT++](#p-L033)：L033，P1，H。
- [DyDiT（原版）](#p-L036)：L036，P1，N。
- [E-DiT](#p-L035)：L035，P1，H。
- [ELT](#p-L026)：L026，P0，H。
- [Emu3](#p-L023)：L023，P2，N。
- [FID / TTUR](#p-L118)：L118，P1，B。
- [FiLM](#p-L131)：L131，P1，B。
- [Flow Matching](#p-L004)：L004，P0，N。
- [Flow-GRPO](#p-L100)：L100，P0，H。
- [FoX / MCoT](#p-L050)：L050，P1，H。
- [Fractal-Guided thesis](#p-L133)：L133，P2，H。
- [FVD](#p-L119)：L119，P1，B。
- [GenEval](#p-L117)：L117，P0，B。
- [GenEval 2](#p-L114)：L114，P0，H。
- [GLIGEN](#p-L088)：L088，P1，N。
- [Grounding DINO](#p-L092)：L092，P2，N。
- [HPSv2](#p-L108)：L108，P0，H。
- [HPSv3 / CoHP](#p-L109)：L109，P1，H。
- [ImageReFL](#p-L101)：L101，P1，H。
- [ImageReward](#p-L106)：L106，P1，H。
- [Improved Precision and Recall](#p-L113)：L113，P1，H。
- [InstructPix2Pix](#p-L091)：L091，P2，N。
- [iREPA](#p-L067)：L067，P0，H。
- [Janus](#p-L020)：L020，P1，N。
- [Janus-Pro](#p-L022)：L022，P2，N。
- [JanusFlow](#p-L021)：L021，P1，N。
- [Knowledge Distillation](#p-L132)：L132，P2，B。
- [Latent Diffusion](#p-L009)：L009，P1，N。
- [LatentUM](#p-L046)：L046，P0，H。
- [LaTtE-Flow](#p-L032)：L032，P0，H。
- [Looped Diffusion Language Models / LoopMDM](#p-L042)：L042，P2，H?。
- [Looped Uniform DLM](#p-L044)：L044，P2，N。
- [Looped-DiT](#p-L027)：L027，P0，H。
- [LoopVAE](#p-L029)：L029，P1，H。
- [LoRA](#p-L130)：L130，P1，B。
- [LPIPS](#p-L120)：L120，P1，N。
- [Mask2Former](#p-L129)：L129，P1，B。
- [MetaQueries](#p-L018)：L018，P1，H。
- [MILR](#p-L051)：L051，P1，H。
- [MME](#p-L127)：L127，P1，N。
- [MME-Unify](#p-L105)：L105，P1，N。
- [MSDDA](#p-L099)：L099，P1，H。
- [ParetoSlider](#p-L098)：L098，P1，H。
- [Pick-a-Pic / PickScore](#p-L107)：L107，P1，H。
- [PonderNet](#p-L041)：L041，P2，N。
- [POPE](#p-L128)：L128，P1，N。
- [PPCL](#p-L034)：L034，P1，H。
- [PreviewDiff](#p-L047)：L047，P0，H。
- [Progressive Seed Pruning / PSP](#p-L083)：L083，P1，H。
- [Prompt-to-Prompt](#p-L087)：L087，P2，N。
- [R3-Bench / R3-Refiner](#p-L048)：L048，P1，H。
- [RAE](#p-L068)：L068，P1，H。
- [RARE](#p-L071)：L071，P0，H。
- [RealUnify](#p-L003)：L003，P0，H。
- [RecA / RECA](#p-L073)：L073，P0，H。
- [Rectified Flow](#p-L005)：L005，P1，N。
- [Rectified Flow Transformers](#p-L008)：L008，P1，N。
- [Reflexion](#p-L064)：L064，P2，N。
- [ReNO](#p-L080)：L080，P1，N。
- [REPA](#p-L065)：L065，P0，H。
- [REPA-E](#p-L066)：L066，P1，H。
- [Reversing the Flow](#p-L059)：L059，P1，H。
- [Reward-diversity / AIG](#p-L102)：L102，P1，H。
- [RIN](#p-L030)：L030，P1，H。
- [SAM](#p-L093)：L093，P2，N。
- [SDEdit](#p-L090)：L090，P1，N。
- [SEED-X](#p-L024)：L024，P2，N。
- [Self-Flow](#p-L070)：L070，P1，H。
- [Self-Refine](#p-L063)：L063，P2，N。
- [Show-o](#p-L013)：L013，P0，H。
- [Show-o2](#p-L014)：L014，P0，H。
- [SigLIP](#p-L122)：L122，P1，B。
- [SiT](#p-L007)：L007，P1，N。
- [SRRL](#p-L055)：L055，P1，H。
- [SRUM](#p-L052)：L052，P1，H。
- [Stepwise-Flow-GRPO](#p-L104)：L104，P1，H。
- [Structured Diffusion](#p-L085)：L085，P1，H。
- [T2I RAE](#p-L069)：L069，P1，H。
- [T2I-CompBench (conference version)](#p-L115)：L115，P1，H。
- [T2I-CompBench++](#p-L116)：L116，P1，N。
- [Transfusion](#p-L015)：L015，P0，H。
- [Turning Internal Gap](#p-L058)：L058，P1，H。
- [Understanding–Generation Transfer](#p-L062)：L062，P0，H。
- [UniAR](#p-L025)：L025，P2，N。
- [UniCorn](#p-L054)：L054，P1，H。
- [UniEdit-I](#p-L056)：L056，P1，H。
- [Unified MLLM Survey](#p-L001)：L001，P1，N。
- [Unified Sparsity](#p-L074)：L074，P1，H。
- [UniGen / CoT-V](#p-L049)：L049，P1，H。
- [UniRect-CoT](#p-L060)：L060，P0，H。
- [Unison](#p-L002)：L002，P0，H。
- [UniT](#p-L061)：L061，P1，H。
- [UniTok](#p-L072)：L072，P2，N。
- [Universal Transformer](#p-L037)：L037，P2，N。
- [VADER / AlignProp后续](#p-L096)：L096，P1，N。
- [Vendi Score](#p-L110)：L110，P1，H。
- [VFScale](#p-L079)：L079，P2，N。
- [VHS](#p-L078)：L078，P0，N。
- [VQA v2](#p-L126)：L126，P1，N。
