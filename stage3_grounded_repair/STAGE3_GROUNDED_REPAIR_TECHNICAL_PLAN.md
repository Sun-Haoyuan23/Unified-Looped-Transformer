# 阶段3完整技术方案：基于理解证据的共享循环修复

## 1. 阶段目标

下一阶段不再以“继续降低单步 flow MSE”“继续搜索共享位置”“继续调整循环触发阈值”为主线，而将研究问题收敛为：

> **统一多模态生成模型能否先利用自身理解通路识别当前生成状态中的具体错误，再通过条件化的共享视觉块重访，对同一中途状态实施具有明确方向的修复，并使该修复在最终图像上产生可重复、可独立验证的收益？**

现阶段已经有三项关键事实：

1. 中途反事实关系证据具有初步可用性，在约 \(t=0.345\) 和 \(t=0.455\) 处已经表现出较强的错误排序能力。
2. 现有“发现错误后增加循环”的方法没有产生稳定修复；generic recurrence 本身尚未被证明具有纠错能力。
3. 单步 flow MSE 的改善已经出现与终图质量相反的趋势，因此局部预测误差不能再作为主要优化目标。

因此下一阶段核心不再是训练一个更准确的“错误检测器”，而是建立：

\[
\boxed{
\text{Error Evidence}
\rightarrow
\text{Repair Action}
\rightarrow
\text{Terminal Improvement}
}
\]

只有当这条因果链成立之后，才重新加入动态控制、参数压缩和 FLOPs 优化。

---

## 2. 核心研究假设

下一阶段明确检验以下四个假设。

### H1：在相同中途状态上，存在能够提高最终结果的修复动作

对于固定生成状态 \(x_t\)，不同 continuation：

\[
x_t\rightarrow a_i\rightarrow I_i
\]

可能具有不同的最终收益：

\[
A(x_t,a_i)
=
R(I_i)-R(I_{\text{base}})
\]

如果对基线已经生成错误的样本，任何合理候选动作都无法稳定产生 \(A>0\)，那么当前问题不是 controller 不够准确，而是 repair action 本身不存在或接口错误。

因此 H1 必须先于动态控制器验证。

### H2：错误证据必须进入第二次共享计算，而不能只决定“算不算第二次”

现有方法近似为：

\[
e(x_t)<0
\Rightarrow
h''=F_\theta(h')
\]

其中 evidence 只负责触发，而第二次共享块并不知道：

- 错误类型；
- 涉及对象；
- 目标关系；
- 相反关系；
- 置信程度；
- 当前时间位置。

下一阶段改为：

\[
h''
=
F_\theta
\left(
h',
C(e,x_t,t)
\right)
\]

其中 Transformer 主权重仍然共享，但第二次访问加入小型条件调制模块。

### H3：训练目标应主要监督最终 continuation 收益，而不是局部速度误差

局部目标：

\[
\|\hat v_t-v_t^{teacher}\|^2
\]

保留为稳定项，但不再作为决定模型优劣的主要目标。

核心监督改为：

\[
\Delta R
=
R(I_{\text{repair}})
-
R(I_{\text{base}})
\]

以及：

\[
P
\left(
I_{\text{repair}}
\succ
I_{\text{base}}
\right)
\]

模型必须学会“这次更新是否最终有用”，而不是只学习“局部向量是否更像教师”。

### H4：只有固定 repair action 被证明有效后，动态 controller 才具有研究意义

最终 controller 的目标不是：

\[
P(\text{current image is wrong})
\]

而应该接近：

\[
Q(x_t,e,a)
=
\mathbb E[
\Delta R-\lambda C
]
\]

即预测：

> 在当前状态和当前错误证据下，执行某一种修复动作最终能改善多少，以及是否值得其计算成本。

---

## 3. 总体技术架构

下一阶段工作名建议使用：

**GroundedFlow-R：Evidence-Conditioned Shared Repair Recurrence**

整个系统拆成四部分：

1. State Capture；
2. Evidence Reader；
3. Repair Operator；
4. Advantage Controller。

前两项已有基础，下一阶段主要重新设计第 3 项，最后再训练第 4 项。

---

## 4. State Capture：严格保存同状态分叉点

在选定 flow timestep \(t\)：

1. 保存当前 latent \(x_t\)；
2. 保存当前 Transformer prefix cache；
3. 保存当前 RNG、CFG 配置、时间网格索引和 solver 状态；
4. 固定后续采样代码路径；
5. 从完全相同状态产生多个 continuation。

必须保证：

\[
x_t^{A}=x_t^{B}
\]

并保证 intervention=0 时：

\[
I_A=I_B
\]

达到逐像素一致。

这一步作为所有下一阶段实验的强制单元测试。

---

## 5. 第一阶段：Repairability Oracle

这是下一阶段最重要的实验。

在这个实验完成之前，不训练新的动态 controller。

### 5.1 目标

回答：

> **如果我们已经知道当前生成结果可能是错的，现有模型附近到底有没有能够改善最终图像的 continuation？**

### 5.2 状态选择

首轮只使用已经有较好证据校准结果的位置：

\[
t_1\approx0.345
\]

和：

\[
t_2\approx0.455
\]

暂时不继续扫描更多 timestep。

### 5.3 每个状态生成的候选 continuation

对完全相同的 \(x_t\)，至少生成：

#### A0：Baseline continuation

不进行任何额外操作：

\[
a_0=\text{normal continuation}
\]

#### A1：Fixed shared revisit

执行当前已有的共享 block 第二次访问。

用于判断：

> generic recurrence 本身是否存在收益。

#### A2：Extra ODE control

不做内部共享循环，而增加近似等计算量的普通 ODE / velocity evaluation。

用于区分：

> 收益来自“递归结构”还是单纯来自“更多计算”。

#### A3：Evidence-conditioned modulation

加入后续定义的证据条件化 repair adapter。

#### A4：Random / mismatched evidence

保持 A3 架构不变，但：

- 随机打乱 evidence；
- 使用其他样本 evidence；
- 使用反事实错误 evidence。

用于验证：

\[
\text{真实 evidence}
>
\text{无关 evidence}
\]

而不是“小模块本身改善结果”。

---

## 6. 第二阶段：Evidence Representation

当前 evidence 主要是标量：

\[
e
=
m(q_{target})
-
m(q_{opposite})
\]

下一阶段保留该标量作为最低配版本，但逐步构造更有方向性的修复条件。

### 6.1 Level 0：Scalar evidence

输入：

\[
c_0=
[
e,\,
t
]
\]

用于验证：

> 即使只知道错误程度，条件 modulation 是否已经优于无条件重复计算。

### 6.2 Level 1：双命题 evidence

不只保存差值，而保存：

\[
c_1=
[
m_{target},
m_{opposite},
m_{target}-m_{opposite},
t
]
\]

避免一个标量丢失：

- target 和 opposite 都低；
- target 和 opposite 都高；
- 两者差相同但不确定性不同；

等状态。

### 6.3 Level 2：Semantic evidence embedding

从理解通路提取对应问题在 MMU 中的 hidden representation：

\[
z_{target}
\]

和：

\[
z_{opposite}
\]

构造：

\[
c_2
=
P[
z_{target};
z_{opposite};
z_{target}-z_{opposite};
t
]
\]

其中 \(P\) 为小型 projection。

该表示开始真正包含：

- object identity；
- relation identity；
- relation polarity；

而不是只有“对或错”的标量。

---

## 7. 第三阶段：Evidence-Conditioned Repair Operator

保持核心研究约束：

> **视觉 Transformer 的主块权重继续共享。**

不允许通过重新加入完整独立 block 偷回原网络容量。

但允许第二次访问存在少量条件化参数，用于恢复：

1. visit identity；
2. error identity；
3. timestep identity。

### 7.1 推荐首选结构：Conditional AdaLN / FiLM

设共享块输入为 \(h\)，evidence embedding 为 \(c\)。

生成：

\[
[\gamma,\beta]
=
MLP(c)
\]

第二次访问前：

\[
\tilde h
=
(1+\gamma)\odot LN(h)+\beta
\]

然后：

\[
h_{repair}
=
F_\theta(\tilde h)
\]

其中：

- 第一次访问仍走原路径；
- 第二次访问使用相同 \(F_\theta\)；
- \(MLP(c)\) 零初始化；
- 初始模型严格恢复 baseline。

这是优先级最高的方案，因为它参数少、容易验证零初始化，同时 evidence 可以直接改变共享块的行为。

### 7.2 备选：Conditional LoRA

第二次访问：

\[
W_{repair}
=
W+
g(c)BA
\]

其中：

- \(W\) 为共享主权重；
- \(BA\) 为低秩残差；
- \(g(c)\) 根据错误证据产生动态强度。

与之前固定 rank residual 不同，本方案的第二次访问残差由 semantic evidence 调制。

### 7.3 禁止同时修改第一次访问

第一阶段 repair 训练期间：

\[
F_\theta^{first}
\]

完全冻结。

只训练：

- evidence projection；
- 第二次访问 modulation；
- 第二次访问 low-rank residual。

必须尽量保证：

\[
I_{\text{short,new}}
\approx
I_{\text{short,baseline}}
\]

理想情况下逐像素完全一致。

---

## 8. 第四阶段：Terminal-Outcome Training

这一阶段需要改变训练数据单位。

过去主要训练：

\[
(x_t,v_t)
\]

下一阶段数据单位改为：

\[
D=
(
x_t,
c_t,
a,
I_{final},
R_{final},
\Delta R
)
\]

### 8.1 建立 paired continuation 数据

对每个保存状态：

\[
s_t=(x_t,c_t)
\]

执行不同 action：

\[
a_0,a_1,\ldots,a_K
\]

得到：

\[
I_0,I_1,\ldots,I_K
\]

然后统一计算最终指标。

至少包含：

- GenEval / 对应任务判定；
- HPSv2；
- PickScore；
- SigLIP；
- intervention cost。

形成：

\[
\Delta R_k
=
R(I_k)-R(I_0)
\]

---

## 9. Reward 设计

下一阶段不能再用单一 PickScore 作为 repair reward。

建议定义分层评价。

### 9.1 第一优先级：目标语义是否修复

对于空间实验：

\[
R_{semantic}
\]

来自最终官方语义判定。

关注四种结果：

| Baseline | Repair | 类型 |
|---|---|---|
| 错 | 对 | Positive repair |
| 错 | 错 | No repair |
| 对 | 对 | Preserved |
| 对 | 错 | Regression |

真正需要最大化：

\[
P(\text{wrong}\rightarrow\text{correct})
\]

并最小化：

\[
P(\text{correct}\rightarrow\text{wrong})
\]

### 9.2 第二优先级：感知与偏好守门

语义翻正不能以明显画质下降换取。

因此增加：

\[
R_{quality}
\]

至少使用：

- HPSv2；
- PickScore。

如果：

\[
\Delta R_{semantic}>0
\]

但：

\[
\Delta HPS \ll 0
\]

则不能计为完整成功，只能计为“semantic repair with quality cost”。

### 9.3 综合 reward

研究阶段可以定义：

\[
R
=
w_sR_{semantic}
+
w_hR_{HPS}
+
w_pR_{Pick}
-
\lambda C
\]

但论文最终应分别报告各项指标，而不是只报告这个人工组合分数。

---

## 10. 训练目标

建议最终训练目标由四部分构成：

\[
L
=
\lambda_1 L_{terminal}
+
\lambda_2 L_{rank}
+
\lambda_3 L_{preserve}
+
\lambda_4 L_{local}
\]

### 10.1 Terminal loss

训练 repair 后的 rollout 获得更高最终奖励。

如果使用可微代理：

\[
L_{terminal}
=
-R(I_{repair})
\]

可以继续使用 truncated rollout 和 gradient checkpointing。

但本轮 reward 必须：

1. 条件化于明确错误 evidence；
2. 只更新 second-visit repair parameters；
3. 与 matched zero-intervention baseline 成对计算。

### 10.2 Pairwise advantage loss

设：

\[
\Delta R
=
R_{repair}-R_{base}
\]

训练：

\[
\hat A_\phi(x_t,c_t,a)
\]

预测真实 advantage：

\[
L_{adv}
=
(\hat A-\Delta R)^2
\]

或者直接训练 pairwise ranking：

\[
P_\phi(
I_{repair}>I_{base}
)
\]

### 10.3 Preservation loss

对基线已经正确或 evidence 表示无需修复的样本：

\[
L_{preserve}
=
\|
\Delta h_{repair}
\|^2
\]

或使用：

\[
\|
I_{repair}-I_{base}
\|
\]

的轻量 latent / trajectory proxy。

核心目的是：

> 没有明确收益时，模型应该学会接近 zero update。

### 10.4 Local teacher loss

原有 velocity / representation loss 保留：

\[
L_{local}
=
\|\hat v-v^{teacher}\|^2
\]

但只作为 regularizer，不能再按该指标选择 checkpoint。

Checkpoint selection 必须使用 terminal paired evaluation。

---

## 11. 两步训练策略

### Phase B1：只训练 Repairer

固定：

- observer；
- base model；
- controller；
- first visit。

所有样本固定执行 second visit。

目的是回答：

> evidence-conditioned repairer 本身能不能工作？

此时不要加入动态触发。

### Phase B2：再训练 Controller

只有 B1 出现稳定的：

\[
wrong\rightarrow correct
\]

后，才训练 controller。

Controller 输入：

\[
(x_t,c_t,t)
\]

输出：

\[
\hat A_{repair}
\]

执行规则：

\[
\text{repair}
\iff
\hat A_{repair}
>
\lambda C_{repair}
\]

这比当前：

\[
e<0\Rightarrow repair
\]

更符合真正目标。

---

## 12. 数据划分

下一阶段必须彻底解决开发集重复使用问题。

建议按 prompt 进行独立划分，而不是按图随机划分。

### 12.1 Repair Train

新增空间关系及后续其他关系提示。

建议规模：

- 100–200 个训练 prompt；
- 每 prompt 2–4 个固定 seed。

主要用于生成 paired continuation。

### 12.2 Repair Validation

完全独立提示：

- 30–50 个 prompt；
- 用于 checkpoint selection。

不得参与 architecture selection。

### 12.3 Mechanism Test

冻结之后再使用：

- 独立空间 prompt；
- 后续完整 GenEval 子集。

现有 12 图空间集合继续作为历史开发机制集，不再把它当成独立 test。

现有 14 图 pilot 同样不得继续承担最终统计验证。

---

## 13. Repairability Oracle 的 Go / No-Go 门槛

在正式训练新模型之前，先完成 oracle experiment。

### Gate A：candidate action 必须存在真实 positive advantage

在 baseline-error 样本中，至少观察到一批：

\[
wrong\rightarrow correct
\]

而不仅仅是自动代理小幅提高。

### Gate B：oracle selection 必须明显优于固定策略

计算：

\[
R_{oracle}
=
\max_a R(a)
\]

若：

\[
R_{oracle}
\approx
R_{baseline}
\]

说明当前 action space 本身缺乏修复能力。

此时应更换 repair interface，而不是训练 controller。

### Gate C：正确样本不能大量被破坏

要求：

\[
P(correct\rightarrow wrong)
\]

明显低于：

\[
P(wrong\rightarrow correct)
\]

否则只是随机扰动。

---

## 14. 必做消融实验

一旦 evidence-conditioned repairer 有正结果，必须同时完成以下对照。

### 14.1 Evidence 消融

- 无 evidence；
- 只有 scalar evidence；
- target/opposite margins；
- MMU hidden evidence；
- shuffled evidence；
- opposite evidence。

目标：

\[
R(real\ evidence)
>
R(shuffled)
\]

### 14.2 Architecture 消融

- 无 second visit；
- generic second visit；
- Conditional AdaLN；
- Conditional LoRA；
- AdaLN + LoRA。

### 14.3 Compute 消融

- extra shared recurrence；
- extra ODE step；
- 相同视觉块调用预算；
- 相同整网 FLOPs 预算。

### 14.4 时间消融

只在 repairer 已经有效之后比较：

- \(t\approx0.345\)；
- \(t\approx0.455\)；
- 双时间点。

不再进行大量 timestep sweep。

---

## 15. 工程实现建议

建议新增独立模块，而不是继续把功能全部塞入已有主脚本。

```text
stage2c/
    capture_repair_states.py
    build_repair_candidates.py
    evidence_encoder.py
    evidence_conditioned_revisit.py
    train_repair_operator.py
    evaluate_repair_pairwise.py
    train_advantage_controller.py
    run_grounded_repair.py
    profile_grounded_repair.py
```

### 15.1 `capture_repair_states.py`

每个状态保存：

```text
prompt_id
prompt
seed
flow_index
true_t
latent
prefix_cache/version
cfg
baseline_prediction
target_question
opposite_question
target_margin
opposite_margin
evidence
baseline_final_image
baseline_geneval
baseline_hps
baseline_pickscore
```

### 15.2 `build_repair_candidates.py`

从相同状态分叉：

```text
baseline
generic_revisit
extra_ode
conditional_revisit
shuffled_evidence
opposite_evidence
```

并输出统一 metadata。

### 15.3 `evaluate_repair_pairwise.py`

必须逐样本输出：

```text
baseline_correct
candidate_correct
semantic_flip
pick_delta
hps_delta
siglip_delta
compute_delta
```

最终自动分类：

```text
positive_repair
negative_flip
quality_only_gain
semantic_gain_quality_loss
unchanged
```

---

## 16. 实验日志规范

以后每一个实验必须至少包含：

```text
experiment_id
git_commit
checkpoint
base_model_hash
prompt_split
seed_list
sampler
time_grid
flow_index
true_t
CFG
first_visit_config
second_visit_config
evidence_type
repair_action
controller
training_objective
trainable_parameters
unique_parameters
FLOPs
wall_time
raw_per_sample_metrics
zero_intervention_baseline
```

所有 paired experiment 在正式评估前自动运行：

\[
\text{zero intervention pixel diff test}
\]

不通过则整个实验作废。

---

## 17. 参数和 FLOPs 策略

下一阶段第一轮允许计算增加。

原因是当前最优先的问题是证明：

\[
\text{repair exists}
\]

而不是证明：

\[
\text{repair is already cheap}
\]

因此效率优化放到 repair mechanism 成立之后。

最终再研究：

1. MMU evidence cache；
2. 单次检查；
3. cheaper evidence head；
4. short-path majority；
5. condition-triggered repair；
6. 外层 flow step reduction；
7. repair 与 ODE budget 重新分配。

---

## 18. 暂停继续投入的路线

现阶段暂停以下搜索：

- COCO flow matching 继续增加训练步数；
- 单纯 velocity imitation；
- 单纯 ILSD 权重扫描；
- rank-256/512/1024 大规模扫描；
- 更多 block replacement position sweep；
- MMU threshold 微调；
- 大量 timestep sweep；
- 继续在 14 图 pilot 上做 checkpoint selection；
- repairer 尚未有效前的大规模 GenEval。

除非新实验直接需要这些变量作为对照，否则不继续投入。

---

## 19. 决策树

下一阶段严格按照以下顺序执行。

### Step 1

完成目前两个待确认实验：

- `coco_single_revisit400_pilot`
- `svd_single_fixed_position12`

只用于关闭历史实验，不据此继续扩展旧路线。

### Step 2

建立完全 matched 的 state-forking pipeline。

验收：

\[
intervention=0
\Rightarrow
pixel\ exact
\]

### Step 3

运行 Repairability Oracle。

如果：

\[
Oracle\not>Baseline
\]

停止 controller 工作，进入 repair interface redesign。

### Step 4

如果 Oracle 显示存在 positive repair candidate：

训练 evidence-conditioned repairer。

首先固定所有样本执行 repair。

### Step 5

如果 repairer 在独立 validation 上出现：

\[
wrong\rightarrow correct
\]

并且 HPSv2 / PickScore 没有系统性下降：

扩大到独立机制测试。

### Step 6

只有完成 Step 5 后：

训练 advantage controller。

### Step 7

最后才进入：

\[
Quality
-
Compute
-
Parameters
\]

联合 Pareto 优化。

---

## 20. 下一阶段的关键成功标准

下一阶段不要求立刻达到完整论文最终指标。

真正的第一个里程碑只有一个：

\[
\boxed{
\text{在完全相同的中途生成状态上，}
\text{真实理解证据能够使共享第二次访问}
\text{比无证据或错误证据更频繁地产生最终正确修复。}
}
\]

具体必须同时观察：

1. baseline wrong → repair correct；
2. real evidence > shuffled evidence；
3. real evidence > generic recurrence；
4. correct baseline 大部分保持正确；
5. 独立偏好代理没有明显系统性下降；
6. matched zero-intervention baseline 有效。

一旦这个现象成立，项目主线就会从：

> “循环模型是否可以节约参数”

升级为：

> **“统一多模态模型能够在生成过程中读取自身理解状态，并用该证据改变共享计算的下一次作用方式，实现生成中的自我纠错。”**

---

## 21. 后续完整论文路径

若上述机制成立，再依次扩展。

### Stage D：从空间关系扩展到错误类型

包括：

- 数量；
- 属性；
- 颜色绑定；
- 空间；
- object presence。

研究不同错误是否需要不同 evidence representation。

### Stage E：动态计算

训练：

\[
Q(x,e,a)
\]

决定：

- skip；
- short；
- repair；
- repair depth。

### Stage F：效率恢复

再优化：

- 唯一参数；
- 平均 FLOPs；
- 实际时延；
- 显存。

### Stage G：完整评价

最后才执行：

- 完整 GenEval；
- GenEval 2；
- 大样本 PickScore/HPSv2；
- 人类盲评；
- FID/CMMD；
- 有效多样性；
- 真实图像理解基准；
- 参数、FLOPs、时延联合曲线。

---

## 22. 本阶段最重要的原则

整个下一阶段需要坚持一句话：

> **不要继续训练模型“更像教师走一步”，而要训练它“在知道自己哪里可能错的情况下，使最终结果真正变好”。**

因此评价单位从：

\[
\text{single-step prediction}
\]

改成：

\[
\text{same-state paired continuation}
\]

训练单位从：

\[
(x_t,v_t)
\]

逐步改成：

\[
(x_t,e,a,\Delta R)
\]

而共享循环也从：

\[
\text{repeat the same computation}
\]

升级为：

\[
\boxed{
\text{reuse the same computation, conditioned on what the model understands is wrong}
}
\]

这应当成为下一阶段所有实验设计、实现和筛选的统一主线。
