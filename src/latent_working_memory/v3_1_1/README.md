# 20261102_压缩预训练与 QA 后训练

创建时间：20261102 16:06:43 Asia/Shanghai

最后修订时间：20261125 10:58:32 Asia/Shanghai

v3.1.1 的 `stage=pretrain` 使用文本重建／续写（AE／LM）；`stage=post` 使用问答（QA），由 `training.update_mode` 选择仅更新压缩器（`compressor_only`）、仅更新边界（`boundary_only`）或联合更新（`joint`）。每次写入将一段正文压缩为固定 K 个记忆槽位（memory slots），已提交的记忆块保持不变；读取器（Reader）根据这些记忆及未压缩的 pending 和 recent 尾段回答问题。

本文说明方法、配置和执行协议；完整 CLI 参数与各任务命令见[运行指南](scripts/README.md)。

[方法概览](#方法概览) · [数据准备与课程](#数据准备与课程) · [配置与运行](#配置与运行) · [验证、独立评估与记录](#验证独立评估与记录) · [实现细节](#实现细节) · [代码与产物](#代码与产物)

## 方法概览

### 记忆与边界

| 符号 | 配置 | 含义 |
|---|---|---|
| K | `model.structure.memory_slots` | 每次提交生成的记忆槽位数 |
| L | `model.structure.pending_stride` | 每次读入的 token 步长，决定边界判断间隔 |
| H | `model.structure.pending_size` | 允许压缩的最短 pending 长度，至少为 K |
| R | `model.structure.recent_size` | 方法默认保留的尾部 token 数 |
| G | `training.objective.group_size` | 每篇文档的随机分段路径数 |

`joint` 和 `boundary_only` 使用 `compression.mode=policy`。先预留正文末尾固定 recent，其余前缀按 L 分成处理单元。在线状态由历史 memory、待提交原文 a 和新到达单元 b 组成。边界模块读取这些已到达的内容，选择继续积累 ab，或将 a 压缩为 K 个 slots，再以 b 开始下一块。a 小于 H 时强制继续；达到 H 后允许策略判断，不限制最大段长。最后一个不足 L 的单元也按实际长度读入，固定 recent 不参与判断。

前缀读取完毕后，若剩余 pending 长度至少为 H，则一次压缩为 K 个 slots；不足 H 则保留原文。收尾由规则决定，不计策略动作。`recent_size=0` 只表示不预留 recent，最终仍可能保留短 pending。

预训练的默认 recent 长度为 0，后训练取 `model.structure.recent_size`。multi／policy 的 `training.compression.recent_size` 只覆盖训练，`null` 沿用默认长度，0 不预留，正整数指定训练长度。训练中的 dev 和独立评估均使用默认长度。single 的 `compression` 仅含 `mode`，在训练和验证中都强制 recent 为 0。

QA 后训练、dev、独立 test、LoCoMo 和 RULER 共用文档问答 system 消息。Writer 只接收正文。Reader 依次读取原文 system 消息、user 角色开头、历史记忆块、未压缩的 pending 和 recent、题目提示、assistant 角色开头。生成期间固定记忆，同一正文的各题可复用记忆块。Writer 与 Reader 的模块及训练状态独立，共享冻结的基座参数；LoRA 仅作用于 Writer。

### 单次压缩预训练

`stage=pretrain` 使用 `training.update_mode=compressor_only`，目标配置为 `training.objective={"type":"ae_lm","strategy":"sft","lm_ratio":0.5,"lm_target_tokens":512}`。本节的单块预设使用 `training.compression={"mode":"single"}`，将完整压缩输入一次写入 K 个 slots，不保留 recent；预训练也支持后文所述的 multi 分段规则。

每条完整 `text` 是一个样本，按 seed 和文本内容固定选择 AE 或 LM，跨 epoch 保持一致。

| 目标 | 压缩输入 | 预测目标 |
|---|---|---|
| AE | 完整文本 | 重建完整文本 |
| LM | 去掉末尾目标 tokens 后的前缀 | 文本末尾的 `lm_target_tokens` 个 tokens |

预设中 `training.objective.lm_ratio=0.5`、`lm_target_tokens=512`。LM 前缀长度不大于 K 时改用 AE；两个目标均追加 EOS 结束标记。每个样本按目标 tokens 平均负对数似然（NLL），再对样本等权平均。

预训练只更新 Writer LoRA 和可学习 memory embeddings。冻结的 Reader 保留损失到 memory 的梯度，边界模块在后训练阶段训练。

### 仅压缩器 QA 后训练

`stage=post`、`training.update_mode=compressor_only` 使用完整正文及 QA，只更新 Writer LoRA 和 memory embeddings。目标配置为 `training.objective={"type":"qa","strategy":"sft"}`，不配置 `backward`，`model.structure.pending_stride`、`pending_size` 与 `boundary_hidden_size` 均为 `null`。多块压缩配置为：

```json
{"mode":"multi","ratios":[2,4,8,16],"recent_size":null}
```

预设的 `model.structure.recent_size=256`，训练覆盖值为 `null`。先固定保留正文末尾实际生效的 recent tokens，其余前缀全部压缩；生效值为 0 时不保留原文。最小压缩段长由 `min(ratios) × K` 确定，示例中为 2K。每一步从 `ratios` 中筛选能放入当前剩余前缀、且压缩后剩余长度为 0 或至少最小段长的倍率。合法候选非空时等概率采样；只有候选为空，才将全部剩余前缀一次压成 K 个 slots 并结束。收尾倍率不加入常规采样集合。

例如还剩 5K 时，4× 会遗留不足 2K 的尾块，只有 2× 合法；先压缩 2K，再将余下 3K 一次收尾。每篇每次训练采样一条路径，各题共享全部记忆和固定 recent。

训练目标仅为原始 QA NLL，先对每题答案 tokens 平均，再对题目、文档平均；不含记忆成本。Reader 冻结并保留到 memory 的梯度，不创建或训练边界模块。`compression.mode=single` 时整篇写入一个记忆块，不保留 recent。

配置的 `init_checkpoint=null` 表示从基座初始化 Writer LoRA 和 memory embeddings；也可加载 AE／LM 预训练或 `compressor_only` checkpoint 中的压缩器。阶段 A 按 `epochs=N` 训练压缩器后，可用该 checkpoint 新建阶段 B 的 `boundary_only` run。阶段切换只加载 Writer 和 memory，新建边界模块、optimizer 和 scheduler；只有 `--resume` 恢复完整运行状态。两段分别设置轮数，不自动判断收敛或交替切换。

### QA 与边界后训练

`training.objective.type=qa`，`strategy` 为 `scst`、`grpo` 或 `ppo`。SCST 和 GRPO 支持 `joint` 与 `boundary_only`；PPO 使用 `boundary_only`，冻结 Writer、memory embeddings 和 Reader，训练边界模块与独立 value head。`group_size` 统一表示每篇正文采样的随机轨迹数，SCST／PPO 至少为 1，GRPO 至少为 2；SCST 额外的一条 greedy 轨迹不计入该数量。

一篇完整文档及其固定 QA 集合是一个训练单位。每条随机路径处理整篇正文，各题共享该路径的记忆。QA NLL 先在每题的答案 tokens 内平均，再依次对题目、随机路径和文档等权平均。

令轨迹编号为 i，奖励事件编号为 t，最后一个事件为 T。D 为 Writer 接收的正文 token 数，R 为实际保留的 recent 长度。事件 t 之后的记忆块数和 pending 长度分别记为 Bᵢₜ 和 Pᵢₜ。令 S=D−R，实际最终记忆成本及最终得分为：

\[
c_{i,T}=\left(\frac{K B_{i,T}+P_{i,T}}{S}\right)^p,
\qquad Z_i=-\ell_i-\alpha c_{i,T}.
\]

其中 ℓᵢ 为最终 QA NLL，`memory_cost_power`（p）为正数，默认 1.0；`memory_cost_weight`（α）为非负数，默认 0.1。固定 recent 不参与成本的分子和分母，但仍供 Reader 回答。p=1 且 pending 长度相同时，同一文档每增加一块的惩罚为 `α × K / (D−R)`。D−R 只在完整轨迹结束后用于奖励结算，策略观察和动作约束不读取该长度。

完整保留比例 `retention_ratio=(K×B+P+R)/D` 独立统计。若正文全部属于 recent，即 D=R，则 `compression_retention_ratio` 和 `memory_cost` 为 0，完整保留比例为 1，奖励只包含最终 QA NLL；训练长度筛选保证 D−R 为正。

`reward_mode` 决定原始奖励的分配。仅 `reward_mode=stepwise` 可配置 `training.objective.stepwise_cost_basis`，支持 `observed` 和 `full`，省略时默认 `observed`；terminal 和 SFT 配置不允许出现该字段。令 Uₜ 为事件 t 后尚未读入的前缀原文长度，不含 recent：

| `stepwise_cost_basis` | 过程成本 cᵢₜ | 初始成本 cᵢ₀ |
|---|---|---|
| `observed` | `((K×Bᵢₜ+Pᵢₜ)/S)^p`，只计已读取文本的保留量 | 0 |
| `full` | `((K×Bᵢₜ+Pᵢₜ+Uₜ)/S)^p`，未读文本按原长计入 | 1 |

两种基准在终局均有 Uₜ=0，最终成本相同；`memory_cost`、`compression_retention_ratio`、完整保留比例和 dev／独立评估统计始终使用实际终局保留量。初始成本只用于计算差分，不是一个奖励事件。

`gamma` 默认 1.0，有效折扣率为：

\[
\bar\gamma=\begin{cases}1,&\text{terminal}\\\gamma,&\text{stepwise}.\end{cases}
\]

terminal 总是覆盖配置中的 gamma。两种奖励公式为：

\[
r_{i,t}^{\mathrm{terminal}}=\mathbf1_{\{t=T\}}Z_i,
\qquad
r_{i,t}^{\mathrm{stepwise}}
=-\alpha(c_{i,t}-c_{i,t-1})-\mathbf1_{\{t=T\}}\ell_i.
\]

stepwise 为初次读入、每次读入与判断、强制收尾分别结算成本差分。`observed` 下，继续积累增加已读占用；压缩判断同时读入新单元，只有净占用减少时才获得成本返还。`full` 下，读入使 U 减少、P 等量增加，自主等待和强制等待的即时记忆奖励均为 0；自主压缩及末尾强制压缩都按实际节省返还成本。p=1 时，把 n 个 pending tokens 压成 K 个 slots 的记忆奖励为 `α×(n−K)/S`。

两种基准都在强制收尾后额外加入一次 QA 奖励，不重复扣完整成本。S>0 时，未折扣原始奖励总和在 `observed` 下为 Zᵢ，在 `full` 下为 Zᵢ+α；γ=1 时评价的最终目标相同。p=1 时，两种基准每步奖励之差只由该位置读入的文本长度决定，SCST 在任意 γ 下的同位置 greedy 回报差均会抵消这部分；GRPO 的奖励标准化和 PPO 的价值目标／GAE 仍可能随基准变化。p≠1 且 γ≠1 时，两种基准还可能改变原始折扣回报的路径排序。

优势算法由 `strategy × reward_mode` 确定。定义从正文位置 t 到终局的折扣回报：

\[
G_{i,t}=\sum_{k=t}^{T}\bar\gamma^{k-t}r_{i,k}.
\]

| 策略 | terminal 优势 | stepwise 优势 |
|---|---|---|
| SCST | `A[i,t] = Z[i] − Z[greedy]`，同轨迹所有自主动作相同 | `A[i,t] = G[i,t] − G[greedy,t]` |
| GRPO | `A[i,t] = (Z[i] − mean(Z)) / (std(Z) + ε)`，同轨迹所有自主动作相同 | 先按同篇采样组标准化真实事件奖励，再计算折扣 reward-to-go |
| PPO | GAE，使用 `γ̄=1` 和配置中的 λ | GAE，使用配置中的 γ 和 λ |

SCST terminal 调用 verl 原生 ReMax 优势函数。每篇从文首额外运行一条 greedy 轨迹，供该篇所有随机路径共用。stepwise 按同一正文读取位置对齐，允许 memory、pending 和自主判断次数不同；即使 greedy 此处强制等待，也保留对应回报。该基线独立于本次采样动作，是位置参考，不是从采样路径当前状态重新执行 greedy 的反事实比较。greedy 不消耗边界采样随机数，同篇动作序列相同的轨迹共用 QA 评分。

GRPO 的 terminal 调用 verl 原生 GRPO，均值和样本标准差来自同篇各轨迹的最终得分 Z；标准差分母为样本数减 1。stepwise 将同篇全部随机轨迹的全部真实事件奖励放在一起计算一组均值 μᵣ 和样本标准差 σᵣ：

\[
\widetilde r_{i,t}=\frac{r_{i,t}-\mu_r}{\sigma_r+\epsilon},
\qquad
\hat A_{i,t}=\sum_{k=t}^{T}\gamma^{k-t}\widetilde r_{i,k}.
\]

该统计保留强制等待、强制收尾及值恰好为零的真实奖励，排除 padding 和初始 c₀ 占位；c₀ 只用于计算第一项成本差分。`advantage_epsilon`（ε）在 GRPO 的两种奖励模式下均使用，默认预设为 `1e-6`。先标准化每步奖励再累积遵循原始过程 GRPO；累积时的可配置 gamma 是本方法的扩展。[DeepSeekMath](https://arxiv.org/html/2402.03300v3#S4.SS1)

PPO 的 `gae_lambda`（λ）必须显式配置，范围为 [0,1]，不受 terminal 覆盖。使用 rollout 时保存的旧价值预测：

\[
\delta_{i,t}=r_{i,t}+\bar\gamma V^{\mathrm{old}}_{i,t+1}-V^{\mathrm{old}}_{i,t},
\qquad
\hat A_{i,t}=\delta_{i,t}+\bar\gamma\lambda\hat A_{i,t+1}.
\]

真正终止后的旧价值预测和优势均为 0。terminal 仅当 λ=1 时，白化前的优势简化为 `Â[i,t]=Z[i]−V_old[i,t]`。λ 控制实际回报与价值预测之间的估计权衡，不能仅根据奖励位置推定其最优值。[GAE](https://arxiv.org/html/1506.02438v6#S3)

PPO 调用 verl 原生 GAE。先固定 critic 目标 `Y=V_old+Â`，再对完整 rollout 批次中全部真实事件的 Â 做优势白化，得到 actor 使用的优势：

\[
A_{i,t}=\frac{\hat A_{i,t}-\operatorname{mean}(\hat A)}
{\sqrt{\operatorname{var}_{\mathrm{sample}}(\hat A)+10^{-8}}}.
\]

白化跨文档与 GPU 汇总，但排除 padding；真实强制事件仍参与统计。critic 目标使用白化前的 Â。若全批仅有一个真实终止事件，则没有自主动作，直接用终止奖励作为 critic 目标并跳过 actor，不对白化统计补造事件。

固定读取事件与自主动作分开处理。当前判断检查新单元加入之前的 pending 长度，达到 H 才允许选择压缩或继续积累；未达到阈值及末尾强制收尾均不产生 actor 损失。自主选择继续积累仍参与 actor 更新。SCST 的位置匹配、回报折扣和 GAE 使用完整真实事件序列，不能先删除强制事件再累计。

奖励结算与优势统计使用 FP32。所有策略的奖励、优势和旧动作概率均停止梯度。三种策略都调用 verl 原生 vanilla 策略损失，仅自主动作参与；沿用配置中的 `clip_epsilon`（预设 0.2），采用原生 dual-clip 系数 3，以及 log 概率比 `[-20,20]` 截断。

SCST／GRPO／PPO 在 terminal 和 stepwise 下统一对每条轨迹的自主动作求和，再对采样轨迹和文档平均：

\[
L_{\pi}=\frac{1}{M}\sum_{d=1}^{M}\frac{1}{G}\sum_{i=1}^{G}
\sum_t m_{d,i,t}\,\ell^{\mathrm{actor}}_{d,i,t}.
\]

\(M\) 为本次更新的实际全局文档数，\(G\) 为 `group_size`，\(m_{d,i,t}\) 为采样路径自己的 `decision_mask`；\(\ell^{\mathrm{actor}}_{d,i,t}\) 为包含上述概率比裁剪和 dual-clip 的单动作策略损失。轨迹内不除以自主动作数。零自主动作路径贡献 0，仍计入 G；整篇没有自主动作的文档仍计入 M。SCST 的 greedy 路径只提供基线，不计入 G。

轨迹内求和会影响 actor 损失及梯度的尺度；`policy_weight`、学习率和其他超参数沿用配置数值。`joint` 另保留 QA NLL 的直接梯度；`boundary_only` 通过奖励使用 QA NLL。

PPO 在完整 rollout 后缓存冻结 Writer 产生的状态特征、旧动作概率、旧价值预测、优势和价值目标。actor 与 critic 分别使用 AdamW、分别裁剪梯度；先完成 critic 的全部内轮，再完成 actor 的全部内轮，两者的内轮数均为 `ppo_epochs`，默认 4。轮内不重算优势或价值目标，不重跑 Writer 和 Reader。`value_loss_weight` 默认 0.5，value head 使用 `boundary_learning_rate`；`policy_weight` 仍控制 actor 损失权重。critic 在包含强制事件的真实状态上学习，actor 只学习自主决策。新一批 rollout 使用更新后的模型。

PPO critic 调用 verl 原生 value clipping，裁剪范围为 0.5：

\[
V_{\mathrm{clip}}=V_{\mathrm{old}}+\operatorname{clip}(V-V_{\mathrm{old}},-0.5,0.5),
\qquad
L_V=\frac12\max\bigl((V-Y)^2,(V_{\mathrm{clip}}-Y)^2\bigr).
\]

价值损失先对真实事件平均，再对轨迹和文档平均。两个独立更新分别使用 `policy_weight × policy_loss` 和 `value_loss_weight × value_loss`。

`training.loop.critic_warmup` 为非负整数，默认 0；仅 PPO 配置可以包含该字段。该值直接对应 verl 从 1 开始的 rollout 批次阈值，当前批次达到阈值时才更新 actor：0／1 均从首批更新 actor，11 表示前 10 批只更新 critic。预热期间继续采样新轨迹，critic 仍使用固定 GAE 目标，不切换 Monte Carlo 目标、不覆盖 λ。

训练同时记录 `rollout_step`、`actor_optimizer_step` 和 PPO 的 `critic_optimizer_step`。PPO 以 critic 的实际 optimizer 更新数作为训练主步数，其他策略使用 actor 的实际更新数；`max_steps`、dev／保存间隔、checkpoint 目录及 SwanLab 横轴沿用该主步数。按数据轮数解析时，PPO 总预算为文档批次数乘 `ppo_epochs`；同批完成后记录最后一步，例如 4、8、12。预热占用原有数据和更新预算，不追加轮次；预热或全局无自主动作时 actor 不更新，其计数可以落后于 critic。

两个 PPO optimizer 的 scheduler 各自按实际更新次数推进，学习率预热步数由 `scheduler.warmup_ratio` 乘解析后的总 optimizer steps 并向上取整得到；总预算包含 `ppo_epochs`。策略温度按训练主步数与原总预算计算，actor 内轮按对应主步数递增。`critic_warmup` 单独按 rollout 批次判断，不能与学习率 warmup 混用。

dev／保存的整数间隔在完整 rollout 更新批次跨过阈值后执行，checkpoint 不保存未完成的批内优化状态。训练指标取相应角色最后一次更新的结果；`step_seconds` 为该批采样与全部更新的总耗时除以本批主步数增量。

默认 α=0.1 来自 `policy-post_k256_20261113` 的 step 69 checkpoint：对 118 篇 dev 文档的固定 1／2／4／8 块候选，按参考答案逐文档选择目标最小项。该冻结模型校准保留了无惩罚候选最优 NLL 收益的 90.72%，相对单块的额外保留比例减少 48.52%，平均块数从 2.551 降至 1.822，其中 55 篇选择多块。记录位于 `artifacts/v3-1/policy-post_k256_20261113/eval/boundary-cost_20261116/alpha-calibration.json`。

### 温度退火

`training.objective.policy_transition` 控制随机策略向确定性策略的过渡，SCST、GRPO 和 PPO 均可使用。`null` 表示固定 T=1；线性退火配置如下：

```json
{
  "method": "temperature_annealing",
  "schedule": "linear",
  "start_temperature": 1.0,
  "end_temperature": 0.2,
  "start_progress": 0.3,
  "end_progress": 0.9
}
```

当前支持 `temperature_annealing` 和 `linear`。温度满足 `start_temperature > end_temperature > 0`，进度满足 `0 <= start_progress < end_progress <= 1`，总更新数至少为 2。

设实际总 optimizer steps 为 S，即将执行的更新编号为 s（从 1 开始），进度为 `(s - 1) / (S - 1)`。开始进度前保持初始温度，区间内线性下降，结束进度后保持终止温度。上述配置在前 30% 保持 T=1，中间 60% 降至 T=0.2，最后 10% 保持 T=0.2。

训练按 `sigmoid(logit / T)` 采样。SCST／GRPO 的 old log-prob、回放 log-prob 和策略熵使用同一温度；同次更新的所有 microbatch 与路径共用 T。PPO 的 old log-prob 固定使用本批采样温度，后续各轮的当前策略按真实 optimizer step 的温度重新计算概率，保证退火能够达到配置终值；记录 `rollout_temperature` 与末轮 `policy_temperature`。greedy、dev 和独立评估按 `logit >= 0` 决定动作，训练内 dev 的熵使用该步当前策略温度。

总步数由数据、普通训练或课程预算及 `max_steps` 解析。进度贯穿整个 run；恢复训练由保存的 optimizer step 和原总预算继续计算温度。

## 数据准备与课程

### 数据格式与计长

两阶段均读取准备后的 `train.jsonl`、`dev.jsonl`。AE／LM 预训练和 QA 训练的数据行分别为：

```json
{"text":"完整预训练文本"}
```

```json
{"text":"完整正文","qas":[{"question":"问题","answer":"答案"}]}
```

QA 后训练每题必填 `question`、`answer`，可成对附带来源字段 `segment_id`、`segment_char_span`，例如 `"segment_id":"seg0","segment_char_span":[0,4]`。区间是相对当前正文的左闭右开字符下标，用于来源与位置分析，并随逐题结果展示。QASPER 的 question/answer 数据同样可读。

数据准备和 train/dev 长度筛选均使用 tokenizer 对完整正文计算的 token 数，不添加 special tokens、系统提示或 chat 前缀。AE／LM 与 `compressor_only` 使用固定分段规则：single 要求正文长度严格大于 K；multi 要求至少 `recent_size + min(ratios) × K`，保证固定保留 recent 后仍有一块合法压缩输入。AE／LM 的 LM 前缀也须满足对应下限，不足时改用 AE。policy 要求至少 `recent_size + pending_size`。train 下限采用训练实际 recent，dev 下限采用模型默认 recent。

`training.data.max_train_data_tokens` 只限制 train，超限样本整条排除。筛选与限量在内存中完成，独立 test 读取完整正文。分段和成本计算使用与长度筛选相同的正文 tokens。

### 准备预训练与后训练数据

两个准备脚本共用以下入口约定：

| 参数 | 用途 |
|---|---|
| `--prepare-data` | 本次读取的来源 JSONL 文件 |
| `--model-path` | 必填，使用该模型的 tokenizer 计长 |
| `--target-samples` | 正整数指定样本数；`null` 使用全部合条件数据，指定数量须在可用范围内 |
| `--seed` | 随机种子，默认 42 |
| `--output-file` | 新建输出 JSONL；同目录保存构造参数与统计，如 `train.jsonl` 对应 `preparation-train.json` |

**预训练 `prepare_pre.sh`** 将 FineWeb Multisegment 的 `text + continuation` 合并，按目标 token 长度随机裁剪连续原文，再对最终 `text` 重新分词计长。

- `--min-tokens`、`--max-tokens` 接受正整数或 `null`；`null` 采用对应来源边界，请求范围与来源长度范围取交集。
- `--length-bins` 默认 10，整数长度不足时减少档数。
- `--length-distribution uniform` 尽量均衡分配各档样本，`random` 逐样本等概率选择档位；档内均随机选择整数目标长度，实际分布由来源容量决定。

**后训练 `prepare_post.sh`** 保留 FineWeb FactQA 的完整正文，按正文长度筛选，从 `task` 题池选择 QA。`--qas-per-sample` 默认最多 10 题，题数较少时全保留，`null` 使用全部 task 题。按保留题数 N 等分全文，依次选择证据最靠近各份中心的题，每题最多一次，最终按证据位置排序。

以下示例准备最多 4096 tokens 的预训练 train/dev，train 取 5000 条：

```bash
for split in train dev; do
  if [ "$split" = train ]; then target_samples=5000; else target_samples=null; fi
  bash src/latent_working_memory/v3_1_1/scripts/prepare_pre.sh \
    --prepare-data "data/fineweb-multisegment-k512-seg1to3x_train32k_20261008/${split}.jsonl" \
    --output-file "data/fineweb_train5k_20261112/${split}.jsonl" \
    --model-path ~/models/Qwen3-4B-Instruct-2507 \
    --seed 42 \
    --target-samples "$target_samples" \
    --max-tokens 4096 \
    --length-bins 10 \
    --length-distribution uniform
done
```

后训练构造命令见[后训练数据准备](scripts/README.md#22-后训练数据)。

### 长度分档与课程

`prepare_length.sh` 读取已准备目录中的指定 `<split>.jsonl`，按完整正文 token 数划分等宽档位，默认 4 档。各条正文及 QA 保持完整，分档结果写入 `<split>/bin_00.jsonl` 等文件，其余 split 原样复制；`preparation-<split>.json` 记录边界、样本数和长度统计。命令见[长度分档](scripts/README.md#23-长度分档)。

`training.data.length_curriculum=true` 或启动参数 `--length-curriculum` 开启课程。每轮按以下过程执行：

1. 从短到长选择当前档位，完整使用本档样本。
2. 从此前各档尽量均衡地随机无放回抽取历史样本，数量为本档样本数乘回放比例，四舍五入；历史不足时取全部。
3. 合并、打乱后切 batch，完成本轮训练。每轮重新抽取历史样本。

`training.loop.curriculum_stage_epochs` 控制每档轮数，默认 1；`curriculum_replay_ratio` 控制回放比例，默认 0.2，范围 `[0,1]`。例如本档 250 条时补充 50 条历史样本。对应 CLI 参数为 `--curriculum-stage-epochs` 和 `--curriculum-replay-ratio`。

课程沿用训练的长度筛选与限量，跳过空档，全部档位完成后结束。关闭课程时，各档合并后按普通 `epochs` 全量随机训练。课程只改变样本调度，G、QA 数、全局 batch 和 dev 协议保持原设置；日志记录档位与阶段内轮数，恢复时继续原样本顺序和游标。

## 配置与运行

### 配置组织

配置顶层为 `stage`、`seed`、`model_name_or_path`、`model`、`compute`、`training`。`stage` 取 `pretrain` 或 `post`；`model_name_or_path` 同时指定基座与 tokenizer，可由 `--model-path` 覆盖。

| 分组 | 职责与主要参数 |
|---|---|
| `model.structure` | 预训练仅 `memory_slots` 为正整数，其余四项均为 `null`。后训练的 `recent_size` 为非负整数。`pending_stride`、`pending_size` 和 `boundary_hidden_size` 在 policy 模式下为正整数，其余模式均为 `null`。 |
| `model.lora` | `lora_rank`、`lora_alpha`、`lora_target_modules` |
| `compute` | 加载精度、attention、混合精度、层内重算、CPU 暂存 |
| `training.data` | 数据目录、长度和数量筛选、课程开关 |
| `training.loop` | 轮数或更新数、课程参数、batch、验证与保存频率、checkpoint 保留数；PPO 另含 critic 预热阈值 |
| `training.compression` | single 仅含 `mode`；multi 含 `mode`、`ratios`、训练 recent 覆盖值；policy 含 `mode`、训练 recent 覆盖值 |
| `training.objective` | 所有模式均含 `type`（`ae_lm/qa`）、`strategy`（`sft/grpo/scst/ppo`），以及该目标适用的参数 |
| `training.update_mode` | 参数更新范围：`compressor_only/boundary_only/joint`；预训练固定为 `compressor_only` |
| `training.dev` | 固定验证倍率 `fixed_ratios`；仅 QA 目标配置生成上限 `max_new_tokens` |
| `training.backward` | 仅 `joint` 策略训练配置：跨段反传长度 `bptt_steps` |
| `training.optimizer` | 按更新范围配置学习率，另含 AdamW 参数和梯度范数裁剪 |
| `training.scheduler` | 学习率预热比例 `warmup_ratio` 与余弦衰减最低比例 `min_lr_ratio` |
| `training.run` | `output_root`、`run_id`；后训练另含 `init_checkpoint` |
| `training.swanlab` | `tracking`、`project`、`group`、`tags` |

multi／policy 的训练 recent 覆盖值为 `null` 时沿用模型默认值，0 或正整数只改变训练；dev 和独立评估始终采用模型默认值。single 强制 recent 为 0。

支持以下组合，`type` 指定监督任务，`strategy` 指定训练方法，`update_mode` 指定可更新的参数：

| `stage` | `objective.type` | `objective.strategy` | `update_mode` |
|---|---|---|---|
| `pretrain` | `ae_lm` | `sft` | `compressor_only` |
| `post` | `qa` | `sft` | `compressor_only` |
| `post` | `qa` | `grpo` 或 `scst` | `joint` 或 `boundary_only` |
| `post` | `qa` | `ppo` | `boundary_only` |

AE／LM 的 `objective` 还包含 `lm_ratio` 和 `lm_target_tokens`；QA SFT 只需 `type` 和 `strategy`。策略训练的 `objective` 另含 `group_size`、`reward_mode`、`gamma`、`policy_transition`、成本与损失参数。仅 stepwise 允许配置 `stepwise_cost_basis=observed/full`，默认 `observed`。GRPO 配置 `advantage_epsilon`；PPO 配置独立的 `gae_lambda`、`ppo_epochs` 和 `value_loss_weight`。`joint` 在策略损失之外保留 QA NLL 的直接反传；`boundary_only` 仅通过奖励使用 QA NLL。

`training.optimizer` 的学习率字段由更新范围决定：`compressor_only` 仅含 `compressor_learning_rate`，`boundary_only` 仅含 `boundary_learning_rate`，`joint` 同时包含两者。PPO 的 critic 与边界共用 `boundary_learning_rate` 数值，分别使用独立 optimizer。仅 `joint` 策略训练包含 `training.backward`；其他模式省略整个分组。

模型结构、目标、优化器及调度参数通过 JSON 设置。新训练的显式 CLI 参数覆盖 JSON，省略时沿用配置。覆盖后的配置保存到计划；解析训练预算后的实际配置写入训练目录、checkpoint 和启用时的 SwanLab config。两阶段均保存完整 `model`，`training` 只保存本阶段适用的字段。源 JSON 保持原样，实际 `training.run.output_dir` 由启动流程生成。

### 配置预设

| 配置 | 用途 |
|---|---|
| [pretrain_default.json](../../../configs/v3_1_1/pretrain_default.json) | 通用 K256 AE／LM 预训练 |
| [post_default.json](../../../configs/v3_1_1/post_default.json) | 通用 K256 GRPO 后训练，4 路采样、固定 T=1 |
| [pretrain_k256.json](../../../configs/v3_1_1/pretrain_k256.json) | FineWeb 5k 预训练与服务器计算设置 |
| [post_k256_grpo.json](../../../configs/v3_1_1/post_k256_grpo.json) | FactQA 长度课程与截断反传，GRPO 4 路采样、固定 T=1 |
| [post_k256_grpo_temp.json](../../../configs/v3_1_1/post_k256_grpo_temp.json) | 同套后训练数据与执行设置，GRPO 4 路采样、线性温度退火 |
| [post_k256_scst_temp.json](../../../configs/v3_1_1/post_k256_scst_temp.json) | 同套后训练数据与执行设置，SCST 单随机路径加 greedy 基线、线性温度退火 |
| [post_k256_conly.json](../../../configs/v3_1_1/post_k256_conly.json) | FactQA 仅压缩器后训练，从合法倍率采样，固定保留 256 tokens recent |
| [post_k256_bonly_grpo.json](../../../configs/v3_1_1/post_k256_bonly_grpo.json) | 冻结压缩器、仅训练边界，GRPO 4 路采样、terminal 奖励 |
| [post_k256_bonly_ppo.json](../../../configs/v3_1_1/post_k256_bonly_ppo.json) | 冻结压缩器，PPO 单随机路径、stepwise 奖励、GAE，成本基准 `observed` |

`post_k256_bonly_ppo.json` 沿用仅边界预设的数据和优化器设置，显式设置 `group_size=1`、`gae_lambda=0.95`、`ppo_epochs=4`、`value_loss_weight=0.5` 和 `critic_warmup=8`。该预设前 7 个 rollout 批次只更新 critic，第 8 批开始更新 actor；通用 `critic_warmup` 默认仍为 0。0.95 是该预设的选择，不是系统默认值；terminal 同样读取这一 λ。

两份 `*_default.json` 的主要设置如下：

| 设置 | 预训练 | 后训练 |
|---|---|---|
| K／pending 步长／最短 pending／recent／边界维度 | `256 / null / null / null / null` | `256 / 256 / 256 / 0 / 128` |
| Writer LoRA | rank 16，alpha 32，目标 `q_proj/k_proj/v_proj/o_proj` | 相同 |
| 计算 | FP32、SDPA；混合精度、层内重算、CPU 暂存均关闭 | 相同 |
| 目标 | `type=ae_lm`、`strategy=sft`，`lm_ratio=0.5`、`lm_target_tokens=512` | `type=qa`、`strategy=grpo`，`group_size=4`、`policy_transition=null` |
| 策略参数 | — | 成本权重 0.1，成本幂 1.0，策略权重 1.0，`clip_epsilon=0.2`，`advantage_epsilon=1e-6` |
| 学习率 | 压缩器 `1e-4` | 压缩器 `2e-5`，边界 `1e-4` |
| AdamW／裁剪 | `weight_decay=0.01`，`beta1=0.9`，`beta2=0.999`，`adam_epsilon=1e-8`，梯度范数上限 1.0 | 相同 |
| 调度 | `warmup_ratio=0.1`，`min_lr_ratio=0.1` | 相同 |
| 跨段反传 | — | `bptt_steps=null`，完整反传 |

K256 实验预设使用 `~/models/Qwen3-4B-Instruct-2507`，开启 FA2、BF16 autocast 和层内重算。它们的执行设置为：

| 设置 | K256 单块预训练 | K256 联合后训练（GRPO／SCST） |
|---|---|---|
| 数据目录 | `data/fineweb_train5k_20261112` | `data/fineweb-factqa_q10_train1k_length3_20261112` |
| 预算 | 普通训练 1 轮 | 3 个长度档位，每档 1 轮，历史回放 20% |
| 每卡 microbatch × 梯度累积 | `16 × 2` | `2 × 8` |
| train 长度上限 | `null` | 12288 tokens |
| 预热比例／最低学习率比例 | `0.1 / 0.1` | `0.1 / 1.0` |
| dev／checkpoint 间隔 | 每 40 steps | 每 20 steps |
| checkpoint 保留上限 | 5 | 6 |
| 跨段反传 | — | `bptt_steps=2` |

仅压缩器 QA 后训练配置 `post_k256_conly.json` 使用 `compressor_learning_rate=1e-4`、`warmup_ratio=0.1`、`min_lr_ratio=0.1`。每卡 `micro_batch_size=4`、`gradient_accumulation_steps=8`，每次更新每卡累计 32 篇文档，全局 batch 为 GPU 进程数乘 32；同篇全部 QA 不额外增加文档 batch。该配置的模型默认 `recent_size=256`，训练覆盖为 `null`，默认训练 5 轮，关闭长度课程。默认 `init_checkpoint` 为 `artifacts/v3-1/pretrain_k256_20261113_copy2/train/checkpoints/global_step_157`，该来源保留 AE／LM 预训练最后第 157 步的压缩器权重，元数据按当前 checkpoint 格式保存；后训练新建 optimizer 和 scheduler。

当前数据含 train 1002 篇／10020 道 QA、dev 118 篇／1180 道 QA，现有长度筛选保留全部样本。全局 batch 为 32 时，每轮 `ceil(1002 / 32)=32` 个 optimizer steps，5 轮共 160 步，预热为 `ceil(0.1 × 160)=16` 步。`dev_every` 和 `save_every` 均为 `after_each_epoch`：在每轮末的 step 32、64、96、128、160 验证并保存 checkpoint，保留全部 5 份。改变 GPU 数或训练轮数时，预热步数随实际总预算自动换算，验证和保存仍在每轮末执行。

基础预设 `post_k256_bonly_grpo.json` 和 `post_k256_bonly_ppo.json` 的 `init_checkpoint` 均为 `artifacts/v3-1/compressor-post_k256_20261120/train/checkpoints/global_step_160`。仅边界阶段默认训练 1 轮、关闭长度课程，每卡 `micro_batch_size=2`、`gradient_accumulation_steps=2`；采用 4 个 GPU 进程时，全局 batch 为 `4 × 2 × 2=16` 篇文档。1002 篇文档形成 63 个 rollout 批次，末批为 10 篇。对应预算如下，GPU 数改变时须重新按全局 batch 换算：

| 设置 | `post_k256_bonly_grpo.json` | `post_k256_bonly_ppo.json` |
|---|---|---|
| 奖励模式 | terminal | stepwise，成本基准 `observed` |
| 每篇随机路径数 | 4 | 1 |
| 训练主步数 | 63 次 actor 更新 | 252 次 critic 更新 |
| 学习率预热 | `ceil(0.1 × 63)=7` 步 | actor／critic 各自预热 `ceil(0.1 × 252)=26` 步 |
| critic 独占更新 | 不适用 | 前 7 批共 28 次 critic 更新 |
| actor 更新上限 | 63 次 | `(63−7) × 4=224` 次，要求预热后每批均有自主动作 |
| `dev_every`／`save_every` | 20 | 80 |
| 验证与 checkpoint 主步数 | 20、40、60、63 | 80、160、240、252 |
| `save_total_limit` | 5，实际保留 4 份 | 5，实际保留 4 份 |

两份预设均在第 20、40、60 个 rollout 批次结束后验证并保存，并在第 63 批轮末额外执行一次，共保留 4 份 checkpoint。`dev.fixed_ratios=[]`，每次 dev 仅评价当前 greedy 边界策略，计算 QA NLL、生成 EM／F1、记忆成本和保留比例等指标。上述整数间隔按当前数据、全局 batch 和 PPO 内轮数设置，改变这些设置后需重新换算。两份预设的 `boundary_learning_rate=1e-4`、`warmup_ratio=0.1`、`min_lr_ratio=1.0`，学习率预热后保持恒定。阶段切换仅加载压缩器，重新初始化边界、PPO value head 及各自 optimizer／scheduler。

边界策略预设均设置 `pending_stride=256`、`pending_size=256`。`post_default.json` 的模型默认 recent 为 0，K256 实验预设为 256，训练覆盖均为 `null`。

两份 `*_temp.json` 均启用线性温度退火：前 30% 进度保持 T=1.0，30%～90% 从 1.0 降至 0.2，最后 10% 保持 T=0.2。进度按整个训练的实际 optimizer steps 计算，跨长度课程档位连续推进。GRPO 使用 `group_size=4`；SCST 使用 `group_size=1`，每篇另计算一条 greedy 路径作为奖励基线，仅随机路径参与反向传播。

### PPO 参数对照配置

以下两组均为可独立加载的完整配置，使用同一 compressor step 160 初始化新的 boundary 与 value head。共同设置为 PPO、`boundary_only`、`reward_mode=stepwise`；双卡运行通过 CLI 指定每卡 `micro_batch_size=2`、`gradient_accumulation_steps=4`，全局 batch 为 16 篇文档。均保留 `gamma=1.0`、`memory_cost_power=1.0`、`warmup_ratio=0.1`、`max_steps=null`，由实际数据和训练轮数解析预算。

#### 基础参数对照

这六份配置分别采用下表设置，其余训练参数沿用基础 PPO 预设。

| 方案 | 配置文件 | 专属设置 |
|---|---|---|
| 完整回报与路径覆盖 | [post_k256_bonly_ppo_gae1_group2.json](../../../configs/v3_1_1/post_k256_bonly_ppo_gae1_group2.json) | `gae_lambda=1.0`、`group_size=2` |
| 过程成本基准 | [post_k256_bonly_ppo_full.json](../../../configs/v3_1_1/post_k256_bonly_ppo_full.json) | 仅 `stepwise_cost_basis=full`，保持 `memory_cost_weight=0.1` |
| 成本基准与权重 | [post_k256_bonly_ppo_full_cost0p03.json](../../../configs/v3_1_1/post_k256_bonly_ppo_full_cost0p03.json) | `stepwise_cost_basis=full`、`memory_cost_weight=0.03` |
| 温度退火 | [post_k256_bonly_ppo_temp0p2.json](../../../configs/v3_1_1/post_k256_bonly_ppo_temp0p2.json) | 线性退火：前 50% 保持 T=1.0，50%～80% 降至 T=0.2，最后 20% 保持 T=0.2 |
| 学习率衰减 | [post_k256_bonly_ppo_decay0p1.json](../../../configs/v3_1_1/post_k256_bonly_ppo_decay0p1.json) | `min_lr_ratio=0.1`，峰值学习率保持 `1e-4` |
| 增加新采样批次 | [post_k256_bonly_ppo_updates2_epochs2.json](../../../configs/v3_1_1/post_k256_bonly_ppo_updates2_epochs2.json) | `ppo_epochs=2`、`epochs=2`、`critic_warmup=15` |

上表中，`full` 和 `full_cost0p03` 使用 `stepwise_cost_basis=full`，其余使用 `observed`；仅 `full_cost0p03` 的成本权重为 0.03，其余为 0.1。除 `gae1_group2` 外均使用 `gae_lambda=0.95`、`group_size=1`。

`post_k256_bonly_ppo_full.json` 与基础 PPO 配置仅有 `stepwise_cost_basis` 一个字段不同，用于比较相同成本权重下的 `observed/full`。它保留基础配置的梯度累积默认值；双卡运行时按上述 CLI 设置覆盖为 4。与 `full_cost0p03` 在相同实际 batch 下比较，可单独观察成本权重的影响。

#### 衰减基线下的质量反馈与采样对照

以下四份以 `post_k256_bonly_ppo_decay0p1.json` 为参数基线，只修改表中参数，仍从上述 compressor step 160 初始化。共同保留 `stepwise_cost_basis=observed`、`min_lr_ratio=0.1`、`boundary_learning_rate=1e-4`、`policy_transition=null`（T=1）、`ppo_epochs=4`、`epochs=1` 和 `critic_warmup=8`。

| 配置文件 | 相对 decay0p1 的修改 | 主要对照 |
|---|---|---|
| [post_k256_bonly_ppo_decay0p1_gae1.json](../../../configs/v3_1_1/post_k256_bonly_ppo_decay0p1_gae1.json) | `gae_lambda=1.0` | 与 decay0p1 比较 GAE 的长程回报传播 |
| [post_k256_bonly_ppo_decay0p1_group2.json](../../../configs/v3_1_1/post_k256_bonly_ppo_decay0p1_group2.json) | `group_size=2` | 与 decay0p1 比较每篇增加随机路径的作用 |
| [post_k256_bonly_ppo_decay0p1_gae1_group2.json](../../../configs/v3_1_1/post_k256_bonly_ppo_decay0p1_gae1_group2.json) | `gae_lambda=1.0`、`group_size=2` | 与 decay0p1_gae1、decay0p1_group2 分别比较路径数和 GAE 的作用 |
| [post_k256_bonly_ppo_decay0p1_gae1_cost0p03.json](../../../configs/v3_1_1/post_k256_bonly_ppo_decay0p1_gae1_cost0p03.json) | `gae_lambda=1.0`、`memory_cost_weight=0.03` | 与 decay0p1_gae1 单独比较成本权重 |

前三份与 decay0p1 构成 `gae_lambda ∈ {0.95, 1.0}`、`group_size ∈ {1, 2}` 的 2×2 对照，成本权重均为 0.1。第四份在 `gae_lambda=1.0`、`group_size=1` 下将成本权重从 0.1 降至 0.03，保持 `observed`，可单独观察成本压力的影响；基础组的 `full_cost0p03` 同时改变成本基准与权重，不能替代这一对照。

#### 共同预算与运行标识

双卡全局 batch 16、训练集 1002 篇时，单轮配置为 63 个 rollout 批次、每批 4 次 PPO 更新；`updates2_epochs2` 为 126 个 rollout 批次、每批 2 次更新。两组配置均为 252 次 critic 更新，前 28 次仅更新 critic，之后最多 224 次 actor 更新；学习率预热均为 26 次更新。`group_size=2` 每篇采样两条路径，保持文档 batch 和更新步数不变，采样及 QA 评分路径数为同轮数 `group_size=1` 的两倍。`updates2_epochs2` 每篇训练两轮，路径数同样为单轮 `group_size=1` 的两倍。

`dev_every=80`、`save_every=80`、`save_total_limit=5`。单轮配置在 step 80、160、240、252 验证并保存；`updates2_epochs2` 还在第一轮末 step 126 验证并保存，共保留 5 份 checkpoint。GPU 数或数据量变化时，须按实际全局 batch 重新核算这些步数。

这些参数预设的 `run_id` 和 SwanLab `group` 均为 `null`。启动时通过 `--run-id` 为各方案指定独立运行名；作为同一组对照运行时，在参与对照的配置中显式填写相同的 `training.swanlab.group`。

### 计算精度与显存

模型要求 full-attention 和固定 default RoPE。所有可训练参数、梯度和优化器状态保持 FP32；冻结基座的加载精度由 `compute.dtype` 控制。

| `compute` 参数 | 行为 |
|---|---|
| `dtype` | `float32` 或 `bfloat16`，统一控制 Writer／Reader 基座加载精度 |
| `attention_implementation` | 默认 `sdpa`，由 PyTorch 选择内核；`flash_attention_2` 指定 FA2 |
| `mixed_precision` | 默认 `false`；开启时要求 `dtype=float32`，用 BF16 autocast 执行采样、前向与 loss 计算 |
| `gradient_checkpointing` | 默认 `false`；开启 Transformer 层内重算以减少保存的激活 |
| `save_on_cpu` | 默认 `false`；带梯度前向通过 `save_on_cpu(pin_memory=True)` 将反向所需张量暂存至 CPU |

BF16 autocast 的 backward 与 optimizer 更新在 autocast 上下文外执行，使用 FP32 参数更新；梯度检查点恢复原前向精度。CPU 暂存支持两阶段，可与混合精度组合；模型参数、返回的 memory 和采样 KV 保留在 GPU，暂存只用于带梯度前向。

FA2 需要 `flash-attn`，并使用 BF16 基座或 BF16 autocast。项目默认的 `torch2.6-cuda12.4` 依赖组在 Linux x86_64 安装 PyTorch 2.6、FlashInfer 和对应的 FA2 wheel，执行 `uv sync --frozen` 即可同步该环境；本地 wheel 优先的安装方式见[主环境说明](../../../README.md#主环境与开发)。保留 FP32 参数时，配置为：

```json
{
  "compute": {
    "dtype": "float32",
    "attention_implementation": "flash_attention_2",
    "mixed_precision": true
  }
}
```

新训练可通过 `--dtype`、`--attention-implementation`、`--gradient-checkpointing`、`--mixed-precision`、`--save-on-cpu` 覆盖对应值。后三项仅传参数名表示开启，也接受小写 `true`／`false`，例如 `--mixed-precision false`。恢复训练和独立评估沿用已保存的计算设置。

### 启动预训练与后训练

从项目根目录运行，使用根目录 `.venv/`。训练与评估脚本默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，外部已设置时沿用。`CUDA_VISIBLE_DEVICES` 显式指定使用的 GPU，每卡一个进程。

新训练指定 `--config`，训练阶段直接读取 JSON 中的 `stage`。下面先启动 K256 AE／LM 预训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/pretrain_k256.json \
  --run-id pretrain_k256_20261112 \
  --epochs 1 \
  --micro-batch-size-per-gpu 16 \
  --gradient-accumulation-steps 2
```

以筛选后 5000 条样本计算，该设置训练 1 轮共 157 个 optimizer steps，在第 40、80、120、157 步验证并保存。最终 checkpoint 记录在 `train/summary.json`。

联合后训练使用新的 run，通过 `--init-checkpoint` 指定压缩器来源：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256_grpo.json \
  --run-id policy-post_k256_20261112 \
  --init-checkpoint artifacts/v3-1/pretrain_k256_20261112/train/checkpoints/global_step_157 \
  --micro-batch-size-per-gpu 2 \
  --gradient-accumulation-steps 8
```

初始化来源与目标 run 须使用相同基座、K 和 LoRA 设置（rank、alpha、目标模块）。`joint/boundary_only` 均可从 AE／LM 预训练或 `compressor_only` checkpoint 加载压缩器。启用 GRPO 温度退火时，将 `--config` 改为 `configs/v3_1_1/post_k256_grpo_temp.json`；使用 SCST 温度退火时，改为 `configs/v3_1_1/post_k256_scst_temp.json`。每次新训练分别设置 `--run-id`，初始化来源由配置中的 `training.run.init_checkpoint` 或显式 `--init-checkpoint` 指定。

阶段 A（`compressor_only`）→ 阶段 B（`boundary_only`）分别通过 `--epochs` 设置 N、M 轮；默认分别为 5 轮和 1 轮，关闭长度课程。两阶段启动后直接训练；阶段 A 每轮末验证并保存 checkpoint，阶段 B 按上述步数间隔验证并保存。阶段 A 默认加载 `pretrain_k256_20261113_copy2` 的第 157 步 checkpoint；可通过 `--init-checkpoint` 指定其他 AE／LM 或 `compressor_only` checkpoint，或在配置中将 `init_checkpoint` 设为 `null`，从基座初始化压缩器。以下阶段 A 示例使用单卡；阶段 B 示例按当前仅边界预设使用四卡，默认加载阶段 A 的第 160 步压缩器，重新初始化边界及 optimizer／scheduler：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256_conly.json \
  --run-id compressor-post_k256_20261120

CUDA_VISIBLE_DEVICES=4,5,6,7 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --config configs/v3_1_1/post_k256_bonly_grpo.json \
  --run-id boundary-only-grpo_k256_20261121
```

使用 PPO 时将阶段 B 的配置改为 `configs/v3_1_1/post_k256_bonly_ppo.json`，并使用独立 `run_id`。若阶段 A 的运行目录或最终步数不同，通过 `--init-checkpoint` 指向其 `train/summary.json` 记录的 checkpoint。边界策略允许一直等待到末尾，因此训练前检查去掉固定 recent 后的完整前缀能否放入 Writer 窗口；固定分段模式在实际 Writer／Reader 前向检查长度。

运行根目录为 `<output_root>/<run_id>/`。显式 `run_id` 原样使用；为 `null` 或省略时，AE／LM 预训练生成 `pretrain_k{K}_{YYYYMMDD-HHMM}`，`compressor_only` 生成 `compressor-post_k{K}_{YYYYMMDD-HHMM}`，`joint/boundary_only` 生成 `policy-post_k{K}_{YYYYMMDD-HHMM}`。时间采用 `Asia/Shanghai`，精确到分钟。

### 训练预算与恢复

全局 batch = GPU 进程数 × 每卡 microbatch × 梯度累积次数，预训练按样本、后训练按文档计数。上例两阶段分别为 32 和 16；随机路径数 G 不计入文档 batch。每轮尾批按实际样本数归约，保留全部样本且在本轮完成更新。

- 普通训练由 `training.loop.epochs` 控制完整轮数。
- 长度课程由 `curriculum_stage_epochs` 控制每档轮数，完成全部档位后结束。
- `training.loop.max_steps` 可进一步限制总更新数；解析后的实际步数写入训练配置和 checkpoint。
- `dev_every`、`save_every` 可设步数间隔或 `after_each_epoch`；每轮结束和最终步也执行 dev 与保存。dev 使用固定验证集。
- `training.scheduler.warmup_ratio` 控制线性预热比例，范围为 `[0,1)`，预设统一为 `0.1`，新训练可由 `--warmup-ratio` 覆盖。完成数据筛选、GPU 数量和训练预算解析后，实际预热步数为 `ceil(warmup_ratio × 总 optimizer steps)`，必须小于总步数；比例为 0 时不预热，启动日志报告实际比例与步数。PPO 的总预算包含 `ppo_epochs`，actor／critic 使用相同预热长度，分别按自身实际更新次数推进；`critic_warmup` 仍独立按 rollout 批次判断。
- `min_lr_ratio` 控制余弦衰减的最低学习率比例，同一 optimizer 的参数组共用调度倍率。`min_lr_ratio=1.0` 时，预热后保持恒定学习率。

恢复同阶段训练使用原 run 最新完整 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/v3_1_1/scripts/train.sh \
  --resume \
  --checkpoint 'artifacts/v3-1/policy-post_k256_20261112/train/checkpoints/global_step_<N>'
```

恢复沿用原计划、输出目录、基座、计算设置、数据、GPU 数量、样本游标和随机状态；其他已知 CLI 实验设置忽略并提示。后训练恢复直接使用自身 checkpoint，预算完成时返回训练摘要。`--resume` 与 `--overwrite` 互斥。

## 验证、独立评估与记录

### SwanLab 与训练内 dev

`training.swanlab.tracking` 默认 `true`。`--tracking online` 开启 SwanLab，`--tracking offline` 仅保存本地产物。project、group、tags 通过 JSON 设置；默认 project 为 `latent-working-memory-v3.1`，group 缺省使用 `run_id`，`job_type=train`。

train 与 dev 共用一个 run，恢复沿用原身份。自动记录阶段、压缩／目标模式及后训练更新方式等 tags；具体超参数记录在 config。

dev 保持模型参数不变，边界策略使用确定性判断。train/dev 曲线横轴为真实 optimizer step，只在实际产生结果时追加标量。首次记录某个指标时注册原生图表并配置标题、坐标轴、配色和合并关系；后续复用面板及历史数据。图表标题与指标显示名统一采用对应 key 的小写下划线名称。Y 轴只标必要单位；NLL、loss、reward、learning_rate、EM／F1 和其他无量纲指标的轴标题留空。资源统计放在 `resources/*`，耗时和显存分别保留 `s`、`GiB` 单位。

### 固定倍率与策略验证

`training.dev.fixed_ratios` 控制固定倍率对照，例如 `[2,4,8,16]` 表示在同一批 dev 上评价四个倍率。当前仅边界预设将其设为 `[]`，只评价 greedy 边界策略。仅 QA 目标配置 `max_new_tokens`，预设值 128 控制 greedy 生成上限。AE／LM 验证只计算 NLL。启用固定倍率验证时，按压缩模式采用对应尾段协议：

- AE／LM 或 `compressor_only` 的 multi：四个倍率使用相同的 recent 长度，预训练取 0，后训练取 `model.structure.recent_size`。对剩余前缀按当前固定倍率写入；若剩余长度不足一个固定倍率块，或写入一块会遗留不足 `min(ratios) × K` 的非空尾块，就将当前剩余前缀整体收尾。例如前缀 5K、固定倍率 4× 时，一次压缩 5K；最后一块实际倍率可略大于主倍率。single 若指定固定倍率对照，不保留 recent。
- policy 的固定倍率基线：同样保留模型默认 recent；按固定倍率处理其余前缀，写入后若会遗留不足 `pending_size` 的非空尾块，就将当前剩余前缀整体收尾。各固定倍率块长至少为 `pending_size`。基线将 recent 之外的前缀全部压缩。

各条件均记录实际段长和倍率。`fixed_ratios=[]` 不关闭 dev：AE／LM 和 `compressor_only` 的 single 验证单块压缩，multi 为每篇固定种子采样一条路径；当前 `recent_size=256` 的两份仅边界预设只验证 greedy 边界策略，同时计算 QA NLL、生成答案并计算 EM／F1。策略配置只有在 recent 为正或固定倍率列表非空时进入此条件验证路径；recent 为 0 且列表为空时使用逐文档损失验证。

阶段 A 预设分别验证四个固定倍率；阶段 B 预设只验证 greedy 边界策略，需要对照时可通过 `fixed_ratios` 加入固定倍率。策略与固定倍率基线使用相同 recent，策略收尾可额外保留不足 `pending_size` 的 pending。进入条件验证的 QA 设置均报告 NLL、EM／F1、块数、未压缩 pending、recent、前缀保留比例及完整保留比例；边界策略模式另报告记忆成本及 `QA NLL + α × cost`，其中成本采用前缀保留比例的 p 次幂。NLL 按题目再文档平均，EM／F1 按题目平均。AE／LM 条件只评价对应重建／续写 NLL，不生成 QA 答案。

新 run 启动后直接训练，这些条件仅在配置的验证间隔、每轮末和最终步计算。启用固定倍率对照时，`boundary_only` 的固定倍率基线在首次正常 dev 时建立，本次进程的后续验证复用；恢复后首次 dev 重建缓存。压缩器可更新的训练每次重新计算固定倍率结果，策略条件每次均使用当前边界参数重新验证。所有验证保留训练 RNG，不改变后续采样。逐篇分段、逐题生成和汇总写入 `train/dev/global_step_<step>/rank_<rank>.jsonl` 与 `summary.json`。

SwanLab 只记录实际计算的条件：当前仅边界预设使用 `dev/policy/*`，四个固定倍率对照启用时另有 `dev/ratio_2/*`、`dev/ratio_4/*`、`dev/ratio_8/*` 和 `dev/ratio_16/*`。压缩统计分为“compression_retention_ratio”“segment_compression_ratio”“memory_blocks”三张原生图，每图只合并同一指标的不同条件；图例保留 `policy`，固定倍率显示为 `2×`、`4×`、`8×`、`16×`；policy 优先，固定倍率按数值升序排列，条件配色在三图间一致。项目共享面板保留历史运行已经记录的同指标条件，policy 加四个固定倍率时每图共 5 个数据源。单图最多 8 个数据源，超过时在更新图表前明确报错，需按实际实验条件拆分。其他 dev 指标同样按指标合并条件；`memory_cost` 与完整 `retention_ratio` 仅保存本地，包含成本的 `objective` 单独展示。

### 指标口径与面板

AE／LM 预训练保存实际选样数量、`loss`、`lm_fraction` 和分项 NLL；`compressor_only` 记录 `loss`、`qa_nll` 与三项压缩统计。train 的“compression_retention_ratio”“segment_compression_ratio”“memory_blocks”分别展示，通用 dev 路径也只为实际产生的裸 `dev/*` 指标配置对应图表；条件验证使用 `dev/{condition}/*`，不重复写裸 dev 指标。三项统计在训练、验证和重传中使用相同的指标键与面板构造。保留率底层数据保留原始比例，百分比仅在图表表达式中换算，不重复绘制原始比例曲线。边界策略模式的 train 面板如下，均使用单 Y 轴；表中指标省略 `train/` 前缀。

| 图表 | 指标与展示 |
|---|---|
| `loss`、`qa_nll`、`policy_loss` | `loss`、`qa_nll`、`policy_loss` 分别展示；仅边界模式的 `loss` 不含 QA 项 |
| `reward` | 随机路径 `reward` 与多样本 `reward_std`。SCST 的 `baseline_reward` 记录 greedy 路径的整轨总奖励，图例为 `greedy_reward` |
| `zero_advantage_rate` | `zero_advantage_rate`：实际优势全部为零的文档比例 |
| `compression_retention_ratio` | `compression_retention_ratio` 保存原始比例，图表以原生表达式乘 100 显示百分比，Y 轴仅标 `%` |
| `segment_compression_ratio` | `segment_compression_ratio`，Y 轴仅标 `×` |
| `memory_blocks` | `memory_blocks`，Y 轴仅标 `blocks`；不是 GPU 显存占用 |
| `policy_entropy` | `policy_entropy` |
| `replay_log_probability_drift` | `replay_log_probability_drift`：各随机路径内最大 log-prob 差的平均值 |
| `policy_temperature` | `policy_temperature`，启用退火时记录 |
| `value_loss` 等 PPO 指标 | `value_loss`、`clip_fraction`、`dual_clip_fraction`、`value_clip_fraction`、`approx_kl`、`ppo_kl`，仅 PPO 记录 |
| `grad_norm` 等梯度指标 | `grad_norm`、`compressor_grad_norm`、`boundary_grad_norm` 分别展示，PPO 另记 `value_grad_norm`，均为相应 optimizer 裁剪前的范数，PPO 的 actor／critic 分别裁剪 |
| `learning_rate` | `compressor_learning_rate`、`boundary_learning_rate` 共用纵轴，PPO 另记 `value_learning_rate` |

PPO 的 `clip_fraction`、`dual_clip_fraction` 与 `value_clip_fraction` 使用原生损失函数的裁剪统计。令 `q=exp(log π_new−log π_old)`，`approx_kl` 保留 `mean(q−1−log q)`，`ppo_kl` 记录原生 `−mean(clip(log q,−20,20))`，后者是采样估计，可以为负。训练进度在本地辅助指标中分别记录 rollout、actor 与 critic 的真实更新计数，不单独上传云端面板。`actor_loss` 与 `critic_loss` 同样保留在本地，分别为两侧已加权损失；`loss` 为两者之和，预热批只包含 critic 损失。

policy 未配置固定倍率且模型默认 recent 为 0 时，dev 只执行 greedy 路径的 NLL 评分，记录 QA NLL、记忆块数、分段压缩倍率、策略熵、压缩区保留率和 `objective`，完整保留比例与记忆成本保存在本地，不生成答案。其余 policy 设置进入条件验证，按上一节组织 `dev/policy/*` 和固定倍率指标。

令 D 为正文长度、R 为 recent 长度、P 为最终未压缩 pending 长度、B 为记忆块数、K 为每块 slots 数。`compression_retention_ratio=(K×B+P)/(D−R)` 衡量压缩区的最终保留比例，包含压缩区内未压缩的 pending，`memory_cost=compression_retention_ratio^p`。`segment_compression_ratio=segment_tokens/K` 衡量实际已压缩段的倍率，pending 和 recent 不参与；`segment_tokens` 是所有已压缩段的平均长度，因此汇总倍率按块加权，没有压缩块时不产生该指标。`memory_blocks=B` 是记忆块数，压缩区保留率与记忆块数均按文档等权平均。`retention_ratio=(K×B+P+R)/D` 是整篇最终保留比例，仅保存在本地。

没有未压缩 pending 时，分段压缩倍率与压缩区保留率在单篇上互为倒数；跨文档汇总后的两条曲线使用不同权重，不保证互为倒数。train 的 QA、记忆块数、压缩统计和奖励只统计随机路径，greedy 奖励单独记录。同时记录 pending 原文峰值和自主动作的平均绝对优势。

边界策略模式的本地 `metrics.jsonl` 还保留辅助统计：train 的 `commit_rate`、`memory_slots`、`segment_tokens`、`retention_ratio`，dev 的 `loss`、`samples`、`commit_rate`、`memory_slots`、`segment_tokens`、`retention_ratio`。`dev/loss` 等于 `dev/qa_nll`，云端展示后者。SCST 单样本的 `reward_std` 为零，仅写本地；边界策略模式不单独记录 `train/samples`。

### 独立评估任务

任务脚本只接受具有边界策略的 `joint/boundary_only` checkpoint；`compressor_only` 当前使用训练内 dev 验证。各任务按 checkpoint 中 `model.structure` 的边界参数和默认 recent 处理完整正文，忽略训练 recent 覆盖值，再由 Reader 根据 memory、未压缩 pending 和 recent 回答。Test 同文档、LoCoMo 同对话的全部 QA 共用一次压缩结果，各题独立生成；RULER 按题目处理。评估计划、日志、明细和汇总保存在本地。

LoCoMo Reader 通过公共输入构造获取训练时的聊天前后缀，由本方法插入记忆块、未压缩尾段和问题；各题共用对话记忆和 Reader 前缀 KV，作答期间不更新记忆。数据与评分见[公共 LoCoMo 说明](../../../benchmarks/locomo/README.md)，实际输入格式随评估配置保存。

独立评估的逐篇记录和汇总保留 `prefix_retention_ratio`、`memory_cost` 和完整 `retention_ratio`；其中 `prefix_retention_ratio` 与训练／dev 的 `compression_retention_ratio` 口径一致，均按文档等权平均；LoCoMo 每段对话只计一次。同文档各题共用压缩成本，不按 QA 数重复累计。

| 任务与脚本 | 数据入口 | 范围与评分 |
|---|---|---|
| Test：`eval_test.sh` | 数据目录下 `test.jsonl` | 完整正文 QA；NLL、EM、F1 |
| LoCoMo：`eval_locomo.sh` | 数据目录下 `locomo10.json` | 类别 1–4，分离对话与问题，复用公共模块的官方评分逻辑 |
| RULER：`eval_ruler.sh` | `<长度>/<子任务>/test.jsonl` | 全部 13 项任务只提取正式题的最终问题。此前的指令、示例与正文按原顺序进入 Writer，Reader 在记忆块与保留尾段之后读取最终问题。 |

Test 的 `qa_nll` 先按每题目标 tokens 平均，再按题目和文档等权平均；`token_nll` 按全部目标 tokens 加权，目标包含模板结束符。EM／F1 按题平均。`--max-test-samples` 限制文档数，`--max-new-tokens` 控制每题生成上限。

### 启动与续评

从 `train/summary.json` 取得最终 checkpoint 名称，填入下面的 `global_step_<N>`：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/v3_1_1/scripts/eval_test.sh \
  --eval-run artifacts/v3-1/policy-post_k256_20261112 \
  --eval-id standard_20261112 \
  --eval-data data/delta-mem-qasper \
  --max-new-tokens 128 \
  --checkpoint 'global_step_<N>' \
  --batch-size-per-gpu 8
```

新建评估必填 `--eval-run` 和 `--eval-data`；前者指向运行根目录。`--eval-id` 缺省生成 `YYYYMMDD-HHMMSS`。`--checkpoint` 接受一个或多个 `global_step_<N>` 名称，省略时选择当时全部完整 checkpoint；多个 checkpoint 按真实 step 串行执行。

每卡批量上限由 `--batch-size-per-gpu` 控制，默认 1；Test 按文档、RULER 按题目组批，LoCoMo 按同一对话的问题组批。LoCoMo 续评会重新压缩仍有未完成问题的对话，仅生成缺失答案。

RULER 通过 `--lengths` 选择长度，默认 `8192 16384 32768`；各长度单独组批，可用 `--batch-size-by-length 8192=8 16384=4 32768=2` 覆盖指定长度的批量，其余沿用通用值。GPU 数与批量属于本次执行设置，可在续评时调整。

续评使用原任务脚本，并传入 `--resume`、原 `--eval-run` 和 `--eval-id`；原任务重定向过输出目录时，同时传入原 `--output-root`。

| 续评内容 | 行为 |
|---|---|
| checkpoint 范围 | 省略 `--checkpoint` 时使用计划累计列表；传入时只调度指定项，并将新项加入累计列表 |
| 数据与协议 | 沿用原数据、预算和 RULER 长度；对应 CLI 设置忽略并提示 |
| 执行进度 | 复用已提交记录，将未完成样本按当前 GPU 数与批量重新分配 |
| 修改协议 | 使用新的 eval-id，或以 `--overwrite` 重建当前任务的计划与全部 checkpoint 结果 |

### 终端进度

主进程按 checkpoint 统一显示总体完成数（历史缓存 + 本轮完成）、百分比、耗时和各 GPU 的进度。每卡显示本轮分片的完成数、阶段、当前批量，RULER 另显示长度；速率和剩余时间按本轮已提交样本估算。Test 在一篇文档的全部 QA 完成后计数，LoCoMo 与 RULER 按题目计数。

交互终端每 5 秒原地刷新，重定向输出时每 30 秒记录一次快照，阶段变化和失败立即显示。逐样本明细与 worker 的完整输出保存在 `eval.log`，带 checkpoint step、GPU 和 rank 前缀。

### 本地看板

```bash
bash src/latent_working_memory/v3_1_1/scripts/dashboard.sh \
  --dashboard-run-dir artifacts/v3-1/policy-post_k256_20261112 \
  --dashboard-eval-id standard_20261112
```

边界诊断只统计策略可自主选择 `compress` 或 `continue` 的位置。每条轨迹保存两个等长数组 `pending_tokens_before`（决策时可压缩的 pending 长度，不含新到达的单元）与 `compress_probs`；`compress_indices` 是实际选择压缩的位置在这两个数组中的下标，从 0 开始。强制等待和 EOF 收尾均排除。greedy 路径以原始 logit 的符号决定动作，实际动作可直接从下标读取。

对于执行过 dev 的训练主步数 S，dev 图覆盖本次完整验证集的 greedy 轨迹，概率使用 T=1。train 图覆盖 S 之后第一批实际训练文档及其所有采样轨迹，概率使用该批真实温度，采样时尚未进行下一次参数更新。PPO 内轮更新和 SCST 的 greedy 基线均不重复计入。若 S 是最终步，则本步只有 dev 图。独立评估按 checkpoint 记录 greedy 轨迹，Test 按文档、LoCoMo 按对话、RULER 按样本各计一次。

看板可直接打开只有训练结果的 run。边界诊断默认对比首、中、末三个有记录的 step；同一步在两张图中使用同色。直方图按 0.1 分箱，每根柱表示该步全部自主决策中落入该区间的比例，最后一区间包含 1。散点图按决策前的 pending 长度展示这些位置的概率，同色折线按 pending 长度递增连接该步在各长度上的平均压缩概率。点击散点、均值标记或使用下拉框可选定一个 pending 长度，竖线标出所选位置。下方直方图按 0.1 分箱，图例显示每个 step 在该长度的自主决策数，各步以各自的决策数为分母。图表按来源分别展示，SwanLab 继续记录训练与 dev 标量。

独立评估按 Test、RULER、LoCoMo 展示。多 checkpoint 比较包含指标表、按训练主步数排列的曲线、资源测量与输出诊断。明细默认选择最新完成的 checkpoint，可查看评估配置、RULER 单步明细、压缩分析与问答。Benchmark 支持接入本地基线，按实际输入、参考答案、基座和评估协议对齐。

压缩区按篇和记忆块分别展示平均、最低、最高压缩倍率，另列每篇平均记忆块数及只生成一个记忆块的比例。输入长度 N 取实际编码后的完整正文 token 数（含模板前缀），B 为记忆块数，K 为每块槽位数，P 为未压缩 pending 长度，R 为固定 recent 长度：

| 统计单位 | 压缩倍率 | 平均方式 |
|---|---|---|
| 整篇文本 | `N / (B × K + P + R)` | 每篇等权；LoCoMo 每段对话只计一次 |
| 单个记忆块 | `block_length / K` | 每块等权，包含末尾强制提交块 |

整篇倍率分布、单块倍率分布和“输入长度—记忆块数”散点图使用当前 checkpoint；RULER 可按长度和子任务筛选。逐文本列表支持排序，默认选中首行，表格高亮与下方的压缩分段条、块信息及 QA 始终对应同一篇文本。分段条内置范围手柄，支持缩放、拖动平移和悬停查看块信息。图表按需生成，不保存图片；首次读取当前 checkpoint 时仅缓存压缩统计和记录位置，完整回答按所选文本读取。

批次耗时按参与样本分摊，decode 耗时按当轮活跃样本分摊，汇总只累计一次批次 GPU 时间；峰值显存为 worker／batch 共享观测。

## 实现细节

### 批量计算与梯度归约

AE／LM 和 `compressor_only` 在一次更新的本卡样本内按 Writer 输入长度降序排列，再切 microbatch。单块 AE／LM 按目标分组前向；多块模式按写入轮次批量压缩，Reader 先向独立 memory 叶子累积梯度，再回传完整 Writer 图；QA 对同篇各题平均。全部 microbatch 完成后按实际全局样本数归约。该排序只作用于当前更新的样本集合。

联合后训练先固定参数、无梯度地采样和评分，再逐轮批量回放各文档的随机路径。Writer 按活跃文档批量执行边界判断与压缩；Reader 每轮从各文档取一道 QA，将梯度反传至独立 memory 叶子后释放 decoder 图。全部题目的 memory 梯度按题数与 G 归一，再与策略项一起回传 Writer，随后释放本轮路径图。

仅边界模式使用冻结 Writer 的状态回放边界 log-prob，跳过 Reader 反传和 memory 梯度回注；固定 Writer 保持 eval 状态，边界模块保持 train 状态。

Writer 与 Reader 均使用左侧 padding、attention mask 和各样本真实位置，损失只统计有效 tokens。分布式归约使用实际全局样本数；尾批中的空 rank 也参与 collective。

边界训练的 `BoundaryTrainer` 继承 verl 0.8.0 的 `main_ppo_sync.PPOTrainer`，直接运行其原生 `step`，由该方法控制 critic → actor 的更新顺序与 `critic_warmup` 判断。任务 hooks 对接同步正文采样及已缓存的奖励、旧概率和旧价值；不重复运行 Writer。SCST terminal、GRPO terminal、PPO GAE 以及策略／价值损失调用原生算法函数，两个 stepwise 相对优势保留任务定义。

`TrainingEngine` 继承 verl `BaseEngine`，RL 每次更新实际调用其 `train_batch` 生命周期：清空梯度、任务前后向、optimizer 更新。AdamW 构造与余弦调度使用原生 helper。任务微批、分段 autograd、FP32 复制式梯度 all-reduce、训练预算、dev、日志和原子单文件 checkpoint 仍由项目实现；当前不使用 `TrainingWorker`、FSDP、Ray worker 或 TransferQueue 服务。多卡进程组使用公共 runtime 的 verl 初始化。

### KV 复用与重算

采样、dev 和独立评估复用原文增量 KV，并保留已提交 memory 的前缀 KV。提交时只压缩 a，排除用于判断的新单元 b；随后续算新 memory 与 b。

固定动作回放中，同一提交段内的判断点共享因果前向，分别读取对应单元末位置。压缩按提交段执行 activation checkpoint，较长可见前缀的边界判断按段重算，答案输出头按块重算。

同一参数下的采样与回放概率比理论值为 1；两种计算路径的浮点差异通过 `replay_log_probability_drift` 记录。

### 跨段反传

仅 `update_mode=joint` 的策略训练配置 `training.backward.bptt_steps`，控制回放时压缩器的跨段梯度跨度。`boundary_only` 冻结压缩器，不配置 `backward`：

| 值 | 梯度传播 |
|---|---|
| `null` | 完整跨段反传 |
| `1` | 每次写入后截断供后续写入与判断使用的历史梯度 |
| 正整数 H | 每 H 次写入为一组，组内完整反传，组间将历史作为常量 |

计数按每条路径的实际写入次数进行，包含末尾强制提交。例如 `2` 将第 1～2 次写入分为一组，第 3～4 次分为下一组。

截断保留历史 memory 的前向数值及 Reader 到每个原始记忆块的直接 QA 梯度，去掉跨组的后续压缩、边界判断到早期 memory 的间接梯度。该设置只用于联合后训练的带梯度回放；梯度范数裁剪由 `training.optimizer.gradient_clip` 在更新前执行。

## 代码与产物

### 模块分工

执行组织沿用 v4.1 的配置、计划、训练恢复与独立评估方式，方法计算位于本模块：

```text
v3_1_1/
  config.py                 模型、计算与训练配置
  cli.py / io.py            共用参数转换与 JSON 原子写入
  execution.py              运行占用锁、GPU 选择与子进程管理
  preparation/
    pretrain.py / post.py   预训练裁剪与后训练 QA 选择
    length.py               正文长度分档与统计
  data.py                   AE／LM 样本与完整文档 QA
  compression.py            单块、多倍率随机与固定倍率分段
  modeling/                 压缩器、边界模块、流式状态与组批
  rollout.py                因果分段、采样与回放
  training/
    cli.py / plans.py       训练参数与计划
    run.py / train.py       启动调度与训练执行
    engine.py               预算、游标与 verl BaseEngine 任务适配
    verl_runtime.py         原生 PPOTrainer.step 的正文轨迹与更新 hooks
    dev.py                  固定倍率／策略验证、逐题生成与冻结基线缓存
    rl.py                   QA／SCST／GRPO 奖励、优势与反传
    advantages.py           原生优势／损失调用及 stepwise 适配
    ppo.py                  全局 GAE 白化、固定 rollout 与独立角色反传
    checkpoint.py           checkpoint 保存与恢复
    tracking.py             SwanLab 指标与面板
  evaluation/
    test.py / locomo.py / ruler.py   任务数据适配、协议、组批与评分
    plans.py                评估计划与累计 checkpoint 范围
    run.py / evaluate.py    checkpoint 调度与单 checkpoint 执行
    worker.py               单卡进程入口
    inference.py            共用正文压缩、Reader 生成与 Test NLL
    records.py / shards.py  结果身份、进度、样本分工与提交
    diagnostics.py          输出诊断
  dashboard.py              本地看板
  dashboard_compression.py  压缩统计、记录索引与分段图
  scripts/                  数据准备、训练、评估和看板入口
```

### 运行目录

下列路径相对于 `<output_root>/<run_id>/`：

| 路径 | 内容 |
|---|---|
| `plan/config.json` | `stage`、`seed`、基座、模型与计算设置 |
| `plan/train.json` | 本阶段训练配置和执行安排，均含 `objective`、`update_mode` 和 `compression` |
| `train/config.json` | 实际训练配置，含解析后的总步数 |
| `train/data.json` | 选样统计及数据、tokenizer、模型和初始化来源绑定信息；`compression_windows.train_recent_size/dev_recent_size` 记录实际生效的 recent 长度 |
| `train/metrics.jsonl` | 逐 optimizer step 的训练、dev 与资源指标 |
| `train/boundary/global_step_<N>/{train,dev}/` | 各 rank 的逐轨迹边界诊断与本次采样信息 |
| `train/summary.json` | 已完成步数与最终 checkpoint 路径 |
| `train/checkpoints/global_step_<N>/state.pt` | 可恢复的训练状态 |
| `plan/eval-<eval-id>/<task>.json` | 独立评估协议及累计 `checkpoints` 列表 |
| `eval/<eval-id>/<task>/global_step_<N>/` | 对应 checkpoint 的评估进度、明细和汇总 |
| `eval/<eval-id>/{test,locomo}/global_step_<N>/boundary.jsonl` | 每个评估正文的自主决策轨迹 |
| `eval/<eval-id>/ruler/global_step_<N>/lengths/<长度>/boundary.jsonl` | 该输入长度下各 RULER 样本的自主决策轨迹，附子任务 |

Checkpoint 固定包含 `config`、`model`、`optimizer`、`scheduler`、`cursor`、`rng`、`boundary_rng`、`binding`。`model` 包含 `memory_embeddings`、`adapter`、`boundary` 和 `value_head`；AE／LM 和 `compressor_only` 的 `boundary` 为 `null`，仅 PPO 的 `value_head` 有参数。`optimizer` 和 `scheduler` 都按 `actor`、`critic` 两个键保存；非 PPO 的 `critic` 为 `null`，PPO 保存两个独立状态。

`cursor` 包含训练主步数 `step`、已完成采样批次 `rollout_step` 和数据游标 `samples`。PPO 的 `step` 等于 critic scheduler 的实际更新次数，其他训练等于 actor scheduler 的更新次数；actor 的独立进度保存在其 scheduler 状态中。随机状态按 rank 保存，恢复时同时核对两侧更新进度、原配置、数据身份和游标，继续原生预热阶段。

评估计划绑定基座、模型结构（含 pending 步长、最短长度和默认 recent）、LoRA、计算设置和 seed，并保存固定数据与协议。任务 `status.json` 的 `selected_checkpoints` 记录本次执行范围，与计划的累计列表分别保存。`progress/worker_<rank>.jsonl` 持续保存已完成样本，flush 后可续评；同目录的 `worker_<rank>.json` 是阶段、批量和长度的展示快照，每次续评重建。完整 details 提交后生成最终 summary。

LoCoMo 入口调用[公共会话评估接口](../../../benchmarks/locomo/README.md)，复用会话分片、同正文组批和 Reader 缓存计算。
