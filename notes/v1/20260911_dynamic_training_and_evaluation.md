# 20260912_动态训练与 QA 评估（16:06:12 UTC+08:00）

创建时间：20260911 15:23:05 UTC+08:00  
最后修订时间：20260912 16:06:12 UTC+08:00

## 1. 训练过程

动态阶段从预训练 checkpoint 初始化，在固定容量的记忆上学习连续写入和问答。训练按指定 tokenizer 的完整文章长度筛选 SQuAD 数据，每篇文章构成一次训练轨迹。数据来源、划分和长度统计见 [动态数据构建](20260911_dynamic_data_construction.md)。

每卡 microbatch 固定为 1 篇文章；`gradient_accumulation_steps=B` 指定一次参数更新累积的文章数，默认为 1，有效文章 batch size 为 B。

双卡时将 B 篇文章分配到两个进程，各自完成轨迹反传后求和同步已按全局 B 归一化的梯度，再共同更新参数。

每个 optimizer step 的流程为：

1. 清空参数梯度，从打乱后的文章序列中依次取 B 篇。
2. 逐篇从空记忆开始，按原始顺序逐段编码，每个完整段落执行一次记忆更新。
3. 每次写入后，抽取当前段落问题和历史段落问题，计算回答损失。
4. 按配置执行完整或截断反向传播；每篇损失按读取数归一化后再除以 B，累积参数梯度。
5. B 篇文章全部完成后，统一裁剪梯度并更新一次参数。

每个 epoch 完整遍历一次符合长度条件的文章，遍历完成后重新打乱。若轮末不足 B 篇，从下一轮继续取文章补足；一次参数更新可以跨越 epoch 边界，不丢弃尾部文章。使用 `--epochs` 指定精确轮数时，仅训练终点允许不足 B 篇，并按实际篇数归一化。

文章之间不延续记忆或计算图，仅累积参数梯度。step、评估间隔和 checkpoint 间隔均按 optimizer 更新次数计量。

## 2. 记忆写入与读取

### 2.1 固定容量的连续更新

记第 t 个段落为 $x_t$，此前累计写入的源 token 数为 $S_t$：

$$
H_t[i]=P_{\mathrm{in}}(\mathrm{LM}_{\mathrm{frozen}}(x_t)[i]) + \mathrm{PE}(S_t+i)
$$

$$
M_t=\mathrm{Writer}(M_{t-1},H_t)
$$

冻结语言模型独立编码当前段落，BOS 的局部位置为 0，正文从 1 开始。提取特征后，通过可训练投影映射到记忆维度，再加入固定的正弦全文位置编码。全文位置按实际写入 tokens 累计，包含段落分隔符，不包含 BOS、问题和答案。

Writer 接收旧记忆和当前段落特征。首次写入分配 K 个记忆位置，之后容量保持为 K；跨段落信息通过记忆传递。全文位置不作为语言模型的 position IDs，但其长距离泛化仍需实验验证。

源长度与记忆位置数之比为 L/K，随写入增加而上升。例如 K=512 时，2K–4K 完整文章对应最终约 4–8 个源 tokens/记忆位置；轨迹初期的比值更小。

### 2.2 QA 读取

读取输入为 BOS、投影后的记忆、问题提示及答案。当前段落原文不直接进入记忆条件下的回答上下文。

同一写入位置的各个问题独立读取相同记忆；问题和答案不写回记忆，也不成为下一题的上下文。训练采用 teacher forcing，仅对参考答案及 EOS 计算损失；多参考问题使用首个参考答案作为训练目标。

### 2.3 可训练参数

| 模块 | 动态阶段 |
|---|---|
| 语言模型基座 | 冻结 |
| 写入特征投影、Writer | 训练 |
| 记忆读取投影、读取 LoRA | 训练 |
| 容量网络 | 冻结 |

读取损失经过冻结语言模型的计算传回记忆，监督写入与更新。新运行继承预训练模型参数，创建独立的 AdamW optimizer。

## 3. 问题采样与损失

每次写入后，从当前段落和此前段落的问题中分别均匀抽样。候选不足时使用全部候选；达到本次轨迹访问上限的问题不再入选。

| 配置 | 含义 | 示例值 |
|---|---|---:|
| `new_count` | 每次写入后的当前问题数上限 | 1 |
| `history_count` | 每次写入后的历史问题数上限 | 1 |
| `max_visits` | 每题在一篇文章轨迹内的访问上限 | 2 |

历史候选包括此前段落的全部问题，不要求之前已被问过。最后一个写入位置使用同样的抽样规则。读取计划在文章开始时确定，每个位置只能选择证据已经到达的问题。

文章顺序由 seed 与 epoch 编号确定，每轮独立打乱；问题抽样由 seed 与累计文章序号确定。第 step 次更新的第 i 篇对应序号 step×B+i（均从 0 开始），同一文章在不同轮次可得到不同问题计划。

恢复训练时，使用 checkpoint 中的累计文章数还原 epoch 与轮内位置，继续原有顺序，并恢复各进程随机状态。checkpoint 核对文章采样策略、数据、配置与进程数。评估固定文章面板及问题种子。

假设文章共选择 $R$ 次读取，第 $r$ 个目标包含 $T_r$ 个 tokens（含 EOS）：


$$
\ell_r=-\frac{1}{T_r}\sum_{j=1}^{T_r}\log p_\theta(y_{r,j}\mid M_{t_r},q_r,y_{r,<j}),
\qquad
\mathcal L_a=\frac{1}{R_a}\sum_{r=1}^{R_a}\ell_{a,r}
$$

文章内每次读取等权，当前问题与历史问题不另加权。一次参数更新的目标为 B 篇文章损失的均值：

$$
\mathcal L_{\mathrm{step}}=\frac{1}{B}\sum_{a=1}^{B}\mathcal L_a
$$

不同文章的读取数、TBPTT 反传次数可以不同，文章权重仍为 1/B。日志中的 `loss` 为文章损失均值，`target_nll` 按本次更新全部目标 tokens 加权。读取数、写入数、token 数和截断次数记录本次更新总量；`article_metrics` 保存逐文章损失及分段记录，`articles_seen` 记录累计处理的文章次数，`epochs_completed` 记录完成轮数，`articles_into_epoch` 记录当前轮已处理的文章数。

## 4. 梯度截断

`bptt_span=0` 使用完整 BPTT，文章结束后对全部读取损失反向传播。正值使用分段 TBPTT，记忆数值贯穿全文，梯度仅在当前分段内传播。

| 配置 | 截断条件 |
|---|---|
| `bptt_unit="tokens"` | 本段累计新写入源 tokens 达到 `bptt_span` |
| `bptt_unit="updates"` | 本段累计记忆更新次数达到 `bptt_span` |

两种模式均在完成当前段落写入及 QA 后检查边界，窗口不随各个 loss 移动。达到边界或文章结束时，对本段损失反向传播，再 detach 记忆。各段统一按文章总读取数及 B 归一化，完成 B 篇后更新参数；没有读取的分段直接 detach。

按 tokens 截断允许超过阈值。例如阈值为 1024、首段为 1500 tokens 时，先完成整段写入、QA 和反传，再截断。单段仍须满足编码窗口预算。

按更新次数截断直接控制递归链深度，但每段覆盖的文本量随段落长度变化。逐文章日志 `bptt_segments` 记录每段实际的 tokens 和 updates，另记录截断次数。

`gradient_checkpointing=true` 时，对完整 QA 读取执行激活重算，以计算换显存；该设置不截断记忆梯度链。

截断后，历史信息仍可用于回答，但后续损失无法跨越边界直接监督早期写入。分段开头的 loss 可回溯范围较短；完整 BPTT 则允许延迟问题沿整条更新链回传。

## 5. QA 评估

各条件共用问题、读取位置、参考答案和生成预算：

| 条件 | 回答输入 | 读取 LoRA |
|---|---|---|
| `memory` | 当前文章在该位置的记忆 + 问题 | 启用 |
| `no_memory` | 问题 | 启用 |
| `wrong_memory` | 另一篇文章完整写入后的同容量记忆 + 问题 | 启用 |
| `gold_paragraph` | 问题所属的原始证据段落 + 问题 | 启用 |
| `gold_paragraph_base` | 同一原始证据段落 + 问题 | 关闭 |

原文条件提供完整证据段落，不标亮答案；历史问题使用其最初所属段落。提示要求根据提供的文本作答，仅输出答案。两个原文条件均不输入记忆，并检查段落、问题和答案预算的总上下文长度。

证据段落是获得正确证据定位帮助的强参照，其与记忆条件的差距包含证据定位和压缩读取条件的差异。错误记忆仅匹配容量，不匹配源长度与更新次数，用于诊断对正确内容的依赖。

评估使用 greedy decoding，遇 EOS 停止，每题采用固定的新 token 上限。指标按各条件及 `all/arrival/delayed` 汇总：

- EM/F1：采用 SQuAD 1.1 归一化规则，多参考取最大值，数值范围 0–1。
- NLL：按目标 tokens 加权，包含 EOS。
- 生成触顶率：达到生成上限且未产生 EOS 的比例。

逐读取结果保存预测、参考答案、证据到达后的 token 距离和更新次数。训练前评估 step 0，之后按间隔及最后一步评估固定内部 dev 面板；独立评估可选择内部 dev 或官方 dev 派生的本地 test。默认面板按长度筛选后的记录顺序取前 N 篇，至少需要两篇不同文章。

## 6. 配置与代码实现

[示例配置](../../configs/v1/dynamic_squad_example.json)采用 2K–4K 完整文章、K=512、文章梯度累积数 B=1、按 1024 个源 tokens 截断、学习率 3e-5、weight decay 0.01、梯度裁剪 1.0 和生成上限 64 tokens。该范围的内部训练集有 64 篇文章；这些参数用于 pilot，尚未验证真实模型效果。

| 文件 | 功能 |
|---|---|
| [squad.py](../../src/latent_working_memory/v1/squad.py) | 根据长度记录选择文章，构造运行时轨迹并采样问题 |
| [dynamic.py](../../src/latent_working_memory/v1/dynamic.py) | 动态配置、训练、两种 TBPTT、五条件 QA 评估及运行入口 |
| [backbone.py](../../src/latent_working_memory/v1/backbone.py) | 段落特征提取、记忆读取与答案生成，支持读取 LoRA 开关 |
| [model.py](../../src/latent_working_memory/v1/model.py) | 联合记忆更新器及正弦位置编码 |
| [test_dynamic.py](../../tests/v1/test_dynamic.py)、[test_backbone.py](../../tests/v1/test_backbone.py) | 截断梯度、分段边界、评估输入、LoRA 开关与动态恢复测试 |

运行入口为 `python -m latent_working_memory.v1.dynamic train` 或 `evaluate`，分别传入 checkpoint、SQuAD token 长度记录、配置文件和输出目录。训练输出包括运行配置、训练日志、dev 汇总与逐读取结果、动态 checkpoint；恢复时核对配置和数据记录，恢复 optimizer、随机状态与步数。

SwanLab 使用 `latent-working-memory-v1` 项目，训练与配对评估共享显式指定的 group。K=1024 的双卡运行使用 [full](../../configs/v1/dynamic_squad_k1024/full.json)、[tokens1024](../../configs/v1/dynamic_squad_k1024/tokens1024.json)、[updates4](../../configs/v1/dynamic_squad_k1024/updates4.json) 三份配置；性能测试入口为 `dynamic_profile.py`，仅写本地日志。
