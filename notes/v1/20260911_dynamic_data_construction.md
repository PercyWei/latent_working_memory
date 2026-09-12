# 20260911_动态训练数据分布与使用流程（17:41:08 UTC+08:00）

创建时间：20260911 10:54:45 UTC+08:00  
最后修订时间：20260911 17:41:08 UTC+08:00

## 1. 数据来源

使用官方 SQuAD 1.1 原始文件 `data/raw/squad/train-v1.1.json`、`dev-v1.1.json`。数据按文章、段落及问题答案对组织，正文、段落顺序、问题、答案和字符标注完整保留。本文的文章指 SQuAD 已发布的同篇文章段落集合，不表示完整 Wikipedia 页面。

准备阶段记录数据划分及指定 tokenizer 下的 token 长度；训练和评估据此选择文章，读取原始文本后再构造写入与问答任务。阶段安排见[动态训练方案](20260910_dynamic_memory_training_plan.md)。

## 2. 数据分布与训练选择

### 统计

使用当前 pretrain 的 Llama-2-7B-Chat tokenizer。每个完整段落追加两个换行后独立分词，文章长度为各段长度之和。统计覆盖全部原始文章与段落，长度包含追加换行的 tokens，不含标题、BOS、问题或答案。

| 指标 | 官方 train | 官方 dev |
|---|---:|---:|
| 文章数 | 442 | 48 |
| 段落数 | 18,896 | 2,067 |
| 问题数 | 87,599 | 10,570 |
| 文章长度最小值 | 822 | 3,395 |
| 文章长度中位数 | 6,625 | 7,694.5 |
| 文章长度 P90 | 13,329.6 | 13,012.1 |
| 文章长度最大值 | 23,405 | 21,331 |
| 每篇段落数中位数 / 最大值 | 36 / 149 | 43.5 / 98 |
| 每篇问题数中位数 / 最大值 | 169 / 817 | 197 / 810 |
| 单段长度中位数 / 最大值 | 168 / 1,070 | 174 / 873 |
| 超过 512 tokens 的段落 | 65 | 15 |
| 超过 4095 tokens 的段落 | 0 | 0 |

P90 为第 90 百分位；长度均以 tokens 计。

| 文章长度区间 | 官方 train 文章数 | 官方 dev 文章数 |
|---|---:|---:|
| ≤1024 | 1 | 0 |
| (1024, 2048] | 7 | 0 |
| (2048, 4096] | 74 | 3 |
| (4096, 8192] | 189 | 27 |
| (8192, 16384] | 149 | 16 |
| (16384, 32768] | 22 | 2 |
| >32768 | 0 | 0 |

官方 train 中 81.4% 的文章流超过 4096 tokens。以每个问题的完整证据段落末尾到文章末尾计算潜在保持距离，中位数为 3,995 tokens，最大值为 23,245；42,901 个问题的距离至少为 4096 tokens，占 49.0%。这些是可安排的延迟距离，不是已经测得的模型记忆表现。


### 训练选择

一篇文章对应一条从空记忆开始的轨迹，一个完整段落对应一次写入。所有原始段落均能在当前 4096-token 基座写入窗口内加 BOS 处理；标题仅用于来源记录。

运行时按完整文章长度选择训练来源。完整轨迹主要位于 4K–16K，另有少量 16K–24K 轨迹；≤2K 的文章在官方 train 中只有 8 篇，内部 dev 和官方 dev 均没有。需要短轨迹时，可从既定来源划分内显式选择文章开头的连续完整段落，并与全文任务分别报告。

文章长度范围、每次读取的问题数、重复次数、记忆容量 K 和反传窗口由运行配置确定。轨迹内 K 固定，累计写入长度为 N_t 时，token/记忆位置比为 N_t/K。前向轨迹长度与 BPTT/TBPTT 反传窗口分别设置，不要求一次运行使用全部数据或对全文执行完整反传。派生前缀、问题和读取位置不计为新的独立文章来源。

## 3. 准备与运行时流程

**读取原始 JSON → 标注检查与来源分组 → 记录划分与 token 长度 → 运行时筛选文章 → 按段落写入 → 按策略读取。**

1. **原始标注检查。** 检查 SQuAD 1.1 标识、文章标题、段落、问题 ID 及答案。验证 `context[answer_start:answer_start + len(text)] == text`，不匹配时明确报错，不搜索其他位置替代。
2. **来源划分。** 同标题或含完全相同段落的文章按传递关系归组。仅来自官方 train 的来源组用 seed `20260907` 确定约 90:10 的内部 train/dev。含官方 dev 的组保留官方 dev 作最终评估，训练侧重叠文章标记 excluded。原始文件不删除记录。
3. **长度记录。** 保存每篇文章的原始位置、来源组、split、段落 token 数、文章总长度和问题数量，供运行时筛选。
4. **选择轨迹。** `SquadDataset.select` 按 split 和闭区间 `[min_tokens, max_tokens]` 返回完整文章标识。`episode` 按需将选中文章转换为内存中的统一 Episode；`paragraph_count` 可显式选择连续前缀。
5. **构造读取候选。** 以答案所在的完整段落作为保守证据范围，在该段提交后将对应问题加入可读取候选池。同一问题的参考答案按原序去除重复文本。
6. **运行时采样。** 每个写入完成后，`sample_reads` 从当前段落问题与更早段落问题中分别抽样。输入参数明确指定 `new_count`、`history_count` 和 `max_visits`，由调用方提供本次轨迹的随机源与访问计数。候选不足时按实际数量读取。
7. **模型使用。** 调用方将已提交记忆和选中问题交给读取路径。参考答案用于 teacher-forcing NLL；问题、gold 和生成回答均不写回记忆。动态 trainer 负责模型前向、反传和本次上下文预算检查。

### 读取的因果边界

当前刚提交段落的问题属于新问题，更早段落的问题属于历史问题；未来段落的问题不可选。历史问题可在后续任一提交边界再次选中，文章末尾使用相同采样规则。完整段落边界是本协议的证据可用位置，不表示最早可回答位置。

访问计数仅对选中的问题增加，达到 `max_visits` 后不再选取。每次执行新 episode 时创建新的计数器。固定随机种子、参数和边界调用顺序可重现同一读取轨迹；评估应固定这些设置。

返回的 read 将 `prefix_end` 设为当前读取位置，保留原始证据范围与稳定问题 ID。记录一次读取时使用 `(episode_id, read_id, prefix_end)` 区分同一问题的不同保持距离。

固定 QA 提示为：

```text
Answer the question using the information stored in memory. Give only the answer.
Question: {original_question}
Answer:
```

训练选择第一个合法参考并附加 EOS，评估对全部合法参考计分。模型执行与损失计算见[动态训练与 QA 评估实现](20260911_dynamic_training_and_evaluation.md)。

## 4. 检查与测试

数据测试覆盖以下行为：

- 答案字符标注错误明确失败；同源文章跨划分隔离。
- 运行时正文 tokens 与原始段落逐一对应，前缀视图仅包含指定前缀。
- 读取只使用已到达证据，抽样可重现、无同次重复，并遵守每题访问上限。
- 运行时采样不改变候选池；文章终点在零读取预算下不产生自动重问。
- 原文或 tokenizer 的段落长度与记录不一致时提示重新计算。

来源隔离检查覆盖当前 SQuAD 数据，未对 FineWeb 或其他 QA 数据集开展跨库重叠核对。

## 5. 实现与入口

| 模块 | 职责 |
|---|---|
| [`data_preparation/squad.py`](../../src/latent_working_memory/data_preparation/squad.py) | 原始数据校验、来源划分及 token 长度记录 |
| [`v1/squad.py`](../../src/latent_working_memory/v1/squad.py) | 完整文章筛选、按需构造内存视图和运行时问题抽样 |
| [`v1/data.py`](../../src/latent_working_memory/v1/data.py) | 复用已有 Episode/Source/Read/Reference 契约 |
| [`test_squad_preparation.py`](../../tests/v1/test_squad_preparation.py) | 数据划分、token 长度及运行时采样测试 |

```bash
.venv/bin/python -m latent_working_memory.data_preparation.squad \
  --train-file data/raw/squad/train-v1.1.json \
  --dev-file data/raw/squad/dev-v1.1.json \
  --tokenizer data/models/Llama-2-7b-chat-hf \
  --output-file data/v1/squad/llama-2-7b-chat_index.json \
  --seed 20260907
```

输出为显式指定的 JSON 文件，例如 `llama-2-7b-chat_index.json`，保存原始文件和本地 tokenizer 路径、来源划分及长度信息。正文与问答仍从原始 JSON 加载。目标文件必须不存在，允许不同 tokenizer 的记录保存在同一目录；源文件或 tokenizer 改变后重新生成对应记录。

最小运行时调用：

```python
from pathlib import Path
import random
from latent_working_memory.v1.squad import SquadDataset, sample_reads

squad = SquadDataset(Path("data/v1/squad/llama-2-7b-chat_index.json"))
documents = squad.select("train", min_tokens=2048, max_tokens=4096)
episode = squad.episode(documents[0])
visits = {}
rng = random.Random(42)
for prefix_end in episode.write_ends:
    # 实际 trainer 先完成对应原文写入，再取得当前读取任务。
    reads = sample_reads(episode, prefix_end, 1, 1, rng, visits, max_visits=2)
```

示例选择 2K–4K 文章，每个边界至多抽取一个新问题和一个历史问题，每题最多读取两次；这些数值仅展示调用方式，正式配置由实验确定。

## 6. 运行记录

SQuAD 数据划分与 token 长度记录保存在 [llama-2-7b-chat_index.json](../../data/v1/squad/llama-2-7b-chat_index.json)。长度统计的关键结果保存在本文第 2 节。

| 划分 | 文章数 | 原始段落数 | 原始问题数 | 输入 tokens |
|---|---:|---:|---:|---:|
| train（官方 train） | 398 | 17,195 | 79,414 | 3,135,319 |
| dev（官方 train） | 44 | 1,701 | 8,185 | 310,729 |
| test（官方 dev） | 48 | 2,067 | 10,570 | 391,606 |
