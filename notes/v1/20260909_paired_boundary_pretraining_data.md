# 20260909_完整句界与随机截断预训练数据（17:04:44 UTC+08:00）

创建时间：20260909 15:58:07 UTC+08:00

最后修订时间：20260909 17:04:44 UTC+08:00

## 构造规则

两套数据共享 FineWeb 来源、质量判定、近重复簇和 train/dev/test 划分。`semantic` 表示输入沿完整句界取样，允许自然话题变化；`random` 表示按 token 位置取连续原文跨度，允许句中截断。两套数据按任务与长度配对，后续训练验证使用 `semantic`。

1. **来源与筛选。** 从服务器已有 `HuggingFaceFW-fineweb/sample-10BT` 读取固定候选池，保留原文及来源字段，应用文档规则、近重复聚类和来源划分。句段规则及已配置的模型判定确定合格连续原文区域。本轮实际配方使用规则筛选；Qwen3.8-27B 的部署与评分仍待完成。
2. **自然父片段。** 以段落、连续句子组、单句和相邻段落组生成父片段 S；每个粒度优先覆盖不同长度。输入上限沿用 1024 tokens，保留完整句界。这里的粒度描述句段边界，单一主题的标注属于独立可选来源。
3. **任务构造。** AE 写入 S 并重建 S。对于具有合法内部句界的 S，随机选择切点，将 S 分成前缀 X 和后缀 Y，生成独立 LM 样本；X 写入记忆，Y 的全部 tokens 与 EOS 计算损失。Y 上限为 256 tokens；合法切点由剩余后缀的长度确定。单句片段提供 AE。
4. **随机截断配对。** 对每个自然样本，从同一合格连续原文区域抽取 token 起点，寻找与自然样本等长的 X；LM 继续提取紧邻且等长的 Y。分词器提供原文字符位置，每次切片重新分词，匹配实际独立编码后的长度。候选起点先排除剩余长度明显不足的位置，再随机尝试至多 64 个起点；边界重分词允许有限位置调整。纯空白跨度继续尝试其他位置，无法匹配时同时移除该对样本。
5. **去重与输出。** 跨来源的相同长输入片段按两套版本共同去重，配对样本保留相同文档和 split。随机版本继承自然样本的粒度作为匹配分层标识。`pair_id` 标识跨版本对应关系；`boundary_variant`、输入、目标、父片段与合格区域的字符范围记录实际构造方式。
6. **自动检查与后置抽查。** 两套数据分别检查输入 token 与原文、LM 连续性、排除区域、来源隔离和合法容量；逐对核对任务、输入与目标 token 数。全部通过后写入两套完成记录。质量抽查通过独立命令对完成数据抽样，复核结果用于下一轮配方调整。

随机样本的长度来自其自然配对样本，具体原文位置随机抽取。两套数据因此具有完全相同的输入、目标联合长度分布；同源文本内容及句界位置可以不同。自然句界偶合与覆盖整个合格区域的样本会保留，句界判定忽略跨度两端的空白，`comparison.json` 单独记录实际句界截断比例和相同跨度比例。

## 数据与训练接口

```text
data/v1/fineweb-paired-20260909/
  semantic/
    train.jsonl  dev.jsonl  test.jsonl
    documents.jsonl  candidates.jsonl  document-decisions.jsonl
    audit.json  preparation.json
  random/
    train.jsonl  dev.jsonl  test.jsonl
    documents.jsonl  candidates.jsonl  document-decisions.jsonl
    audit.json  preparation.json
  comparison.json
```

每个 episode 包含一次写入和一个 AE 或 continuation read。训练按长度区间、文档、粒度、样本和容量采样，容量由实际写入 token 数确定。每次参数更新分别按 AE、LM 有效样本数归一化，再加权相加。梯度累积的归一化基于整个参数更新批次，SwanLab 分别统计实际存在的任务损失及监督量。评估面板同时按任务、粒度与长度分层，自由重建名额在 AE 样本中计数。

构造命令：

```bash
uv run python -m latent_working_memory.data_preparation \
  --config configs/v1/pretrain_a800.json \
  --recipe configs/data_preparation/fineweb.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir data/v1/fineweb-paired-20260909
```

训练入口的 `--data-dir` 指向 `data/v1/fineweb-paired-20260909/semantic`。两套版本各自具有完整的训练数据接口；配对字段与固定来源划分用于后续边界对照实验。

## 实现与验证

- `data_preparation/fineweb.py`：完整自然片段与内部前缀—后缀任务构造。
- `data_preparation/truncation.py`：原文 token 跨度随机抽取、精确长度匹配与双版本统计。
- `data_preparation/pipeline.py`：共享筛选、配对去重、文档配额与两套数据发布。
- `data_preparation/audit.py`、`inspection.py`：单任务 episode 自动检查与独立质量抽查。
- `v1/data.py`、`sampling.py`、`training.py`、`evaluation.py`、`tracking.py`：独立 AE/LM 样本的索引、损失、恢复、评估和可视化。

本地 66 项 v1 测试通过。验证包含精确原文恢复、输入／目标逐对等长、随机截断、LM 写入前缀、同源划分、梯度累积等价性、真实小型 Llama 的训练与精确恢复。服务器已完成两套数据的构造及全量自动检查。

实际数据位于服务器 `a800:/data/bywei/projects/latent_working_memory/data/v1/fineweb-paired-20260909/`。本地 `artifacts/v1/data-preparation/fineweb-paired-20260909/` 保存准备报告、比较结果、测试记录与抽查样本。

两套版本各自包含：

| 划分 | 源文档数 | AE 样本 | LM 样本 | 合计 |
|---|---:|---:|---:|---:|
| train | 10,000 | 139,608 | 86,414 | 226,022 |
| dev | 512 | 7,200 | 4,410 | 11,610 |
| test | 512 | 7,230 | 4,455 | 11,685 |

两个版本的输入与目标 token 数逐对完全一致。随机版本训练输入的句界截断比例为 AE 91.9933%、LM 90.2527%；与自然父片段原文范围完全相同的比例分别为 7.7646%、9.4985%。构造过程中共排除 64 对无法匹配长度的样本和 36 对跨来源重复片段。

训练集的实际长度分布如下。两套版本在每个区间的样本数相同，训练访问权重由采样配置独立确定。

| 输入长度 | 每套样本数 | 每套样本占比 | 训练目标访问占比 |
|---|---:|---:|---:|
| 1–32 | 75,484 | 33.40% | 10% |
| 33–128 | 106,057 | 46.92% | 30% |
| 129–512 | 39,551 | 17.50% | 40% |
| 513–1024 | 4,930 | 2.18% | 20% |

确定性采样预演的 1,000 次访问在上述区间分别为 100、301、385、214 次；两套版本的对应样本与容量一致。抽查链路已分别抽取 200 个随机样本和 200 个按任务、粒度、长度分层的样本，判定字段留待独立复核。
