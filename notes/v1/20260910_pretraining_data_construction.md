# 20260910_预训练数据构建策略与实现

创建时间：20260910 15:44:35 UTC+08:00
最后修订时间：20260912 23:14:23 UTC+08:00

## 1. 数据来源

使用 HuggingFaceFW/FineWeb 的 `sample-10BT` 子集。FineWeb 是从 Common Crawl 网页提取、清洗的英语文本语料，`sample-10BT` 为约 100 亿 tokens 的抽样子集。原始 Parquet 已保存在服务器项目目录下的 `data/raw/HuggingFaceFW-fineweb/sample-10BT/`。

每条记录对应一篇网页文档，包含 `text`、文档 ID、URL、抓取日期及来源信息。构建过程保留原文和字符位置，AE 与 LM 的目标均直接截取原文。当前候选池预算为 100,000 篇文档；按固定 seed 混合读取文件、row group 和批内记录。

## 2. 数据类型与构建规则

### 数据类型

令 X 为写入记忆的文本，Y 为 X 后紧邻的原文。任务类型与边界类型交叉组合为四类数据：

| 类型 | 输入与目标 | 边界规则 |
|---|---|---|
| semantic AE | 写入 X，重建 X | X 从完整句界开始并在完整句界结束 |
| semantic LM | 写入 X，续写 Y | X 的起点、X/Y 切点及 Y 的终点均为完整句界 |
| random AE | 写入 X，重建 X | 从原文 token 位置随机截取 X，可在词或句子内部截断 |
| random LM | 写入 X，续写 Y | 随机选择 X 起点及 X/Y 长度，允许随机边界截断 |

AE 与 LM 独立抽样，每个 episode 只有一次写入和一个读取目标；代码中的 LM 任务名为 `continuation`。semantic 与 random 共用来源划分和任务内长度配额，但具体文本独立抽取，不要求逐条对应或长度完全相同。相同来源可提供多个片段，各类样本也可能重叠。

注：

- 共用来源划分：某篇原文属于 train，它在 semantic 和 random 中生成的所有样本都属于 train。
- 共用任务内长度配额：以 AE 为例，LM 同理。假如在 semantic AE 训练集中，32–64 tokens 的样本要求有 14,000 条，random AE 对应区间也要求有 14,000 条。

### 长度与配额

所有长度按训练基座 tokenizer 对最终文本切片独立分词计算，不含提示与特殊 token。本轮使用 Llama-2-7B-Chat tokenizer。

- AE 的 X、LM 的 X 与 Y 均为 **32–4096 tokens**；LM 满足 **0.3 ≤ |X| / (|X| + |Y|) ≤ 0.7**，X 与 Y 对应相邻字符区间。
- 按 X 长度划分七档：32–64、65–128、129–256、257–512、513–1024、1025–2048、2049–4096。
- 每个 split 的四类数据分别均衡分配七档配额。总数不能被七整除时，余数依次分配到前几档，各档最多相差一条；通过长度检查及去重的入选样本计入配额。

## 3. 主构造流程

**固定来源池 → 基本属性检查、去重与来源划分 → semantic 构造 → random 构造 → 自动审计。**

1. 检查来源字段和最小文档长度；按 ID、规范化 URL、全文及近重复关系聚类。每簇选择代表并固定 split，全部派生样本继承来源划分。
2. 按各 split、任务及 X 长度档的剩余配额选择候选。AE 与 LM 使用独立随机序列，每篇文档最多尝试 `candidates_per_document` 次。
3. semantic 随机选择保守句界起点，在各目标长度档内选择句界终点，形成连续原文 X；单句与多句使用同一候选规则。LM 在 X 后选择符合长度及比例约束的句界终点形成紧邻 Y；random 在 token 位置抽取连续原文。最终切片独立分词，检查 32–4096 tokens 及 LM 比例约束。
4. 按任务及规范化 X/Y 去重，长输入登记阻止跨 split 重复；random 同时查询 semantic 的输入登记。符合配额的候选直接入选，状态写入 `sample-decisions.jsonl`。
5. 补足各档后检查原文连续性、字符跨度、token 长度、来源隔离及配额，写入完成记录。random 完成后复用两类数据已完成的审计结果，核对来源契约与长度档数量。

数据用途是 AE 忠实重建及 LM 对记忆条件的适配。内容风格、自然上下文依赖、话题变化及表述水平保留为原始数据属性。

### semantic 边界规则

`segmentation.py` 使用 pySBD 提议边界，并实施保守切点处理。分句副本将换行等长映射为空格，保留引号；所有样本仍从未经改写的原文按字符偏移截取。

候选终点须有句末标点，允许标点后的闭引号或括号。省略号结尾作为不确定切点并入后续句段。句界右侧为小写续接等不确定情况时合并为更长句段。文档起始的小写残片及缺少可靠句末的尾部不提供外部切点。该规则提高边界精度，也会减少某些小写风格和无标点文本的 semantic 候选；random 保留这些原文的采样可能。

AE 检查 X 起点和终点；LM 检查 X 起点、X/Y 切点及 Y 终点。内部出现标题或列表属于原文结构。自动审计确认切点符合构造规则，真实句界正确率由独立复核估计。

## 4. 抽样检查

抽样检查在数据构造完成后独立执行，按检查目标预先定义判定标准、抽样方案和验收条件。检查结果用于评估构造流程并定位问题。

1. **抽取样本。** 随机面板对成品样本无放回均匀抽样，用于估计总体表现；分层诊断面板覆盖任务与长度档，并限制同一文档的重复出现，用于定位系统性问题。
2. **准备检查材料。** 保留样本 X/Y、来源文档和字符位置，并根据检查目标提供必要的原文上下文，使复核者能够区分原文属性与构造操作造成的问题。
3. **逐条复核。** 根据预先约定的标准记录通过、失败或存疑，同时保存理由与复核者信息。存疑项保持待确认状态。
4. **汇总与反馈。** 汇总随机面板的通过比例、未决数量及统计不确定性，按任务和长度分析分布。分层面板用于诊断，同文档样本的相关性及抽样代表性单独分析。发现系统性问题后优化构造流程，再使用独立样本复核。

当前检查目标是 semantic 样本的外部句界：AE 检查 X 起点和终点，LM 检查 X 起点、X/Y 切点及 Y 终点。全部切点正确才判为通过，任一切点位于句中则判为失败。检查依据切点两侧的原文；自然指代、叙述尚未结束和内容风格属于原文属性。

## 5. 实现、调度与恢复

代码位于 `src/latent_working_memory/data_preparation/`。

| 模块 | 职责 |
|---|---|
| `config.py`、`__main__.py` | 专用构造配置、长度配额及分阶段入口 |
| `sources.py`、`quality.py`、`dedup.py` | 本地 Parquet 读取、属性检查与来源去重 |
| `segmentation.py`、`fineweb.py`、`truncation.py` | 保守句界、semantic/random 独立任务构造 |
| `pipeline.py`、`resume.py` | 候选窗口、配额接收及原子恢复 |
| `audit.py` | 原文、来源、长度和规则切点审计 |
| `inspection.py` | 独立边界抽样与复核统计 |

CPU 预取下一窗口的分句与分词，主线程处理当前窗口并按顺序接收。每个窗口刷入样本和来源后原子更新 `progress.json`。同一构造契约使用 `--resume` 恢复，回退未提交尾部并重做该窗口。新构造使用独立输出目录。

正式配置保存在 `configs/data_preparation/`。单份 JSON 包含 `data`（tokenizer、数据来源、seed、来源划分与读取提示）和 `recipe`（候选预算、长度及任务配额）两部分；构造入口直接读取这份配置。

数据构造、主实验训练与评估共用项目根目录 `.venv/`，依赖由根目录 `pyproject.toml` 和 `uv.lock` 管理。环境准备使用 `uv sync --frozen`，执行入口为：

```bash
uv run --frozen python -m latent_working_memory.data_preparation \
  --config configs/data_preparation/fineweb-4096-doc100k.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir /path/to/new-data
```

阶段依次为 `sources`、`semantic`、`random`，默认 `all`。输出包括共享 `sources.jsonl` 与 `source-pool.json`，各版本的 train/dev/test、documents、sample-decisions、audit、preparation，以及最终 comparison。

### 实验数据选择

`python -m latent_working_memory.data_preparation.experiment` 从构造完成的样本中，按来源比例、任务与长度配额选择实验数据，并验证训练及原文评估的上下文预算。输入为来源／配比配置 `--spec`、训练配置 `--config` 和新目录 `--output-dir`。

输出按数据集名称组织为 `<输出目录>/<数据集名称>/`。来源数据集保存 `dev.jsonl`、`test.jsonl`；配置中的训练数据集保存 `train.jsonl`。同名来源与训练数据集合并到一个目录，并使用同一份 `preparation.json` 记录全部划分的数量、数据身份、来源和契约；同名训练数据集的配比须为该来源的 100%。混合训练集保存训练划分，评估显式引用各来源数据集。

根目录的 `selection.json` 记录选择配置、各任务／长度档配额、筛选统计，以及训练目录和评估目录映射。当前实验使用 `data/v1/fineweb-4096-doc100k_20260910/derived/pretrain-data-comparison-2048_20260910/`，具体布局、配置与复现命令见 [预训练数据类型对比实验](20260911_pretraining_data_comparison.md)。

## 6. 运行记录

### 配置与产物

本轮使用 FineWeb `sample-10BT` 和 Llama-2-7B-Chat tokenizer，构造与自动审计在 CPU 上执行。

| 设置 | 值 |
|---|---|
| 正式构造配置 | `configs/data_preparation/fineweb-4096-doc100k.json` |
| tokenizer | `/data/bywei/models/meta-llama/Llama-2-7b-chat-hf` |
| 来源候选预算 | 100,000 篇；本轮去重后保留 99,998 篇 |
| 来源 seed | 20260907 |
| train/dev/test 来源比例 | 0.9 / 0.05 / 0.05 |
| X/Y 长度范围 | 各 32–4096 tokens |
| LM 的 X 占 X+Y 比例 | 0.3–0.7 |
| X 长度档上限 | 64、128、256、512、1024、2048、4096 |
| 每篇文档候选尝试上限 | 64 |
| 候选窗口 | 4 篇文档 |

semantic/random × AE/LM 四种组合的目标与实际数量一致：

| 每种组合 | train | dev | test |
|---|---:|---:|---:|
| 样本数 | 98,000 | 2,100 | 2,100 |
| 每个 X 长度档 | 14,000 | 300 | 300 |

服务器项目目录为 `/data/bywei/projects/latent_working_memory`，以下路径相对此目录：

| 内容 | 位置 |
|---|---|
| 构造代码与阶段调度 | `src/latent_working_memory/data_preparation/` |
| 主环境 | `.venv/` |
| 本地 FineWeb Parquet | `data/raw/HuggingFaceFW-fineweb/sample-10BT/` |
| 数据成品、准备元数据及审计报告 | `data/v1/fineweb-4096-doc100k_20260910/` |

目录名记录 FineWeb 来源、X/Y 各自的 token 长度上限 4096、来源候选预算 `doc100k` 和创建日期。`doc100k` 表示最多读取 100,000 篇候选文档，包含随后被过滤或去重的文档。本轮保留 99,998 篇来源文档，semantic 和 random 各生成 204,400 条样本（train 196,000 + dev 4,200 + test 4,200），两类合计 408,800 条；实际数量保存在来源及数据准备元数据中。训练 run 名称中的规模按实际训练集样本数。

### 生成命令

以下命令采用当前源码入口与正式配置。执行目录为服务器项目根目录；FineWeb 原文与 tokenizer 使用已有本地文件。环境首次安装或依赖更新时在项目根目录执行 `uv sync --frozen`；环境就绪后的运行命令为：

```bash
cd /data/bywei/projects/latent_working_memory
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

python -u -m latent_working_memory.data_preparation \
  --config configs/data_preparation/fineweb-4096-doc100k.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir data/v1/fineweb-4096-doc100k_20260910 \
  --stage all
```

`all` 依次完成 `sources`、`semantic` 和 `random`，包括各阶段自动审计及最终联合检查。执行日志可由 shell 重定向到 `artifacts/v1/data-preparation/<本次构造名称>/build.log`，构造命令只依赖源码、正式配置、主环境及输入数据。

分阶段执行时，在相同环境中使用：

```bash
python -u -m latent_working_memory.data_preparation \
  --config configs/data_preparation/fineweb-4096-doc100k.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir data/v1/fineweb-4096-doc100k_20260910 \
  --stage sources

python -u -m latent_working_memory.data_preparation \
  --config configs/data_preparation/fineweb-4096-doc100k.json \
  --output-dir data/v1/fineweb-4096-doc100k_20260910 \
  --stage semantic

python -u -m latent_working_memory.data_preparation \
  --config configs/data_preparation/fineweb-4096-doc100k.json \
  --output-dir data/v1/fineweb-4096-doc100k_20260910 \
  --stage random
```

整轮执行与分阶段执行是两种启动方式。`sources` 创建新的来源池；后两阶段依次复用该池。未完成的 semantic 或 random 阶段保持原配置并追加 `--resume`，从对应 `progress.json` 恢复。现有目标目录已完成构造；重新构造须替换为尚未存在的输出目录。独立抽样检查通过 `data_preparation.inspection` 另行执行。

### 结果

构造耗时约 2 小时 55 分钟。semantic 和 random 各生成 204,400 条，总计 408,800 条。

原文一致性、来源隔离、长度、LM 比例、任务与长度档配额审计均通过，semantic 规则切点检查通过。独立边界抽查待执行，真实句界正确率待复核。
