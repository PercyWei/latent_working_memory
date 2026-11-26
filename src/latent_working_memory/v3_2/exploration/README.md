# 20261126_v3-2 探索实验

创建时间：20261126 17:22:38 Asia/Shanghai

最后修订时间：20261126 23:50:37 Asia/Shanghai

本目录按探索任务组织代码。每项任务使用独立的运行入口，专用实现放在对应子目录中。

| 探索任务 | 运行入口 | 实现位置 |
|---|---|---|
| 数据统计 | [run_data_statistics.py](run_data_statistics.py) | [data_statistics/](data_statistics/) |
| 统计展示 | [data_statistics/dashboard.sh](data_statistics/dashboard.sh) | [show_data_statistics.py](show_data_statistics.py) |

## 数据统计

用 CPU 统计文本结构、证据分布和局部密度，筛选值得进入压缩探针实验的数据。只加载 Fast Tokenizer，不加载模型权重；这些属性不等于模型的边界敏感度。

### 数据与位置

下表目录相对于项目根目录，也可通过 `--dataset-dir` 指定其他位置。程序根据数据集和 split 读取目录中的对应文件，不自动下载。

| `--dataset` | `--dataset-dir` | `--split` | 读取文件 |
|---|---|---|---|
| `fineweb_factqa` | `data/fineweb-factqa-k512-seg1to3x_train1000_01-20261008` | `train` / `dev` / `test` | `<split>.jsonl` |
| `locomo` | `data/benchmarks/locomo` | 省略 | `locomo10.json` |
| `longmemeval` | `data/benchmarks/longmemeval` | `longmemeval_s_cleaned` / `longmemeval_oracle` | `<split>.json` |
| `qasper` | `data/raw/qasper` | `train` / `dev` / `test` | `qasper-<split>-v0.3.json` |

LoCoMo、LongMemEval 的准备方法见[公共 benchmark 说明](../../../../benchmarks/README.md#2-数据准备)。QASPER 使用 [AllenAI 官方数据](https://allenai.org/data/qasper)：解压 [train/dev](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz) 或 [test](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz) 后，将原始 JSON 放入表中目录。`data/delta-mem-qasper/` 缺少所需证据标注，不能作为本模块的 QASPER 输入。

各适配器的取样与证据规则：

- **FineWeb FactQA** 每条轨迹（原文的一个连续正文窗口）为一条记录。三个 split 均统计 `role="task"` 的任务题，`gate` 题只计入来源题数。证据和答案的字符区间均相对于窗口正文 `text`，自然段按正文中的非空行划分。
- **LoCoMo**：每段完整对话一条记录，只选类别 1–4。按原 session 顺序保留日期、说话人、正文和图片描述，以 `dia_id` 定位证据发言；空白／分号分隔的引用会拆开，错误编号不猜测修复。
- **LongMemEval**：每题的完整历史一条记录，按日期排序，并保持会话 ID 与正文对应。证据统一取 `answer_session_ids` 指定的完整会话，包含日期；不使用覆盖不完整的 `has_answer` 轮次标签。拒答题单列为不可回答。
- **QASPER**：每篇论文一条记录，保留标题、摘要、小节标题和正文。证据只用 `answer.evidence` 的完整段落，匹配时忽略首尾空白，保留原始区间。按原顺序选择第一份可答、纯文本且可唯一定位的标注，不合并替代证据，也不把 `extractive_spans` 答案当证据；仅引用标题／小节名的标注记为定位失败。

缺失、定位失败、不可回答或仅有不受支持的图表证据均记录原因。不同数据集的证据粒度不同，报告分组展示，不能直接把劈裂率混排为敏感度排名。

### 运行

在项目根目录运行，例如：

```bash
uv run -m latent_working_memory.v3_2.exploration.run_data_statistics \
  --dataset locomo \
  --dataset-dir data/benchmarks/locomo \
  --tokenizer /path/to/Qwen3-4B-Instruct-2507 \
  --output-dir artifacts/v3-2/exploration/locomo_statistics
```

省略 `--split` 时，程序按上表从左到右读取所选数据集的全部划分，各划分的指标分别汇总。所有列出的划分文件都须就位，缺少文件会在统计开始前报错。只统计一个划分时可传入其名称，例如 `--dataset qasper --split dev`。LoCoMo 没有划分，省略 `--split` 即读取 `locomo10.json`。

分段和末尾参数默认扫描下表中的取值集合，也可用空格分隔的值覆盖，例如：`--blocks 4 8 --min-block-tokens 32 --max-block-tokens 512 none --recent-tokens 0 256`。只传一个值即可统计单一设置。

### 完整参数

| 参数 | 必填／默认值 | 含义与约束 |
|---|---|---|
| `--dataset` | 必填 | 数据源，取值见上方数据表 |
| `--dataset-dir` | 必填 | 数据集目录，文件布局见上方数据表 |
| `--split` | 全部划分 | 不同数据集支持的 split 见上表 |
| `--tokenizer` | 必填 | 本地目录或 Hugging Face 模型 ID；需 Fast Tokenizer |
| `--output-dir` | 必填 | 输出目录，必须尚不存在 |
| `--max-samples` | 不限制 | 按读取顺序处理前 N 条文档（过滤前），所有划分共用此上限；正整数 |
| `--min-tokens` | 不限制 | 整篇长度下限（含端点）；指定时为正整数 |
| `--max-tokens` | 不限制 | 整篇长度上限（含端点）；指定时为正整数 |
| `--blocks` | `4 8 16` | 目标块数集合；正整数 |
| `--min-block-tokens` | `32 128 256` | 单块长度下限集合；正整数 |
| `--max-block-tokens` | `512 1024 2048 none` | 单块长度上限集合；`none` 表示不限 |
| `--window-tokens` | `128` | 密度窗口大小；正整数，不足半窗的尾窗舍弃 |
| `--recent-tokens` | `0 128 256 512` | 末尾覆盖范围集合；非负整数，`0` 关闭 |

所有 token 长度均由当前 tokenizer 在 `add_special_tokens=False` 下计算；同时指定长度上下限时，下限不得超过上限。JSON 数据源仍先解析完整文件，`--max-samples` 只限制处理条数。块数和块长参数只用于指标计算，不执行压缩或生成切片；分段不可行不会额外排除文档。

块数、最短块和最长块取笛卡尔积，默认共 36 组；最长块小于最短块的组合跳过并记入 `config.json`，若全部非法则报错。末尾范围独立统计 4 档，不与分段组合相乘。每篇只分词、定位证据和计算密度一次。

### 输出

输出目录须为新目录，包含：

- `report.md`：基础指标、分段扫描、末尾扫描及各项分母。
- `summary.json`：各参数组合分别汇总；QA 先在文档内平均，再按文档等权汇总。
- `documents.jsonl`：来源、基础指标和扫描明细，不复制正文或 token 序列。
- `config.json`：本次参数、数据位置和 tokenizer。

逐文档的 `segmentations` 保存各组合的分段可行性；逐题的 `blocks` 保存各块数下的原始等长切分几何，`recent` 保存各末尾范围的交集。汇总时只在等长分段可行的组合中使用切分风险指标。多组结果分别报告，不重复累计样本数。

### 指标口径

| 指标组 | 计算内容 |
|---|---|
| 长度与结构 | 总 tokens、结构单元数、单元长度 P50/P90；等长分段是否可行，以及是否存在恰好 B 块的合法自然边界分段 |
| 证据分布 | 每题证据片段数、去重叠后的 token 总数、最早到最晚证据的跨度；片段数不是推理跳数 |
| 切分风险 | 切点是否严格落入证据片段内部，以及证据涉及几个块；相接证据不算内部劈裂 |
| 局部密度 | 每窗每百词的实体提及代理和数字表达式数量，分别报告篇内均值、CV、P10/P90 |
| 证据与背景 | 与本题证据相交的窗口归为证据组，其余归为背景组；报告两侧平均密度及其差值 |
| 末尾覆盖 | 任一证据与最后 `recent_tokens` 相交的题目比例 |

实体指标是英文 **capitalized mention proxy**（大写形式的提及代理），不是 NER 或事实计数：连续大写开头的词合为一次提及，排除句首单个非全大写词。句首按句末标点和换行近似判断；数字支持整数、小数、千分位、日期和百分比。特征在全文提取，按起始 token 分配窗口，避免窗界截断人名或数字。

窗口等权计算均值和总体标准差，分位数使用线性插值。没有有效窗口或缺少某一侧密度时，相应指标为 `null`；密度全为零时均值为 0、CV 为 `null`。等长分段不可行的组合，其切分风险指标在汇总中记为缺失；各组分别记录有效分母。

范围选择（角色／类别）与证据检查分别记账。整篇长度不合格时，所选 QA 标为 `document_excluded`；没有可用证据 QA 时，文档标为 `no_answerable_qa`。分段不可行和指标缺失不额外排除样本。本入口统计最终输入正文，不做截取、拼接或语义可回答性验证。

### 代码与扩展

| 模块 | 职责 |
|---|---|
| [run_data_statistics.py](run_data_statistics.py) | 本统计任务的参数解析、分词、长度过滤和写出 |
| [data_statistics/adapters/](data_statistics/adapters/) | 每个数据源一个读取模块，定位结构与证据，保留来源 |
| [data_statistics/schema.py](data_statistics/schema.py) | `Document`、`Question` 和计算参数 |
| [data_statistics/metrics.py](data_statistics/metrics.py) | 接收统一文档与 token offsets，计算指标 |
| [data_statistics/report.py](data_statistics/report.py) | 聚合、分母与报告 |

新增数据实现独立读取函数，返回 `Document`，并在 `run_data_statistics.py` 的 `ADAPTERS` 中注册读取函数与按读取顺序排列的 split 到文件名的映射。适配器接收 `dataset_dir, split` 并读取对应文件；无划分的数据集使用空映射，适配器只接收 `dataset_dir`，`Document.split` 为 `None`。所有字符区间相对同一 `text`，采用左闭右开区间；结构单元有序且不重叠，证据允许重叠。用 `unit_kind`、`evidence_granularity` 标明粒度，以 `exclusion_reason` 保留失败题，勿静默丢弃。`source_ref` 保存原始位置；QA 范围计数使用 `source_qa_count`、`selected_qa_count` 和 `qa_selection_rule`。

若接入切片，应先重定位证据；缺失必需证据时标为 `incomplete_evidence`。重复来源保留原文标识，当前报告仅做描述统计，不把切片当独立样本检验显著性。

检查：`uv run pytest -q tests/v3_2`。

## 统计展示

展示脚本读取多个统计输出目录的 `config.json` 和 `summary.json`，不重新计算指标。使用当前统计入口生成的结果：

```bash
bash src/latent_working_memory/v3_2/exploration/data_statistics/dashboard.sh \
  --run \
    FineWeb-FactQA artifacts/v3-2/exploration/fineweb-factqa_statistics \
    LoCoMo artifacts/v3-2/exploration/locomo_statistics \
    QASPER artifacts/v3-2/exploration/qasper_statistics
```

一个 `--run` 后可连续传入多对“名称 目录”，也可重复使用 `--run`。名称须非空且唯一，含空格时加引号。启动脚本自动定位项目根目录并使用项目环境；结果目录的相对路径也从项目根目录解析。

页面提供三个视图：

- **基础指标**：选择文档／QA、指标和统计量，显示对比柱状图与有效分母。
- **分段扫描**：固定块长约束，对比不同块数下的劈裂率、占用块数或分段可行率。
- **末尾覆盖**：按末尾 token 范围对比覆盖率，独立于分段配置。

页面上方列出 tokenizer、密度窗口、长度筛选和样本量；来源标签保留结构／证据粒度。各运行分别展示，未运行的组合和缺失值不补零，明细表保留有效分母。
