# 20261126_LongMemEval 公共评估

创建时间：20261126 10:49:01 Asia/Shanghai

最后修订时间：20261127 20:01:48 Asia/Shanghai

本目录维护 LongMemEval 的数据读取、按题生成、问答评分和结果格式。[官方仓库](https://github.com/xiaowu0162/LongMemEval) · [清洗版数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)

## 评估约定

S 和 Oracle 使用相同的 500 道题，各自保存预测和分数，通过原始 `question_id` 对齐。数据默认位于 `data/benchmarks/longmemeval/`。[数据下载](../README.md#2-数据准备)

| 设置（`--split`） | 官方文件 | 每题的历史 |
|---|---|---|
| `longmemeval_s_cleaned`（默认） | `longmemeval_s_cleaned.json` | 证据会话和干扰会话，约 40 场会话、11.5 万个 token |
| `longmemeval_oracle` | `longmemeval_oracle.json` | 作者为该题选出的证据会话 |

- **输入范围**：每题使用所选设置的完整会话历史、问题和提问日期。参考答案、题型、问题 ID 和证据标注不进入模型输入。
- **答案生成**：同一完整历史的题目分在同一卡，历史只准备一次。按组取题，不足一批时加入其他历史；新历史一起准备，各题独立贪心解码，每题默认最多生成 128 个新 token。S 和 Oracle 使用相同的模型及生成设置。
- **评分与续评**：回答齐全后，裁判逐题判定正误。续评沿用原数据和配置，补齐缺失的回答与判定。

## 公共接口

### 数据与输入构造

`load_samples(benchmark, data_dir)` 按原始题序返回 `LongMemEvalSample`。其中 `id` 保留原始 `question_id`，`question_type` 和 `answer` 供评分使用。公共层从 `sample.input`（`LongMemEvalInput`）构造正文与问题，交给生成适配器。[数据入口](../data.py) · [类型与格式](data.py)

| 输入字段 | 含义 |
|---|---|
| `sessions` | 按日期排序的会话，日期相同时保留原始顺序。每场会话含 `date` 和 `turns`，每轮含 `role` 和 `content` |
| `question` | 原始问题 |
| `question_date` | 该设置中记录的提问日期 |

`format_history(item)` 以 `Session date: <日期>` 开始每场会话，按原始轮次逐行写入 `user: <正文>` 或 `assistant: <正文>`。`format_question(item)` 依次写入 `Question date:` 和 `Question:`。两部分的顺序固定为历史在前、提问在后。

### 生成与调度

| 接口 | 约定 |
|---|---|
| `load_adapter(settings)` | 根据可序列化为 JSON 的方法配置加载模型，返回生成适配器。`settings` 记录模型或 checkpoint 身份、提示格式和解码设置 |
| `prepare_contexts(documents)` | 每项为 `(完整历史, 待答问题列表)`。按输入顺序为每份历史返回一份正文状态及准备好的问题 |
| `answer_batch(states, questions, complete)` | 从每题对应的正文状态并行生成，完成一题即回调保存。空回答仍作为预测进入评分 |
| `run_generation(...)` | 按原始题序汇总回答并返回 `predictions.jsonl` 的路径 |
| `run_evaluation(...)` | 按完整历史分配生成任务，在全部回答保存后启动公共评分，输出总体准确率 |

`add_evaluation_arguments(parser, output_root, checkpoint_layout=False)` 定义公共参数。按模型或 checkpoint 组织结果的方法传入 `checkpoint_layout=True`，并向 `run_evaluation` 传入 `checkpoint_name` 和评估设置前缀 `output_name`。Full-context 使用模型目录名作为 `checkpoint_name`，`output_name=""` 时默认结果目录名仅含所选 split。分词、模型窗口检查和生成由适配器负责；输入超出窗口时应报告错误。[生成接口](generate.py) · [调度入口](evaluate.py)

| 参数 | 含义与默认值 |
|---|---|
| `--split` | 选择 S 或 Oracle，默认 `longmemeval_s_cleaned` |
| `--benchmark-data` | 输入数据目录，默认 `data/benchmarks/longmemeval` |
| `--output-root` | 输出根目录，默认值由接入方提供 |
| `--output-id` | 结果目录名；Full-context 默认 `<split>`，ICAE、AutoCompressors 和 SDCC 默认 `<压缩设置>_<split>`，其他方法默认 `<output_name>_<YYYYMMDD-HHMM>` |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--max-new-tokens` | 每题最多生成的新 token 数，默认 `128` |
| `--resume` | 沿用原运行进度，默认关闭。续评时可调整 GPU 数和每卡批量 |

## 问答评分

- **裁判输入**：项目使用 `gpt-6-luna`。按题型选用官方提示词，将问题、参考答案和模型回答放入一条 user 消息。偏好题的参考答案是评分标准；问题 ID 含 `_abs` 时优先使用不可回答题的提示词。[官方判题脚本](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)
- **判定规则**：`judge_answer(sample, hypothesis)` 使用 `reasoning_effort=none`、`temperature=0` 和 `max_completion_tokens=10` 请求裁判。按官方规则，裁判回复转为小写后包含 `yes` 即判为正确。[裁判实现](judge.py)
- **独立评分**：`uv run -m benchmarks.longmemeval.score --run-dir <设置目录>` 读取该目录的 `evaluation_config.json` 和完整的 `predictions.jsonl`。`--benchmark-data` 可指定同一份数据的本地目录。`--resume` 在核对数据、预测和裁判配置一致后，只补评尚未保存判定的题目。[评分入口](score.py)

评分脚本自动读取项目根目录的 `.env`，已导出的同名环境变量优先。`OPENAI_API_KEY` 必填；`OPENAI_BASE_URL` 默认使用官方地址。使用兼容服务时，基础地址须支持上述模型和请求参数，脚本会在地址后拼接 `/chat/completions`。

```dotenv
OPENAI_API_KEY=替换为你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 产物与指标

ICAE、AutoCompressors 和 SDCC 写入 `<output-root>/eval/<checkpoint_name>/<output-id>/`，默认 ID 为 `<压缩设置>_<split>`，例如 `mem128_cr4_longmemeval_s_cleaned`。Full-context 写入 `<output-root>/eval/<model_name>/<output-id>/`，`model_name` 取 `--model-path` 指向的目录名，默认 ID 为 `<split>`。显式指定 `--output-id` 时以该值作为完整目录名；在相同命令后加 `--resume` 即可续评，自定义 ID 须沿用原值。其他方法写入 `<output-root>/eval/<output-id>/<split>/`，续评时指定原 ID。

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法、适配器、推理配置、设置名，以及数据目录、文件名、来源版本、题数和内容指纹 |
| `progress/` | 多卡调度按题保存已生成的回答，汇总后保留各卡日志 |
| `predictions.jsonl` | 按原始题序保存预测，每行包含 `question_id` 和 `hypothesis` |
| `scoring_config.json` | 裁判模型与请求参数、官方评分代码版本、设置名、数据与预测指纹 |
| `details.jsonl` | 每次判题成功后保存。每行包含预测、参考答案、题面元数据和 `autoeval_label`（裁判 `model` 和布尔判定 `label`） |
| `summary.json` | 全部题目评分完成后的评估配置、裁判配置和汇总指标 `metrics` |

`summarize_scores(records)` 汇总逐题判定，准确率按 0–1 保存。[官方指标汇总](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/print_qa_metrics.py)

| `metrics` 字段 | 统计口径 |
|---|---|
| `score` | 全部 500 道题中判为正确的比例，每题权重相同 |
| `groups` | 六个 `question_type` 各自的题数与准确率 |
| `task_averaged_accuracy` | 六个题型准确率的算术平均，每类权重相同 |
| `abstention` | 30 道不可回答题（Abstention）的题数与准确率。它们是历史证据不足的跨题型子集，仍计入所属题型和总体分数 |

## 结果展示

### 跨运行比较

[公共看板脚本](scripts/dashboard.sh) 使用当前环境的 Python 启动 Streamlit。启动前激活已安装看板依赖的环境。`--run-file` 读取目录清单，`--run` 直接接收结果目录；至少提供一个参数，也可同时使用以合并输入。

将需要比较的已评分目录保存为 `configs/benchmarks/longmemeval_runs.txt`，例如：

```text
# 每行一个结果目录
artifacts/reproductions/icae/eval/mistral_7b_ft_icae/mem128_cr4_longmemeval_s_cleaned
artifacts/baselines/full_context/eval/Qwen3-4B-Instruct-2507/longmemeval_s_cleaned
artifacts/reproductions/icae/eval/mistral_7b_ft_icae/mem128_cr4_longmemeval_oracle
```

清单使用 UTF-8 编码，每行一个直接包含 `summary.json` 和 `details.jsonl` 的目录；忽略空行和以 `#` 开头的整行注释。文件中的路径含空格时不用加引号。清单须非空，目录不递归展开。

```bash
source .venv/bin/activate
bash benchmarks/longmemeval/scripts/dashboard.sh \
  --run-file configs/benchmarks/longmemeval_runs.txt

# 也可直接指定运行
bash benchmarks/longmemeval/scripts/dashboard.sh \
  --run artifacts/reproductions/icae/eval/mistral_7b_ft_icae/mem128_cr4_longmemeval_s_cleaned \
        artifacts/baselines/full_context/eval/Qwen3-4B-Instruct-2507/longmemeval_s_cleaned
```

`--run DIR [DIR...]` 可连续传入多个目录，也可重复该参数。两种输入的相对路径均以项目根目录为基准，也支持绝对路径和 `~`；命令行路径含空格时加引号。运行别名统一由目录名自动生成，重名时补充父路径。所有输入中的结果目录须互不重复。

S 和 Oracle 各自使用带边框的比较区，区内按保存的配置列出“方法 → checkpoint／模型 → 评估设置”复选框，默认全选；支持整组或逐项勾选，只有一个 checkpoint 或没有评估参数时简化层级。运行按方法、checkpoint 自然顺序及评估参数数值排序；配置相同的重复运行用别名区分。S／Oracle 和裁判配置不作为压缩设置。

每个比较区的主图展示所选运行的总体、六类题型和不可回答题准确率，图例显示完整标签，柱顶显示分数。同一方法与 checkpoint 使用同一色系，不同评估设置用不同深浅；颜色按全部输入预先分配，勾选时保持固定。“详细统计”列出运行身份、裁判、题数和准确率，题型宏平均保留在表中；配置可展开查看。

底部汇集两个比较区的已选运行，按题型或“仅看不可回答题”筛选，通过原始 `question_id` 对照同题回答与裁判判定。问题和参考答案展示一次，回答每行最多三列，支持同时比较 S 和 Oracle。

参与比较的运行须有相同的题目 ID、问题、题型和参考答案，逐题记录须完整且题号唯一；同一 S／Oracle 设置还须有相同的数据来源、指纹和提问日期。看板读取已保存的分数，图表、统计与逐题回答随勾选同步更新。[展示接口](dashboard.py) · [共用布局](../comparison_dashboard.py)

### 方法看板接入

公共跨运行看板通过 [comparison_runs.py](../comparison_runs.py) 从方法配置提取 checkpoint／模型与评估参数。接入新方法时，在该模块明确其配置字段；任务名、结果目录名和运行别名不用于推断评估设置。

方法维护自己的 Streamlit 页面和启动脚本，通过 `show_runs(runs)` 嵌入 LongMemEval 的指标、运行配置和逐题回答对照。页面入口负责设置页面标题和选择结果目录。[公共展示组件](dashboard.py)

| 接口 | 用法 |
|---|---|
| `show_runs(runs)` | 接收非空的 `(展示标签, Path 类型的设置目录)` 序列，标签须各不相同。读取已评分产物、核对运行间的数据和题目一致性，展示准确率、配置及逐题回答 |
| `load_runs(runs)` | 接收与 `show_runs` 相同的标签与目录序列，核对运行间的数据和题目一致性，返回运行字典列表；每项包含 `label`、`directory`、`summary` 和 `records` |
| `load_run(label, directory)` | 读取一个设置目录，返回含 `label`、`directory`、`summary` 和 `records` 的字典，其中 `records` 按原始 `question_id` 索引 |
| `comparison_scores(metrics)` | 返回按主图顺序排列的“中文指标名 → 准确率”字典，供公共看板和 checkpoint 看板共用 |
| `show_scores(runs, colors=None, show_chart=True)` | 默认展示总体、六类题型和不可回答题准确率柱图，以及包含题型宏平均的“准确率统计”表。`colors` 指定运行标签对应的颜色，`show_chart=False` 时仅展示表格 |
| `show_answers(runs)` | 接收同样的运行对象列表，按题型和不可回答条件筛选，展示同题回答与裁判判定 |

跨运行校验集中在 `load_runs`，`show_runs` 复用该加载接口。自定义布局先调用 `load_runs`，再将返回值分别交给 `show_scores` 和 `show_answers`，即可分开展示概览与逐题内容。在方法的 Streamlit 页面中可直接调用下面的代码，同时展示同一次运行的 S 和 Oracle。

```python
from pathlib import Path

from benchmarks.longmemeval.dashboard import show_runs

run_dir = Path("artifacts/my-method/eval/run-1")
show_runs([
    ("My method · S", run_dir / "longmemeval_s_cleaned"),
    ("My method · Oracle", run_dir / "longmemeval_oracle"),
])
```

ICAE、AutoCompressors 和 SDCC 使用 `benchmarks.checkpoint_dashboard.run_dashboard(method, title, render_memory, argv=None)`。`--output-dir` 可连续接收多个 checkpoint 目录，也可重复指定该参数。LongMemEval-S 和 Oracle 各自列出两级复选框，默认全选。可勾选整个 checkpoint 或其下单独的 `output_id`，所有子项和已选数量始终可见。主图展示所选运行的总体、六类题型和不可回答题准确率。顶部图例列出完整运行标签，柱顶显示百分比分数。[checkpoint 看板入口](../checkpoint_dashboard.py)

运行按 checkpoint 名、压缩参数数值、`output_id` 排序。每个 checkpoint 使用一种色系，不同设置的深浅按全部输入预先分配。“详细统计”内的准确率表保留题型宏平均，与底部的逐题回答、裁判判定随选择同步更新。筛选前后配色一致，清空选择后该任务显示选择提示。[多目录命令与排序规则](../locomo/README.md#方法看板接入)

按运行 ID 保存结果的方法使用 `benchmarks.locomo.dashboard.run_dashboard(method, title, render_results, argv=None, qasper=False)`，`--output-dir` 指向运行目录。入口选择一个已评分任务，核对方法与任务后调用 `show_runs` 展示 LongMemEval；`render_results` 回调负责 LoCoMo 的方法专属页面。[公共方法入口](../locomo/dashboard.py) · [LaCT 接入示例](../../reproductions/lact/evaluation/dashboard.py)

方法环境需安装 Streamlit 等看板依赖，启动脚本使用该环境的 Python。以已有的 LaCT 页面为例，在项目根目录运行下面的命令，`--output-dir` 替换为实际评估运行目录。

```bash
PYTHONPATH=.:src python -m streamlit run reproductions/lact/evaluation/dashboard.py -- \
  --output-dir artifacts/lact/eval/run-1
```
