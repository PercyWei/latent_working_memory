# 20261126_LongMemEval 公共评估

创建时间：20261126 10:49:01 Asia/Shanghai

最后修订时间：20261127 10:43:03 Asia/Shanghai

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

`add_evaluation_arguments(parser, output_root)` 定义下列公共参数。分词、模型窗口检查和生成由适配器负责；输入超出窗口时应报告错误。[生成接口](generate.py) · [调度入口](evaluate.py)

| 参数 | 含义与默认值 |
|---|---|
| `--split` | 选择 S 或 Oracle，默认 `longmemeval_s_cleaned` |
| `--benchmark-data` | 输入数据目录，默认 `data/benchmarks/longmemeval` |
| `--output-root` | 输出根目录，默认值由接入方提供 |
| `--output-id` | 运行 ID，默认 `<output_name>_<YYYYMMDD-HHMM>`，续评时须指定原 ID |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--max-new-tokens` | 每题最多生成的新 token 数，默认 `128` |
| `--resume` | 沿用原运行进度，默认关闭。续评时可调整 GPU 数和每卡批量 |

## 问答评分

- **裁判输入**：项目使用 `gpt-6-luna`。按题型选用官方提示词，将问题、参考答案和模型回答放入一条 user 消息。偏好题的参考答案是评分标准；问题 ID 含 `_abs` 时优先使用证据不足题的提示词。[官方判题脚本](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)
- **判定规则**：`judge_answer(sample, hypothesis)` 使用 `reasoning_effort=none`、`temperature=0` 和 `max_completion_tokens=10` 请求裁判。按官方规则，裁判回复转为小写后包含 `yes` 即判为正确。[裁判实现](judge.py)
- **独立评分**：`uv run -m benchmarks.longmemeval.score --run-dir <设置目录>` 读取该目录的 `evaluation_config.json` 和完整的 `predictions.jsonl`。`--benchmark-data` 可指定同一份数据的本地目录。`--resume` 在核对数据、预测和裁判配置一致后，只补评尚未保存判定的题目。[评分入口](score.py)

评分脚本自动读取项目根目录的 `.env`，已导出的同名环境变量优先。`OPENAI_API_KEY` 必填；`OPENAI_BASE_URL` 默认使用官方地址。使用兼容服务时，基础地址须支持上述模型和请求参数，脚本会在地址后拼接 `/chat/completions`。

```dotenv
OPENAI_API_KEY=替换为你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 产物与指标

每个设置的结果目录为 `<output-root>/eval/<output-id>/<split>/`。

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
| `abstention` | 30 道证据不足题的题数与准确率；这些题也计入所属题型和总体分数 |

## 结果展示

### 跨运行比较

公共看板脚本使用当前环境的 Python 启动 Streamlit。启动前激活已安装看板依赖的环境。`--run` 后按“展示标签、已评分的设置目录”成对传参，可连续指定多组，也可重复使用 `--run`。[公共看板脚本](scripts/dashboard.sh)

```bash
source .venv/bin/activate
bash benchmarks/longmemeval/scripts/dashboard.sh \
  --run 方法A-S artifacts/method-a/eval/run-1/longmemeval_s_cleaned \
        方法B-S artifacts/method-b/eval/run-1/longmemeval_s_cleaned \
        方法A-Oracle artifacts/method-a/eval/run-1/longmemeval_oracle
```

每个目录直接包含 `summary.json` 和 `details.jsonl`，展示标签须非空且各不相同，含空格时加引号。侧栏选择运行，页面展示总体准确率、题型宏平均、证据不足题准确率、分题型结果和裁判模型。按题型或证据不足条件筛选后，可通过原始 `question_id` 并排查看同题回答、参考答案及裁判判定。

参与比较的运行须有相同的题目 ID、问题、题型和参考答案，同一设置还须有相同的数据指纹。看板读取已保存的分数，支持不同方法、模型及 S／Oracle 设置的运行并列展示。[展示接口](dashboard.py)

### 方法看板接入

方法维护自己的 Streamlit 页面和启动脚本，通过 `show_runs(runs)` 嵌入 LongMemEval 的指标、运行配置和逐题回答对照。页面入口负责设置页面标题和选择结果目录。[公共展示组件](dashboard.py)

| 接口 | 用法 |
|---|---|
| `show_runs(runs)` | 接收非空的 `(展示标签, Path 类型的设置目录)` 序列，标签须各不相同。读取已评分产物、核对运行间的数据和题目一致性，展示准确率、配置及逐题回答 |
| `load_run(label, directory)` | 读取一个设置目录，返回含 `label`、`summary` 和 `records` 的字典，其中 `records` 按原始 `question_id` 索引 |
| `show_scores(runs)` | 接收 `load_run` 返回对象的列表，展示总体准确率、题型宏平均、证据不足题准确率和分题型表格 |
| `show_answers(runs)` | 接收同样的运行对象列表，按题型和证据不足条件筛选，展示同题回答与裁判判定 |

自定义布局使用后两个接口时，由调用方保证各运行题目对齐。完整的跨运行校验集中在 `show_runs`。在方法的 Streamlit 页面中可直接调用下面的代码，同时展示同一次运行的 S 和 Oracle。

```python
from pathlib import Path

from benchmarks.longmemeval.dashboard import show_runs

run_dir = Path("artifacts/my-method/eval/run-1")
show_runs([
    ("My method · S", run_dir / "longmemeval_s_cleaned"),
    ("My method · Oracle", run_dir / "longmemeval_oracle"),
])
```

使用公共方法入口 `benchmarks.locomo.dashboard.run_dashboard(method, title, render_results, argv=None, qasper=False)` 时，`--output-dir` 指向包含设置子目录的上一级运行目录。入口枚举已评分任务，核对方法与设置后，将 LongMemEval 交给 `show_runs` 展示，`render_results` 回调负责 LoCoMo 的方法专属页面。[公共方法入口](../locomo/dashboard.py) · [LaCT 接入示例](../../reproductions/lact/evaluation/dashboard.py)

方法环境需安装 Streamlit 等看板依赖，启动脚本使用该环境的 Python。以已有的 LaCT 页面为例，在项目根目录运行下面的命令，`--output-dir` 替换为实际评估运行目录。

```bash
PYTHONPATH=.:src python -m streamlit run reproductions/lact/evaluation/dashboard.py -- \
  --output-dir artifacts/lact/eval/run-1
```
