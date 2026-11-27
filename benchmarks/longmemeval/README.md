# 20261126_LongMemEval 公共评估

创建时间：20261126 10:49:01 Asia/Shanghai

最后修订时间：20261127 20:33:44 Asia/Shanghai

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

激活已安装 Streamlit 的环境后，在项目根目录运行以下任一命令。将 `path/to/...` 替换为实际路径。

```bash
bash benchmarks/longmemeval/scripts/dashboard.sh --run path/to/run-a path/to/run-b
bash benchmarks/longmemeval/scripts/dashboard.sh --run-file configs/reproductions/benchmarks/longmemeval_runs.txt
```

- `--run DIR [DIR ...]` 直接传入一个或多个结果目录，可重复使用。
- `--run-file FILE` 读取 UTF-8 清单，每行一个结果目录，忽略空行与以 `#` 开头的注释行。

两种入口可单独或合用。结果目录须直接包含 `summary.json` 和 `details.jsonl`，相对路径以项目根目录为基准，重复目录会报错。

看板按方法、checkpoint／模型和评估设置分组，默认全选。方法按首次出现的顺序排列，合用时先读 `--run`，再按行读 `--run-file`。ICAE 优先按评估配置中的训练阶段排序，`pretrained` 排在 `finetuned` 前。同一方法内再按 checkpoint 名自然排序和评估参数数值排序。勾选会同步更新图表、统计和逐题对照。

S 和 Oracle 分别比较总体、六类题型和不可回答题的准确率。统计表补充题型宏平均、题数和裁判信息，逐题区支持跨设置对照同题回答。看板按题号核对问题、题型和参考答案，同一设置还须使用相同的数据和提问日期。

### 方法看板接入

[公共展示组件](dashboard.py) 的 `show_runs` 和 `load_runs` 均接收非空的 `(展示标签, Path 类型的结果目录)` 序列，标签须唯一。

| 接口 | 用途 |
|---|---|
| `show_runs(runs)` | 读取并校验结果，展示准确率、配置和逐题回答 |
| `load_runs(runs)` | 返回校验后的运行列表，每项含 `label`、`directory`、`summary` 和按 `question_id` 索引的 `records` |
| `show_scores(runs, colors=None, show_chart=True)` | 展示已加载运行的指标。设 `show_chart=False` 时仅展示统计表 |
| `show_answers(runs)` | 对照已加载运行的同题回答 |

新方法接入跨运行看板时，在[运行配置解析](../comparison_runs.py)中定义模型、checkpoint 和评估参数的读取规则。
