# 20261126_LongMemEval 公共评估

创建时间：20261126 10:49:01 Asia/Shanghai

最后修订时间：20261126 11:18:00 Asia/Shanghai

本目录提供原版 LongMemEval 清洗数据的读取、批量生成和官方问答评分。S 和 Oracle 使用同一组 500 道题，分别保存生成结果和指标，以原始 `question_id` 对齐比较。[官方仓库](https://github.com/xiaowu0162/LongMemEval) · [清洗版数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)

## 数据与输入

从项目根目录运行以下命令下载两个文件。公共数据目录默认为 `data/benchmarks/longmemeval/`。

```bash
bash benchmarks/scripts/download_benchmarks.sh --benchmark longmemeval
```

| 设置 | 官方文件 | 每题的历史 |
|---|---|---|
| `longmemeval_s` | `longmemeval_s_cleaned.json` | 证据会话和干扰会话，约 40 场会话、11.5 万 tokens |
| `longmemeval_oracle` | `longmemeval_oracle.json` | 作者为该题选出的证据会话 |

每题有独立的对话历史及提问时间。两个设置各读本文件中的原始时间，按日期先后排列会话，同一时刻保持文件顺序。每场会话保留所有用户和助手消息及轮次顺序。模型输入只含会话日期、角色、消息正文、问题和提问日期。题型、含 `_abs` 的问题 ID、原始会话 ID、`has_answer`、`answer_session_ids` 及参考答案均留在生成接口之外。

## 方法接入与生成

`benchmarks.data.load_samples(benchmark, data_dir)` 返回该设置的样本。公共生成器按原始题序将样本的 `input` 交给适配器，每题单独建立记忆状态。方法实现一个模型加载函数 `load_adapter(settings)`，返回具有以下接口的对象。[输入类型](data.py) · [生成接口](generate.py)

| 接口 | 约定 |
|---|---|
| `answer_batch(inputs: list[LongMemEvalInput]) -> list[str]` | 按输入顺序为每题返回一条回答，保留空回答进入评分 |
| `item.sessions` | 按时间排列的会话。每个会话有 `date` 和 `turns`，每轮含 `role` 和 `content` |
| `item.question`、`item.question_date` | 记忆方法先按时间处理历史，读完后用这两个字段回答 |

方法负责模型加载、提示模板、分词、记忆和生成。将模型与 checkpoint 身份、实际提示与解码设置（含生成预算）写入 `settings`。同一方法在两个设置上使用相同配置，保留完整输入；模型窗口不足时应显式报告。批量大小仅控制每次送入适配器的题数。

方法入口可直接调用 `run_generation`，也可用以下公共 CLI。示例中的 `my_method.adapter:load_adapter` 和 `settings.json` 分别由接入方法提供。将 `--benchmark` 改为 `longmemeval_oracle` 可在同一 `output-id` 下生成 Oracle 的回答。

```bash
uv run --frozen python -m benchmarks.longmemeval.generate \
  --benchmark longmemeval_s --method my-method \
  --adapter my_method.adapter:load_adapter --settings settings.json \
  --output-root artifacts/my-method --output-id longmemeval-eval --batch-size 1
```

每批完成后原子保存回答。中断后用相同的模型配置、数据设置和运行目录追加 `--resume`，生成器从下一道题继续。生成阶段可在各方法自己的环境中运行。

## 官方评分

评分脚本会自动从项目根目录的 `.env` 读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。每项配置均优先使用 shell 中已导出的同名环境变量。项目的 Git 忽略规则已包含该文件。

```dotenv
OPENAI_API_KEY=替换为你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
```

`OPENAI_API_KEY` 必填。`OPENAI_BASE_URL` 默认为 `https://api.openai.com/v1`，使用官方接口时可省略。使用兼容接口时，填写支持 `gpt-4o-2024-08-06` 的 API 基础地址，脚本会在末尾拼接 `/chat/completions`。

```bash
uv run --frozen python -m benchmarks.longmemeval.score \
  --run-dir artifacts/my-method/eval/longmemeval-eval/longmemeval_s
```

每道待评题调用一次裁判，发送问题、参考答案或偏好 rubric 以及模型回答。中断后追加 `--resume`，沿用已经落盘的判定。在另一台机器评分时可追加 `--benchmark-data data/benchmarks/longmemeval`，公共层会核对数据内容与生成时一致。使用自有调度器时，也可按原始 ID 将预测与 `load_samples` 读取的样本配对，调用 `judge_answer(sample, hypothesis)`，再用 `summarize_scores(details)` 汇总下表格式的判定。[评分接口](judge.py)

裁判采用作者的 `gpt-4o-2024-08-06` 和原始分题型提示词，单条 user 消息，`temperature=0`、`n=1`、`max_tokens=10`。沿用官方脚本的判定规则，裁判回复转为小写后含 `yes` 计为正确。原始问题 ID 含 `_abs` 时，优先使用证据不足题的专用提示。S 和 Oracle 各自评分。[官方判题脚本](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py) · [官方指标汇总](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/print_qa_metrics.py)

## 产物与指标

每个设置的运行目录为 `<output-root>/eval/<output-id>/<benchmark>/`。

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法、适配器、推理配置、设置名，以及原始数据的目录、文件名、来源 revision、题数和内容指纹 |
| `predictions.jsonl` | 按原始题序保存的官方输入格式，每行含 `question_id` 和 `hypothesis` |
| `scoring_config.json` | 裁判模型、官方评分代码 revision、设置名、数据与预测指纹，供续评核对 |
| `details.jsonl` | 每次判题成功后保存。每行保留预测、原始参考答案、题面元数据和官方格式的 `autoeval_label`（`model`、布尔 `label`） |
| `summary.json` | 全部题目判定完成后的数据与模型配置、裁判配置和 `metrics` |

`metrics.score` 为总体准确率，即全部 500 题中答对的比例。`groups` 给出六个 `question_type` 各自的题数与准确率，`task_averaged_accuracy` 为六类准确率的算术平均。`abstention` 单列 30 道证据不足题的题数与准确率，这些题仍计入各自题型和总体分母。各项分数按 0–1 保存。
