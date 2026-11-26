# 20261126_LongMemEval 公共评估

创建时间：20261126 10:49:01 Asia/Shanghai

最后修订时间：20261126 15:03:56 Asia/Shanghai

本目录提供原版 LongMemEval 清洗数据的读取、批量生成和问答评分。S 和 Oracle 使用同一组 500 道题，分别保存生成结果和指标，以原始 `question_id` 对齐比较。[官方仓库](https://github.com/xiaowu0162/LongMemEval) · [清洗版数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f)

## 数据与输入

从项目根目录运行以下命令下载两个文件。公共数据目录默认为 `data/benchmarks/longmemeval/`。

```bash
bash benchmarks/scripts/download_benchmarks.sh --benchmark longmemeval
```

| 设置 | 官方文件 | 每题的历史 |
|---|---|---|
| `longmemeval_s_cleaned` | `longmemeval_s_cleaned.json` | 证据会话和干扰会话，约 40 场会话、11.5 万 tokens |
| `longmemeval_oracle` | `longmemeval_oracle.json` | 作者为该题选出的证据会话 |

每题有独立的对话历史及提问时间。两个设置各读本文件中的原始时间，按日期先后排列会话，同一时刻保持文件顺序。每场会话保留所有用户和助手消息及轮次顺序。模型输入只含会话日期、角色、消息正文、问题和提问日期。题型、含 `_abs` 的问题 ID、原始会话 ID、`has_answer`、`answer_session_ids` 及参考答案均留在生成接口之外。

## 运行基线

七个基线共用同一份数据和裁判。各自的评估脚本先在方法环境中生成并保存回答，生成进程退出后自动通过 `uv` 在项目根环境中调用裁判，最后输出准确率。运行前请配置裁判 API，模型参数、checkpoint 和推理窗口见各方法 README。[裁判配置](#问答评分)

| 方法 | 评估入口 | 配置与运行示例 |
|---|---|---|
| Full-context | [eval_longmemeval.sh](../../src/latent_working_memory/full_context/scripts/eval_longmemeval.sh) | [Qwen 完整上下文](../../src/latent_working_memory/full_context/README.md) |
| ICAE v2 | [eval_longmemeval.sh](../../reproductions/icae/scripts/eval_longmemeval.sh) | [预训练与指令微调 checkpoint](../../reproductions/icae/README.md) |
| AutoCompressors | [eval_longmemeval.sh](../../reproductions/autocompressors/scripts/eval_longmemeval.sh) | [Llama-2-7B-6k 与分段长度](../../reproductions/autocompressors/README.md) |
| SDCC | [eval_longmemeval.sh](../../reproductions/sdcc/scripts/eval_longmemeval.sh) | [Qwen3-4B 压缩器及压缩率](../../reproductions/sdcc/README.md) |
| LaCT | [eval_longmemeval.sh](../../reproductions/lact/scripts/eval_longmemeval.sh) | [LaCT checkpoint](../../reproductions/lact/README.md) |
| E²-TTT | [eval_longmemeval.sh](../../reproductions/e2_ttt/scripts/eval_longmemeval.sh) | [E²-TTT checkpoint](../../reproductions/e2_ttt/README.md) |
| In-Place TTT | [eval_longmemeval.sh](../../reproductions/in_place_ttt/scripts/eval_longmemeval.sh) | [Qwen3-4B 32K 与长输入设置](../../reproductions/in_place_ttt/README.md) |

下面的命令用完整上下文基线评估 S。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --output-id qwen3-4b-longmemeval --batch-size-per-gpu 1
```

- **数据设置**：使用 `--split longmemeval_oracle` 可在同一 `--output-id` 下评估 Oracle。两个设置分别保存回答和分数。
- **生成参数**：每张可见 GPU 加载一个模型副本，按题分工，默认每卡批量为 1。每题贪心生成，默认最多 128 个新 token，可用 `--max-new-tokens` 调整。
- **中断续跑**：在原命令中使用原 `--output-id` 并追加 `--resume`。生成中断时先补齐缺失的回答再评分。回答已齐时跳过模型推理，直接对尚无判定的题目评分。续跑时保持数据与生成设置一致，GPU 数量与每卡批量可调整。

Full-context 和三种 TTT 基线可在单卡上对多道独立问题组批。ICAE、AutoCompressors 和 SDCC 在每卡收到的批次内逐题压缩历史、读出回答并释放该题记忆，多卡之间并行。

七个入口均按日期拼接完整会话，每场以 `Session date: <日期>` 开头，逐行保留 `user: <正文>` 和 `assistant: <正文>`。提问日期与问题放在全部历史之后。对本项目的 Qwen3-4B-Instruct-2507 完整上下文输入，计入 QA 系统提示与聊天模板后，S 每题为 99,491–110,826 tokens，Oracle 为 259–22,567 tokens。其他方法使用自己的 tokenizer、历史编码和窗口限制。

## 方法接入

`benchmarks.data.load_samples(benchmark, data_dir)` 返回该设置的样本。生成器只将样本的 `input` 交给适配器，每题单独建立记忆状态，最终按原始题序汇总。方法实现一个模型加载函数 `load_adapter(settings)`，返回具有以下接口的对象。[输入类型与文本格式](data.py) · [生成接口](generate.py)

| 接口 | 约定 |
|---|---|
| `answer_batch(inputs: list[LongMemEvalInput]) -> list[str]` | 按输入顺序为每题返回一条回答，保留空回答进入评分 |
| `item.sessions` | 按时间排列的会话。每个会话有 `date` 和 `turns`，每轮含 `role` 和 `content` |
| `item.question`、`item.question_date` | 记忆方法先按时间处理历史，读完后用这两个字段回答 |

方法负责模型加载、提示模板、分词、记忆和生成。将模型与 checkpoint 身份、实际提示与解码设置（含生成预算）写入 `settings`。同一方法在两个设置上使用相同配置，保留完整输入；模型窗口不足时应显式报告。批量大小仅控制每次送入适配器的题数。

方法入口用 `add_evaluation_arguments` 添加公共参数，用 `run_evaluation` 串联多卡生成与公共评分。自有调度器若只需生成回答，可调用 `run_generation` 或以下 CLI，提供所用的适配器和配置。[基线调度入口](evaluate.py)

```bash
uv run --frozen python -m benchmarks.longmemeval.generate \
  --split longmemeval_s_cleaned --method my-method \
  --adapter my_method.adapter:load_adapter --settings settings.json \
  --output-root artifacts/my-method --output-id longmemeval-eval --batch-size 1
```

## 问答评分

S 和 Oracle 默认使用 `gpt-6-luna` 逐题评分。脚本按题型选用作者的原始提示词，将问题、参考答案（偏好题使用评分标准）和模型回答放在一条 user 消息中。问题 ID 含 `_abs` 时优先使用证据不足题的专用提示。裁判以 `reasoning_effort=none`、`temperature=0` 回答 yes 或 no，输出上限为 10 个 tokens（`max_completion_tokens=10`）。按官方判定规则，完整回复转为小写后含 `yes` 即计为正确。[官方判题脚本](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py) · [官方指标汇总](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/print_qa_metrics.py)

评分脚本会自动从项目根目录的 `.env` 读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。每项配置均优先使用 shell 中已导出的同名环境变量。项目的 Git 忽略规则已包含该文件。

```dotenv
OPENAI_API_KEY=替换为你的密钥
OPENAI_BASE_URL=https://api.openai.com/v1
```

`OPENAI_API_KEY` 必填。`OPENAI_BASE_URL` 默认为 `https://api.openai.com/v1`，使用官方接口时可省略。使用兼容服务时，填写支持 `gpt-6-luna` 及上述请求参数的 API 基础地址，脚本会在末尾拼接 `/chat/completions`。

独立评分入口可用于自有生成流程或已有预测。在项目根目录运行，`--run-dir` 指向包含 `evaluation_config.json` 和完整 `predictions.jsonl` 的设置目录。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/my-method/eval/longmemeval-eval/longmemeval_s_cleaned
```

评分中断后追加 `--resume`。裁判配置、数据和预测均与上次一致时，脚本从第一道尚未保存判定的题目继续。要用新裁判复评已有回答，可将 `evaluation_config.json` 和 `predictions.jsonl` 复制到新的设置目录，将 `--run-dir` 指向该目录开始评分。在另一台机器评分时可追加 `--benchmark-data data/benchmarks/longmemeval` 指定数据目录。

使用自有调度器时，可按原始 ID 将预测与 `load_samples` 读取的样本配对，调用 `judge_answer(sample, hypothesis)`，再用 `summarize_scores(details)` 汇总下表格式的判定。[评分接口](judge.py)

## 产物与指标

每个设置的运行目录为 `<output-root>/eval/<output-id>/<benchmark>/`。

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法、适配器、推理配置、设置名，以及原始数据的目录、文件名、来源 revision、题数和内容指纹 |
| `progress/` | 基线生成时按题原子保存已完成的回答，汇总后保留每张卡的运行日志 |
| `predictions.jsonl` | 按原始题序保存的官方输入格式，每行含 `question_id` 和 `hypothesis` |
| `scoring_config.json` | 裁判模型、推理强度、温度、输出上限、官方评分代码 revision、设置名、数据与预测指纹，供续评核对 |
| `details.jsonl` | 每次判题成功后保存。每行保留预测、原始参考答案、题面元数据和官方格式的 `autoeval_label`（`model`、布尔 `label`） |
| `summary.json` | 全部题目判定完成后的数据与模型配置、裁判配置和 `metrics` |

`metrics.score` 为总体准确率，即全部 500 题中答对的比例。`groups` 给出六个 `question_type` 各自的题数与准确率，`task_averaged_accuracy` 为六类准确率的算术平均。`abstention` 单列 30 道证据不足题的题数与准确率，这些题仍计入各自题型和总体分母。各项分数按 0–1 保存。

## 查看结果

各方法的 `scripts/dashboard.sh --output-dir <output-root>/eval/<output-id>` 可查看本次运行中已评分的设置。跨运行比较使用公共看板，每个 `--run` 指定展示标签和包含 `summary.json`、`details.jsonl` 的设置目录。下面的示例比较同一模型的 S 和 Oracle。

```bash
run_dir=artifacts/baselines/full_context/eval/qwen3-4b-longmemeval
uv run --frozen bash benchmarks/longmemeval/scripts/dashboard.sh \
  --run Qwen-S "$run_dir/longmemeval_s_cleaned" \
  --run Qwen-Oracle "$run_dir/longmemeval_oracle"
```

看板展示总体准确率、六类题表现、证据不足题准确率及裁判模型，并按原始问题 ID 并排显示各运行的提问日期、回答和判定。
