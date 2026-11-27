# 20261019_Full-context 基线评估

创建时间：20261019 20:16:32 Asia/Shanghai（UTC+08:00）

最后修订时间：20261127 16:39:18 Asia/Shanghai

Full-context 基线用原始模型读取完整输入，贪心生成答案。支持 LoCoMo、LongMemEval-S、LongMemEval-Oracle、QASPER 和 RULER。[公共数据准备](../../../benchmarks/README.md)

## 一、运行环境

从项目根目录运行，脚本使用项目的 `.venv`。推理默认以 BF16 加载权重，关闭模型级自动混合精度（autocast），注意力使用 FlashAttention 2。[主环境准备](../../../README.md#主环境与开发)

## 二、benchmark 评估

| 完整评估脚本共同参数 | 用途与默认值 |
|---|---|
| `--resume` | 在原评估命令后添加即可续评，沿用原模型、输出根目录和自定义 ID |
| `--model-path` | 本地基座模型目录，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--dtype` | 权重精度，支持 `bf16`（默认）和 `fp32` |
| `--attention-implementation` | 注意力实现，支持 `flash_attention_2`（默认）和 `sdpa`。FP32 权重使用 FlashAttention 2 时须开启 `--mixed-precision` |
| `--mixed-precision` | 为 FP32 权重启用 BF16 autocast，默认关闭 |
| `--seed` | 随机种子，默认 `42` |
| `--benchmark-data` | 输入数据目录。RULER 首次运行时必填，其他任务的默认路径见各任务小节 |
| `--output-root` | 输出根目录，默认 `artifacts/baselines/full_context` |
| `--output-id` | 默认 `<task>` |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |

- **运行方式**：每张可见 GPU 加载一个模型副本。完整输入与回答预算之和须在模型的位置窗口内。[运行环境](#一运行环境) · [数据下载](../../../benchmarks/README.md)
- **输入与解码**：使用基座模型的聊天模板和对应任务的系统提示，将完整会话历史或论文正文与问题放在同一条 user 消息中，从 assistant 起始标记贪心生成回答。Qwen Instruct 关闭 thinking。[输入构造](../prompts.py)
- **产物保存**：每次评估的配置、进度和结果统一写入 `<output_root>/eval/<model_name>/<output_id>/`。`<model_name>` 取 `--model-path` 指向的目录名，`<task>` 标识评估任务，默认路径见各任务小节。

<a id="locomo"></a>

### 1. LoCoMo

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/locomo` |

- **评估范围**：覆盖 10 段完整对话中类别 1–4 的 1,540 道题。[LoCoMo 评估协议](../../../benchmarks/locomo/README.md)
- **推理方式**：按对话分配 GPU，每段对话的公共前缀只预填一次。各题从该前缀的 KV 缓存独立生成回答。
- **回答长度与评分**：每题最多生成 50 个 token。按官方规则逐题计算 F1，再对所有题目取平均。
- **产物保存**：评估结果默认写入 `<output_root>/eval/<model_name>/locomo/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_locomo.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --batch-size-per-gpu 2
```

<a id="longmemeval-s-与-oracle"></a>

### 2. LongMemEval

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/longmemeval` |
| `--split` | 支持 `longmemeval_s_cleaned`（默认）和 `longmemeval_oracle` |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |

- **评估范围**：S 和 Oracle 使用相同的 500 道题。S 提供证据会话和干扰会话，Oracle 只提供证据会话。
- **输入长度**：按示例模型 Qwen3-4B-Instruct-2507 的 tokenizer 统计，计入系统提示和聊天模板后，S 每题为 99,491–110,826 个 token，Oracle 为 259–22,567 个 token。
- **推理方式**：相同的完整历史归到同一卡。模型按日期顺序读入会话历史，以最多 4,096 个 token 为一块并行预填新历史。各题从对应的前缀 KV 出发，读入提问日期和问题后并行作答。
- **自动评分**：生成全部回答后，脚本调用 `gpt-6-luna`，按官方分题型提示词逐题判定正误，汇总准确率。裁判密钥 `OPENAI_API_KEY` 可在项目根目录的 `.env` 中配置。[裁判配置与评分口径](../../../benchmarks/longmemeval/README.md#问答评分)
- **产物保存**：评估结果默认写入 `<output_root>/eval/<model_name>/<split>/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_longmemeval.sh \
  --split longmemeval_s_cleaned \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --batch-size-per-gpu 2
```

| 独立评分代码参数 | 用途与默认值 |
|---|---|
| `--run-dir` | 必填，所选设置的结果目录 |
| `--benchmark-data` | 默认沿用生成时记录的数据目录 |
| `--resume` | 用于续评分，默认关闭 |

- **独立评分**：已有完整回答时，可在项目根环境中直接评分，无需重新运行 Full-context 推理。

```bash
uv run -m benchmarks.longmemeval.score \
  --run-dir artifacts/baselines/full_context/eval/Qwen3-4B-Instruct-2507/longmemeval_s_cleaned
```

### 3. QASPER

| 完整评估脚本参数 | 用途与默认值 |
|---|---|
| `--benchmark-data` | 默认 `data/benchmarks/qasper` |
| `--split` | 支持 `test`（默认） |
| `--max-new-tokens` | 每题最多生成的 token 数，默认 `128` |

- **评估范围**：test 包含 416 篇论文、1,451 道题。[QASPER 评估协议](../../../benchmarks/qasper/README.md)
- **输入长度**：按示例模型 Qwen3-4B-Instruct-2507 的 tokenizer 统计，计入系统提示和聊天模板后，test 每题为 893–29,902 个 token。加上默认回答预算后均在模型的 262,144 个 token 窗口内。
- **推理方式**：按论文分配 GPU，每篇论文的公共前缀只预填一次。各题从该前缀的 KV 缓存独立生成回答。
- **评分口径**：在本地逐题计算官方 Answer F1。每题取各参考答案中的最高分，再对所有题目取平均。
- **产物保存**：评估结果默认写入 `<output_root>/eval/<model_name>/qasper_<split>/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_qasper.sh \
  --split test \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --batch-size-per-gpu 2
```

## 三、RULER 评估

新建 RULER 评估时，基座模型、精度、注意力实现、随机种子和每卡批量的默认值见[共同参数表](#二benchmark-评估)。

首次运行须用 `--benchmark-data` 指定基座模型对应的数据目录。`--lengths` 默认选择 8192、16384、32768 三档，完整输入和回答预算还须满足所选长度档位的上限。调度器按题分卡，将长度档位和生成预算相同的题目组批。`--batch-size-by-length` 按长度覆盖每卡批量，其余档位沿用 `--batch-size-per-gpu`。

- **产物保存**：评估结果默认写入 `<output_root>/eval/<model_name>/ruler/`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_ruler.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --benchmark-data data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507 \
  --lengths 8192 16384 32768 \
  --batch-size-by-length 8192=32 16384=32 32768=16
```

续评时沿用原模型目录和运行 ID。脚本从评估结果目录的 `eval.json` 读取原配置，可按长度调整每卡批量。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_ruler.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --resume --batch-size-by-length 8192=8 16384=4 32768=2
```

## 四、查看结果

评分完成后，每次评估目录中的 `summary.json` 保存汇总指标，`details.jsonl` 按原始题序保存回答及评分。LoCoMo、LongMemEval 和 QASPER 的 `evaluation_config.json` 记录模型身份、精度与聊天模板。

| 任务 | 看板指标与运行记录 |
|---|---|
| LoCoMo | 总体及分题型 F1，可按对话查看逐题结果。`execution.json` 记录执行状态，`work.json` 和 `progress/` 保存任务分配与进度 |
| LongMemEval-S / Oracle | 总体、分题型、题型宏平均及证据不足题准确率，可查看逐题判定和裁判模型。`scoring_config.json` 保存裁判配置 |
| QASPER | 总体及分答案类型的 Answer F1，可按论文查看问题、参考答案、模型回答和逐题分数。`predictions.jsonl` 按原始题序保存模型回答 |
| RULER | 按长度档位和子任务查看准确率。`summary.json` 另含推理批量、耗时与峰值显存，`eval.json`、`status.json`、`eval.log` 和 `progress/` 保存配置与运行记录 |

方法看板的 `--output-dir` 指向一个模型目录 `<output_root>/eval/<model_name>/`。看板扫描 `*/summary.json`，根据其中的评估信息识别任务。先选任务（含 RULER），再选对应的 `output_id` 查看该次评估。

```bash
bash src/latent_working_memory/full_context/scripts/dashboard.sh \
  --output-dir artifacts/baselines/full_context/eval/Qwen3-4B-Instruct-2507
```

如需按题号并排比较 LongMemEval 的两种设置，可将各自的评分目录传给公共看板。

```bash
uv run --frozen bash benchmarks/longmemeval/scripts/dashboard.sh \
  --run "Full-context S" artifacts/baselines/full_context/eval/Qwen3-4B-Instruct-2507/longmemeval_s_cleaned \
  --run "Full-context Oracle" artifacts/baselines/full_context/eval/Qwen3-4B-Instruct-2507/longmemeval_oracle
```
