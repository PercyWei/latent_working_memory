# 20261019_Full-context 基线评估

创建时间：20261019 20:16:32 Asia/Shanghai（UTC+08:00）

最后修订时间：20261123 18:27:04 Asia/Shanghai

Full-context 基线使用原始模型和完整上下文评估 LoCoMo 与 RULER。模型通过原生 chat template 编码统一的 QA system 提示和包含完整题面的 user 消息，再按任务规定的生成预算进行贪心解码。

[提示原文](../prompts.py) · [数据与评分](../benchmarks/README.md)

## 1. 启动评估

准备数据后，在项目根目录按任务启动评估。`CUDA_VISIBLE_DEVICES` 指定物理 GPU，每张卡加载一份完整模型。

```bash
# LoCoMo
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_locomo.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --benchmark-data data/benchmarks/locomo \
  --attention-implementation flash_attention_2 \
  --mixed-precision \
  --batch-size-per-gpu 32

# RULER
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_ruler.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --benchmark-data data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507 \
  --attention-implementation flash_attention_2 \
  --mixed-precision \
  --lengths 8192 16384 32768 \
  --batch-size-by-length 8192=32 16384=32 32768=16
```

| 任务 | 分卡与批量推理 |
|---|---|
| LoCoMo | 按会话分卡。每段对话只预填充一次，各题复用正文的 KV 缓存，可在同一批次独立解码。 |
| RULER | 按题目分卡。长度档位和生成预算相同的题目组成一批。 |

完整输入与生成预算之和须在模型窗口内，RULER 还须满足所选长度档位的上限。直接调用 Python 时，使用 `python -m latent_working_memory.full_context.evaluate` 并指定 `locomo` 或 `ruler` 子命令。

## 2. 评估参数

共同参数：

| 参数 | 用途／默认值 |
|---|---|
| `--model-path` | 本地基座模型，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--benchmark-data` | benchmark 数据目录 |
| `--output-root` | 模型结果的父目录，默认 `artifacts/baselines/base` |
| `--dtype` | 权重精度，支持 `fp32`（默认）和 `bf16` |
| `--attention-implementation` | 支持 `sdpa`（默认）和 `flash_attention_2` |
| `--mixed-precision` | PyTorch BF16 autocast，默认关闭 |
| `--batch-size-per-gpu` | 每卡批量上限，默认 `1` |
| `--seed` | 默认 `42` |

专属参数：

| 参数 | 用途／默认值 |
|---|---|
| **eval_locomo.sh** |
| - |
| **eval_ruler.sh** |
| `--lengths` | RULER 评估长度档位，默认 `8192 16384 32768` |
| `--batch-size-by-length` | RULER 各长度档位的每卡批量上限，格式为一个或多个 length=batch，如 8192=8 16384=4 32768=2；未指定的档位沿用 `--batch-size-per-gpu` |

## 3. 评估产物

结果保存在 `<output-root>/<模型目录名>/<任务>/`。例如，Qwen3-4B-Instruct-2507 的 LoCoMo 默认结果目录为 `artifacts/baselines/base/Qwen3-4B-Instruct-2507/locomo/`。

| 文件 | 内容 |
|---|---|
| `eval.json` | 模型、数据与评估配置，`protocol.system_prompt` 保存实际使用的 QA 提示 |
| `summary.json` | 评估配置、汇总指标与资源统计 |
| `details.jsonl` | 按原始题序保存的模型回答、参考答案与逐题评分 |
| `status.json`、`eval.log` | 执行状态与运行日志 |

数据、模型和评估配置一致时，直接读取已完成的结果，无需 GPU。换卡或调整批量也会复用原计时，测量不同设置时应分别指定 `--output-root`。

`summary.json` 的 `resources` 记录实际预填充批量（`batch_sizes`）和每批解码题数（`decode_batch_sizes`）。`wall_seconds` 是从启动推理进程到合并结果的耗时，`generation_seconds` 是各卡生成耗时之和，共享计算时间分摊到各题。峰值显存取各卡最大值。

## 4. 本地看板

在项目根目录启动看板，用 `--output-dir` 指定单个模型的评估结果目录。看板从该目录下的 `locomo/` 和 `ruler/` 读取已完成的任务。

```bash
bash src/latent_working_memory/full_context/scripts/dashboard.sh \
  --output-dir artifacts/baselines/base/Qwen3-4B-Instruct-2507
```

启动后访问 `http://127.0.0.1:8501`，选择要查看的任务。LoCoMo 可按题型查看 F1，RULER 可按长度档位和子任务查看准确率。两者均可逐题查看模型回答、参考答案与得分。
