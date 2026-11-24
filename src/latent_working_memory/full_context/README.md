# 20261019_Full-context 基线评估

创建时间：20261019 20:16:32 Asia/Shanghai（UTC+08:00）

最后修订时间：20261124 15:43:38 Asia/Shanghai

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
  --batch-size-per-gpu 2

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
| LoCoMo | 按对话分卡。每张卡一次预填充一段对话，分批并行回答该段的问题。全部答完后释放正文 KV，再处理下一段。 |
| RULER | 按题目分卡。长度档位和生成预算相同的题目组成一批。 |

完整输入与生成预算之和须在模型窗口内，RULER 还须满足所选长度档位的上限。直接调用 Python 时，使用 `python -m latent_working_memory.full_context.evaluate` 并指定 `locomo` 或 `ruler` 子命令。

中断后用原任务脚本加 `--resume` 续跑。用原 `--model-path` 和 `--output-root` 定位结果目录，默认值可省略。续跑沿用原计划的数据、精度、随机种子和 RULER 长度档位，只将未完成的题目按本次 GPU 和批量重新分配。

```bash
# LoCoMo 续跑，每卡同时回答同一段对话的 2 道题
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/full_context/scripts/eval_locomo.sh \
  --resume --batch-size-per-gpu 2

# RULER 续跑，调整不同长度的批量
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_ruler.sh \
  --resume --batch-size-by-length 8192=8 16384=4 32768=2
```

## 2. 评估参数

共同参数：

| 参数 | 用途／默认值 |
|---|---|
| `--model-path` | 本地基座模型，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--benchmark-data` | benchmark 数据目录 |
| `--output-root` | 模型结果的父目录，默认 `artifacts/baselines/base` |
| `--resume` | 恢复已有计划并补评未完成题目，默认关闭 |
| `--dtype` | 权重精度，支持 `fp32`（默认）和 `bf16` |
| `--attention-implementation` | 支持 `sdpa`（默认）和 `flash_attention_2` |
| `--mixed-precision` | PyTorch BF16 autocast，默认关闭 |
| `--batch-size-per-gpu` | 每卡并行题数上限，默认 `1`。LoCoMo 只在同一段对话内组批 |
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
| `progress/worker_*.jsonl` | 每张卡逐题追加已完成记录，供中断后续跑 |
| `status.json`、`eval.log` | 执行状态与运行日志 |

已有计划须用 `--resume` 打开。所有题目都已完成时，脚本直接读取或补建汇总，无需 GPU。要从头测量另一组设置，请指定新的 `--output-root`。

`summary.json` 的 `resources` 记录实际预填充批量（`batch_sizes`，LoCoMo 为 `[1]`）和每批解码题数（`decode_batch_sizes`）。`wall_seconds` 和 `evaluation_seconds` 分别记录本次调度的墙钟耗时和最长单卡评估循环耗时。`generation_seconds` 汇总历次已完成题目的生成耗时，峰值显存取这些题目记录的最大值。

## 4. 本地看板

在项目根目录启动看板，用 `--output-dir` 指定单个模型的评估结果目录。看板从该目录下的 `locomo/` 和 `ruler/` 读取已完成的任务。

```bash
bash src/latent_working_memory/full_context/scripts/dashboard.sh \
  --output-dir artifacts/baselines/base/Qwen3-4B-Instruct-2507
```

启动后访问 `http://127.0.0.1:8501`，选择要查看的任务。LoCoMo 可按题型查看 F1，RULER 可按长度档位和子任务查看准确率。两者均可逐题查看模型回答、参考答案与得分。

统一的多方法 LoCoMo 入口见[公共评估说明](../benchmarks/locomo/README.md)，支持按方法注册加载与计算逻辑。
