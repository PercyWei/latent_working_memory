# 20261019_Full-context 基线评估

创建时间：20261019 20:16:32 Asia/Shanghai（UTC+08:00）

最后修订时间：20261126 10:05:16 Asia/Shanghai

Full-context 基线使用原始模型和完整上下文评估 LoCoMo 与 RULER，按任务规定的生成预算进行贪心解码。

LoCoMo 的 reader 输入由公共输入构造器套用模型聊天模板：`system(QA 系统提示) → user(完整对话 + 简答指令与问题) → assistant 起始标记`。同一对话只预填一次公共前缀 KV，每批问题从该前缀创建独立缓存分支。

[系统提示与输入构造](../prompts.py) · [数据准备与评分](../../../benchmarks/README.md) · [LoCoMo 公共接口](../../../benchmarks/locomo/README.md)

## 一、启动评估

推理默认以 BF16 加载模型权重，关闭模型级自动混合精度（autocast）。注意力使用 FlashAttention 2。

在项目根目录运行，使用项目 `.venv` 环境。

```bash
# LoCoMo
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_locomo.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --output-id Qwen3-4B-Instruct-2507_20261126 \
  --batch-size-per-gpu 2

# RULER
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_ruler.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --benchmark-data data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507 \
  --lengths 8192 16384 32768 \
  --batch-size-by-length 8192=32 16384=32 32768=16
```

| 任务 | 调度与推理 |
|---|---|
| LoCoMo | 使用公共调度器，按完整对话分卡，在同一对话内组批回答；整段完成后释放正文 KV |
| RULER | 按题目分卡，长度档位和生成预算相同的题目组成一批 |

完整输入与生成预算之和须在模型窗口内；RULER 还须满足所选长度档位的上限。Python 入口分别为 `latent_working_memory.full_context.eval_locomo` 和 `latent_working_memory.full_context.eval_ruler`。

### 中断续评

LoCoMo 使用原 `--output-id` 和相同的模型、数据及推理参数，加 `--resume` 补齐未完成题目；GPU 数量和批量可调整。使用 FP32 权重和 BF16 autocast 的运行在续评时需带上 `--dtype fp32 --mixed-precision`。以下示例续评上述 BF16 运行。

```bash
CUDA_VISIBLE_DEVICES=0 bash src/latent_working_memory/full_context/scripts/eval_locomo.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --output-id Qwen3-4B-Instruct-2507_20261126 \
  --batch-size-per-gpu 2 \
  --resume
```

RULER 通过原 `--model-path` 和 `--output-root` 定位结果目录，`--resume` 沿用已保存的数据、精度、随机种子和长度档位：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash src/latent_working_memory/full_context/scripts/eval_ruler.sh \
  --resume --batch-size-by-length 8192=8 16384=4 32768=2
```

## 二、评估参数

共同参数：

| 参数 | 用途／默认值 |
|---|---|
| `--resume` | 补评未完成题目 |
| `--model-path` | 本地基座模型，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--benchmark-data` | 数据目录；LoCoMo 默认 `data/benchmarks/locomo`，RULER 首次运行必填 |
| `--output-root` | 评估产物根目录，默认 `artifacts/baselines/full_context` |
| `--dtype` | 权重精度，支持 `bf16`（默认）和 `fp32` |
| `--attention-implementation` | 注意力实现，支持 `flash_attention_2`（默认）和 `sdpa` |
| `--mixed-precision` | 混合精度计算，默认关闭 |
| `--batch-size-per-gpu` | 每卡评估批量大小，默认 `1` |
| `--seed` | 默认 `42` |

任务专属参数：

| 参数 | 用途／默认值 |
|---|---|
| LoCoMo |
| `--output-id` | 运行标识，默认 `<模型名>_<YYYYMMDD-HHMM>` |
| RULER |
| `--lengths` | 评估长度档位，默认 `8192 16384 32768` |
| `--batch-size-by-length` | 按长度覆盖每卡批量，如 `8192=8 16384=4 32768=2`；未指定档位沿用 `--batch-size-per-gpu` |

## 三、评估产物

| 任务 | 结果目录 | 配置与执行记录 |
|---|---|---|
| LoCoMo | `<output-root>/eval/<output-id>/locomo/` | 公共接口保存 `evaluation_config.json`、`execution.json`、任务分配和逐题续评记录 |
| RULER | `<output-root>/<模型目录名>/ruler/` | `eval.json` 保存配置；`status.json`、`eval.log` 和 `progress/` 保存状态、日志与续评记录 |

两项任务均保存 `summary.json` 和按原始题序排列的 `details.jsonl`。LoCoMo 使用公共结果格式，记录总体与分题型 F1、模型回答和参考答案；模型身份、精度和聊天模板记录在 `settings` 中。完整字段见[产物规范](../../../benchmarks/locomo/README.md#产物规范)。

RULER 的 `summary.json` 另外记录推理批量、耗时和峰值显存。所有题目完成后，续评直接读取记录并汇总，无需加载模型。另测一组设置时，LoCoMo 使用新的 `--output-id`，RULER 使用新的 `--output-root`。

## 四、查看结果

`--output-dir` 指向包含任务子目录的评估运行目录。LoCoMo 页面复用公共展示组件，支持按题型和对话查看 F1、逐题回答与参考答案；RULER 页面按长度档位和子任务展示准确率。

```bash
# LoCoMo
bash src/latent_working_memory/full_context/scripts/dashboard.sh \
  --output-dir artifacts/baselines/full_context/eval/Qwen3-4B-Instruct-2507_20261126

# RULER
bash src/latent_working_memory/full_context/scripts/dashboard.sh \
  --output-dir artifacts/baselines/full_context/Qwen3-4B-Instruct-2507
```
