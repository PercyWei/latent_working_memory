# 20261019_Full-context 基线评估

创建时间：20261019 20:16:32 Asia/Shanghai（UTC+08:00）

最后修订时间：20261020 14:50:30 Asia/Shanghai（UTC+08:00）

使用原始基座模型完整读取 RULER 或 LoCoMo 输入，不压缩上下文、不加载 v4 checkpoint。数据读取与官方评分复用公共 [benchmarks](../benchmarks/) 模块。结果保存到本地并在终端打印，图表由 v4 独立看板展示。

## 1. 运行

先按 [benchmark 数据准备说明](../v4/benchmarks/README.md#2-手动准备数据) 下载素材并生成 RULER 样本。在项目根目录执行：

```bash
# LoCoMo
bash src/latent_working_memory/baselines/scripts/run_full_context.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --benchmark locomo --gpus 0 \
  --run-id qwen3-4b_full-context \
  --benchmark-data data/benchmarks/locomo

# RULER
bash src/latent_working_memory/baselines/scripts/run_full_context.sh \
  --model-path ~/models/Qwen3-4B-Instruct-2507 \
  --benchmark ruler --gpus 0 \
  --run-id qwen3-4b_full-context \
  --benchmark-data data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507 \
  --lengths 8192 16384 32768
```

| 参数 | 用途／默认值 |
|---|---|
| `--benchmark` | 必填，`ruler` 或 `locomo` |
| `--benchmark-data` | 对应原生数据目录；首次评估必填，已有同类配置时沿用 |
| `--lengths` | 仅用于 RULER；已有配置时沿用，首次默认 `8192 16384 32768` |
| `--model-path` | 本地基座模型，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--backbone-dtype` | `bfloat16`（默认）或 `float32` |
| `--attention-implementation` | `sdpa`（默认）或 `eager` |
| `--seed` | 默认 `42` |
| `--gpus` | 单个物理 GPU 编号，默认 `0` |
| `--device` | `cuda`（默认）或 `cpu` |
| `--run-id` | 基线运行编号，默认上海时间 `YYYYMMDD-HHMMSS` |
| `--output-root` | 产物父目录，默认 `artifacts/baselines` |

- shell 入口统一从项目根目录运行，相对路径以此为基准；Python 入口为 `latent_working_memory.baselines.full_context`。
- 同一 run 沿用已有公共模型与 seed 配置，显式冲突时报错；RULER 与 LoCoMo 分别保存评估配置，同类计算协议固定。
- 使用一次 chat template、greedy 生成和官方任务预算。完整输入与生成预算必须满足模型窗口；RULER 同时受指定长度限制，不截断或缩减预算。
- 相同协议的完成结果可以复用；指标见 `summary.json`，逐样本预测与评分见 `details.jsonl`，执行状态与日志分别见 `status.json`、`eval.log`。

## 2. 产物与 v4 对照

```text
artifacts/baselines/full-context/<run-id>/
├── plan/
│   ├── config.json                 公共模型配置与 seed
│   └── <ruler|locomo>.json          同类评估配置
└── eval/
    └── <ruler|locomo>/
        ├── eval.json               实际评估配置快照
        ├── status.json             执行状态
        ├── summary.json            指标汇总
        ├── details.jsonl           逐样本预测与评分
        └── eval.log                运行日志
```

通过 v4 的 Streamlit + ECharts 看板加入对照，用必填参数 `--run-dir` 指定单次 run，在页面内选择方法、eval-id 和 checkpoint，仅加载当前打开的评估任务：

```bash
bash src/latent_working_memory/v4/scripts/dashboard.sh \
  --run-dir artifacts/v4/<run-id> \
  --baseline-results \
    full-context artifacts/baselines/full-context/qwen3-4b_full-context/eval \
    reference artifacts/baselines/full-context/reference/eval
```

- `--baseline-results` 按“标签＋结果目录”成对传入，标签用于图例且不可重复；目录须为基线运行的 `eval/`，当前仅支持 full-context 结果格式。
- 页面可勾选本次传入的基线，不保存绑定；不传时不展示。缺少当前 benchmark 汇总的基线跳过；路径、格式或比较协议错误在页面提示，不影响评估计算及已有结果。
- 对照要求基座模型、精度、attention 实现、窗口、实际样本及生成评分设置一致；RULER 长度范围也须一致，各自的 seed 保留在结果中。基线仍使用上述目录结构，v4 的 eval-id 不改变基线结果路径。
- 单 checkpoint 的 benchmark 图加入分组柱，多 checkpoint 图加入水平虚线，不改变 v4 的评估结果和真实 step 顺序。看板直接读取本地汇总，无需重新运行评估，更多用法见 [v4 benchmark 说明](../v4/benchmarks/README.md#4-看板与-full-context-基线)。
