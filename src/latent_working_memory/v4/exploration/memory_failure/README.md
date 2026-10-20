# 20261020_LoCoMo 记忆失败诊断

创建时间：20261020 16:26:40 Asia/Shanghai（UTC+08:00）

最后修订时间：20261020 16:32:05 Asia/Shanghai（UTC+08:00）

当前路径实现第一组实验：在同一 checkpoint、同一批问题和生成协议下，分辨上下文删除、记忆读取及局部 TTT 更新各自的影响。

## 实验依据与边界

目标 checkpoint 为 `artifacts/v4/am-query_20261019/train/conditioned-query-m32/checkpoints/global_step_444`。此前全量 1540 题结果中，完整上下文 F1 为 41.9634%，正常 TTT 为 4.8228%；正常 TTT 有 327 条空白输出，其中 292 条立即生成 EOS。这些是原始全量评估的观察，不是本实验子集的结果。

本实验固定外层权重，仅改变下面的推理行为。它用于定位失效环节，不直接证明记忆容量不足、训练长度外推失败或 AM 目标失效；这些结论还需要后续针对性实验。

| 条件 | 保留的原始 KV | 记忆更新 | Q/O 记忆读取修正 |
|---|---|---|---|
| `full-context` | 完整上下文 | 无 | 无；调用原生基座生成 |
| `no-read` | 正常 pending/recent 裁剪 | 正常 TTT | 整个 prefill 与 decode 均关闭 |
| `fixed-initial` | 正常 pending/recent 裁剪 | 保持 checkpoint 学到的初始 slots \(M_0\) | 首次裁剪后正常启用 |
| `ttt` | 正常 pending/recent 裁剪 | 正常 TTT | 首次写入后正常启用 |

`no-read` 仍执行写入并收集 source，使禁用读取成为唯一主动干预；读取关闭后，后续隐藏状态和写入输入自然也会变化。这个条件适合质量归因，其耗时不代表一个专门优化的无记忆滑窗实现。`fixed-initial` 继续删除相同范围的 KV/source 并推进边界计数，确保启用读取的时机一致；不计算或伪造 AM loss。

所有条件都从相同的完整 prompt 开始。`no-read` 的保留 KV 仍可能包含被删除历史在先前计算中留下的信息，不能把它解释为“模型完全没见过历史”。

## 固定设置与结果解释

- 从 LoCoMo 类别 1–4 中，每个 conversation 选 20 题，标准 10 个 conversation 共 200 题。每个对话内按类别占比分配配额，用最大余数法取整，再按抽样 seed 与样本 ID 的稳定哈希选择；不依据模型对错选题。
- 抽样 seed 默认为 42；模型、窗口、精度、attention 实现、记忆结构、局部更新超参数及生成 seed 均沿用 checkpoint。每题独立初始化记忆。
- 使用公共 LoCoMo prompt、chat template、关闭 thinking、greedy 解码、50 tokens 预算及官方分类别 F1。参考答案仅用于评分。
- 抽样名单及完整评估协议保存到 `plan.json`。改变 checkpoint、数据或抽样设置必须换输出目录；不同条件共享同一计划。
- 总体 F1 是所选题的平均值。各对话等量抽样后的类别比例与全量可能不同，报告中的子集分数不替代全量 1540 题分数。

报告按题配对计算以下四个差值，并显示更好／相同／更差的题数；差值以百分点展示，不作统计显著性声明：

| 差值 | 主要回答的问题 |
|---|---|
| `full-context − no-read` | 保留完整上下文相对于当前流式裁剪路径有多大收益？ |
| `ttt − no-read` | 正常记忆路径相对关闭读取带来净收益还是净伤害？ |
| `ttt − fixed-initial` | 在读取模块相同的前提下，局部更新记忆带来什么变化？ |
| `fixed-initial − no-read` | 仅使用学到的初始记忆及读取修正，是否已改变输出质量？ |

若 `no-read` 明显高于 `ttt`，优先检查记忆读取对生成的干扰；若 `fixed-initial` 明显高于 `ttt`，进一步检查更新后的记忆及其对后续隐藏状态的影响。若三种裁剪条件都很低，而完整上下文正常，则当前证据首先指向删掉原始 KV 后缺乏有效补偿。上述情况可能同时发生；`ttt − fixed-initial` 也包含更新后读取引起的后续状态变化，不能把它单独当成 AM 写入保真度。

## 运行

从已同步的服务器项目仓库根目录执行，使用根目录 `.venv/`。单进程串行运行四个条件：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python \
  -m latent_working_memory.v4.exploration.memory_failure.evaluate \
  --checkpoint artifacts/v4/am-query_20261019/train/conditioned-query-m32/checkpoints/global_step_444 \
  --benchmark-data data/benchmarks/locomo \
  --output-dir artifacts/v4/exploration/locomo-memory-failure_20261020/global_step_444
```

默认读取 checkpoint 配置中的本地基座路径。`--conditions` 可指定一个或多个条件；不使用 `torchrun`。

### 复用已有完整评估

已有完整 LoCoMo 结果时，用 `--full-context-results`、`--ttt-results` 分别指向包含 `summary.json` 与 `details.jsonl` 的结果目录。下面是此次诊断产物在服务器上的路径：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python \
  -m latent_working_memory.v4.exploration.memory_failure.evaluate \
  --checkpoint artifacts/v4/am-query_20261019/train/conditioned-query-m32/checkpoints/global_step_444 \
  --benchmark-data data/benchmarks/locomo \
  --output-dir artifacts/v4/exploration/locomo-memory-failure_20261020/global_step_444 \
  --full-context-results artifacts/baselines/full-context/qwen3-4b-inst-2507_20261019/eval/locomo \
  --ttt-results artifacts/v4/am-query_20261019/eval/conditioned-query-m32/am-query_20261019-eval/locomo/global_step_444
```

导入要求公共评估器的完整结果，并核对基座配置、数据身份、prompt 长度、参考答案、生成预算、token 解码与评分；TTT 还核对完整模型配置、checkpoint 路径／step 及训练 run。导入后只提取固定名单，并重算子集统计，不复制全量 F1。导入的两个条件无需加载基座权重；另外两个条件仍须实际推理。

上述两份原始结果已通过默认固定 200 题的导入校验，重算 F1 分别为完整上下文 42.9671%、正常 TTT 6.1394%。这是已有输出的子集统计，`no-read` 与 `fixed-initial` 尚待计算。

复用来源与条件绑定到各自的 `eval.json`，恢复时保留原参数。已经以重新计算方式启动的条件不能改成导入；需要改变来源时使用新输出目录。

### 双卡分开补算

先用上面的导入命令加 `--conditions full-context ttt` 完成两组已有结果导入，再在两个终端分别执行：

```bash
# 终端一：物理 GPU 0
CUDA_VISIBLE_DEVICES=0 uv run --frozen python \
  -m latent_working_memory.v4.exploration.memory_failure.evaluate \
  --checkpoint artifacts/v4/am-query_20261019/train/conditioned-query-m32/checkpoints/global_step_444 \
  --benchmark-data data/benchmarks/locomo \
  --output-dir artifacts/v4/exploration/locomo-memory-failure_20261020/global_step_444 \
  --conditions no-read

# 终端二：物理 GPU 1
CUDA_VISIBLE_DEVICES=1 uv run --frozen python \
  -m latent_working_memory.v4.exploration.memory_failure.evaluate \
  --checkpoint artifacts/v4/am-query_20261019/train/conditioned-query-m32/checkpoints/global_step_444 \
  --benchmark-data data/benchmarks/locomo \
  --output-dir artifacts/v4/exploration/locomo-memory-failure_20261020/global_step_444 \
  --conditions fixed-initial
```

每个条件有独立文件锁，可同时计算不同条件。完整结果直接复用；中断后重复原命令，已刷新的完整 JSONL 行保留，从下一题继续。首次未完成的尾行会重新生成。

| 参数 | 含义 |
|---|---|
| `--checkpoint` | 必填，含 `state.pt` 的 v4 checkpoint 目录 |
| `--benchmark-data` | 必填，含 `locomo10.json` 的数据目录 |
| `--output-dir` | 必填，四条件共用的此次诊断产物目录 |
| `--conditions` | 默认全部四条件，可选择部分条件 |
| `--samples-per-conversation` | 默认 20，每个对话选题数；不得超过任一对话可用题数 |
| `--selection-seed` | 默认 42，只控制固定子集选择 |
| `--full-context-results`／`--ttt-results` | 可选，完整公共评估结果的目录 |
| `--device` | 默认 `cuda`；CPU 小模型检查可用 `cpu` |

## 展示与产物

评估命令完成后自动生成 `report/index.html`，浏览器可直接离线打开。仅有部分条件完成时也可生成报告，缺失条件显示“待完成”。单独重建报告不需要模型、checkpoint 或原始数据：

```bash
uv run --frozen python \
  -m latent_working_memory.v4.exploration.memory_failure.report \
  --output-dir artifacts/v4/exploration/locomo-memory-failure_20261020/global_step_444
```

报告包含总体及分类 F1、四组配对差值、生成行为计数与占比、来源及抽样表，以及可按类别、行为、问题或答案文本筛选的四条件并列回答。行为标记为：空白、立即 EOS、生成触顶、包含固定无信息回答片段、复制固定指令片段、至少 12 个相邻零；精确定义在页面中显示。标记可以重叠，不用于剔除样本或修改 F1。

```text
global_step_444/
├── plan.json
├── full-context/
├── no-read/
├── fixed-initial/
├── ttt/
│   ├── eval.json
│   ├── status.json
│   ├── details.jsonl
│   └── summary.json
└── report/
    ├── index.html
    ├── summary.json
    ├── comparison.csv
    └── details.csv
```

四个条件的目录结构相同。`summary.json` 是条件完成标记；报告只读取已完成条件，不将中途题数不同的结果并列比较。逐题记录保留原始输出、token IDs、停止原因、F1 和耗时；新计算的条件额外保存裁剪边界数和实际记忆更新数。二者包括 prompt 与 decode 的事件，固定初始记忆的实际更新数为零。

运行代码在 [evaluate.py](evaluate.py)，推理干预在 [model.py](model.py)，本地报告在 [report.py](report.py)。正式训练与 benchmark 入口继续使用原有实现。
