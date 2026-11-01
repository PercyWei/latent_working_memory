# 20261014_v4 RULER 与 LoCoMo 评估

创建时间：20261014 11:04:16 Asia/Shanghai（UTC+08:00）

最后修订时间：20261101 17:31:30 Asia/Shanghai

数据下载、RULER 样本生成与公共评分协议见[公共 benchmark 指南](../../benchmarks/README.md)。本页说明 v4 的 checkpoint 评估与结果看板；每题独立初始化记忆，按 checkpoint 的窗口和写入设置流式读取。

## 1. 运行评估与查看结果

将 `global_step_N` 替换为要评估的 checkpoint 的实际目录：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --eval-type benchmark --benchmark ruler --gpus 0 \
  --checkpoint artifacts/v4/<run-id>/train/frozen-query-gen-m32/checkpoints/global_step_N \
  --benchmark-data data/benchmarks/ruler-v1/Qwen3-4B-Instruct-2507 \
  --lengths 8192 16384 32768

bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --eval-type benchmark --benchmark locomo --gpus 0 \
  --checkpoint artifacts/v4/<run-id>/train/frozen-query-gen-m32/checkpoints/global_step_N \
  --benchmark-data data/benchmarks/locomo
```

- 独立评估统一使用 `--evaluate-only`。本节须设置 `--eval-type benchmark`，并通过 `--benchmark` 指定 `ruler` 或 `locomo`；该参数默认 `null`，在 benchmark 模式下不可为空。省略 `--eval-type` 时为 test 评估，仅传 benchmark 名称不会自动切换。
- `--checkpoint` 后可列出同一 run 的多个目录；或改用 `--eval-run artifacts/v4/<run-id>/train/frozen-query-gen-m32/`，选择其中仍保留 `state.pt` 的全部 checkpoint。两者互斥；拒绝重复、跨 run 或缺少 `state.pt` 的 checkpoint。多 GPU 时每卡一个进程并行评估不同 checkpoint，命令示例见[运行指南](../scripts/README.md#22-单独评估)。
- `--eval-id` 默认 `<run-id>-eval`，同一 run、方法下可包含 test、RULER 和 LoCoMo；各任务分别固定设置，换设置时使用新标识。默认沿用 `plan/<method-dir>/<eval-id>/<ruler|locomo>.json` 中的数据目录；首次计算没有该配置时，必须传入 `--benchmark-data`。RULER 指向包含各长度子目录的目录，LoCoMo 指向包含 `locomo10.json` 的目录。
- 人工确认不同模型可复用同一批 RULER 样本时，用 `--benchmark-data` 指定已有模型目录，不自动判断 tokenizer 是否一致。
- benchmark 模式不接受 test 专用的 `--test-data`、`--max-test-samples` 和 `--max-new-tokens`，`--lengths` 仅用于 RULER。以上评估参数不用于训练或恢复；下载脚本的同名 `--benchmark` 参数用于选择下载范围，见[公共数据准备说明](../../benchmarks/README.md#2-数据准备)。
- 模型配置与 memory 权重取自 checkpoint，tokenizer 从配置中的本地基座模型目录加载；GPU 通过 `--gpus` 指定。
- 产物统一由 checkpoint 推导到原实验的 `eval/<method-dir>/<eval-id>/<ruler|locomo>/global_step_N/`，单个、多个 checkpoint 与 `--eval-run` 使用同一规则。
- 同一 eval-id 的同类任务共用配置，记录实际数据、模型／分词、生成与官方评分设置，不重复保存 seed。评估始终读取公共 `config.json` seed，不接受评估 `--seed`；快照与结果保存实际值，与已有结果不一致时报错。对选中的 checkpoint，已有完整且设置一致的结果直接复用；设置不一致则报错。
- RULER 的 `--lengths` 按集合比较，允许保持或扩展为超集，缩减或替换集合时报错。扩展后更新任务配置的目标长度，按 checkpoint × length 复用完整结果，只补算缺失长度；已有长度的样本及其余设置仍须一致。
- 每个长度保存到 `global_step_N/lengths/<length>/{summary.json,details.jsonl}`；checkpoint 顶层的同名文件汇总实际完成的长度。失败时保留已完成长度，重跑补缺；缺失长度必须有权重才能计算，原有结果仍可独立查看。LoCoMo 缺少完整汇总时从头计算。
- 评估仅在终端输出重要指标并保存本地结果，不使用 SwanLab；看板直接读取已有汇总，不依赖训练是否完成、云端身份或 checkpoint 权重。
- 任务级 `eval/<method-dir>/<eval-id>/<benchmark>/status.json` 保存任务信息、本轮选择 `selected_checkpoints`、整体状态与进度 `progress`，以及 `checkpoints` 中以 `global_step_N` 为键的最近详细执行状态；补评估保留未选中的旧条目，整体状态与进度仅统计本轮选择。各 checkpoint 的日志保存在 `global_step_N/eval.log`；某个 checkpoint 失败后继续其余 checkpoint，最后汇总结果并报错。
- 复用项目的 checkpoint 加载、流式生成和资源记录，评分复用公共 benchmark 模块中的固定官方实现；不计算答案 NLL。


## 2. 看板与 full-context 基线

启动 Streamlit + ECharts 看板，必须用 `--run-dir` 指定单次 run；页面内选择方法、eval-id 和 checkpoint，仅加载当前打开的评估任务，无需重跑评估：

```bash
bash src/latent_working_memory/v4/scripts/dashboard.sh \
  --run-dir artifacts/v4/<run-id>
```

先按 [full-context 运行说明](../../baselines/README.md) 独立计算基线，再按“标签＋基线运行的 `eval/` 目录”成对传给看板：

```bash
bash src/latent_working_memory/v4/scripts/dashboard.sh \
  --run-dir artifacts/v4/<run-id> \
  --baseline-results \
    full-context artifacts/baselines/full-context/qwen3-4b_full-context/eval \
    reference artifacts/baselines/full-context/reference/eval
```

- `--baseline-results` 是看板应用参数，不参与评估计算。标签用于图例且不可重复；可传入 `eval/<benchmark>/summary.json` 布局的 `eval/`，或直接含 `summary.json` 的单任务／单 checkpoint 目录，支持公共 benchmark、v4 与 v4.1 结果。
- 页面可勾选本次传入的基线；不保存绑定，不传时只展示当前方法。基线缺少当前 benchmark 的 `summary.json` 时跳过；路径、格式或比较协议错误在页面提示，已保存结果不受影响。
- 质量对照核对基座模型标识、实际输入与参考答案、输入／解码／生成预算及评分口径；基座按路径末级名称识别，模型父目录、算法、窗口、精度、attention 实现、设备和批量不参与硬性判等。数据按比较指纹核对，旧结果沿用可核实的原指纹。
- RULER 单 checkpoint 展示各长度 Mean 的总览折线，以及可切换长度的 13 项任务分组柱图；任务横轴使用缩写，悬停显示完整名称，只显示实际完成的长度。LoCoMo 按 QA 类别展示分组柱。
- 多 checkpoint 按真实累计 step 升序绘图，基线使用虚线。RULER 可比较共同长度；Overall 按各 checkpoint 实际完成的长度范围重算基线，缺任一长度时不显示对应基线 Overall，缺失不补零。
- 耗时和显存需按实际硬件、精度、批量及计时范围另行比较。跨方法展示不改变结果复用、失败重试和恢复的严格校验。
- 基线先于当前方法排列，RULER 中同一方法跨长度和任务保持同色。补评估或调整基线后点击“刷新结果”重新读取，不缓存。
- 看板直接显示汇总中保存的实际 seed 和指标；同步时保留 `<run-id>/eval/` 目录结构即可，无需 `plan/`、训练记录或权重。逐样本记录 `details.jsonl` 按需读取。更多启动参数见[运行指南](../scripts/README.md#25-评估看板)。
