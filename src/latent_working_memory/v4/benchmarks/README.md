# 20261014_RULER 与 LoCoMo 评估

创建时间：20261014 11:04:16 Asia/Shanghai（UTC+08:00）

最后修订时间：20261020 19:40:50 Asia/Shanghai（UTC+08:00）

## 1. 数据位置与评估协议

| Benchmark | 本地数据 | 默认评估范围 | 评分 |
|---|---|---|---|
| RULER v1 | `data/benchmarks/ruler-v1/<model-name>/<length>/<task>/test.jsonl` | 8192、16384、32768 tokens；完整 13 项任务 | 官方必答项召回／任一参考答案命中，按任务等权汇总 |
| LoCoMo | `data/benchmarks/locomo/locomo10.json` | 全部会话的 QA 类别 1–4 | 官方分类别 F1 与按题平均 F1 |

- 以下指令从项目根目录执行；数据下载和 RULER 样本生成由用户手动运行。
- RULER 保留官方 `input`、`outputs` 等字段；LoCoMo 保留官方完整 JSON，不转换为训练用的 `{text,qas}`。
- 每题独立初始化记忆，按 checkpoint 的窗口和写入设置流式读取，再执行 greedy 生成。参考答案只用于评分。
- 输入只应用一次 Qwen chat template，关闭 thinking；不套用训练数据的长度筛选，不截断 benchmark 内容。
- RULER 按任务使用官方生成预算：NIAH 128、VT 30、CWE 120、FWE 50、QA 32 tokens。LoCoMo 为 50 tokens。
- LoCoMo 按 session 时间顺序保留日期、说话人、正文和 `blip_caption`；不同问题不继承之前生成的答案。类别 1–4 分别为 multi-hop、temporal、open-domain、single-hop；不评估 adversarial 类别 5。

## 2. 手动准备数据

- 下载与生成使用独立入口，均从项目根目录执行，无需 GPU。
- 先执行 `uv sync --frozen` 安装项目锁定依赖；从官方来源下载还需系统 `git`、`git-lfs`、`curl`，脚本不自动安装依赖。

```bash
uv sync --frozen
bash src/latent_working_memory/v4/scripts/download_benchmarks.sh --benchmark all
```

**原始数据下载：[download_benchmarks.sh](../scripts/download_benchmarks.sh)**

- RULER 固定采用 [NVIDIA/RULER v1 源码](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)，下载 essay、SQuAD、HotpotQA、CWE 词表及 NLTK 分词资源；不加载 tokenizer，不生成样本。
- NLTK 的 `punkt` 和 `punkt_tab` 固定下载到项目 `data/raw/nltk_data/`。生成脚本将 `NLTK_DATA` 设置为该目录的绝对路径，官方生成器子进程继承此设置。
- NLTK 下载默认允许使用已配置的代理（`NLTK_ALLOW_PROXIED_URLOPEN=1`），可通过同名环境变量覆盖；下载失败立即退出，不进入交互重试。
- LoCoMo 下载 [官方正式数据](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)，下载后可直接评估，无需额外构造脚本。

| 参数 | 填法与默认值 |
|---|---|
| `--benchmark` | `all`、`ruler` 或 `locomo`，默认 `all` |
| `--data-root` | LoCoMo 数据父目录，默认 `data/benchmarks`；写入 `locomo/locomo10.json` |
| `--ruler-source-dir` | RULER 官方源码及下载素材目录，默认 `data/raw/ruler-v1-source` |

- 已下载素材和已有 LoCoMo 文件直接复用。只下载其中一个 benchmark 时，将 `--benchmark all` 改为 `ruler` 或 `locomo`。
- HotpotQA 默认通过 `https://hf-mirror.com` 下载，可用 `HF_ENDPOINT` 覆盖下载源；数据路径和 revision 保持不变。
- 镜像可能重定向至 Hugging Face。仍无法连接时，使用可用代理，或在其他机器下载同一文件，保存为 `<ruler-source-dir>/scripts/data/synthetic/json/hotpotqa.json` 后重跑脚本。

通过 ModelScope 分发时，将下列目录一起保存，并按项目根目录的相对路径还原；`nltk_data/` 保留完整目录结构：

```text
data/
├── raw/
│   ├── ruler-v1-source/          # RULER 源码与已下载素材
│   └── nltk_data/
│       └── tokenizers/
│           ├── punkt/
│           └── punkt_tab/
└── benchmarks/
    └── locomo/
        └── locomo10.json
```

还原后跳过下载脚本，直接执行下方 RULER 生成命令；生成只读取源码与素材，`ruler-v1-source/` 无需保留 `.git`。LoCoMo 可直接评估。

**RULER 样本生成：[generate_ruler.sh](../scripts/generate_ruler.sh)**

- 使用上述已下载素材，调用官方生成器构造全部 13 项任务；仅加载 tokenizer，在 CPU 上控制文本长度。

```bash
bash src/latent_working_memory/v4/scripts/generate_ruler.sh \
  --model-path "$HOME/models/Qwen3-4B-Instruct-2507" \
  --lengths 8192 16384 32768 \
  --num-samples 100 --seed 42
```

| 参数 | 填法与默认值 |
|---|---|
| `--ruler-source-dir` | 已下载的 RULER 源码及素材目录，默认 `data/raw/ruler-v1-source` |
| `--output-dir` | 默认 `data/benchmarks/ruler-v1/<model-name>`；显式指定时直接使用该目录，不附加模型名 |
| `--model-path` | 与训练一致的 tokenizer 来源；本地路径或 HF repo ID，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--lengths` | 空格分隔的长度列表，默认 `8192 16384 32768` |
| `--num-samples` | 每个长度、每项任务的样本数，默认 `100` |
| `--seed` | 样本生成的随机种子，默认 `42` |

- `<model-name>` 取 `--model-path` 去掉尾部斜杠后的最后一级目录名或 HF 仓库名，默认 `Qwen3-4B-Instruct-2507`；不同模型自动使用各自目录，原始素材共享。
- 相对目录以项目根目录为准；路径含空格时加双引号。更换下载目录时，生成阶段须传入相同的 `--ruler-source-dir`。
- 已有 RULER 长度目录拒绝覆盖，包括失败生成留下的目录；重试前将失败的长度目录移出输出目录，或指定新的 `--output-dir`。同一模型修改长度、样本数或 seed 等设置并需重建已有长度时，用新的 `--output-dir` 保存样本。
- RULER 的 `<length>` 表示完整输入与该任务生成预算的总长度上限；脚本预留 chat template 开销，官方生成器另行预留输出预算。
- 官方生成器将 `answer_prefix` 单独保存；评估只读取 `input`，不拼回该前缀。多值／多 query 任务保持一条输入、一次回答。
- 同一模型的比较方法应使用同一批生成样本。评估自定义位置时，`--benchmark-data` 指向 RULER 的 `--output-dir` 或 LoCoMo 的 `<data-root>/locomo`。
- 完整生成参数执行 `bash src/latent_working_memory/v4/scripts/generate_ruler.sh --help` 查看；下载参数见上表。

## 3. 运行评估与查看结果

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
- benchmark 模式不接受 test 专用的 `--test-data`、`--max-test-samples` 和 `--max-new-tokens`，`--lengths` 仅用于 RULER。以上评估参数不用于训练或恢复；下载脚本的同名 `--benchmark` 参数仍按第 2 节选择下载范围。
- 模型配置与 memory 权重取自 checkpoint，tokenizer 从配置中的本地基座模型目录加载；GPU 通过 `--gpus` 指定。
- 产物统一由 checkpoint 推导到原实验的 `eval/<method-dir>/<eval-id>/<ruler|locomo>/global_step_N/`，单个、多个 checkpoint 与 `--eval-run` 使用同一规则。
- 同一 eval-id 的同类任务共用配置，记录实际数据、模型／分词、生成与官方评分设置，不重复保存 seed。评估始终读取公共 `config.json` seed，不接受评估 `--seed`；快照与结果保存实际值，与已有结果不一致时报错。对选中的 checkpoint，已有完整且设置一致的结果直接复用；设置不一致则报错。
- RULER 的 `--lengths` 按集合比较，允许保持或扩展为超集，缩减或替换集合时报错。扩展后更新任务配置的目标长度，按 checkpoint × length 复用完整结果，只补算缺失长度；已有长度的样本及其余设置仍须一致。
- 每个长度保存到 `global_step_N/lengths/<length>/{summary.json,details.jsonl}`；checkpoint 顶层的同名文件汇总实际完成的长度。失败时保留已完成长度，重跑补缺；缺失长度必须有权重才能计算，原有结果仍可独立查看。LoCoMo 缺少完整汇总时从头计算。
- 评估仅在终端输出重要指标并保存本地结果，不使用 SwanLab；看板直接读取已有汇总，不依赖训练是否完成、云端身份或 checkpoint 权重。
- 任务级 `eval/<method-dir>/<eval-id>/<benchmark>/status.json` 保存任务信息、本轮选择 `selected_checkpoints`、整体状态与进度 `progress`，以及 `checkpoints` 中以 `global_step_N` 为键的最近详细执行状态；补评估保留未选中的旧条目，整体状态与进度仅统计本轮选择。各 checkpoint 的日志保存在 `global_step_N/eval.log`；某个 checkpoint 失败后继续其余 checkpoint，最后汇总结果并报错。
- 复用项目的 checkpoint 加载、流式生成和资源记录，评分沿用上述固定官方源码；不计算答案 NLL。


## 4. 看板与 full-context 基线

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

- `--baseline-results` 是看板应用参数，不参与评估计算。标签用于图例且不可重复；目录须为基线运行的 `full-context/<run-id>/eval/`，当前仅支持 full-context 格式。
- 页面可勾选本次传入的基线；不保存绑定，不传时只展示当前方法。基线缺少当前 benchmark 的 `summary.json` 时跳过；路径、格式或比较协议错误在页面提示，已保存结果不受影响。
- 基座模型、精度、attention 实现、窗口、实际样本及生成评分协议须一致；数据目录可以不同，按样本指纹校验。基线不会改变 v4 评估协议。
- RULER 单 checkpoint 展示各长度 Mean 的总览折线，以及可切换长度的 13 项任务分组柱图；任务横轴使用缩写，悬停显示完整名称，只显示实际完成的长度。LoCoMo 按 QA 类别展示分组柱。
- 多 checkpoint 按真实累计 step 升序绘图，基线使用水平虚线。RULER 可选择 Overall 或具体长度，并另选一个 checkpoint 查看总览及任务明细；Overall 使用各 checkpoint 实际完成的长度，范围显示在汇总表中，不额外对齐。
- 基线先于当前方法排列，RULER 中同一方法跨长度和任务保持同色。补评估或调整基线后点击“刷新结果”重新读取，不缓存。
- 看板直接显示汇总中保存的实际 seed 和指标；同步时保留 `<run-id>/eval/` 目录结构即可，无需 `plan/`、训练记录或权重。逐样本记录 `details.jsonl` 按需读取。更多启动参数见[运行指南](../scripts/README.md#25-评估看板)。
