# 20261014_RULER 与 LoCoMo 评估

创建时间：20261014 11:04:16 Asia/Shanghai（UTC+08:00）

最后修订时间：20261016 18:51:46 Asia/Shanghai（UTC+08:00）

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
- 先安装系统 `git`、`git-lfs`、`curl`，再执行 `uv sync --frozen` 安装项目锁定依赖；脚本不自动安装依赖。

```bash
uv sync --frozen
bash src/latent_working_memory/v4/scripts/download_benchmarks.sh --benchmark all
```

**原始数据下载：[download_benchmarks.sh](../scripts/download_benchmarks.sh)**

- RULER 固定采用 [NVIDIA/RULER v1 源码](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)，下载 essay、SQuAD、HotpotQA、CWE 词表及 NLTK 分词资源；不加载 tokenizer，不生成样本。
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
- 已有 RULER 长度目录拒绝覆盖；同一模型修改长度、样本数或 seed 等设置并需重建已有长度时，用新的 `--output-dir` 保存样本。
- RULER 的 `<length>` 表示完整输入与该任务生成预算的总长度上限；脚本预留 chat template 开销，官方生成器另行预留输出预算。
- 官方生成器将 `answer_prefix` 单独保存；评估只读取 `input`，不拼回该前缀。多值／多 query 任务保持一条输入、一次回答。
- 同一模型的比较方法应使用同一批生成样本。评估自定义位置时，`--benchmark-data` 指向 RULER 的 `--output-dir` 或 LoCoMo 的 `<data-root>/locomo`。
- 完整生成参数执行 `bash src/latent_working_memory/v4/scripts/generate_ruler.sh --help` 查看；下载参数见上表。

## 3. 运行评估与查看结果

将 `global_step_N` 替换为要评估的 checkpoint 的实际目录：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --eval-type benchmark --benchmark ruler --gpus 0 \
  --checkpoint artifacts/v4/<run-id>/train/conditioned-query-m32/checkpoints/global_step_N \
  --lengths 8192 16384 32768

bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --eval-type benchmark --benchmark locomo --gpus 0 \
  --checkpoint artifacts/v4/<run-id>/train/conditioned-query-m32/checkpoints/global_step_N
```

- 独立评估统一使用 `--evaluate-only`。本节须设置 `--eval-type benchmark`，并通过 `--benchmark` 指定 `ruler` 或 `locomo`；该参数默认 `null`，在 benchmark 模式下不可为空。省略 `--eval-type` 时为 test 评估，仅传 benchmark 名称不会自动切换。
- `--checkpoint` 后可列出同一 run 的多个目录；或改用 `--eval-run artifacts/v4/<run-id>/train/conditioned-query-m32/`，扫描其中全部 `checkpoints/global_step_*/state.pt`。两者互斥；拒绝重复、跨 run 或缺少 `state.pt` 的未完成 checkpoint。多个 checkpoint 按数字 step 升序在单卡上顺序评估，命令示例见[运行指南](../scripts/README.md#22-单独评估)。
- RULER 默认按 checkpoint 中原始 `model_name_or_path` 的末级名称读取对应模型目录；人工确认不同模型可复用同一批样本时，用 `--benchmark-data` 指定已有模型目录，不自动判断 tokenizer 是否一致。
- `--benchmark-data` 可覆盖表中的 benchmark 根目录；RULER 直接指向包含各长度子目录的目录，LoCoMo 指向包含 `locomo10.json` 的目录。
- benchmark 模式不接受 test 专用的 `--test-data` 和 `--max-new-tokens`，`--lengths` 仅用于 RULER。以上评估参数不用于训练或恢复；下载脚本的同名 `--benchmark` 参数仍按第 2 节选择下载范围。
- 模型配置、tokenizer 和 memory 权重取自 checkpoint；只需选择一张 GPU。
- 产物统一由 checkpoint 推导到原实验的 `eval/<method-dir>/<ruler|locomo>/global_step_N/`，单个、多个 checkpoint 与 `--eval-run` 使用同一规则。
- 同类评估共用父目录的 `eval.json`，记录实际数据内容、模型／分词、生成与官方评分设置；已有协议后须保持一致，配置冲突时报错。每个 checkpoint 另存协议快照、逐题预测及计算完成标记 `summary.json`。
- 相同 checkpoint、相同协议且已有完整汇总时直接复用，不重新加载模型或生成答案；缺少汇总时在原目录从头计算。本地结果先于在线发布保存，发布失败后重跑原命令可复用结果补传。
- tracking、设备及单次／批量展示方式不改变计算协议，可先离线计算再在线发布，也可将单 checkpoint 结果用于批量评估。
- 在线追加要求训练已完成、云端原 SwanLab run 为 `FINISHED`。单 checkpoint 用汇总图展示任务／长度或 QA 类别分数，批量评估可上传原运行全部已完成保存的 checkpoint，使用真实 checkpoint step 的原生增量曲线。`--tracking disabled` 仅保存本地结果。
- 各 checkpoint 的日志和执行状态位于 `plan/<method-dir>/<benchmark>-evaluation/global_step_N.log`、`global_step_N-result.json`。批量评估另存 `plan/<method-dir>/<benchmark>-evaluation-result.json`；某个 checkpoint 失败后继续其余 checkpoint，最后汇总结果并报错。
- 复用项目的 checkpoint 加载、流式生成、资源记录和发布流程，评分沿用上述固定官方源码；不计算答案 NLL。
