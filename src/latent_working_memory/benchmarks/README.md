# 20261014_RULER 与 LoCoMo 数据准备与协议

创建时间：20261014 11:04:16 Asia/Shanghai（UTC+08:00）

最后修订时间：20261109 00:01:38 Asia/Shanghai

本目录提供 RULER 与 LoCoMo 的公共数据准备、读取和评分，供不同方法及 full-context 基线共用。

## 1. 数据位置与评估协议

| Benchmark | 本地数据 | 默认评估范围 | 评分 |
|---|---|---|---|
| RULER v1 | `data/benchmarks/ruler-v1/<model-name>/<length>/<task>/test.jsonl` | 8192、16384、32768 tokens；完整 13 项任务 | 官方必答项召回／任一参考答案命中，按任务等权汇总 |
| LoCoMo | `data/benchmarks/locomo/locomo10.json` | 全部会话的 QA 类别 1–4 | 官方分类别 F1 与按题平均 F1 |

- 以下指令从项目根目录执行；数据下载和 RULER 样本生成由用户手动运行。
- RULER 保留官方 `input`、`outputs` 等字段；LoCoMo 保留官方完整 JSON，不转换为训练用的 `{text,qas}`。
- 评估采用 greedy 生成，参考答案只用于评分；模型与记忆机制由各方法的评估入口管理。
- 本项目 Instruct 模型的输入只应用一次 Qwen chat template，关闭 thinking；不套用训练数据的长度筛选，不截断 benchmark 内容。TTT baselines 的 base 续写协议及任务范围另见 [评估数据准备](../../../reproductions/20261108_ttt_data_preparation.md)。
- RULER 按任务使用官方生成预算：NIAH 128、VT 30、CWE 120、FWE 50、QA 32 tokens。LoCoMo 为 50 tokens。
- LoCoMo 按 session 时间顺序保留日期、说话人、正文和 `blip_caption`；不同问题不继承之前生成的答案。类别 1–4 分别为 multi-hop、temporal、open-domain、single-hop；不评估 adversarial 类别 5。

## 2. 数据准备

- 下载与生成使用独立入口，均从项目根目录执行，无需 GPU。
- 先执行 `uv sync --frozen` 安装项目锁定依赖；官方下载入口还需系统 `curl`、`tar`，脚本不自动安装依赖。

```bash
uv sync --frozen
bash src/latent_working_memory/benchmarks/scripts/download_benchmarks.sh --benchmark all
```

**原始数据下载：[download_benchmarks.sh](scripts/download_benchmarks.sh)**

- RULER 固定采用 [NVIDIA/RULER v1 源码](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)，下载 essay、SQuAD、HotpotQA、CWE 词表及 NLTK 分词资源；不加载 tokenizer，不生成样本。
- **适用场景**：未从 ModelScope 获取素材时，运行此入口从官方来源准备数据。每次执行均重新下载所选 benchmark 的源码与素材，覆盖同路径文件。
- **RULER 源码**：下载上述固定提交的源码压缩包并解包覆盖，CWE 词表单独从对应的官方 LFS 地址下载。
- **NLTK 资源**：强制重新下载 `punkt` 和 `punkt_tab`，保存到 `data/raw/nltk_data/tokenizers/`。生成脚本设置绝对路径 `NLTK_DATA`，供官方生成器子进程使用。
- NLTK 下载默认允许使用已配置的代理（`NLTK_ALLOW_PROXIED_URLOPEN=1`），可通过同名环境变量覆盖；下载失败立即退出，不进入交互重试。
- LoCoMo 下载 [官方正式数据](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)，下载后可直接评估，无需额外构造脚本。

| 参数 | 填法与默认值 |
|---|---|
| `--benchmark` | 支持 `all`（默认）、`ruler` 或 `locomo` |
| `--data-root` | LoCoMo 数据父目录，默认 `data/benchmarks` |
| `--ruler-source-dir` | RULER 官方源码及下载素材目录，默认 `data/raw/ruler-v1-source` |

- HotpotQA 默认通过 `https://hf-mirror.com` 下载，可用 `HF_ENDPOINT` 覆盖下载源；数据路径和 revision 保持不变。
- 镜像可能重定向至 Hugging Face。仍无法连接时，配置可用代理或更换 `HF_ENDPOINT` 后重新运行。

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

从 ModelScope 还原完整目录后，直接执行下方 RULER 生成命令；LoCoMo 可直接评估。RULER 源码与素材应来自上述固定版本，目录位置不同时向生成脚本传入 `--ruler-source-dir`。

**RULER 样本生成：[generate_ruler.sh](scripts/generate_ruler.sh)**

- 使用上述已下载素材，默认调用官方生成器构造全部 13 项任务；仅加载 tokenizer，在 CPU 上控制文本长度。

```bash
bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh \
  --model-path "$HOME/models/Qwen3-4B-Instruct-2507" \
  --lengths 8192 16384 32768 \
  --num-samples 100 --seed 42
```

| 参数 | 填法与默认值 |
|---|---|
| `--ruler-source-dir` | 已下载的 RULER 源码及素材目录，默认 `data/raw/ruler-v1-source` |
| `--output-dir` | 默认 `data/benchmarks/ruler-v1/<model-name>`；显式指定时直接使用该目录，不附加模型名 |
| `--model-path` | 与训练一致的 tokenizer 来源；本地路径或 HF repo ID，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--prompt-format` | `chat`（默认）应用现有 Qwen 聊天协议；`base` 使用原始文本与 `answer_prefix`，按默认 special tokens 预留 BOS 等开销 |
| `--tasks` | 空格分隔的 RULER 任务名称；默认全部 13 项 |
| `--lengths` | 空格分隔的长度列表，默认 `8192 16384 32768` |
| `--tokens-to-generate` | 可选，统一覆盖各任务的生成预算；默认沿用官方任务预算 |
| `--num-samples` | 每个长度、每项任务的样本数，默认 `100` |
| `--seed` | 样本生成的随机种子，默认 `42` |

- `<model-name>` 取 `--model-path` 去掉尾部斜杠后的最后一级目录名或 HF 仓库名，默认 `Qwen3-4B-Instruct-2507`；不同模型自动使用各自目录，原始素材共享。
- 相对目录以项目根目录为准；路径含空格时加双引号。更换下载目录时，生成阶段须传入相同的 `--ruler-source-dir`。
- 已有 RULER 长度目录拒绝覆盖，包括失败生成留下的目录；重试前将失败的长度目录移出输出目录，或指定新的 `--output-dir`。同一模型修改长度、样本数或 seed 等设置并需重建已有长度时，用新的 `--output-dir` 保存样本。
- RULER 的 `<length>` 表示完整输入与该任务生成预算的总长度上限；`chat` 模式预留聊天包装开销，`base` 模式预留编码时的 special tokens，任务生成器另行预留输出预算。
- 官方生成器将 `answer_prefix` 单独保存；本项目现有聊天评估只读取 `input`，不拼回该前缀。`base` 评估须将 `input + answer_prefix` 编码为完整输入。多值／多 query 任务保持一条输入、一次回答。
- `base` 模式的 NIAH 使用 [本地 NVIDIA 生成器副本](vendor/ruler_niah.py)，修正 512-token essay 无法缩短到 500 词以下及超长重试不退出的问题；原始素材不变，其他任务继续使用下载的官方生成器。
- 同一模型的比较方法应使用同一批生成样本。评估自定义位置时，`--benchmark-data` 指向 RULER 的 `--output-dir` 或 LoCoMo 的 `<data-root>/locomo`。
- 完整生成参数执行 `bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh --help` 查看；下载参数见上表。

## 3. 评估入口

各方法共用上述数据目录及公共读取、评分模块，使用各自的 checkpoint 与运行参数：

| 方法 | 运行说明 |
|---|---|
| v4 | [Benchmark 评估与看板](../v4/benchmarks/README.md) |
| v4.1 | [独立评估与结果看板](../v4_1/scripts/README.md#4-独立评估) |
| Full-context 基线 | [基线运行说明](../baselines/README.md) |

- **数据选择**：RULER 指向包含长度子目录的 `data/benchmarks/ruler-v1/<model-name>/`，LoCoMo 指向包含 `locomo10.json` 的目录。
- **可比性**：同一模型的不同方法使用同一批生成样本及评分协议；评估结果、checkpoint 组织和可视化由各方法负责。
