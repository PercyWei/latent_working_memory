# 20261014_公共 Benchmark 数据准备

创建时间：20261014 11:04:16 Asia/Shanghai（UTC+08:00）

最后修订时间：20261127 15:31:23 Asia/Shanghai

本文介绍 RULER、LoCoMo、LongMemEval 和 QASPER 的数据下载与 RULER 样本生成。以下命令均从项目根目录执行。

## 1. 数据概览

| Benchmark | 本地数据 | 准备内容 |
|---|---|---|
| RULER v1 | `data/benchmarks/ruler-v1/<model-name>/<length>/<task>/test.jsonl` | 下载素材后生成样本；默认 8192、16384、32768 tokens 三档、13 项任务，每档每项 100 条 |
| LoCoMo | `data/benchmarks/locomo/locomo10.json` | 完整原始 JSON，含 10 段对话及类别 1–5 的 QA 标注 |
| LongMemEval-S / Oracle | `data/benchmarks/longmemeval/<split>.json`，其中 `<split>` 为 `longmemeval_s_cleaned` 或 `longmemeval_oracle` | 同时下载 S 与 Oracle，各含相同的 500 道题及对应上下文 |
| QASPER v0.3 | `data/benchmarks/qasper/qasper-test-v0.3.json` | test 原始 JSON，含 416 篇论文、1,451 道题 |

## 2. 数据准备

数据下载和 RULER 样本生成由用户手动运行，无需 GPU。下载依赖系统 `curl`、`tar` 及项目环境：

```bash
uv sync --frozen
bash benchmarks/scripts/download_benchmarks.sh --benchmark all
```

**原始数据下载：[download_benchmarks.sh](scripts/download_benchmarks.sh)**

下载入口覆盖所选 benchmark 的同路径文件。RULER 下载素材后还需生成样本；其余 benchmark 可直接读取原始 JSON 评估。

- **RULER v1**：采用固定提交的 [NVIDIA/RULER v1 源码](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)。
  - 下载源码及 essay、SQuAD、HotpotQA、CWE 词表，保存到 `--ruler-source-dir`。
  - 下载 NLTK 的 `punkt`、`punkt_tab`，保存到 `data/raw/nltk_data/tokenizers/`。
- **LoCoMo**：下载[官方 `locomo10.json`](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)。
- **LongMemEval-S / Oracle**：下载作者[清洗版](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)的 S 与 Oracle JSON。
- **QASPER v0.3**：从[测试集压缩包](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz)提取 `qasper-test-v0.3.json`。

| 参数 | 填法与默认值 |
|---|---|
| `--benchmark` | 支持 `all`（默认）、`ruler`、`locomo`、`longmemeval` 或 `qasper` |
| `--data-root` | LoCoMo、LongMemEval 和 QASPER 的数据父目录，默认 `data/benchmarks` |
| `--ruler-source-dir` | RULER 官方源码及下载素材目录，默认 `data/raw/ruler-v1-source` |

`HF_ENDPOINT` 可统一覆盖 HotpotQA 和 LongMemEval 的下载地址；未设置时，前者使用 `https://hf-mirror.com`，后者使用 `https://huggingface.co`。

通过 ModelScope 分发或还原数据时，保留下列完整目录：

```text
data/
├── raw/
│   ├── ruler-v1-source/          # RULER 源码与已下载素材
│   └── nltk_data/                # punkt、punkt_tab 及其目录结构
└── benchmarks/
    ├── locomo/
    │   └── locomo10.json
    ├── longmemeval/
    │   ├── longmemeval_s_cleaned.json
    │   └── longmemeval_oracle.json
    └── qasper/
        └── qasper-test-v0.3.json
```

**RULER 样本生成：[generate_ruler.sh](scripts/generate_ruler.sh)**

使用上述固定版本的源码与素材，仅加载 tokenizer 构造样本。源码目录不在默认位置时，传入对应的 `--ruler-source-dir`。

```bash
bash benchmarks/scripts/generate_ruler.sh \
  --model-path "$HOME/models/Qwen3-4B-Instruct-2507" \
  --lengths 8192 16384 32768 \
  --num-samples 100 \
  --seed 42
```

| 参数 | 填法与默认值 |
|---|---|
| `--ruler-source-dir` | 已下载的 RULER 源码及素材目录，默认 `data/raw/ruler-v1-source` |
| `--output-dir` | 默认 `data/benchmarks/ruler-v1/<model-name>`；显式指定时直接使用该目录，不附加模型名 |
| `--model-path` | 与训练一致的 tokenizer 来源；本地路径或 HF repo ID，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--prompt-format` | `chat`（默认）或 `base`，输入协议见下文 |
| `--tasks` | 空格分隔的 RULER 任务名称；默认全部 13 项 |
| `--lengths` | 空格分隔的长度列表，默认 `8192 16384 32768` |
| `--tokens-to-generate` | 可选，统一覆盖各任务的生成预算；默认沿用官方任务预算 |
| `--num-samples` | 每个长度、每项任务的样本数，默认 `100` |
| `--seed` | 样本生成的随机种子，默认 `42` |

- **输出目录**：`<model-name>` 取模型路径末级目录名或 HF 仓库名。已有长度目录拒绝覆盖；修改设置或失败重试时，移走对应目录或指定新的 `--output-dir`。
- **长度预算**：`<length>` 包括完整输入与生成预算。`chat` 为聊天模板预留空间，`base` 为 BOS 等特殊 token 预留空间。默认生成预算为 NIAH 128、VT 30、CWE 120、FWE 50、QA 32 tokens。
- **输入格式**：保留官方 `input`、`outputs`、`answer_prefix` 等字段。聊天评估只使用 `input`；`base` 评估使用 `input + answer_prefix`。多值／多 query 任务保持一条输入、一次回答。
- **生成器**：`base` 的 NIAH 使用[本地 NVIDIA 生成器副本](vendor/ruler_niah.py)，其余任务使用下载的官方生成器。

## 3. 评估入口

RULER 与 LoCoMo 的评估用法见各方法文档：

| 方法 | 运行说明 |
|---|---|
| v4 | [Benchmark 评估与看板](../src/latent_working_memory/v4/benchmarks/README.md) |
| v4.1 | [独立评估与结果看板](../src/latent_working_memory/v4_1/scripts/README.md#4-独立评估) |
| Full-context 基线 | [基线运行说明](../src/latent_working_memory/full_context/README.md) |

### 公共 LoCoMo 评估

Full-context、ICAE v2 和 v3.1.1 的公共接口与使用说明见 [LoCoMo 文档](locomo/README.md)。

### 公共 LongMemEval 评估

Full-context、ICAE v2、AutoCompressors、SDCC、LaCT、E²-TTT 和 In-Place TTT 均提供 `scripts/eval_longmemeval.sh`。[使用说明](longmemeval/README.md) · [裁判配置](longmemeval/README.md#问答评分)

### 公共 QASPER 评估

Full-context、ICAE v2、AutoCompressors 和 SDCC 均通过 `scripts/eval_qasper.sh` 评估 test。[公共协议与续评参数](qasper/README.md) · [Full-context](../src/latent_working_memory/full_context/README.md) · [ICAE](../reproductions/icae/README.md) · [AutoCompressors](../reproductions/autocompressors/README.md) · [SDCC](../reproductions/sdcc/README.md)
