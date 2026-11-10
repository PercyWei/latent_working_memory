# 20261108_TTT baselines 评估数据准备

创建时间：20261108 23:41:27 Asia/Shanghai

最后修订时间：20261110 16:50:53 Asia/Shanghai

本文记录 LaCT、E²-TTT 和 In-Place TTT 所需评估数据的下载与构造。全部命令从项目根目录执行，仅需 CPU；模型路径沿用 `~/rivermind-data/models/reproductions/`，数据直接保存到指定的本地目录。

## RULER

### 任务与规模

| 方法 | 任务 | 总长度档位（tokens） | 每任务、每长度样本数 | 唯一样本总数 | 生成预算 |
|---|---|---|---:|---:|---:|
| LaCT 760M（Momentum 与两个 Muon 变体） | S-NIAH-1/2/3、MK-NIAH-1、MQ-NIAH、MV-NIAH，共 6 项 | 4096、8192、16384、32768 | 500 | 12000 | 128 |
| E²-TTT 1.3B | S-NIAH-1、S-NIAH-2，共 2 项 | 512、1024、2048、4096、8192、16384 | 500 | 6000 | 128 |
| In-Place TTT Qwen3-4B | RULER v1 全部 13 项 | 4096、8192、16384、32768、65536、131072、262144 | 100 | 9100 | 64 |

LaCT 的 500 条采用 [NVIDIA RULER 默认设置](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/config_tasks.sh)，原文未明确每项样本数。表中长度包含完整输入与生成预算。

LaCT 的以下三个 checkpoint 共用一份数据，构造时使用 Momentum 目录中的 tokenizer：

- `lact-momentum-rope-prenorm-nheads4-chunk2048-760m`
- `lact-muon-nope-postnorm-nheads2-chunk2048-760m`
- `lact-muon-nope-prenorm-nheads2-chunk2048-760m`

E²-TTT 的 MLP、SwiGLU 两个 1.3B checkpoint 共用一份 RULER 数据；In-Place TTT 的第三方 32K TTT 与 baseline 共用一份 RULER 数据。

### 下载原始素材

已有 `data/raw/ruler-v1-source/` 和 `data/raw/nltk_data/` 时跳过下载，否则首次准备时需执行：

```bash
bash src/latent_working_memory/benchmarks/scripts/download_benchmarks.sh --benchmark ruler
```

该入口下载官方源码、Paul Graham essays、SQuAD、HotpotQA、词表及 NLTK 资源，三种方法共享这些原始素材。

### 构造命令

LaCT 使用 Momentum 目录的 tokenizer 构造，供其三个 760M checkpoint 使用：

```bash
bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh \
  --model-path ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --output-dir data/benchmarks/ruler-v1/reproductions/lact-760m \
  --prompt-format base \
  --tasks niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multiquery niah_multivalue \
  --lengths 4096 8192 16384 32768 \
  --num-samples 500 \
  --seed 42
```

E²-TTT 使用 MLP checkpoint 的 tokenizer 构造，供其两个 1.3B checkpoint 使用：

```bash
bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh \
  --model-path ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --output-dir data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b \
  --prompt-format base \
  --tasks niah_single_1 niah_single_2 \
  --lengths 512 1024 2048 4096 8192 16384 \
  --num-samples 500 \
  --seed 42
```

In-Place TTT 使用第三方 TTT checkpoint 的 tokenizer 构造，供其 TTT 和 baseline 使用：

```bash
bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh \
  --model-path ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-dir data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b \
  --prompt-format base \
  --lengths 4096 8192 16384 32768 65536 131072 262144 \
  --num-samples 100 \
  --seed 42 \
  --tokens-to-generate 64
```

三种方法使用 `base` 文本续写协议，按各自 tokenizer 重新生成；已有 Qwen3-Instruct 数据不直接复用。未指定 `--tasks` 时生成全部 13 项任务。

每条命令的 `--output-dir` 下保存 `<length>/<task>/test.jsonl`，保留官方 `input`、`answer_prefix`、`outputs` 等字段。已有同名长度目录拒绝覆盖；需要重建时指定新的 `--output-dir`。

## E²-TTT：SQuAD completion 与 SWDE

MLP 与 SwiGLU 两个 1.3B checkpoint 共用以下数据，使用完整 validation split：

| 数据集 | 样本数 | 保存目录 |
|---|---:|---|
| [SQuAD completion](https://huggingface.co/datasets/hazyresearch/based-squad) | 2984 | `data/benchmarks/based-squad/` |
| [SWDE](https://huggingface.co/datasets/hazyresearch/based-swde) | 1111 | `data/benchmarks/based-swde/` |

若尚未安装 `hf`，先执行 `uv tool install huggingface_hub`。下载命令：

```bash
hf download hazyresearch/based-squad \
  --repo-type dataset \
  --local-dir data/benchmarks/based-squad

hf download hazyresearch/based-swde \
  --repo-type dataset \
  --local-dir data/benchmarks/based-swde
```

两个目录均包含 `data/validation-00000-of-00001.parquet`，保留原始 `text` 和 `value` 等字段，无需自行构造。RULER 素材中的 `squad.json` 不能替代 SQuAD completion。

原生 SWDE task 使用的 `hazyresearch/based-swde-v2` 当前跳转到 `hazyresearch/based-swde`；上述命令使用实际仓库名称。
