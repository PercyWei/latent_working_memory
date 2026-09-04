# C-DIC 论文复现（20260904 22:47:18 CST）

创建时间：20260904 16:19:08 CST（UTC+08:00）

最后修订时间：20260904 22:47:18 CST（UTC+08:00）

本目录用于分阶段复现 Context-Driven Incremental Compression（C-DIC）。由于当前没有公开的官方实现，所有论文未明确的行为均记录在 `ASSUMPTIONS.md`。

## 当前状态

已完成不依赖 GPU 的 R1 核心机制：

- 持久化 thread identity 与 revision lineage；
- cosine similarity 与 exponential recency decay；
- multi-state threshold retrieval 与 top-1 fallback；
- 确定性的 insert/replace write-back；
- one-hop retrieval-aware credit plan；
- 可审计的 turn trace 与 adapter-driven inference loop；
- checkpoint schema 检查与 LoRA rank 推断；
- inference-only ICAE v1 adapter；
- 多轮 JSONL smoke CLI。

R2 训练代码已实现：

- 官方 MSC `session_4/train.txt` episode loader；
- teacher-forced response loss 与 gold-response compression；
- one-hop ra-TBPTT autograd path；
- frozen generator，仅训练 compressor LoRA 与 compression-token embeddings；
- episode-level AdamW step、gradient clipping 配置和确定性 shuffle；
- checkpoint/resume、resolved config、metrics 与 memory trace；
- pilot 和论文规模 JSON 配置。

ICAE adapter 已在 A800 上通过真实 Llama-2-7B-Chat、公开 checkpoint 和五轮对话的工程 smoke test。MSC 训练代码已通过本地单元测试、官方数据 schema 检查、单卡 autograd/checkpoint smoke test 和双卡官方 MSC pilot。完整训练尚未完成，因此不能视为已经复现论文训练结果。

## 目录结构

- `src/cdic_repro/`：论文核心机制、ICAE adapter 与运行接口；
- `configs/paper.yaml`：论文默认参数和显式复现选择；
- `configs/msc_pilot_a800.json`：两条 episode、每条八轮的 GPU pilot；
- `configs/msc_paper_a800.json`：论文规模两 epoch 配置；
- `tests/`：retrieval、memory transition、credit assignment 和端到端状态机的 CPU tests；
- `UPSTREAM.md`：论文版本、代码开放状态和复现边界；
- `ASSUMPTIONS.md`：论文未说明的细节及当前选择。

## 本地核心检查

核心测试不依赖 tensor framework，可使用根项目环境运行：

```bash
PYTHONPATH=reproductions/cdic/src uv run pytest -q reproductions/cdic/tests
uv run ruff check reproductions/cdic/src reproductions/cdic/tests
```

## 服务器环境

C-DIC compatibility environment 依赖同级 ICAE reproduction，但使用独立的 `reproductions/cdic/.venv`。两者共享依赖版本、ICAE path dependency 和 uv cache，不共享 Python interpreter 或虚拟环境。

`reproductions/cdic/.python-version` 已固定 Python 3.10，因此创建环境时不需要传入 ICAE 环境的 interpreter：

```bash
export UV_CACHE_DIR=/data/bywei/cache/uv
export CUDA_VISIBLE_DEVICES=0

uv sync --project reproductions/cdic --frozen
uv run --project reproductions/cdic --no-sync \
  pytest -q reproductions/cdic/tests
```

后续实验默认仅使用物理 GPU 0 和 1；单进程 smoke test 优先使用 GPU 0。

加载 7B 模型前，先检查公开 checkpoint：

```bash
uv run --project reproductions/cdic --no-sync cdic-inspect-checkpoint \
  /data/bywei/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt
```

`cdic-run-dialogue` 接收 JSONL 输入，每行至少包含 `query`，可选 `id`。输出包含 generated response 和该 turn 的完整 memory trace。

## 多轮 GPU smoke test

GPU test 使用真实 Llama-2-7B-Chat、ICAE checkpoint 和五轮合成对话，检查 strict checkpoint load、latent shape/有限值、retrieval、insert/replace、thread revision、trace 和峰值显存。模型路径、device、生成参数、测试对话和 artifact 路径统一保存在 `configs/gpu_smoke_a800.json`；不提供该配置参数时，测试自动 skip。

```bash
uv run --project reproductions/cdic --no-sync pytest -q -s \
  reproductions/cdic/tests/test_gpu_multiturn_smoke.py \
  --cdic-gpu-config reproductions/cdic/configs/gpu_smoke_a800.json
```

该测试只验证真实模型上的工程链路和状态转移，不将未经过 MSC 训练的 ICAE initialization 当作 C-DIC 论文效果。

截至 20260904 21:26:56 CST，环境变量版与 JSON 配置版均已通过真实 GPU test；JSON 配置版结果为：

- `1 passed`，耗时约 173 秒；
- 五轮 latent 均为 `[128, 4096]`、bfloat16 且数值有限；
- 峰值 GPU memory 为 14,143,527,424 bytes，约 13.17 GiB；
- trace 覆盖 `initialize`、`insert`、`replace` 和 top-1 fallback；
- 第三轮成功从旧 thread 回答 `ZETA-4827`；
- 更新为 `OMEGA-7319` 后未正确合并和召回，说明未经过 MSC 训练的 ICAE initialization 尚不具备论文要求的稳定 thread revision 能力。

服务器报告保存在：

```text
/data/bywei/projects/latent_working_memory/artifacts/cdic/20260904_multiturn_gpu_smoke/gpu_smoke_report.json
```

报告 SHA256：`a732f168897c80d9dcc8c570139161834b10c061506b2898e51ea2c8ed8b93c7`。

## MSC 训练

训练 loader 使用官方 `session_4/train.txt` 的 1001 条完整多 session records，按原始 session 顺序展开，并将相邻 utterances 组成 `(query, gold response)`。官方数据包含 122 个无配对尾 utterance 和 1 个空 utterance；默认丢弃无法形成 response 的尾项，并将空文本替换为 `__SILENCE__`，统计写入 `data_summary.json`。

准备官方 `msc_v0.1`：

```bash
mkdir -p /data/bywei/datasets/msc/raw
curl -L --fail --retry 5 \
  https://parl.ai/downloads/msc/msc_v0.1.tar.gz \
  -o /data/bywei/datasets/msc/raw/msc_v0.1.tar.gz
echo "e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60  /data/bywei/datasets/msc/raw/msc_v0.1.tar.gz" \
  | sha256sum -c -
tar -xzf /data/bywei/datasets/msc/raw/msc_v0.1.tar.gz \
  -C /data/bywei/datasets/msc/raw
```

先只验证数据，不加载模型：

```bash
uv run --project reproductions/cdic --no-sync cdic-train-msc \
  --config reproductions/cdic/configs/msc_pilot_a800.json \
  --validate-data-only
```

运行最小 GPU pilot：

```bash
uv run --project reproductions/cdic --no-sync \
  torchrun --standalone --nproc-per-node=2 \
  -m cdic_repro.train_msc \
  --config reproductions/cdic/configs/msc_pilot_a800.json
```

pilot 通过后运行 seed 42 的论文规模训练：

```bash
uv run --project reproductions/cdic --no-sync \
  torchrun --standalone --nproc-per-node=2 \
  -m cdic_repro.train_msc \
  --config reproductions/cdic/configs/msc_paper_a800.json
```

从 checkpoint 恢复：

```bash
uv run --project reproductions/cdic --no-sync \
  torchrun --standalone --nproc-per-node=2 \
  -m cdic_repro.train_msc \
  --config reproductions/cdic/configs/msc_paper_a800.json \
  --resume-from /data/bywei/checkpoints/cdic/msc_paper_seed42/checkpoints/step-000050.pt
```

训练使用一个 seed 42 模型的同步双卡 data parallel：每张卡处理一个 episode，NCCL 同步并平均 trainable gradients 后共同执行 optimizer step。每卡 batch size 为 1，global batch size 为 2；这与论文单卡 global batch size 1 存在明确差异。

训练目录包含 `config.resolved.json`、`data_summary.json`、`trainable_parameters.json`、按 rank 分开的 `metrics.rankXX.jsonl`、`memory_trace.rankXX.jsonl` 和 `checkpoints/`。checkpoint 保存 trainable model state、AdamW state、各 rank RNG state 与训练位置，默认只保留最近两个 step checkpoints。不使用 `--resume-from` 时，程序拒绝写入非空 output directory。

模型权重、数据集、predictions 和 checkpoints 均保存在 Git 仓库之外。
