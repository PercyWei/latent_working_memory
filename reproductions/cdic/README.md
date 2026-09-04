# C-DIC 论文复现（20260904 17:08:27 CST）

创建时间：20260904 16:19:08 CST（UTC+08:00）

最后修订时间：20260904 21:26:56 CST（UTC+08:00）

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

ICAE adapter 已在 A800 上通过真实 Llama-2-7B-Chat、公开 checkpoint 和五轮对话的工程 smoke test。MSC 数据流水线、实际 ra-TBPTT autograd graph 和训练闭环尚未实现。ICAE v1 上游 inference 示例不完整，因此本项目不直接调用该脚本。

## 目录结构

- `src/cdic_repro/`：论文核心机制、ICAE adapter 与运行接口；
- `configs/paper.yaml`：论文默认参数和显式复现选择；
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

模型权重、数据集、predictions 和 checkpoints 均保存在 Git 仓库之外。
