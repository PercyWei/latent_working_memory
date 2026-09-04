# C-DIC 论文复现（20260904 17:08:27 CST）

创建时间：20260904 16:19:08 CST（UTC+08:00）

最后修订时间：20260904 19:52:38 CST（UTC+08:00）

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

ICAE adapter 仍需在 A800 上使用真实 Llama-2-7B-Chat 和公开 checkpoint 验证。MSC 数据流水线、实际 ra-TBPTT autograd graph 和训练闭环尚未实现。ICAE v1 上游 inference 示例不完整，因此本项目不直接调用该脚本。

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

C-DIC compatibility environment 依赖同级 ICAE reproduction，复用 Python 3.10、定制 PEFT 和 PyTorch `2.0.1+cu118`：

```bash
export UV_CACHE_DIR=/data/bywei/cache/uv
export CUDA_VISIBLE_DEVICES=0
uv sync --project reproductions/cdic --frozen
uv run --project reproductions/cdic --no-sync pytest -q
```

后续实验默认仅使用物理 GPU 0 和 1；单进程 smoke test 优先使用 GPU 0。

加载 7B 模型前，先检查公开 checkpoint：

```bash
uv run --project reproductions/cdic --no-sync cdic-inspect-checkpoint \
  /data/bywei/checkpoints/icae/v1/llama-2-7b-chat-finetuned-icae_zeroweight_llama2.pt
```

`cdic-run-dialogue` 接收 JSONL 输入，每行至少包含 `query`，可选 `id`。输出包含 generated response 和该 turn 的完整 memory trace。

模型权重、数据集、predictions 和 checkpoints 均保存在 Git 仓库之外。
