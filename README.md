# latent_working_memory（20260906 17:05:27 CST）

最后修订时间：20260906 17:05:27 CST（UTC+08:00）

本项目用于研究 streaming mutable latent working memory，并开展 matched-budget context compression 实验。论文复现与新方法分开管理；ICAE v1 和 C-DIC 分别位于 `reproductions/icae/` 与 `reproductions/cdic/`，各自使用独立的 `uv` 环境。

## 本地开发

```bash
uv sync --frozen
uv run pytest
```

## 服务器目录

项目相关的数据、checkpoint 和实验产物均放在服务器项目根目录 `/data/bywei/projects/latent_working_memory` 下：

```text
data/raw/pwc/          PwC 原始数据
data/raw/msc/          MSC 原始数据与归档
checkpoints/icae/v1/  ICAE v1 公开 checkpoint
checkpoints/cdic/      C-DIC pilot 与完整训练 checkpoint
artifacts/             生成结果、测试报告和日志
```

上述目录均已由 `.gitignore` 排除。Llama-2-7B-Chat 基础模型仍保存在共享目录 `/data/bywei/models/`，Hugging Face 与 `uv` cache 仍保存在 `/data/bywei/cache/`。

迁移前位于 `/data/bywei/datasets/`、`/data/bywei/checkpoints/` 和 `/data/bywei/logs/cdic/` 下的相关旧路径暂时保留为 symlink，以兼容已有 checkpoint 中记录的绝对路径；新配置统一使用项目内路径。

## ICAE 环境

ICAE lock 面向 Linux x86-64，使用与原始代码依赖栈兼容的 PyTorch CUDA 11.8 wheel。服务器驱动支持 CUDA 13.0，并可向后兼容该 runtime。环境安装不会自动下载模型或数据。

```bash
uv sync --project reproductions/icae --frozen
uv run --project reproductions/icae icae-check-environment
```

在 GPU 服务器运行前，参见 `reproductions/icae/README.md`。

## C-DIC 复现

C-DIC 的论文实现位于 `reproductions/cdic/`。当前已完成 retrieval、recency、write-back、state lineage、trace、one-hop credit assignment、ICAE adapter、MSC loader、双卡训练与 checkpoint/resume。A800 多轮 GPU smoke test、双卡 MSC pilot 和 seed 42 的两 epoch 完整训练均已完成。

[20260906 评估口径对齐](notes/reproduction_results/20260906_c_dic_evaluation_alignment_record.md) 已在相同 32 个 validation episodes 上比较初始化与训练后模型，并分列轮次范围、EOS 分母和 ROUGE recall/F1。全部轮次 PPL 从 16.9929 升至 23.0159，session 5 最后一轮从 22.5589 升至 28.2935；当前结果不支持论文效果已复现。

```bash
PYTHONPATH=reproductions/cdic/src uv run pytest -q reproductions/cdic/tests
uv sync --project reproductions/cdic --frozen
```
