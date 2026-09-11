# C-DIC 论文复现（20260909 15:22:35 CST）

创建时间：20260904 16:19:08 CST（UTC+08:00）

最后修订时间：20260909 15:22:35 CST（UTC+08:00）

本目录用于分阶段复现 Context-Driven Incremental Compression（C-DIC）。由于当前没有公开的官方实现，所有论文未明确的行为均记录在 `ASSUMPTIONS.md`。

## 当前状态

已完成不依赖 GPU 的 R1 核心机制：

- 持久化 thread identity 与 revision lineage；
- cosine similarity 与 exponential recency decay；
- multi-state threshold retrieval 与 top-1 fallback；
- 确定性的 insert/replace write-back；
- one-hop retrieval-aware credit plan；
- 可审计的 turn trace 与 adapter-driven inference loop；
- ICAE canonical checkpoint 与固定 LoRA rank 128 校验；
- C-DIC 内置的现代 ICAE 实现及训练、推理 adapter；
- 多轮 JSONL smoke CLI。

R2 训练代码已实现：

- 官方 MSC `session_4/train.txt` episode loader；
- teacher-forced response loss 与 gold-response compression；
- 可配置的 revision-chain gradient window，默认 `gradient_window_size=1`，对应 one-hop ra-TBPTT；
- frozen generator，仅训练 compressor LoRA 与 compression-token embeddings；
- episode-level AdamW step、gradient clipping 配置和确定性 shuffle；
- checkpoint/resume、resolved config、metrics 与 memory trace；
- pilot 和论文规模 JSON 配置。

旧 ICAE 后端曾在 A800 上通过真实 Llama-2-7B-Chat、公开 checkpoint 和五轮对话的工程 smoke test。MSC 训练代码也曾通过本地单元测试、官方数据 schema 检查、单卡 autograd/checkpoint smoke test 和双卡官方 MSC pilot。seed 42 的两 epoch 完整训练已完成，共执行 1002 个 optimizer steps；最终 checkpoint 位于 `checkpoints/cdic/msc_paper_seed42/checkpoints/final.pt`。这些历史结果证明旧训练链路可运行，不代表本次现代 ICAE 后端已经通过服务器复验，论文效果仍需单独评估。

`gradient_window_size` 表示一个 latent state 的计算图最多包含多少次连续 compression。默认值 `1` 保持原复现的 one-hop 行为；大于 `1` 时，仅 argmax write path 可跨 revision 继续反传，到达上限后 detach 并开始新的图分段。因此它是 bounded/chunked TBPTT，而不是逐轮平移的严格 sliding window。

## 目录结构

- `src/cdic_repro/`：retrieval、write-back、memory state 等论文核心机制与推理入口；
- `src/cdic_repro/icae/`：现代 ICAE 实现、checkpoint 和 C-DIC adapter；
- `src/cdic_repro/experiments/msc/`：MSC 数据、训练与评估的完整工作流；
- `src/cdic_repro/experiments/`：checkpoint、分布式运行和生成指标等可复用实验基础设施；
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

## SwanLab 可视化

训练入口和评估汇总入口支持 `--swanlab-mode disabled|offline|online`，默认不记录；所有 runs 默认保存在专用的 `latent-working-memory-cdic-repro` project 中。启用记录时必须通过 `--swanlab-group` 指定实验系列。训练 run 标记为 `train`，集中展示过程曲线；评估和配对比较只写本地 JSON，完成后由唯一的 `evaluation-summary` run 用柱状图和表格统一展示，避免为单个最终值生成折线图。代码自动附加 `scope:reproduction`、`method:cdic` 和 `data:msc`，实验性质等额外标签通过可重复的 `--swanlab-tag` 添加；学习率、阈值和 seed 等具体参数只保存在 config 中。

在线模式复用服务器已有登录，离线数据保存在对应 run 输出目录的 `swanlab/`，实验 ID、group、job type、tags 和链接写入 `swanlab.json`。恢复同一输出目录时会校验这些组织信息并续接该实验，也可通过 `--swanlab-run-id` 指定已有实验。接口行为参见 [SwanLab 初始化与续接文档](https://docs.swanlab.cn/api/py-init.html)和[实验分组文档](https://docs.swanlab.cn/guide_cloud/experiment_track/grouping.html)。

训练看板只记录 mean turn NLL、gradient norm、有效反传比例、检索命中率、平均检索状态数、最终 memory 状态数、step 时间和峰值显存；两个检索指标排除尚无 memory 的 episode 首轮。多卡训练由主进程记录所有活跃 rank 的聚合值。评估汇总看板分别展示 PPL、BLEU、ROUGE-L F1、on-topic rate、平均检索状态数、平均 memory 状态数，以及配对比较的 NLL 差和改善轮次比例；不同量纲不会混在同一张图中，精确数值另存于表格和本地 `report.json`。

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run --project reproductions/cdic --no-sync \
  torchrun --standalone --nproc-per-node=2 \
  -m cdic_repro.experiments.msc.train \
  --config reproductions/cdic/configs/msc_paper_a800.json \
  --swanlab-mode online \
  --swanlab-project latent-working-memory-cdic-repro \
  --swanlab-group cdic-msc-paper-seed42 \
  --swanlab-tag study:paper-reproduction \
  --swanlab-tag scale:full

uv run --project reproductions/cdic --no-sync \
  python -m cdic_repro.experiments.msc.report \
  --config <evaluation-summary-config.json> \
  --swanlab-mode online \
  --swanlab-project latent-working-memory-cdic-repro \
  --swanlab-group cdic-msc-paper-seed42 \
  --swanlab-tag study:paper-reproduction \
  --swanlab-tag scale:full
```

## 服务器环境

C-DIC 直接依赖当前 PyTorch、Transformers 和 PEFT，并在 `cdic_repro.icae` 中维护所需的 ICAE 实现，不再从同级 ICAE 复现仓库导入运行时代码。

当前 lockfile 解析到 CUDA Toolkit 13.0 runtime，符合服务器最高支持 CUDA 13.0 的约束。

`reproductions/cdic/.python-version` 已固定 Python 3.10，因此创建环境时不需要传入 ICAE 环境的 interpreter：

```bash
export UV_CACHE_DIR=/data/bywei/cache/uv
export CUDA_VISIBLE_DEVICES=0

uv sync --project reproductions/cdic --frozen
uv run --project reproductions/cdic --no-sync \
  pytest -q reproductions/cdic/tests
```

后续实验默认仅使用物理 GPU 0 和 1；单进程 smoke test 优先使用 GPU 0。

ICAE checkpoint 使用转换后的 direct state dict，只保存当前参数名下的 LoRA 与 memory/control token embeddings；不接受上游 zero-placeholder 格式或额外 wrapper。MSC 的 `data_root` 固定指向项目内 `data/raw/msc`，loader 读取其下的 `msc/msc_dialogue/`。

`cdic-run-dialogue` 接收 JSONL 输入，每行至少包含 `query`，可选 `id`。输出包含 generated response 和该 turn 的完整 memory trace。

## 多轮 GPU smoke test

GPU test 使用真实 Llama-2-7B-Chat、ICAE checkpoint 和五轮合成对话，检查 strict checkpoint load、latent shape/有限值、retrieval、insert/replace、thread revision、trace 和峰值显存。模型路径、device、生成参数、测试对话和 artifact 路径统一保存在 `configs/gpu_smoke_a800.json`；不提供该配置参数时，测试自动 skip。

```bash
uv run --project reproductions/cdic --no-sync pytest -q -s \
  reproductions/cdic/tests/test_gpu_multiturn_smoke.py \
  --cdic-gpu-config reproductions/cdic/configs/gpu_smoke_a800.json
```

该测试只验证真实模型上的工程链路和状态转移，不将未经过 MSC 训练的 ICAE initialization 当作 C-DIC 论文效果。

截至 20260904 21:26:56 CST，旧 ICAE 后端的环境变量版与 JSON 配置版均已通过真实 GPU test；JSON 配置版结果为：

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
mkdir -p /data/bywei/projects/latent_working_memory/data/raw/msc
curl -L --fail --retry 5 \
  https://parl.ai/downloads/msc/msc_v0.1.tar.gz \
  -o /data/bywei/projects/latent_working_memory/data/raw/msc/msc_v0.1.tar.gz
echo "e640e37cf4317cd09fc02a4cd57ef130a185f23635f4003b0cee341ffcb45e60  /data/bywei/projects/latent_working_memory/data/raw/msc/msc_v0.1.tar.gz" \
  | sha256sum -c -
tar -xzf /data/bywei/projects/latent_working_memory/data/raw/msc/msc_v0.1.tar.gz \
  -C /data/bywei/projects/latent_working_memory/data/raw/msc
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
  -m cdic_repro.experiments.msc.train \
  --config reproductions/cdic/configs/msc_pilot_a800.json
```

pilot 通过后运行 seed 42 的论文规模训练：

```bash
uv run --project reproductions/cdic --no-sync \
  torchrun --standalone --nproc-per-node=2 \
  -m cdic_repro.experiments.msc.train \
  --config reproductions/cdic/configs/msc_paper_a800.json
```

从 checkpoint 恢复：

```bash
uv run --project reproductions/cdic --no-sync \
  torchrun --standalone --nproc-per-node=2 \
  -m cdic_repro.experiments.msc.train \
  --config reproductions/cdic/configs/msc_paper_a800.json \
  --resume-from /data/bywei/projects/latent_working_memory/checkpoints/cdic/msc_paper_seed42/checkpoints/step-000050.pt
```

训练使用一个 seed 42 模型的同步双卡 data parallel：每张卡处理一个 episode，NCCL 同步并平均 trainable gradients 后共同执行 optimizer step。每卡 batch size 为 1，global batch size 为 2；这与论文单卡 global batch size 1 存在明确差异。

训练目录包含 `config.resolved.json`、`data_summary.json`、`trainable_parameters.json`、按 rank 分开的 `metrics.rankXX.jsonl`、`memory_trace.rankXX.jsonl` 和 `checkpoints/`。checkpoint 保存 trainable model state、AdamW state、各 rank RNG state 与训练位置，默认只保留最近两个 step checkpoints。不使用 `--resume-from` 时，程序拒绝写入非空 output directory。

基础模型保存在共享模型目录；数据、predictions、checkpoints 和日志保存在项目根目录下的 Git-ignored `data/`、`artifacts/` 与 `checkpoints/`。

迁移后的实际资源路径为：

- MSC：`data/raw/msc/`；
- ICAE v1 checkpoint：`checkpoints/icae/v1/`；
- C-DIC checkpoint：`checkpoints/cdic/`；
- C-DIC 日志：`artifacts/cdic/logs/`。

历史 `config.resolved.json` 保留训练时记录的旧绝对路径，避免改写实验 provenance。服务器上的旧路径已改为指向新目录的 symlink，因此现有 checkpoint 仍可恢复；新运行统一使用当前配置中的项目内路径。

## MSC held-out pilot

使用同一模型进程在 MSC session 4 validation 的相同样本上依次评估 ICAE initialization 和 C-DIC final：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/cdic --no-sync \
  pytest -q -s reproductions/cdic/tests/test_gpu_msc_heldout.py \
  --cdic-msc-eval-config reproductions/cdic/configs/msc_heldout_pilot_a800.json
```

首轮 8 episodes × 8 turns pilot 已完成。训练后 token-weighted loss 略升，且 threshold `0.8` 下 cross-episode false accept rate 从 12.5% 升至 100%；该结果只构成诊断信号，需扩大样本并补充 threshold-independent 指标。详细记录见 `notes/reproduction_results/20260905_c_dic_evaluation_reproduction_record.md`。

## MSC 统一评估

`msc/evaluate.py` 统一承担 Table 1 MSC 评估与 initialization/final 对齐比较。入口固定加载 session 5 数据：session 1 只构建 gold memory，sessions 2–5 执行 teacher-forced PPL 与 greedy generation。配置中的 `condition` 决定是否加载 C-DIC 训练 checkpoint，`split` 可选择 `valid` 或 `test`，因此同一执行协议可以直接用于 Table 1 test 设置下的训练前后比较。

每轮只运行一次生成，再分别汇总 sessions 2–5 全部轮次、每 session 最后一轮和 session 5 最后一轮；PPL 区分含／不含 EOS，ROUGE 区分 recall／F1。逐 token NLL、目标 ID 和协议均保存，允许不重跑模型即可重新汇总。`episode_count` 为 `null` 时评估全部 episodes；指定数量且 `sample_seed` 为 `null` 时取数据集前 N 条，提供 seed 时执行固定随机抽样。

固定抽样的 32 条 validation 对齐评估：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/cdic --no-sync \
  python -m cdic_repro.experiments.msc.evaluate \
  --config reproductions/cdic/configs/msc_alignment_initialization_a800.json

CUDA_VISIBLE_DEVICES=1 uv run --project reproductions/cdic --no-sync \
  python -m cdic_repro.experiments.msc.evaluate \
  --config reproductions/cdic/configs/msc_alignment_final_a800.json

PYTHONPATH=reproductions/cdic/src uv run python -m cdic_repro.experiments.msc.evaluate \
  --compare <initialization_dir> <final_dir> --output <comparison.json>
```

单项评估和配对比较不会创建 SwanLab run。全部结果完成后，准备一个汇总配置，将展示名称映射到已有的 `summary.json` 和 comparison JSON：

```json
{
  "artifact_dir": "/data/bywei/projects/latent_working_memory/artifacts/cdic/cdic-msc-evaluation-summary-20260910",
  "evaluations": {
    "initialization-th0.80": "/path/to/initialization_threshold80/summary.json",
    "initialization-th0.85": "/path/to/initialization_threshold85/summary.json",
    "final-lr2e-4-th0.80": "/path/to/final_baseline/summary.json"
  },
  "comparisons": {
    "lr2e-4-th0.80": "/path/to/baseline_comparison.json"
  }
}
```

汇总只读取现有 JSON，不加载模型或占用 GPU：

```bash
uv run --project reproductions/cdic --no-sync \
  python -m cdic_repro.experiments.msc.report \
  --config <evaluation-summary-config.json> \
  --swanlab-mode online \
  --swanlab-project latent-working-memory-cdic-repro \
  --swanlab-group cdic-msc-hyperparameter-screen-20260909 \
  --swanlab-tag study:ablation \
  --swanlab-tag scale:full
```

Table 1 test 两条 episode pilot 的训练前后配对评估：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project reproductions/cdic --no-sync \
  python -m cdic_repro.experiments.msc.evaluate \
  --config reproductions/cdic/configs/table1_msc_initialization_pilot_a800.json

CUDA_VISIBLE_DEVICES=1 uv run --project reproductions/cdic --no-sync \
  python -m cdic_repro.experiments.msc.evaluate \
  --config reproductions/cdic/configs/table1_msc_pilot_a800.json

PYTHONPATH=reproductions/cdic/src uv run python -m cdic_repro.experiments.msc.evaluate \
  --compare \
  /data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_table1_msc_pilot/initialization \
  /data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_table1_msc_pilot/final \
  --output /data/bywei/projects/latent_working_memory/artifacts/cdic/20260905_table1_msc_pilot/comparison.json
```

test split 可用于最终报告，但 threshold、生成长度等选择应先在 validation 上完成，避免用测试集调参。设置 `num_shards > 1` 时，各分片结果写入 `artifact_dir/shard-XX-of-YY/`；配对比较时传入 initialization 和 final 对应的同编号分片目录。

initialization 指相同 C-DIC 状态机加载公开 ICAE 权重，不等同于论文的 ICAE incremental baseline。Appendix K 明确使用 session 5 最后一轮；Table 1 的具体轮次、split 和指标库仍未完全确认。操作、分母和结果见 [20260906 评估口径对齐记录](../../notes/reproduction_results/20260906_c_dic_evaluation_alignment_record.md)。

本次两条件均完成 753 个计分轮次。全部轮次 PPL 从 initialization 的 16.9929 升至 final 的 23.0159；session 5 最后一轮从 22.5589 升至 28.2935。三种计分范围的 paired episode bootstrap 区间均支持 NLL 恶化；final 在 sessions 2–5 的 753 次检索全部走 fallback/insert，后续应优先审计跨 session 的检索与压缩状态。

此前 final-only 的两 episode Table 1 pilot 已完成，PPL 高于论文、BLEU 低于论文。由于作者未公开 prompt serialization、instruction initialization 和 metric 实现，暂不启动全量运行。协议与结果记录见 `notes/experiment_designs/20260905_c_dic_table1_reproduction_plan.md` 和 `notes/reproduction_results/20260905_c_dic_evaluation_reproduction_record.md`。
