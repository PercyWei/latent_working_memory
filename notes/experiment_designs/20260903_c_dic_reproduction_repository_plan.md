# 20260903_C-DIC 复现与项目仓库结构规划

创建时间：20260903 19:58:48 CST（UTC+08:00）

最后修订时间：20260903 21:22:33 CST（UTC+08:00）

状态：根项目与 ICAE v1 迁移已完成；C-DIC 尚未实现

## 术语

- **Faithful reproduction（忠实复现）**：仅实现论文明确描述的 C-DIC 机制，并记录所有不得不补充的假设，不混入本项目的新方法。
- **Mechanism reproduction（机制复现）**：验证检索、写回、memory 演化和 retrieval-aware truncated backpropagation through time（ra-TBPTT，检索感知截断反向传播）的行为。
- **Result reproduction（结果复现）**：在论文数据、模型和指标上复现主要表格、消融与效率趋势。
- **Compatibility island（兼容环境）**：为旧依赖单独维护的 `uv` 项目。它通过标准文件接口与主项目通信，不要求与主项目共享同一个 Python 环境。
- **Run manifest（运行清单）**：记录代码版本、配置、数据版本、模型、依赖、硬件和随机种子的机器可读文件。

## 代码开放状态

截至 20260903 19:58 CST，未找到 C-DIC 官方代码仓库：

- [论文 arXiv 页面](https://arxiv.org/abs/2606.12411)及可下载的 arXiv 源文件没有代码链接；
- 精确标题、`C-DIC`、arXiv ID 和多位作者姓名的 GitHub 搜索均未发现官方实现；
- 作者 [Jaehyeok Kim 的个人页面](https://stevejaehyeok.github.io/)将 C-DIC 标记为 `Code (Coming soon!)`；
- 已检查的公开作者仓库中没有名称或说明与 C-DIC 对应的项目。

因此当前工作属于 paper-based reimplementation，而不是运行作者代码。后续应定期复查作者页面；若官方代码发布，保留当前实现，并新增逐模块差异审计，不直接覆盖。

可用上游资源：

- [ICAE 官方仓库](https://github.com/getao/icae)：提供 C-DIC 使用的 latent compressor 基础代码与 Llama-2-7B-Chat checkpoint；
- [MSC 官方项目](https://parl.ai/projects/msc/)：主要训练与评估数据；
- [REALTALK 官方仓库](https://github.com/danny911kr/REALTALK)：论文使用的长对话零样本评估数据；
- [LongMemEval 官方仓库](https://github.com/xiaowu0162/LongMemEval)：辅助长程问答评估。

## 当前可复现边界

论文明确给出的核心设置包括：

- frozen `Llama-2-Chat-7B` generator；
- 由公开 ICAE checkpoint 初始化的 LoRA compressor；
- 每个 thread state 使用 128 个 compression tokens；
- cosine similarity、检索阈值 $\tau=0.8$、recency decay $\alpha=0.05$；
- batch size 1，AdamW，learning rate $2\times10^{-4}$，MSC 上训练 2 epochs；
- 单张 A100 80GB，论文报告约 17 GPU hours；
- seeds 42、43、44；
- MSC 训练，REALTALK 与 LongMemEval 零样本评估。

尚未充分说明、必须进入 assumption ledger（假设清单）的内容包括：

- 具体 ICAE checkpoint 文件及其校验值；
- pooling function $\psi$ 最终使用 mean、CLS 还是其他实现；
- LoRA rank、target modules、dropout 和初始化；
- query 的 compressor 输入模板与 special tokens；
- 多个 retrieved states 的拼接顺序、位置编码和最大长度处理；
- MSC/REALTALK 的完整序列化模板、speaker 顺序与截断规则；
- ra-TBPTT 的具体 autograd graph 构造和训练循环；
- optimizer 的 weight decay、warmup、gradient accumulation、gradient clipping；
- BLEU smoothing、ROUGE 实现版本和文本规范化；
- MSC-QA 构造数据及 GPT-4o judge prompt；
- latency 测量中的生成长度、warmup、同步和缓存设置。

上述缺口意味着第一阶段应以机制一致和定性趋势为目标，不能预设能够精确复现全部数值。

## 仓库结构

建议采用“轻量主项目 + 独立论文复现环境”的结构：

```text
latent_working_memory/
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md
├── configs/
│   ├── data/
│   ├── models/
│   ├── methods/
│   └── experiments/
├── src/latent_working_memory/
│   ├── contracts/
│   │   ├── batches.py
│   │   ├── memory.py
│   │   └── run_outputs.py
│   ├── data/
│   │   ├── schemas.py
│   │   ├── msc.py
│   │   ├── realtalk.py
│   │   └── longmemeval.py
│   ├── methods/
│   │   ├── full_context.py
│   │   ├── no_memory.py
│   │   ├── truncation.py
│   │   ├── rag.py
│   │   ├── whole_prefix.py
│   │   └── streaming_baselines.py
│   ├── evaluation/
│   │   ├── generation.py
│   │   ├── qa.py
│   │   ├── memory_diagnostics.py
│   │   └── efficiency.py
│   ├── instrumentation/
│   │   ├── memory_trace.py
│   │   └── resource_usage.py
│   └── cli/
│       ├── prepare_data.py
│       ├── run_method.py
│       └── evaluate.py
├── reproductions/
│   ├── icae/
│   │   ├── README.md
│   │   ├── UPSTREAM.md
│   │   ├── pyproject.toml
│   │   ├── uv.lock
│   │   ├── src/icae/
│   │   ├── src/icae_repro/
│   │   ├── vendor/peft/
│   │   └── tests/
│   └── cdic/
│       ├── README.md
│       ├── UPSTREAM.md
│       ├── ASSUMPTIONS.md
│       ├── pyproject.toml
│       ├── uv.lock
│       ├── .python-version
│       ├── configs/
│       ├── patches/
│       ├── src/cdic_repro/
│       │   ├── icae_adapter.py
│       │   ├── memory_state.py
│       │   ├── retrieval.py
│       │   ├── writeback.py
│       │   ├── compressor.py
│       │   ├── trainer.py
│       │   ├── inference.py
│       │   └── export.py
│       └── tests/
├── experiments/
│   ├── suites/
│   └── manifests/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── notes/
├── data/          # gitignored: raw、interim、processed
├── artifacts/     # gitignored: checkpoints、predictions、memory traces
└── reports/       # 聚合后的表格、图和可审计结果摘要
```

### 结构边界

- `reproductions/cdic/` 只放忠实复现代码、论文配置和不可避免的兼容补丁。
- `reproductions/icae/` 保存 C-DIC 所需的 ICAE v1 与定制 PEFT，并维护独立锁文件。
- C-DIC 后续将 ICAE 作为明确的本地上游依赖，不复制第二份 ICAE 源码。
- 新的 latent allocation、gap 方法和研究消融不得直接修改 `cdic_repro`；应作为主项目中的独立 method 实现。
- `src/latent_working_memory/contracts/` 定义数据 batch、memory snapshot 和运行产物格式，不依赖 Transformers 或 PyTorch。
- 数据预处理和指标尽量由主项目统一提供，防止每种方法使用不同样本、序列化或分母。
- C-DIC 与主项目通过标准产物通信，不要求两个环境互相 import 模型代码。

## 标准运行产物

每个 method run 输出同一目录结构：

```text
artifacts/runs/<run_id>/
├── manifest.json
├── config.resolved.yaml
├── predictions.jsonl
├── memory_trace.jsonl
├── metrics.json
├── resource_usage.json
└── logs/
```

关键字段：

- `manifest.json`：Git commit、dirty status、uv lock hash、Python、CUDA、GPU、模型 revision、数据 hash 和 seed；
- `predictions.jsonl`：sample、turn、prompt、reference、prediction 和 token counts；
- `memory_trace.jsonl`：每轮 retrieval scores、retrieved IDs、write target、insert/replace、slot count、state bytes 和 detach edges；
- `resource_usage.json`：write/read wall time、peak allocated/reserved GPU memory 和 OOM；
- `metrics.json`：原始 numerator、denominator、聚合值和 metric implementation version。

比较实验只读取这些产物。这样即使某个方法使用独立环境，也能进入同一评测流水线。

## uv 环境规划

### 主项目

根目录使用一个轻量 `uv` 项目，负责数据、评测、编排和兼容性较好的基线：

- 提交 `.python-version`、`pyproject.toml` 和 `uv.lock`；
- 使用 `uv run` 执行命令；
- 将 `pytest`、`ruff` 等放入 `dev` dependency group；
- 将开放式评测或绘图依赖放入独立 dependency groups，避免基础安装过重；
- CI 使用 `uv sync --frozen` 和 `uv lock --check`。

配置建议使用 typed YAML + Pydantic，先不引入 Hydra。只有配置组合数量显著增加时再评估 Hydra，避免早期运行路径过度隐式化。

### ICAE 与 C-DIC 兼容环境

ICAE v1 官方说明要求 Transformers 4.31.0 和其定制 PEFT；这与未来较新的 Transformer/PEFT 方法可能冲突。因此 ICAE 已在 `reproductions/icae/` 单独维护：

- 独立 `pyproject.toml` 与 `uv.lock`；
- Python 3.10；
- ICAE 上游固定 commit，定制 PEFT 以本地 path dependency 或明确 patch 形式安装；
- PyTorch 2.0.1 使用官方 cu118 wheel；CUDA 13.0-capable driver 向后兼容该 runtime，且该版本更接近 ICAE 的 2023 依赖栈；
- GPU 机器使用 `uv sync --project reproductions/icae --frozen`；
- macOS 仅同步主项目和 CPU 测试依赖，不尝试运行 7B 训练环境。

C-DIC 实现阶段可建立独立 `reproductions/cdic/pyproject.toml`，通过本地 path dependency 引用 ICAE package，并生成自己的 `uv.lock`，避免在 C-DIC 中重新 vendor ICAE。

不建议直接把 ICAE 整仓复制进本项目。优先记录 upstream commit，并只保存必要补丁；若上游代码必须修改，补丁应能够从干净 commit 重放。

## C-DIC 模块分解

### ICAE adapter

职责：加载 Llama-2-7B-Chat、公开 ICAE checkpoint、learnable compression tokens 和 LoRA compressor；导出统一的 `compress()` 与 `generate()` 接口。

先验证：

- checkpoint 能无缺失键加载；
- 单段文本能生成形状为 `[128, hidden_size]` 的 state；
- frozen generator 只读取 latent state 和当前 query；
- bfloat16 前向结果有限且可重复。

### Memory state

显式保存：

- 唯一 `state_id`；
- latent tensor；
- 创建轮次、最后检索轮次和 recency counter；
- 是否连接 autograd graph；
- 可选 provenance，仅用于审计，不输入模型。

不要仅用 list index 代表 slot identity，否则 replace 和反向 credit path 容易混淆。

### Retrieval

实现论文公式：pooled query/state cosine similarity 乘以 $e^{-\alpha\Delta t_i}$；记录完整 score vector、阈值命中集合和 top-1 fallback。

pooling 未明确，应将 `mean`、`last`、`CLS-like` 设为显式配置，并把选择写入假设清单。主结果只能使用预先选定的一种，其他选择作为敏感性分析。

### Write-back

严格区分：

- `insert`：最大 similarity 低于阈值；
- `replace`：替换 argmax state，而不是逐 latent token 插值；
- retrieved support set：用于 generation/compression 的多 slot 集合；
- write target：仅 argmax state。

EMA 和 2-layer gate 作为论文消融实现，不与主 C-DIC 路径混用。

### ra-TBPTT trainer

需要专门测试以下梯度语义：

- on-topic replace 时，梯度只沿 argmax write edge 回传一跳；
- 其他 retrieved states 可参与 forward，但 stop-gradient；
- off-topic top-1 fallback 可参与 generation，但对应旧 state detach；
- 更早历史 state 不保留完整计算图；
- teacher-forced training 使用 gold response，closed-loop inference 使用 generated response。

## 复现阶段

### R0：环境与上游冻结

- 固定论文版本、ICAE commit、checkpoint URL/revision 和数据 revision；
- 生成 checksum；
- 完成单 GPU import、checkpoint load 和单样本前向；
- 记录所有未公开细节。

完成标准：从干净 checkout 使用 `uv sync --frozen` 可重复运行 smoke test。

### R1：机制复现

- 使用小模型或合成 tensor 测试 retrieval、recency、insert/replace 和 state identity；
- 用短对话验证 memory trace；
- 用 gradient assertions 验证 ra-TBPTT；
- 验证 ICAE incremental 连续重压缩会出现退化趋势。

完成标准：算法状态转移与论文公式一致，全部无 GPU 单元测试通过，7B smoke test 通过。

### R2：MSC 最小训练闭环

- 只运行 MSC 的小训练子集和固定 validation subset；
- 验证 loss、slot growth、retrieved slot count 和生成结果；
- 冻结所有 generator 参数，只更新论文允许的 compressor/LoRA/token 参数。

完成标准：训练稳定、checkpoint 可恢复、固定样本输出可复查。

### R3：MSC 主结果与消融

优先比较：

1. no memory、truncation、full context；
2. ICAE one-shot、incremental、append；
3. C-DIC；
4. 去除 incremental compression、ra-TBPTT、memory threading；
5. $\tau$、$\alpha$ 和 compression length 敏感性。

三随机种子完成后，先判断是否复现论文的相对排序和 failure mode，再讨论绝对数值差异。BLEU、ROUGE、PPL 必须保存 numerator/denominator 与具体实现。

### R4：零样本迁移和长程效率

- REALTALK per-session；
- REALTALK all-sessions 与 turn cap latency；
- LongMemEval-S；
- total slots、retrieved slots、scoring time、generation time 和 state bytes。

GPT-4o judge 相关结果需保存 model snapshot、prompt、原始判断和重试策略；若无法得到论文 prompt，应明确标记为 protocol approximation。

### R5：Gap 与后续方法比较

在 C-DIC 机制复现稳定后，接入已有 whole-prefix/streaming gap 方案。所有方法固定：

- causal prefix；
- reader 与生成设置；
- 总 latent 数和持久状态字节；
- query/probe 集；
- 数据 partition 与随机种子。

比较集合至少包括：

- whole-prefix ICAE；
- ICAE incremental 与 append；
- C-DIC；
- 简单 recurrent/global-resampler baseline；
- 后续 proposed writer。

C-DIC 原论文的 memory bank 会随 topic shift 增长，因此论文复现结果与 matched-total-budget gap 实验必须分开：前者保留原机制，后者增加明确的总容量控制，标记为 `C-DIC-budgeted`，不能冒充论文原方法。

## 测试与审计重点

- unit tests 不加载 7B 模型，使用小 tensor 或 tiny causal LM；
- integration tests 覆盖一个完整 turn 的 `retrieve → generate → compress → write-back`；
- regression tests 固定论文公式、默认参数和关键 trace；
- GPU tests 使用显式 marker，普通 `pytest` 不触发模型下载；
- 数据下载、模型下载与运行分离，测试不得隐式访问网络；
- 所有生成路径记录 tokenizer、chat template 和 special-token IDs；
- 对官方代码未来发布预留 `upstream comparison` 报告：API、权重加载、数据、训练图和指标逐项比较。

## 下一步

下一步是在 A800 服务器上执行 ICAE 环境检查和无下载 import smoke test；确认 PyTorch、Transformers 与定制 PEFT 兼容后，再配置 Llama-2/ICAE checkpoint 路径并开始单样本前向。C-DIC 仍应从 R0/R1 机制实现开始，不直接进入完整训练或全部 baseline。
