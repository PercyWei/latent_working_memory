# 20260907 可增长 Latent Working Memory v1 实施总计划（16:58:30 UTC+08:00）

创建时间：20260907 16:58:30 UTC+08:00

最后修订时间：20260907 21:50:07 UTC+08:00

状态：M0–M2 已完成；M3 与 P0 的工程链路已用真实 Transformers/PEFT tiny Llama 通过 CPU 集成测试。服务器 Llama-2-7B-Chat smoke test、16-episode P0 训练和独立 dev 关卡尚未运行，因此当前停在进入 M5 之前。

本文把 [v1 框架设计](../20260907_growing_latent_working_memory_framework_v1.md) 转换为工程边界、依赖顺序和验收关卡。研究目标、公式、数据定义和实验口径以框架设计为准；本文只决定如何落地，不另行发明第二套方法定义。

## 1. 总体决策

v1 采用**版本目录内完整纵向切片**。代码、配置、测试、数据、checkpoint、运行产物和后续实施记录都显式包含 `v1`，未来 v2 与 v1 并列，而不是在原文件中不断增加条件分支。现有框架设计稿保留在用户给出的原路径，避免打断已有引用；它通过文件名中的 `v1` 和本文链接绑定该版本。

当前只实现 v1，不创建空的 `v2/`，也不先设计通用 plugin、registry、兼容 loader 或基类体系。多个真实版本出现后，仅将已经证明契约相同的代码提升为共享模块；实验假设、状态格式、模型结构和训练流程默认留在各版本内部。这样既保留 v1 的可复现性，也避免为了未知 v2 过早抽象。

版本边界遵循以下规则：

- Python import 路径固定为 `latent_working_memory.v1`。
- 手写配置固定放在 `configs/v1/`，产物中的 resolved config 记录 `framework_version: "v1"`。
- v1 loader 只读取 v1 checkpoint、runtime memory 和 capacity labels；版本不匹配时直接失败，不做隐式迁移。
- 跨版本比较只消费导出的 predictions、metrics、trace 摘要和 resource usage，不读取对方内部 latent state。
- `reproductions/icae/` 与 `reproductions/cdic/` 保持独立兼容环境；v1 不从其中 import 训练实现。

## 2. 目标目录

```text
src/latent_working_memory/
├── __init__.py
└── v1/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── state.py
    ├── backbone.py
    ├── model.py
    ├── objectives.py
    ├── rollout.py
    ├── capacity.py
    ├── baselines.py
    ├── training.py
    ├── evaluation.py
    ├── checkpoint.py
    ├── prepare_data.py
    ├── train.py
    └── evaluate.py
configs/
└── v1/
    └── pilot.json
tests/
└── v1/
    ├── test_config.py
    ├── test_data.py
    ├── test_state.py
    ├── test_model.py
    ├── test_objectives.py
    ├── test_rollout.py
    ├── test_capacity.py
    ├── test_checkpoint.py
    └── test_evaluation.py
notes/v1/
data/v1/                 # gitignored
checkpoints/v1/          # gitignored
artifacts/v1/<run_id>/   # gitignored
```

根 package 不提供“当前版本”别名。调用方必须显式 import `v1`，避免将来新增版本后旧命令悄然改变行为。测试目录镜像代码版本；不把 v1 测试继续堆在根目录的 `test_growing_memory_*.py` 中。

## 3. 模块边界

| 模块 | 唯一职责 | 不应承担的职责 |
|---|---|---|
| `config.py` | 读取唯一 JSON 配置、构造 typed config、检查 v1 内部约束并输出 resolved config | 不选择实验结果最优值，不兼容其他版本配置 |
| `data.py` | `Episode/Event/Probe`、JSONL、事实真值、64-token cells 与 update partitions | 不加载模型，不让 probe/event 标注进入 writer |
| `state.py` | `MemoryState`、合法增长动作、`seen_tokens` 更新与只读状态检查 | 不保存原文、query、日志字段或模型参数 |
| `backbone.py` | tokenizer、冻结 teacher、冻结 `E0`、学生 projection/reader LoRA、答案位置对齐 | 不管理 rollout、optimizer 或容量动作 |
| `model.py` | memory 初始化、joint updater、growth value network | 不根据未来 probe 选择动作，不持久化训练状态 |
| `objectives.py` | gold NLL、token-level KL、partition 对称 KL 及其聚合 | 不执行 optimizer step，不缓存带梯度的 backbone features |
| `rollout.py` | 单 episode 的 encode → choose → update → read 状态机及 trace | 不拥有数据集划分和 checkpoint schema |
| `capacity.py` | 同根反事实分支、资源代理、target JSONL 与 value regression | 不更新被冻结的学生，不用 teacher 输出计算标签质量 |
| `baselines.py` | `global_at_K`、`fixed_joint`、`scheduled_joint`、`append_only` 的独立配对 | 不复用完整方法训练后的 reader 参数 |
| `training.py` | P0/P1/P2b/P3 循环、TBPTT、梯度边界和 optimizer | 不隐藏 phase 切换，不静默改变增长策略 |
| `evaluation.py` | EM/NLL/PPL、partition、容量、byte-token area 与实测成本汇总 | 不调参，不把 teacher 当 gold |
| `checkpoint.py` | v1 模型 checkpoint 与 runtime memory 的单一 `.pt` 格式 | 不提供跨版本 fallback 或猜测缺失字段 |
| 三个入口模块 | 参数解析并调用上述模块 | 不重复实现业务逻辑 |

依赖方向固定为：

```text
config/data/state
        ↓
backbone/model
        ↓
objectives/rollout
        ↓
training/capacity/baselines
        ↓
evaluation 与命令入口
```

低层模块不得反向 import 训练器或命令入口。`baselines.py` 与主学生共享 v1 内的 block 定义可以减少机械重复，但必须创建独立参数实例，不能共享训练后的 reader 或 writer。

## 4. v1 的最小契约

实现只建立当前训练路径实际使用的类型：

- `ExperimentConfig`：正文全部运行参数及 `framework_version="v1"`；启动时一次性严格检查。
- `Episode/Event/Probe`：对应唯一 JSONL 数据格式。
- `MemoryState(values, seen_tokens)`：部署时唯一持久状态。
- `ReaderOutput`：目标 token logits、逐 token NLL、目标长度；生成结果由单独方法返回。
- `CapacityTarget`：同一学生 checkpoint 下的 policy features、合法动作、分项 cost 与 delta。
- `RolloutTrace`：状态大小、动作、prefix、probe 和成本的审计记录，不进入模型输入。

不创建泛型 `BaseMemory`、通用 trainer protocol、任意模型 adapter registry 或多格式序列化层。若未来 v2 的实际实现需要相同契约，再通过有针对性的重构提取共享部分，并用 v1 回归测试证明行为未变。

## 5. 命令与运行产物

最终首版保留三个显式入口。当前批次已经实现可实际生成数据的 `prepare_data`；`train` 与 `evaluate` 将随 M3–M5 创建，不预先放置只能报错的空入口。

```bash
python -m latent_working_memory.v1.prepare_data \
  --config configs/v1/pilot.json \
  --output-dir data/v1

CUDA_VISIBLE_DEVICES=0 python -m latent_working_memory.v1.train \
  --phase p0 \
  --config configs/v1/pilot.json \
  --data-dir data/v1 \
  --output-dir artifacts/v1/<run_id>

CUDA_VISIBLE_DEVICES=0 python -m latent_working_memory.v1.evaluate \
  --config configs/v1/pilot.json \
  --checkpoint checkpoints/v1/<checkpoint>.pt \
  --data-dir data/v1 \
  --output-dir artifacts/v1/<run_id>
```

`train --phase` 的合法值只覆盖已实现阶段：`p0`、`p1`、`p2_labels`、`p2_value`、`p3`。其中 `p2_labels` 固定完整学生并生成反事实标签，`p2_value` 只训练价值网络；实现时不得把二者合成一个会悄然改动学生参数的循环。

每个 run 至少输出：

```text
artifacts/v1/<run_id>/
├── config.resolved.json
├── manifest.json
├── logs/
├── predictions.jsonl
├── memory_trace.jsonl
├── metrics.json
└── resource_usage.json
```

训练阶段按需增加 checkpoint 引用、capacity targets 与 optimizer progress。`manifest.json` 记录 Git commit/dirty status、`framework_version`、数据 split、模型/tokenizer 实际来源与 revision、seed、CUDA/PyTorch/Transformers/PEFT 版本和物理 GPU。下载资源完整性 hash 或 manifest 不属于默认流程。

## 6. 实施里程碑与关卡

### M0：版本骨架与配置（已完成）

创建 v1 package、pilot JSON 与当前实现使用的硬依赖。配置加载必须拒绝错误的 `framework_version`、未知字段、矛盾维度、非法增长动作和不合法概率；resolved config 可完整复现默认值。命令入口只随对应功能创建，不放置空壳。

推进条件：CPU 上配置测试通过，package 可导入；尚不加载 7B 模型。

### M1：数据、状态与持久化（已完成）

实现合成事实流、event 真值状态机、probe 生成、token cell 编译、partition 生成、`MemoryState` 以及两种 `.pt` 持久化路径。先用少量固定样本覆盖 set/retract/history/compose，再扩大到 256/64/64。

推进条件：不同 update partitions 复用完全相同的 cell features/source positions；prefix gold 正确；read-only 路径不改变 state；保存恢复后下一次更新输入一致。

### M2：纯张量 writer 与容量网络（已完成）

实现 sinusoidal source/slot 编码、初始化、birth slots、3 层 Pre-LN joint updater、增长 action mask 和 2051 维 value features。单元测试使用缩小但结构相同的维度运行，不引入伪 PyTorch 或依赖 fallback。

推进条件：`g=0/8/16` 形状与 dtype 正确；旧 tensor 不原位修改；新位置对旧 memory 有梯度，旧位置对当前 chunk 有梯度；非法动作直接失败或在策略选择处被 mask。

### M3：Teacher、encoder 与学生 reader 链路（工程实现完成，服务器关卡待验证）

接入 Llama-2-7B-Chat、冻结 teacher/`E0`、`W_in/P` 和 reader LoRA；实现 teacher/student prompt、answer-relative logits 对齐、gold NLL 与 token KL。基础 encoder 明确禁用学生 adapter。

推进条件：服务器单卡 smoke test 证明 `T/E0` 无梯度且输出不随 reader LoRA 更新变化；梯度能到达 `W_in/U/P/LoRA`；EOS 两种统计口径可复算。所有 GPU 命令显式使用物理 GPU 0 或 1。

### M4：P0 单次压缩预热（训练驱动完成，真实 P0 实验待运行）

完成一次前缀写入、三类 probes、teacher cache 和 checkpoint/resume；先过拟合 16 个 train episodes，再在独立 dev 上比较 teacher、no-memory 和学生 memory。

推进条件：学生显著利用 memory，且从保存点恢复能得到相同下一 step；若此关失败，不进入递归训练。

### M5：P1 在线联合训练与 partition 辅助项

实现自身状态 rollout、随机增长、8-cell TBPTT、4-cell 监督点、episode 末 optimizer step，以及 `[4]`、`[2,2]`、交替 `[1,1,1,1]` 加后续 2 cells 的一致性分支。

推进条件：长 rollout 中 loss/状态有限；梯度和 detach 边界测试通过；固定容量与随机扩容都能跑通；不同 partition 的共同 prefix 可配对评估。

### M6：P2 容量标签与价值回归

冻结完整学生，从同一 root 展开 `0/8/16`，固定后续 `g=0`，计算 gold NLL 与三项成本，保存与学生 checkpoint 绑定的 targets；随后只训练 value network。

推进条件：重复生成的标签确定一致；学生参数无变化；更换 writer 或 reader checkpoint 会使旧标签校验失败；held-out dev 报告 delta 误差、动作准确性与 regret。若更大容量没有稳定收益，停止在本阶段审计 writer/reader，不用策略结果掩盖问题。

### M7：P3 交替训练

实现“固定策略训练学生 → 固定新学生刷新 targets → 训练策略”的两轮流程。每轮产物保持独立，不能覆盖 P1/P2 checkpoint。

推进条件：每轮能从 episode 边界恢复；策略只由当前 `M/H/N` 选择动作；旧 targets 不会跨学生版本复用；报告触顶率和增长轨迹。

### M8：对照与完整系统评估

按 `no_memory → fixed_joint → scheduled_joint → append_only → global_at_K` 的顺序补齐对照，每个可训练方法拥有独立 reader。最后统一输出质量、partition、容量、持久 bytes、byte-token area、实测 encode/write/read、TTFT、峰值显存及训练成本。

推进条件：比较发生在共同 prefix，训练预算和 reader 归属可审计；完成 `global_at_K` 前不声称接近同容量全量压缩，固定/追加基线未被击败时收缩对应结论。

## 7. 测试分层

- **CPU unit**：配置、JSONL、事实真值、cell/partition、state、动作 mask、成本公式、checkpoint 字段和指标聚合。
- **CPU tensor integration**：缩小 `d_mem/layers/heads` 的真实 PyTorch updater、value network、梯度与非原位语义；本地 tiny Llama 使用标准 Transformers/PEFT 构造、保存和重新加载，不以伪模块替代模型接口。
- **单卡 model smoke**：真实 tokenizer/Llama 权重上的 prompt、E0、teacher/student 对齐、LoRA 梯度和 greedy generation；使用 `CUDA_VISIBLE_DEVICES=0`。
- **单卡 training smoke**：每个 phase 只跑极少 episodes/steps，验证 checkpoint/resume 与产物。
- **双卡并行实验**：GPU 0/1 分别运行独立条件；v1 不因有两张卡就引入 DDP。
- **回归实验**：固定小数据与 seed，检查 partition 配对、capacity target、resume 后下一步和导出指标。数值容差由 dtype/设备明确规定，不用“测试能跑完”替代语义断言。

根项目的常规 `pytest` 保留 CPU 测试；需要模型权重或 GPU 的测试使用显式 marker 和单独命令，缺少硬依赖时正常报错，不以伪模块模拟成功。

## 8. 关键依赖顺序与停止点

关键路径是 `M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7`。`global_at_K` 不阻塞 P0–P3，但阻塞“接近全量压缩”的最终结论；其他基线可以在 M5 后并行开发，但不能共享已训练学生参数。

以下是明确停止点：

- teacher/full-text 在数据上不可靠：修正数据或 prompt，暂缓蒸馏训练。
- P0 不能超过 no-memory：修正 memory-reader 接口，暂缓在线递推。
- P1 中固定更大 K 没有收益：审计 updater、reader 和训练状态分布，暂缓价值网络。
- P2 的真实 action delta 接近噪声：扩大标签或调整 horizon 的实验须单独记录，暂不把 value accuracy 当主结果。
- scheduled 或 append-only 达到相同质量—成本：保留结果，缩小对学习策略或联合重组的贡献主张。

## 9. 首个实现批次

第一批已经完成并以提交 `3a18d6a` 固定 M0–M2：版本目录、严格配置、合成数据与状态契约、checkpoint、joint updater、value network、纯张量目标/指标、feature rollout 及 CPU 测试。

第二批已经实现 `backbone.py`、`training.py` 和 `train.py` 的实际职责：一份冻结基础 Llama 由 teacher、`E0` 和学生 reader 共享；teacher/`E0` 显式禁用 LoRA，学生 reader 启用 `q_proj/v_proj` LoRA；`W_in/P` 保持可训练；teacher/student 只切取答案相对位置 logits。P0 会从 16 起按 8 抽样容量，将完整 cell 前缀以 `g=0` 写入一次，联合优化 gold NLL 与 teacher KL，并持久化只含 `W_in/U/P/LoRA/V` 的 checkpoint。Teacher cache 同时绑定模型、revision、dtype、tokenizer、完整序列化输入和目标长度；resume 恢复模型、optimizer 与 RNG。独立 dev 输出 teacher、学生 memory、学生 no-memory 三者含/不含 EOS 的 token-weighted NLL/PPL、greedy prediction 与 EM；run 同时记录分段日志、memory trace 和资源用量。

当前根测试集共 45 项通过，新增 v1 代码的 Ruff 与格式检查通过，lockfile 与 `pyproject.toml` 一致。测试覆盖 tiny Llama 的 adapter 隔离、cell 分组不变性、答案位置对齐、完整梯度路径、greedy generation、teacher cache 绑定、P0 下一 step 精确恢复，以及通过 `save_pretrained/from_pretrained` 的一整次 P0 运行与恢复。这里证明的是接口和状态机正确，不等同于 7B 质量关卡已经通过。

下一步应先在服务器执行 M3 单卡 smoke 和 16-episode P0 overfit/dev 对照。只有 dev 上 `student_memory` 明显优于同一 reader 的 `student_no_memory`，且恢复后的下一 step 一致，才进入 M5；否则按停止条件审计 prompt、writer-reader 接口或训练信号。第三批才实现 P1/P2/P3，不预先创建 `baselines.py` 或 `evaluate.py` 空壳。

## 10. v1 完成定义

只有同时满足以下条件，v1 才算工程完成：

1. P0–P3 可从明确命令启动、在 episode 边界恢复，并生成完整可审计产物。
2. 部署 rollout 只持有 `MemoryState`、当前 chunk 和固定配置，query/答案/未来输入不进入 writer 或策略。
3. 核心梯度、partition、capacity labels、版本拒绝和保存恢复均有自动测试。
4. 主方法与全部必要基线使用独立、匹配训练预算的 reader。
5. dev/test 评估覆盖设计稿第 9 节的指标与否证条件，并区分 pilot 信号、正式结果和不支持的结论。
6. v1 的代码、配置和产物不因新增 v2 而改变；任何共享化重构都需先通过完整 v1 回归。
