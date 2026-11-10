# 20261110_E²-TTT 推理分项计时与缓存存储诊断

创建时间：20261110 10:59:11 Asia/Shanghai

最后修订时间：20261110 11:16:38 Asia/Shanghai

## 目标与范围

测量官方 MLP／SwiGLU 1.3B checkpoint 的 prefill、普通单 token decode 和跨 TTT chunk 更新的耗时，检查逻辑尾段缓存是否继续持有整个前缀的底层存储，确定显存与速度优化的优先级。本轮只添加独立诊断入口，不修改官方模型、依赖安装或正式 `eval.sh`，不实施缓存压紧、编译或融合优化。

使用 `zcy-memory` 的单张 A800、物理 GPU 0，显式设置 `CUDA_VISIBLE_DEVICES=0`；两个 checkpoint 顺序执行，沿用 `~/rivermind-data/envs/ttt-e2_ttt/`。代码通过 GitHub 同步后从服务器项目仓库运行；执行前保存计划、Git commit／未提交状态、checkpoint 配置及实际依赖来源。

## 测量设置

- 真实输入：RULER 的 `niah_single_1`、`niah_single_2`，长度 512／4K／16K，各取索引 1 的样本（第 2 条）。沿用原 HFLM 编码、`input + answer_prefix`、greedy、128-token 生成上限、EOS 与停止条件，逐条创建独立 cache。
- 边界输入：从真实输入的 token 序列截取 511／512 tokens，各生成最多 4 tokens，只诊断 chunk 边界，不作为 benchmark 分数。
- 每个模型先预热。分别执行无 hook 的原始生成及带诊断的同输入生成，比较完整输入／生成 IDs；速度参照使用无 hook wall，带 hook 的分项时间单独记录。
- 每次 forward 记录输入长度与 CUDA event；按调用前实际 TTT 缓冲长度区分 prefill、普通 decode 和填满 512-token chunk 的 decode。prefill 产生第一个输出 token，生成 128 tokens 通常对应 127 次 decode forward。
- CUDA event 的 device span 包含 CPU 未及时提交后续算子时的等待，不解释为纯 kernel 活动时间；分项记录用于阶段归因，不冒充正式 benchmark 实测。
- prefill 结束和首个 decode 结束记录各层 attention KV、TTT 缓冲、fast weights 与 momentum 的 shape／stride／逻辑字节／底层存储字节。按底层 storage 去重，只保存元数据，不保留额外 tensor 引用；底层存储与逻辑大小的差额不能直接视为可释放的峰值显存。

## 测量结果

两种 checkpoint 各完成 6 条真实输入及 2 条边界输入，配对输入／完整生成 IDs 全部一致，三类阶段均被覆盖。真实输入均生成 128 tokens，执行 127 次普通 decode；这些输入的尾段在生成期间未填满 512，因此边界更新成本由独立短输入测量。

下表为每长度两条真实输入的均值；峰值显存取两条中的最大 PyTorch allocated。wall 来自无 hook 运行，分项跨度来自另一轮配对诊断，不能将它们直接相减解释未归因耗时。wall 不包含 CPU 分词、事后文本解码及评分。

| checkpoint | 长度 | 原始生成 wall | prefill 跨度 | 127 次普通 decode 跨度 | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| MLP | 512 | 7.343 秒 | 0.059 秒 | 6.955 秒 | 3.13 GiB |
| MLP | 4K | 8.047 秒 | 0.414 秒 | 7.164 秒 | 4.73 GiB |
| MLP | 16K | 9.765 秒 | 1.846 秒 | 7.108 秒 | 9.96 GiB |
| SwiGLU | 512 | 7.214 秒 | 0.061 秒 | 7.110 秒 | 3.24 GiB |
| SwiGLU | 4K | 7.810 秒 | 0.475 秒 | 7.262 秒 | 4.92 GiB |
| SwiGLU | 16K | 9.437 秒 | 2.119 秒 | 7.227 秒 | 10.10 GiB |

### 缓存存储

16K prefill 结束时，两种模型具有相同的尾段存储冗余。下表为两条输入的均值，计算口径为该类别去重后的底层 storage 字节减去逻辑 tensor 字节。

| 类别 | 超额底层存储 |
|---|---:|
| attention KV | 2.823 GiB |
| TTT 缓冲 | 2.926 GiB |
| fast weights／momentum | 0 |
| 合计 | **5.749 GiB** |

四个 16K 快照均不存在跨类别 storage 共享或模型参数别名，故此处可相加描述该时点的冗余量。首个 decode 后，两类冗余均归零：attention 缓存的 `roll` 与 TTT 缓冲的 `cat` 创建了新的短存储。这确认了尾部视图持有全长存储的问题，但 **5.749 GiB 不是已经实现的峰值节省**，仍需压紧后的配对测量。

### 普通读出与边界更新

511-token 前缀的首个 decode 会填满 chunk；后续两步为普通读出，测得以下 forward 跨度：

| checkpoint | 普通 decode／步 | 跨 chunk 更新步 |
|---|---:|---:|
| MLP | 55.94 ms | 106.66 ms |
| SwiGLU | 57.50 ms | 118.08 ms |

普通 decode 在真实输入上约为 55–57 ms／步，随输入长度变化较小。即使在 16K，普通 decode 仍占本轮诊断 forward 跨度约 77%–79%。优化完整 chunk 更新主要影响 prefill 和偶发边界步，不能据此预期显著加速当前的 128-token 普通生成。

### 算子采样

`profile_operations.py` 对真实 4K prefill、普通 decode、511 前缀后的更新步各进行一次独立 capture；每项先按相同路径预热，随后用 fresh cache 执行原始 forward。只累计非 annotation 的实际 CUDA 活动，不使用 CPU event 关联的 device 时间；每项均剔除了一个 CUDA annotation。未导出完整 trace。

| 阶段／指标 | MLP | SwiGLU |
|---|---:|---:|
| 普通 decode：CUDA 活动数 | 1,875 | 1,923 |
| 普通 decode：CUDA 活动累计时长 | 8.36 ms | 8.55 ms |
| 普通 decode：`_local_scalar_dense`／DtoH 次数 | 0／0 | 0／0 |
| 4K prefill：`_local_scalar_dense`／DtoH 次数 | 2,016／2,016 | 2,520／2,520 |
| 边界 decode：`_local_scalar_dense`／DtoH 次数 | 288／288 | 360／360 |

普通 decode 的活动包括小矩阵运算、类型转换、逐元素操作、120 次 TTT 缓冲拼接及 48 次窗口 `roll`。CUDA 活动数包含拷贝等事件，不等于 kernel 数。profiler 本身会增加调度开销，不能把上述活动时长与无 profiler 的跨度相减计算 CPU 占比或推导加速上限；结合大量短活动及阶段计时，结果支持优先减少普通 decode 的调度与启动开销。

标量回读与源码中的梯度裁剪条件、动态尾段长度及切片一致，集中发生于完整 chunk 更新。`is_nonzero`、`item`、`_local_scalar_dense` 可能为同一次读取的嵌套事件，不能将其计数相加当作同步次数。

## 优化优先级

1. **压紧尾部缓存。**同时检查 TTT 缓冲与 attention 窗口存储；显存冗余已确认，只复制需保留的尾段，不改变 fast weights 或更新时序。
2. **编译固定形状的普通读出及其周边运算。**优先减少归一化、激活和逐元素操作的独立调度，避免把不断增长的缓冲与整个生成循环一起编译。实际速度和数值一致性需配对验证。
3. **减少完整更新中的同步。**保留原裁剪公式与动态尾段选择，移除不必要的 Python 条件和重复标量读取，主要改善 prefill／边界步。
4. **再评估缓冲预分配与 batching。**不能直接提高当前 harness 的 batch size；多样本 padding 路径尚需保证 attention 与 TTT 状态隔离。

本轮未实施上述优化，没有更新完整 benchmark 的耗时外推，也未启动正式评估。

## 入口与产物

阶段诊断入口为 `e2_ttt/scripts/profile_inference.py`，算子入口为 `e2_ttt/scripts/profile_operations.py`。产物已同步到本地与服务器项目的 `artifacts/reproductions/e2_ttt/inference-profiling_20261110/`：`mlp/`、`swiglu/` 保存阶段计时与缓存元数据，`mlp-operations/`、`swiglu-operations/` 保存算子汇总。记录计划、逐输入生成与一致性及执行状态，不保存大张量或完整 trace，不计入正式评估。

阶段测量使用 `48d8ee43`，算子采样使用 `89a7929c`，执行时服务器 Git 状态均为 clean。两轮均从服务器项目仓库运行，官方 `upstream/`、环境安装与正式评估入口未修改。测量于 20261110 11:12:11 Asia/Shanghai 完成，GPU 进程已退出；结束后物理 GPU 0 占用为 0 MiB。

在服务器仓库根目录运行：

```bash
source ~/rivermind-data/envs/ttt-e2_ttt/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/reproductions/e2_ttt/upstream:$PWD/reproductions/e2_ttt/upstream/lm-evaluation-harness"

for variant in mlp swiglu; do
  python -u reproductions/e2_ttt/scripts/profile_inference.py \
    --checkpoint "$HOME/rivermind-data/models/reproductions/e2_ttt/${variant}-1.3b-15b" \
    --ruler-data data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b \
    --output-dir "artifacts/reproductions/e2_ttt/inference-profiling_20261110/$variant"

  python -u reproductions/e2_ttt/scripts/profile_operations.py \
    --checkpoint "$HOME/rivermind-data/models/reproductions/e2_ttt/${variant}-1.3b-15b" \
    --ruler-data data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b \
    --output-dir "artifacts/reproductions/e2_ttt/inference-profiling_20261110/${variant}-operations"
done
```
