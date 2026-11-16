# 20261115_LaCT 与 E²-TTT 的 padding 推理

创建时间：20261115 20:45:30 Asia/Shanghai

最后修订时间：20261116 10:47:03 Asia/Shanghai

## 实现范围

本轮实现使本地优化路径支持标准左 padding 的不等长 prefill 和逐 token decode。20261115 的验证阶段仍沿用等长评估组批；20261116 已将[长度排序与相邻组批](20261116_ttt_length_sorted_batching.md)接入两个正式评估入口，TTT 状态调度和本笔记记录的模型计算保持不变。

| 方法 | 本地实现 | 原始对照 |
|---|---|---|
| LaCT | `lact/incremental/`，`--inference-mode incremental` | `lact/upstream/`，`--inference-mode original` |
| E²-TTT | `e2_ttt/optimized/`，`--inference-mode optimized` | `e2_ttt/upstream/`，`--inference-mode original` |

没有 padding 历史的全有效 mask 继续使用已有等长路径。带 padding 时，投影、FFN 和 attention 仍批量执行；TTT 按每条样本的真实 token 序列调用已有计算函数。未修改 LaCT 的 `ttt.py`，也未修改 E²-TTT 的 `mlp.py`、`swiglu.py`、`tail.py` 或两者上游源码。

## 位置、缓存与更新

padding 不进入真实 KV 或待提交的 TTT 数据，也不推进真实位置、Momentum、衰减和更新计数。每条样本独立保存真实长度、KV 和 TTT 状态；HF 生成接口使用的物理列数另行维护。mask 为 0 的 decode 行不推进其状态。

设已处理的真实 token 数为 $n>0$，chunk 大小为 $C$：

| 方法 | 已提交更新次数 | 末尾待提交 token 数 | 提交时序 |
|---|---|---|---|
| LaCT | $\lfloor(n-1)/C\rfloor$ | $n-C\lfloor(n-1)/C\rfloor$ | 满 chunk 在下一条真实 token 到来时提交 |
| E²-TTT 推理 | $\lfloor n/C\rfloor$ | $n\bmod C$ | 填满 chunk 的 token 仍读旧权重，随后提交，新权重用于下一 token |

attention 使用有效 token 的独立序列边界，保持原 causal mask、滑动窗口和 RoPE 规则。E²-TTT 还保留固定 FLA 0.4.1 缓存的既有行为：层 0 推进全局计数后，后续已有缓存层的 decode RoPE 偏移相应增加。本轮没有顺带修正这一原 B1 行为。

## 验证入口

在服务器仓库根目录激活对应方法环境，再运行。两条命令均仅检查小样本正确性，不执行完整 benchmark。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=reproductions/lact:reproductions/lact/upstream/lact_llm:src \
  ~/rivermind-data/envs/ttt-lact/bin/python reproductions/lact/scripts/validate_padding.py \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --ruler-data data/benchmarks/ruler-v1/reproductions/lact-760m \
  --length 8192 --teacher-steps 4 --generation-tokens 128 \
  --output-dir artifacts/reproductions/padding-validation_20261115/lact-momentum

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=reproductions/e2_ttt:reproductions/e2_ttt/upstream:reproductions/e2_ttt/upstream/lm-evaluation-harness:src \
  ~/rivermind-data/envs/ttt-e2_ttt/bin/python reproductions/e2_ttt/scripts/validate_padding.py \
  --checkpoint ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --ruler-data data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b \
  --length 8192 --teacher-steps 4 --generation-tokens 128 \
  --output-dir artifacts/reproductions/padding-validation_20261115/e2-mlp-native-dtype
```

输出目录须为新目录。每个 checkpoint 分别运行，替换模型路径和输出目录；LaCT 两个 Muon 变体使用同一验证入口。

## 验证记录

服务器为 `zcy-memory-v4.2`，物理 GPU 0，NVIDIA L40S 48GB。验证以同一 checkpoint、同一精度下的已有无 padding 优化 B1 路径为参考。原始实现保留，但不能把此对照写成已逐项重验官方实现。

检查覆盖 $[C-1,C,C+1]$、$[2C-1,2C,2C+1]$ 的混合长度、固定真实续接、B1 左 padding、padding 内容变化、固定形状的样本隔离和 mask 为 0 的 decode 行。另选三条真实 8K RULER prompt 比较 128-token greedy 生成与原生评分。

状态结构、有限值、真实更新进度、padding 内容忽略和同形状样本隔离是必要检查；跨 batch 的 logits、状态数值误差、argmax、完整生成 IDs 与评分分别记录。允许浮点舍入差异不等于承诺逐元素或逐 token 相同。脚本的 `passed` 只表示必要检查通过，需结合数值与生成字段判断适用性。

20261115 在先执行边界验证、再执行真实生成的调用流程中，默认编译设置下的结果如下。固定续接只统计混合长度 B3 与各自 B1 的比较；真实生成每模型三条，预算为 128 tokens。

| 模型 | 状态、进度与严格隔离 | 固定续接 argmax 相同 | 完整生成 IDs 相同 | 逐题评分相同 |
|---|---|---:|---:|---:|
| LaCT Momentum 760M | 通过 | 34/34 | 3/3 | 3/3 |
| LaCT Muon PostNorm 760M | 通过 | 34/34 | 0/3 | 2/3 |
| LaCT Muon PreNorm 760M | 通过 | 34/34 | 2/3 | 3/3 |
| E²-TTT MLP 1.3B | 通过 | 34/34 | 3/3 | 3/3 |
| E²-TTT SwiGLU 1.3B | 通过 | 34/34 | 3/3 | 3/3 |

五个模型的跨 batch 逐元素 `allclose` 均未全部成立，不能据上表声称状态逐位相同。Muon PostNorm 的 `ruler/8192/niah_single_2/62` 在 B1 得分为 100，混合长度 B3 为 0，第二个生成 token 即出现差异。这记录了该调用流程中的评分差异，不能据此把变化归因于 padding。后续完整输入控制实验见[Muon PostNorm 的 batch 数值差异](20261116_lact_muon_postnorm_batch_numerics.md)。

### LaCT 数值诊断

为区分 padding 逻辑与执行路径，额外比较了相同输入形状下的旧路径、强制新路径，并用完全相同的归一化 hidden 重放同一 attention 层。

- 默认编译设置下，仅增加诊断调用就会改变后续路径的误差分布；日志记录到 `_ttt_read` 的编译缓存达到上限后回退，以及 stride／dtype guard 差异。因此不能将同形状的差异笼统解释为 batch 大小变化，也不能据此断言编译器有错误。
- 在新进程设置 `TORCH_COMPILE_DISABLE=1` 后，Momentum 和 Muon PostNorm 在 2,047、4,095-token 两个长度的 B1、等长 B3 中，新旧路径的逐层状态与 logits 全部完全一致；同一 hidden 的逐算子重放也完全一致。这支持本次 padding 调度保持了原计算与状态逻辑。
- Muon PostNorm 的真实混合长度生成在关闭编译后，IDs 为 2/3 相同、评分仍为 2/3 相同，仍是同一条样本从 100 变为 0。因此其跨 batch 评分变化不能仅归因于编译缓存。20261116 的完整输入对照进一步确认，旧无 padding batch 本身已足以改变该题评分，详见[后续定位](20261116_lact_muon_postnorm_batch_numerics.md)。

另确认 [FlashAttention 2.7.4.post1](https://github.com/Dao-AILab/flash-attention/blob/v2.7.4.post1/csrc/flash_attn/flash_api.cpp#L608-L609) 会在窗口覆盖整批 `max_seqlen_k` 时切换计算分支。混合长度可能改变内核选择，即使某条短样本的数学注意力范围未变；这是一项已知数值路径差别，不是对全部生成差异的解释。

本轮未默认关闭编译，也未为 Muon 增加自动降级或改变精度。Momentum、两个 E²-TTT 模型已有本轮小样本一致性依据；Muon 跨 batch 的完整生成和评分仍不保证相同。后续定位没有发现该案例的 padding 状态逻辑错误，不能将此处评分变化作为 padding 降分的独立证据。本轮不是完整 benchmark，不据三条样本估计准确率变化。

### 环境与产物

LaCT 使用 Python 3.11、PyTorch 2.6.0+cu124、FlashAttention 2.7.4.post1、Transformers 4.57.3；E²-TTT 使用 Python 3.12.3、PyTorch 2.8.0+cu128、FlashAttention 2.8.3、Transformers 4.57.1；FLA 均为 0.4.1。服务器原 E²-TTT 环境为空，本轮按既有固定版本补齐，`uv pip check` 通过。官方 Torch／FlashAttention wheel 仅经本地传输加速，其余依赖使用 PyPI 镜像，未改变版本约束；LaCT 环境未修改。

实现提交为 `bf98f7e2`，E²-TTT 原生读出 dtype 修复为 `602dd7bd`，逐算子诊断扩展为 `656b5d50`。所有 GPU 运行均从 Git 同步后的服务器仓库执行，运行时 Git 状态为 clean。原始源码、LaCT `ttt.py` 和 E²-TTT 三个 TTT 计算文件均与改造前一致。

配置、逐样本预测、边界比较与日志位于 `artifacts/reproductions/padding-validation_20261115/`，汇总为 `validation_summary.json`。其中 `lact-*` 三个主目录和 `e2-*-native-dtype/` 对应上表；`*-eager-*` 为关闭编译的独立诊断，不能混作默认推理结果。仅保存 JSON 与日志，没有 tensor dump。

E²-TTT 当前 packed RoPE 会按整批有效 token 总数扩展 cos/sin 缓存，可能增加长 batch 显存；本轮未优化该开销，也未据此给出新的吞吐或 batch 上限建议。
