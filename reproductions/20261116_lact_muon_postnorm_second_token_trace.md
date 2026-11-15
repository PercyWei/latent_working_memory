# 20261116_LaCT Muon PostNorm 第二个生成 token 的数值分歧

创建时间：20261116 02:18:13 Asia/Shanghai

最后修订时间：20261116 02:18:13 Asia/Shanghai

## 结论

对既有无 padding 增量路径的同一条完整输入，首次 decode 的第 0 层 QKV 投影已经出现 B1／B3 数值差异：焦点输入完全相同，4,608 个输出元素中有 5 个不同。该层 attention 与 memory 相加后的 `o_proj` 输入又完全相同，但输出投影重新产生 1 个元素的差异。两个投影均通过同输入、同权重、仅改变 batch 形状的独立重放复现。

因此，本例已经确认了具体的线性投影数值差异，不需要 padding 或新的 TTT 更新才能产生。尚未证明其中某一个元素是最终 `2`／`5` 排序变化的唯一原因，也未定位到底层具体 CUDA kernel 或 cuBLAS 算法。

## 条件与记录有效性

沿用[完整输入控制实验](20261116_lact_muon_postnorm_batch_numerics.md)：`zcy-memory-v4.2` 的物理 GPU 0、L40S 48GB；Muon PostNorm 760M，实际四个 fast-weight heads；float32 参数、BF16 autocast；焦点 `ruler/8192/niah_single_2/62` 的全部 7,435 tokens，B3 为该输入的三份复制。各模式独立新进程，第一次模型调用为 128-token greedy 生成，没有预热。

两组均先生成相同的 token ID `28705`。本次定位的 forward 输入就是这个 token，输出用于选择第二个生成 token。此时各层已提交三次更新，待提交长度由 1,291 增至 1,292；没有执行新的 commit 或 Muon 更新。

先做完整模块采样，再缩小记录范围：

| 记录方式 | B1 前两步完整 logits／128 IDs 与原参照一致 | B3 前两步完整 logits／128 IDs 与原参照一致 | 用途 |
|---|---|---|---|
| 全部模块采样 | 是／是 | 否／是 | 仅探索候选位置，不视为原 B3 的精确轨迹 |
| 只记录目标 decode 的第 0 层 QKV、`o_proj` | 是／是 | 是／是 | 本文主要证据 |

最小记录保留原顶层 logits hook 的 CPU 回读与状态检查，仅在两个投影实际计算完成后，直接将其完整输入输出按原 dtype 复制到 CPU。没有额外前向、没有在生成前重放，也没有保存整份 KV 或 fast weights。

## 同一个 token 在哪里开始不同

表中比较 B1 和 B3 的焦点行，层编号从 0 开始。两个模式的对应权重相同。

| 首次 decode 的位置 | 不同元素数 | 最大绝对差 |
|---|---:|---:|
| 第 0 层 QKV 输入，float32 | 0／1,536 | 0 |
| 第 0 层 QKV 输出，BF16 | 5／4,608 | 0.0001220703125 |
| 第 0 层 `o_proj` 输入，BF16 | 0／1,536 | 0 |
| 第 0 层 `o_proj` 输出，BF16 | 1／1,536 | 0.000244140625 |

Q、K、V 分别有 1、3、1 个元素不同。虽然 QKV 更早不同，但其对当前计算的影响没有保留到本层 `o_proj` 输入；不能据此认定这 5 个元素导致第二个 token 分叉。不同 K／V 还可能进入后续缓存，本实验不判断它们对更后续 token 的影响。

`o_proj` 的第 642 个元素（从 0 编号）为：B1 `0.059814453125`，B3 `0.0595703125`。这是在完全相同的融合输入上重新产生的差异，随后进入 residual／MLP 计算。

## 固定输入的投影重放

对线性投影，设输入行是 $x$，相同权重和偏置为 $W,b$。精确实数运算下：

\[
X_1=[x],\qquad X_3=[x;x;x],\qquad
(X_1W^\top+b)_0=(X_3W^\top+b)_0.
\]

实际测试在完整生成结束后进行：先按照现场 dtype、shape、stride 恢复输入，再调用原投影模块；接着固定焦点输入，将它分别复制成 B1、B3。原模型、autocast 精度和参数不变。

- 两次独立进程中，QKV 与 `o_proj` 按各自现场完整 batch 重放，输出均逐元素一致。
- 固定同一份 QKV 输入，B1／B3 重放再次出现相同位置的 5 个元素差异；最大绝对差与现场相同。
- 固定同一份 `o_proj` 输入，B1／B3 重放再次只在第 642 个元素不同，两个值与现场完全相同。
- B3 的三行输入彼此相同，重放输出的三行也彼此相同；没有借助其他样本内容产生差异。

这将局部差异定位到 BF16 autocast 下、随 batch 形状变化的线性投影执行。没有抓取底层算法选择，因此不进一步指定某个 CUDA kernel，也不将其判为方法公式或样本隔离错误。

原第二步输出仍为：

| 模式 | `2` 的 logit | `5` 的 logit | 单题得分 |
|---|---:|---:|---:|
| B1 | 25.125 | 25.125 | 0 |
| 无 padding 等长 B3 | 24.750 | 24.875 | 100 |

局部投影重放证明了 batch 形状足以产生数值差；它不是将整条后续计算固定后的因果干预，不能把最终排序变化全部归给第 642 个元素。完整模块采样中的后续层误差只用于探索，因其 B3 logits 未通过精确参照，不作为原轨迹的逐层放大证据。

## 入口与产物

`lact/scripts/trace_muon_decode.py` 记录完整模块采样；`compare_decode_traces.py` 在 CPU 比较，并明确标记参照检查失败时不可据此归因。主要定位使用：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=reproductions/lact:reproductions/lact/upstream/lact_llm:src \
  ~/rivermind-data/envs/ttt-lact/bin/python reproductions/lact/scripts/trace_muon_qkv.py \
  --checkpoint ~/rivermind-data/models/reproductions/lact/lact-muon-nope-postnorm-nheads2-chunk2048-760m \
  --ruler-data data/benchmarks/ruler-v1/reproductions/lact-760m \
  --source-plan artifacts/reproductions/padding-validation_20261115/lact-muon-postnorm/plan.json \
  --mode old_B1 \
  --reference-dir artifacts/reproductions/lact-muon-postnorm-batch-diagnosis_20261116/compiled/old_B1 \
  --output-dir artifacts/reproductions/lact-muon-postnorm-token2-trace_20261116/minimal/old_B1
```

B3 将命令中的三个 `old_B1` 替换为 `old_equal_B3`，每组单独进程、使用新的输出目录。

本地与服务器产物均位于 `artifacts/reproductions/lact-muon-postnorm-token2-trace_20261116/`：`plan.json` 记录两阶段计划和完成状态；`full_trace_comparison.json` 保留探索结果及参照检查失败标记；`minimal/` 保存有效的真实投影、重放统计、logits 和生成结果；`minimal_comparison.json` 汇总两组精确对照。

完整采样代码提交为 `0a091dc2`，最小记录和重放为 `b1a890e8`，GPU 运行时服务器仓库均为 clean。本轮只新增诊断与记录，未修改官方实现、本地模型公式、精度、更新时序或正式组批逻辑；没有运行完整 benchmark。
