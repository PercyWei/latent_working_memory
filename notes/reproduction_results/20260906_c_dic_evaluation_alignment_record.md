# 20260906 C-DIC 评估口径对齐记录（20260906 17:05:27 CST）

创建时间：20260906 16:48:26 CST（UTC+08:00）

最后修订时间：20260906 17:05:27 CST（UTC+08:00）

状态：32 个 validation episodes 的两条件评估、配对校验和离线分析均已完成；训练后 NLL 在三种计分范围均恶化，不能由评测口径单独解释原有差距。

## 目的与结论边界

本次承接 [训练记录](20260904_c_dic_training_reproduction_record.md) 和 [初步评估](20260905_c_dic_evaluation_reproduction_record.md)，先对齐评估样本、轮次范围、指标分母和 initialization/final 对照。模型仍为已有 seed 42、2 epochs、step 1002 checkpoint，本次不改变训练参数。

这里的“对齐”指可审计的内部对照和论文已公开评估条件的对齐，不等于已恢复作者全部 Table 1 实现。论文 Appendix K、Table 16/17 明确使用 MSC session 5 最后一轮；Table 1 的具体 split 聚合和目标轮次仍未公开确认。不能仅因某种口径更接近论文数字就选择它。

## 运行前固定的协议

| 项目 | 固定选择 |
|---|---|
| 数据 | 官方 MSC `session_5/valid.txt` |
| 样本 | 使用 `random.Random(20260906).sample` 无放回抽取 32 episodes，再按原始文件顺序运行；保存具体 episode IDs |
| 历史 | 每个 episode 展开 sessions 1–5 的全部 paired turns；session 1 只构建 gold memory；保留原 loader 的相邻配对与无配对尾项处理 |
| 计分 | sessions 2–5 全部轮次生成并计算 NLL；从同一份逐轮输出派生全部轮次、每 session 最后一轮、session 5 最后一轮三个视图 |
| 配对 | initialization 为公开 ICAE 权重运行相同 C-DIC 状态机；final 为现有训练后权重；不是论文的 ICAE incremental baseline |
| likelihood | gold-response teacher forcing；只监督 response；分别保存内容 token 与最后一个 ICAE stop token 的 NLL |
| PPL | 全部 response tokens 加权、含 EOS 为主；同步报告去掉 EOS 后的 PPL；不把每轮 PPL 直接平均 |
| 文本指标 | corpus BLEU-4，casefold regex 分词，无 smoothing；ROUGE-1/2/L 同时报 macro recall 与 macro F1，无 stemming |
| 生成 | greedy、最多 128 tokens；保留 ICAE `model.eos_id=1` 和既有扩展词表停止规则 |
| 截断 | 不截断 episode 的 turn 数；保留原 adapter 每次文本编码最多 512 tokens 的限制 |
| 输入接口 | 空初始 memory；压缩使用 `<s>[INST] query [/INST] response </s>`；生成保留 `[FT] query [FT]` |
| 检索 | mean pooling、threshold 0.8、decay 0.05、score-desc、无额外 retrieval cap |
| 设备 | A800 物理 GPU 0 运行 initialization，GPU 1 运行 final；每个进程内均为 `cuda:0` |

本地 loader 预核对得到 sessions 1–5 分别有 238、189、185、190、189 个 paired turns。因此每个条件预期评 753 turns，每 session 最后轮视图为 128 turns，session 5 最后轮视图为 32 turns。实际运行与这些分母完全一致；本地抽样 ID 与服务器保存的 32 个 ID 逐项一致。所选 episodes 共丢弃 5 个无配对尾项、无空文本替换。该核对针对评估覆盖范围，不涉及下载文件完整性检查。

输入模板、instruction seeding、指标库细节和 Table 1 split 聚合仍属于原始复现假设。本次把它们固定为相同条件，不通过 validation 数字调优，也不改变 checkpoint。论文正文将 R-1/R-2 描述为 recall，因此 recall 与 F1 分列，不再混用名称。

## 操作与产物

- 代码基线：`506f153d87b07056bfd3bf6cb0785e24282216bb` 加本次工作区修改。
- 新入口：`reproductions/cdic/src/cdic_repro/eval_aligned_msc.py`。
- 配置：`reproductions/cdic/configs/msc_alignment_initialization_a800.json` 和 `msc_alignment_final_a800.json`。
- `response_loss(..., collect_token_nll=True)` 仅增加评估诊断输出；原训练调用不改变。运行时检查逐 token NLL 的均值重构原 response loss，绝对误差容限 `1e-5`。
- 本地验证：首批针对性 16 tests passed；随后项目检查 59 passed、2 个 GPU tests skipped（未传各自配置），根 pytest 配置对旧 gpu/slow markers 有 4 条 warning；新增 episode bootstrap 检查后，对齐模块 6 tests passed。Ruff、`git diff --check` 通过。
- 服务器已有未提交文件，因此评估使用独立代码快照，不覆盖服务器 checkout。
- 服务器根目录：`/data/bywei/projects/latent_working_memory/artifacts/cdic/20260906_evaluation_alignment/`。
- 每个条件保存 `protocol.json`、`runtime.json`、`predictions.jsonl`、`summary.json`；根目录保存两条件对比与运行日志。完成后回收至本地同名 `artifacts/cdic/20260906_evaluation_alignment/`，逐轮结果可离线重算。
- 续跑必须匹配已保存协议；已完成轮次重新构建 gold history 而不重复计分；合并要求完整且严格匹配的目标 ID、query、reference 和 token 分母。
- GPU 运行固定使用 `code/` 快照；离线分析版本另存于 `analysis/eval_aligned_msc.py`，增加 paired episode bootstrap（2000 次重采样，seed 20260906，95% percentile 区间），不影响模型运行。

服务器实际使用以下固定目录运行；`launch.sh` 分别设置物理 GPU 0/1、快照 `PYTHONPATH` 和两个配置，记录开始／结束时间与退出码：

```bash
cd /data/bywei/projects/latent_working_memory/artifacts/cdic/20260906_evaluation_alignment
bash launch.sh
```

离线配对汇总：

```bash
PYTHONPATH=reproductions/cdic/src .venv/bin/python -m cdic_repro.eval_aligned_msc \
  --compare artifacts/cdic/20260906_evaluation_alignment/initialization \
            artifacts/cdic/20260906_evaluation_alignment/final \
  --output artifacts/cdic/20260906_evaluation_alignment/comparison.json
```

## 旧 pilot 的同输出重汇总

从 20260905 pilot 的原始 `predictions.shard00.jsonl` 重算，未重新生成、未改 checkpoint。以下是同一批 test 样本改变计分范围的结果，不与新 validation 样本混为“修改前后”。

| 范围 | 回复数 | response tokens，含 EOS | PPL | BLEU-4 | R-L F1 | R-1 F1 / recall | R-2 F1 / recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| sessions 2–5 全部轮次 | 48 | 1703 | 18.8383 | 0.00838 | 0.15451 | 0.16822 / 0.15494 | 0.02469 / 0.02305 |
| 每 session 最后一轮 | 8 | 248 | 17.7204 | 0.02086 | 0.15852 | 0.17094 / 0.16975 | 0.03341 / 0.03262 |
| session 5 最后一轮 | 2 | 74 | 11.1996 | 0.00000 | 0.15937 | 0.17172 / 0.15142 | 0.02532 / 0.02500 |

计分范围确实显著改变 PPL，但不能同时恢复所有指标：最后两轮的无 smoothing BLEU 为 0，且从 F1 改看 recall 没有消除 ROUGE 差距。旧输出没有逐 token NLL，因此不从它推算去 EOS 的 PPL。原始输出与重算 JSON 保存在本次目录的 `legacy_pilot/`。

## 最终结果

两条件各完成 753 个目标，总计 1506 次 response likelihood／generation；两进程退出码均为 0，无缺失或重复目标。query、reference、轮次标记和 response token 分母逐项一致。最长被计分 response 为 95 tokens（含 EOS），没有 response 触及 512-token 编码上限。两个条件均已退出，GPU 0/1 显存恢复至各 9 MiB。

| 条件 | 开始时间，CST | 完成时间，CST | 墙钟耗时，含加载 |
|---|---|---|---:|
| initialization | 20260906 16:49:27 | 20260906 17:03:26 | 839 秒 |
| final | 20260906 16:49:27 | 20260906 17:00:55 | 688 秒 |

运行环境为 PyTorch 2.0.1+cu118、A800-SXM4-80GB；stop token ID 为 1。final 的 checkpoint progress 确认为 epoch 2、step 1002。

### PPL 与成对差异

| 计分范围 | 回复数 | 内容 / EOS tokens | init PPL，含 EOS | final PPL，含 EOS | init PPL，去 EOS | final PPL，去 EOS |
|---|---:|---:|---:|---:|---:|---:|
| sessions 2–5 全部轮次 | 753 | 23810 / 753 | 16.9929 | 23.0159 | 17.1577 | 24.5058 |
| 每 session 最后一轮 | 128 | 3990 / 128 | 18.5840 | 24.0800 | 18.6095 | 25.5572 |
| session 5 最后一轮 | 32 | 1063 / 32 | 22.5589 | 28.2935 | 22.5346 | 30.2200 |

以下 Δ 为 final − initialization 的 token-weighted NLL，正值表示恶化。置信区间按 episode 成对重采样，保留同一 episode 内轮次依赖；不是把 753 个 turns 当成独立样本。

| 范围 | ΔNLL | paired episode bootstrap 95% 区间 | mean turn NLL 降低的目标数 |
|---|---:|---|---:|
| 全部轮次 | +0.30339 | [+0.24914, +0.35830] | 170 / 753 |
| 每 session 最后一轮 | +0.25908 | [+0.18820, +0.33466] | 29 / 128 |
| session 5 最后一轮 | +0.22650 | [+0.10252, +0.35478] | 8 / 32 |

三种范围的区间都高于 0，支持在这批固定 validation episodes 上，训练后 response prediction 退化。该结论不依赖选择“全部轮次”还是“最后一轮”。按 sessions 2/3/4/5 单独汇总，PPL 也分别从 15.6079/19.1825/16.1004/17.6067 升到 21.5106/25.3985/21.6103/24.1238。

EOS 不能解释高 PPL：全部轮次的 EOS 平均 NLL 从 2.52773 降到 1.15292，但内容 PPL 从 17.1577 升到 24.5058。包含 EOS 反而减轻了表观退化。新 validation 样本的最终轮 PPL 高于全部轮次，也说明旧 2-episode pilot 中“最后轮更好”的现象不能外推。

### 文本指标

ROUGE 以下顺序为 L / 1 / 2，recall 与 F1 分列；BLEU 是固定分词和无 smoothing 的 corpus BLEU-4。

| 范围 | 条件 | BLEU-4 | ROUGE F1，L / 1 / 2 | ROUGE recall，L / 1 / 2 |
|---|---|---:|---|---|
| 全部轮次 | init | 0.01027 | 0.11478 / 0.13488 / 0.01983 | 0.11883 / 0.13668 / 0.01914 |
| 全部轮次 | final | 0.00916 | 0.13533 / 0.14718 / 0.01921 | 0.12071 / 0.12968 / 0.01688 |
| 每 session 最后一轮 | init | 0.01047 | 0.11048 / 0.12822 / 0.01517 | 0.11357 / 0.13047 / 0.01505 |
| 每 session 最后一轮 | final | 0.00916 | 0.13498 / 0.14677 / 0.01941 | 0.12438 / 0.13336 / 0.01815 |
| session 5 最后一轮 | init | 0.00942 | 0.10797 / 0.12862 / 0.01916 | 0.11666 / 0.13739 / 0.02012 |
| session 5 最后一轮 | final | 0.01100 | 0.14322 / 0.15247 / 0.02035 | 0.12051 / 0.12678 / 0.01798 |

不能概括为“所有指标都下降”：部分 ROUGE F1 和最终轮 BLEU 提高，但 likelihood 退化，全部轮次的 BLEU、R-1/R-2 recall 下降。init 有 28/753 条空字符串输出，final 为 0；final 又出现重复模板，三种 teacher 句子的精确输出次数分别为 49、48、39，合计 136/753。空输出属于生成结果，不算执行失败。

### 检索与写回行为

| 项目，sessions 2–5 | init | final |
|---|---:|---:|
| 达到阈值并执行 replace | 14 / 753 | 0 / 753 |
| top-1 fallback 并执行 insert | 739 / 753 | 753 / 753 |
| peak score，最小 / 中位 / 最大 | 0.36715 / 0.45579 / 0.81876 | 0.48366 / 0.56646 / 0.62424 |
| 最终写回后的平均 memory slots | 30.03125 | 29.96875 |

训练后，所有被计分轮次都未达到 0.8 阈值。其行为是在 fallback context 上继续压缩，同时不断插入新槽位，没有在 sessions 2–5 中形成 on-topic revision。由 episode 总 turn 数与最终槽位数反推，final 的每个 episode 恰好只有 1 次 replace，均发生于未计分的 session 1；该反推不确定其在 session 1 中的具体轮次。

这与论文 Appendix K 默认阈值下平均约 3.501 个槽位的状态形态差异很大，但样本/split 不同，不能当成严格逐样本比较。此前 8×8 pilot 观察到的早期高分不能直接代表跨 session 行为；本次两种权重在后续 sessions 中都几乎只走 fallback，final 未修复这一问题。

### 本次可得结论与下一步边界

1. 已固定评估样本、完整历史、三个计分范围和指标分母；原 2-episode 数值对轮次选择敏感，但口径差异不足以解释本次配对结果。
2. 在本次 32 个 validation episodes 上，训练后 NLL 的退化稳定存在；去掉 EOS 也不能消除。不能把更新参数和完成训练当成效果复现。
3. 后续应优先审计“为什么跨 session 后几乎不再触发 replace”，并区分 compressor/FT 向量的梯度贡献。此次没有通过调整阈值、模板或 checkpoint 来追分，也没有重训。
4. 这不是完整 Table 1 test-set 复现。论文 Table 1 的 PPL 8.431、BLEU 0.023 等仅作背景参考；作者未公开的输入与指标实现、split 聚合仍需确认。validation 的置信区间描述本次配对样本的不确定性，不证明具体退化机制。

完整机器可读结果：[配对汇总](../../artifacts/cdic/20260906_evaluation_alignment/comparison.json)、[行为诊断](../../artifacts/cdic/20260906_evaluation_alignment/analysis/diagnostics.json)。两条件的逐轮输出、协议、运行日志和代码快照保存在同一 artifact 目录。

## 参考依据

- [C-DIC arXiv v1](https://arxiv.org/pdf/2606.12411v1)：Eq. 7、Algorithm 1、Appendix B/K、Table 1/16/17。
- [原 Table 1 复现计划](../experiment_designs/20260905_c_dic_table1_reproduction_plan.md)。
