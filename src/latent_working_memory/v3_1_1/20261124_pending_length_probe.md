# 20261124_边界输入的待压缩长度探针

创建时间：20261124 11:23:40 Asia/Shanghai

最后修订时间：20261124 11:51:32 Asia/Shanghai

冻结 PPO checkpoint，检验边界判断器实际输入 `hidden[-1]` 能否预测当前待压缩长度。只训练轻量探针，不更新 Writer、压缩器或原边界判断器，不计算 QA。

## 实验设置

- checkpoint：`boundary-post_k256_ppo-decay0p1-gae1-cost0p03_20261122/train/checkpoints/global_step_252`。该点在已有 dev 诊断中仍有自主压缩，适合同时观察首次压缩前后的事件。
- 保留原 train 的 1,002 篇与 dev 的 118 篇文档，二者文档互斥。train 内按文档划分 80% 拟合、20% 调参；选定超参数后在全部 train 上重新拟合，最后报告 dev。
- 每篇文档采样一条轨迹，train/dev 均使用 checkpoint 配置对应的 `temperature=1.0`，`recent_size=256`、`pending_stride=256`、`pending_size=256`，随机种子 42。实际温度以 checkpoint 配置为准，不从运行名推断。
- 精确提取 `event_features[1:-1][decision_mask]`。标签为到达新单元之前的 `pending_tokens_before`；强制等待与 EOF 不作为样本。首次自主 compress 所在事件归入“首次压缩前”，之后的事件归入“首次压缩后”。
- 特征与目标仅用拟合文档计算均值和标准差。Ridge 在 `[0.1, 1, 10, 100, 1000, 10000]` 中选调参 MSE 最低值；目标为平方误差之和加 `alpha * ||w||²`，使用 float64 求解。
- MLP 为 `Linear(2560,128) → SiLU → Linear(128,1)`，AdamW，学习率 `1e-3`、weight decay `1e-4`、batch size 256，最多 100 epoch，patience 10。按最佳 epoch 数重新初始化并在全部 train 上拟合。
- 所有自主决策等权，报告 R²、MAE（tokens）、事件数、文档数和目标标准差。统一拟合的模型分别在全部、首次压缩前、首次压缩后统计；各组 R² 分母使用该组自己的标签均值。另报告训练均值与 `read_tokens` 单变量线性回归基线。

## 输入协议与产物

该历史 checkpoint 的 Writer 输入包含固定聊天提示与 `Document:` 前缀，当前训练代码已采用纯正文输入。本实验明确使用 checkpoint 原始输入协议，以其 tokenizer binding 和 train/dev 的原始 `tokenized_content` 校验还原结果，原 checkpoint 不作修改。实验结论对应这套历史输入分布。

实验目录为 `a100:/data/percyw/projects/latent_working_memory/artifacts/v3-1/pending-length-probe_20261124/`。先保存 `plan.json`，再采集特征；完整轨迹输入保存为 `documents.pt`，唯一格式如下：

```text
{checkpoint, tokenization, data_identity,
 documents: [{id, split, input_ids}]}
```

两个 GPU worker 按文档分片，分别保存 `features/shard_0.pt`、`features/shard_1.pt`，每个文档包含原 dtype 的特征矩阵、pending 长度、首次压缩后标记、已读长度和自主动作。采集过程保存 `features/progress_*.json`。探针目录 `probe/` 保存计划、训练日志、模型参数、逐位置预测和最终 `metrics.json`；大特征仅保存在服务器。

采集入口：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m latent_working_memory.v3_1_1.evaluation.pending_length_features \
  --checkpoint <checkpoint> --documents-file <experiment>/documents.pt \
  --output-dir <experiment> --batch-size 2 --shard-index 0 --num-shards 2
```

另一个 worker 使用物理 GPU 1 和 `--shard-index 1`。全部采集完成后：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m latent_working_memory.v3_1_1.evaluation.pending_length_probe \
  --feature-dir <experiment>/features --output-dir <experiment>/probe --device cuda --seed 42
```

## 结论口径

高 dev R² 支持长度信息在该 checkpoint 的边界输入中可读出；首次压缩后的结果更能检验对 pending 重置的适应。低 R² 只能说明本实验的样本分布和轻量探针未能准确读出长度，不能证明特征完全不包含长度信息。首次压缩前 pending 与已读位置近乎一致，单看该组的高分不足以证明能追踪压缩后的 pending。

## 实际结果

20261124 完成采集与拟合。GPU 0–1 采集耗时约 23 分钟，探针拟合约 4 秒；没有执行 QA 或更新原模型。采集代码提交为 `5151c424`。原始输入的 tokenizer binding 与 train/dev 的 tokenized_content 均精确匹配 checkpoint；保存的逐位置预测重算 R² 与汇总一致。

特征为 2,560 维 float32，共 22,619 个自主决策位置，其中 train 20,218 个、dev 2,401 个。内部调参使用 802 篇拟合文档、200 篇调参文档，最终在全部 1,002 篇 train 上重新拟合；dev 的 118 篇文档始终未参与探针拟合或调参。

Ridge 选择 `alpha=10000`。MLP 在内部调参集上最佳为第 1 epoch，第 11 epoch 触发 patience=10 的早停，最终重新初始化并用全部 train 拟合 1 epoch。调参阶段训练 MSE 持续下降、调参 MSE 未改善，存在泛化限制。Ridge 的最优值落在预设网格上界；本次结果不是探针能力的穷尽搜索。

以下 MAE 的单位均为 tokens。同一探针按子集报告结果，没有分别训练前后两个探针。

| 数据 | 范围 | 文档数 | 决策数 | Ridge R² | Ridge MAE | MLP R² | MLP MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 全部 | 1,002 | 20,218 | 0.331 | 233.0 | 0.400 | 220.7 |
| train | 首次压缩前 | 1,002 | 2,338 | 0.242 | 318.6 | 0.354 | 285.4 |
| train | 首次压缩后 | 999 | 17,880 | 0.349 | 221.8 | 0.408 | 212.2 |
| dev | 全部 | 118 | 2,401 | 0.243 | 251.6 | 0.248 | 249.1 |
| dev | 首次压缩前 | 118 | 292 | 0.139 | 355.6 | 0.176 | 334.8 |
| dev | 首次压缩后 | 118 | 2,109 | 0.262 | 237.2 | 0.259 | 237.3 |

dev 总体标签均值为 550.9、标准差 402.8 tokens。训练均值常数基线的 dev R² 为 −0.0002、MAE 为 293.8；全局 `read_tokens` 线性基线的 dev R² 为 0.0005、MAE 为 295.0。两个探针的 MAE 相比训练均值基线约降低 14–15%。这里比较的是训练均值基线，并非 MAE 最优的常数预测器。

采样分布明显偏向短 pending：

| dev pending 范围 | 文档数 | 决策数 | Ridge MAE | MLP MAE |
|---|---:|---:|---:|---:|
| 256–768 | 118 | 2,022 | 185.1 | 185.2 |
| 1,024–1,792 | 107 | 352 | 539.9 | 525.1 |
| 2,048–3,328 | 16 | 27 | 1,476.0 | 1,441.1 |
| ≥4,096 | 0 | 0 | — | — |

84.2% 的 dev 决策点 pending 小于 1,024；≥2,048 仅有 27 个点。首次压缩后的最大 pending 为 2,304，全部 dev 的最大值为 3,328。本次 sampled 轨迹的自主压缩率为 44.2%，与 dashboard 中 greedy 轨迹的状态分布不同。

**结论：`hidden[-1]` 中存在能跨文档泛化的长度信号，但本次轻量探针读取精度有限。** 整体 R² 约 0.25、MAE 接近一个 stride；首次压缩后的 R² 约 0.26，说明信号并不限于首次压缩前。MLP 与 Ridge 几乎持平，单个 seed 的微小差别不支持非线性头更强或更弱的判断。

这既不能坐实“hidden 几乎不包含长度”，也不能证明输入瓶颈就是策略坍缩的原因。首次压缩后的事件占训练位置的 88.4%，前后指标还受样本量和标签分布影响；全局已读位置线性基线较弱也不能排除其他位置捷径。对 dashboard 的 greedy 长 pending 状态、新的纯正文输入协议，以及其他 checkpoint，本次均没有给出可读性结论。

完整数值见实验目录的 `probe/metrics.json`，逐位置预测与拟合参数分别保存在 `probe/predictions.pt`、`probe/probe.pt`。本地仅同步计划、汇总和训练日志；输入、特征、预测和探针参数保留在服务器。
