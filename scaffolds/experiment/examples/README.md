# 20261031_产物结构示例

创建时间：20261031 16:19:44 Asia/Shanghai

最后修订时间：20261031 17:46:40 Asia/Shanghai

本目录以 v4.1 为依据，用一组手工构造的小数据展示计划、训练状态、checkpoint 与评估记录之间的关系。**这些文件是文档示例，配置是字段片段，不是可加载的训练配置；指标、耗时、token IDs、文档 ID 和指纹均为合成值，不代表真实 GPU 实验结果。** `<...>` 字符串表示被省略的真实内容；设备没有执行，显存没有测量。代码模板不读取或运行本目录数据；模板自身的可运行配置和数据见 [template/](../template/README.md)。

所有产物引用统一使用虚拟绝对运行目录 `/example/latent_working_memory/artifacts/example-run`；本目录的 `run/` 对应这个目录。实际运行须记录解析后的真实绝对路径。JSON 使用 UTF-8，JSONL 每行一个完整对象并以换行结尾；示例中的时间安排也仅用于说明记录结构。README 的创建与修订时间为实际编写时间。

```text
run/
├── plan/
│   ├── config.json
│   ├── train.json
│   └── eval-auto/test.json
├── train/
│   ├── config.json
│   ├── data.json
│   ├── metrics.jsonl
│   ├── summary.json
│   ├── status.json
│   └── checkpoints/global_step_2/state.structure.json
└── eval/auto/test/
    ├── status.json
    └── global_step_2/
        ├── eval.json
        ├── work.json
        ├── progress/worker_0.jsonl
        ├── details.jsonl
        └── summary.json
```

| 文件 | 展示的职责 |
|---|---|
| `plan/config.json` | 共同设置，顶层恰为 `seed`、`model`、`teacher`；方法字段是 v4.1 的真实字段片段。 |
| `plan/train.json` | 启动前固定的训练预算、数据选择与执行安排。 |
| `plan/eval-auto/test.json` | 同一训练 run 的自动 test 计划；`training_run` 是训练目录字符串。 |
| `train/config.json` | 选样后使用的训练配置；真实产物保存完整配置。 |
| `train/data.json` | `binding` 绑定选样内容、原始文档集合与 teacher identity；`train`、`dev` 保存各自筛选统计。teacher identity 此处仅保留字段片段。 |
| `train/metrics.jsonl` | 每个完成的 optimizer step 追加一行，按 `train/*`、`dev/*`、`resources/*` 分区。 |
| `train/summary.json` | 训练完成后的 step、最后一个 checkpoint 与训练目录。 |
| `train/status.json` | 控制进程记录命令、物理 GPU 分配、起止时间及退出状态。 |
| `state.structure.json` | `state.pt` 的可读结构说明，保留真实的八个顶层键；tensor、optimizer 与 RNG 状态以文字代替。 |
| `eval/auto/test/status.json` | 任务级调度状态，包含选中 checkpoint、各 worker 与完成计数。 |
| `eval.json` | 当前 checkpoint 的请求身份，包含数据选择、方法和评估协议；`training_run` 是含 `directory` 的对象。 |
| `work.json` | 本次分工生成时的快照；`workers[0]` 是 worker 0 的样本 ID 列表。 |
| `progress/worker_0.jsonl` | 每个完成样本追加 `record` 与设备、显存信息，供失败后恢复。 |
| `details.jsonl` | 去掉进度外层的逐题记录，按原始选样顺序合并。 |
| `summary.json` | 请求身份加聚合指标、输出诊断、停止原因与资源统计；仅在所有目标样本完整后写出。 |

示例训练使用 1 个 rank、每卡 microbatch 2、梯度累积 1，全局 batch 为 2；4 条 train QA 完成 1 个 epoch，得到 2 个 optimizer steps。`plan/train.json` 的 `max_steps=null` 表示未另设 step 上限，`train/config.json` 与 checkpoint 配置中的 `max_steps=2` 是从选样数推导出的实际预算。第 1 步没有执行 dev，因此没有 `dev/*`；第 2 步完成 epoch 并保存 checkpoint，此时数据游标已经进入 `epoch=1, offset=0`。

Test 的 2 条记录都来自独立于 train/dev 的文档。`work.json` 保留分工时的 `completed_samples=0, pending_samples=2`，即使任务最终完成也不改写；最终状态见任务级 `status.json`，逐题完成证据见 `progress/`。进度记录中的 `record` 与 `details.jsonl` 完全一致。

两题 EM/F1 分别为 1 和 0，因此汇总均为 0.5。NLL（负对数似然）按答案 token 数加权：`(1×4 + 2×6) / 10 = 1.6`，perplexity 为 `exp(1.6)`；生成 token 总数为 `2+3=5`。样本耗时合计 1.4 秒，生成阶段合计 0.9 秒，NLL 阶段合计 0.5 秒；调度墙钟时间单独记为 2 秒，不与并行样本耗时混用。

真实 checkpoint 名为 `state.pt`，仅在 optimizer step 完成后原子提交。**`state.structure.json` 不是 checkpoint，严禁交给 `torch.load`。** 此处没有二进制权重，也没有可恢复的随机状态。恢复时真实文件中的 `rng` 与 `query_rng` 都必须各有一个条目对应每个训练 rank；`cursor.step`、`scheduler.last_epoch` 与目录 `global_step_2` 应一致。`config`、`binding` 和嵌套状态中的省略部分不构成新的持久化格式。

示例关闭远程记录（`tracking=disabled`），因此没有 `swanlab.json`。实际开启 SwanLab 时，本地完整配置、增量指标与 run 身份由正式训练代码保存；云端记录不能替代这些本地产物。日志、锁文件与可重建的可视化文件未在此目录展开。
