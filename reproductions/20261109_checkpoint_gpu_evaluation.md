# 20261109_TTT checkpoint GPU 验证与评估

创建时间：20261109 11:46:00 Asia/Shanghai

最后修订时间：20261109 12:18:53 Asia/Shanghai

## 本轮计划

用户已授权使用 `zcy-memory` 当前空闲的物理 GPU 0。所有 GPU 命令显式设置 `CUDA_VISIBLE_DEVICES=0`，各方法顺序运行，沿用已安装的独立环境和原始模型代码；不新增测试。

本轮验证模型完整加载、原始 GPU 算子、真实样本生成、计分及结果导出，并按用户补充要求估算完整评估耗时。运行验证使用真实评估样本，单独保存输出和资源记录，不计入正式 benchmark 分数。下表保留完整评估范围；本轮没有启动全量评估，也没有缩减或修改原评估数据。

| 方法 | 首批 checkpoint | 正式评估范围 |
|---|---|---|
| LaCT | Momentum 760M | 六项 RULER，4K／8K／16K／32K，每任务每长度 500 条，共 12000 条 |
| E²-TTT | MLP 1.3B、SwiGLU 1.3B | 两项 RULER，512–16K，共 6000 条；完整 SQuAD completion 2984 条、SWDE 1111 条；每个 checkpoint 独立评估 |
| In-Place TTT | 第三方 Qwen3-4B 32K TTT 及 baseline | 十三项 RULER，4K–256K，每任务每长度 100 条，共 9100 条；两个 checkpoint 分别评估 |

正式运行使用各自 `eval.sh`，保留原有 prompt、生成预算、截断规则和评分实现。LaCT 保留原 `use_cache=False` 完整前缀生成；In-Place TTT 超过 32K 的输入标注为长度外推。公开 checkpoint 与原论文模型的差异沿用各方法 README 的说明。

运行验证保存于服务器项目 `artifacts/reproductions/gpu-validation_20261109/`；正式结果保存于 `artifacts/reproductions/<method>/eval/<checkpoint>_20261109/`。记录计划、逐样本输出、实际耗时、GPU 显存及成功或失败状态；只有完成完整范围才汇总正式指标。

## 验证结果

GPU 0 为 A800-SXM4-80GB，开始前显存占用 0 MiB、利用率 0%。三个独立环境及五个 checkpoint 的加载、真实 GPU 生成和原生计分均通过运行检查；这表示流程可执行，不代表答案全部正确或论文指标已复现。

| 方法／checkpoint | 实际验证范围 | PyTorch 峰值分配显存 |
|---|---|---:|
| LaCT Momentum 760M | 严格权重加载；4K 六项 RULER 生成与计分；8K／16K／32K 单项生成与计分，均生成完整 128 tokens | 5.03 GiB |
| E²-TTT MLP 1.3B | SQuAD／SWDE 各两条；RULER 六个长度各两条；另以原生 evaluator 验证 SQuAD、SWDE、两项 512 RULER 的计分与导出 | 9.96 GiB |
| E²-TTT SwiGLU 1.3B | SQuAD／SWDE 各两条；RULER 六个长度各两条 | 10.12 GiB |
| In-Place TTT 32K TTT | 4K 十三项 RULER；8K–256K 各一个 NIAH 样本；生成预算 64 tokens，原生评分通过 | 70.72 GiB |
| In-Place TTT 32K baseline | 与 TTT 对照相同的验证范围 | 62.87 GiB |

E²-TTT 原生 evaluator 的结果导出与 `CachingLM` 逐样本缓存写入成功。三个正式入口均已提供 `--resume`：LaCT／In-Place TTT 复用已保存预测并重新汇总，E²-TTT 使用随附 harness 的 SQLite 生成缓存；续评核对原运行设置。尚未通过人为中断实验验证恢复行为，未新增测试。

## 运行中发现的问题

- **In-Place TTT 256K 的显存碎片。** 默认分配器首次运行失败：PyTorch 已分配 58.69 GiB、保留但未分配 16.12 GiB，申请新的 4.75 GiB 时 GPU 空闲仅 3.94 GiB。设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后，按 128K→256K 顺序重试成功，TTT 的 256K 样本耗时 128.80 秒，峰值分配／保留显存为 70.72／77.04 GiB。baseline 的 256K 样本耗时 126.62 秒，峰值分配／保留为 62.87／68.49 GiB。评估入口默认采用该分配设置；模型、SDPA backend、生成预算与 tokenizer 保持一致。该设置的作用见 [PyTorch 2.8 内存管理说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#optimizing-memory-usage)。
- **第三方 32K 权重的长输入输出退化。** TTT 在所选 64K／128K／256K NIAH 样本上未检出目标，128K／256K 出现无意义输出；baseline 在所选 256K 样本上输出换行。它们均能完成生成和评分，但长输入质量存在问题。当前权重只在 32K 训练，不能将这些少量观察直接解释为原论文方法的效果；具体退化原因尚未定位，未修改 RoPE 或 TTT 更新规则。
- **tokenizer 警告是误报。** Transformers 4.57.3／4.57.6 对这两个本地 Qwen3 checkpoint 触发了 Mistral regex 警告。CPU 核对表明，实际 backend 与 Qwen 正则一致；两个环境在十三项 4K 首样本及两条 Unicode／数字文本上的 token IDs 完全相同。保留原分词行为，不设置 `fix_mistral_regex=True`，无需重建数据。
- **LaCT 的首次编译成本。** 首个 4K 样本耗时 62.33 秒，随后独立测量的预热样本耗时 31.66 秒；增长的生成长度会触发原 `torch.compile` 重编译，并达到其默认缓存上限。预热后 4K 单样本均值为 15.23 秒，估时采用稳定阶段数据，保留原完整前缀重算路径。

## 完整评估耗时估算

假设独占当前 GPU 0、batch size 1、沿用本轮生成实现和输出预算，按各长度的实际样本耗时外推：

\[
\widehat T=\sum_\ell N_\ell\bar t_\ell.
\]

其中 \(N_\ell\) 为完整评估中该长度的样本数，\(\bar t_\ell\) 为验证样本的稳定耗时。LaCT 每长度 3000 条，E²-TTT RULER 每长度 1000 条，In-Place TTT 每长度 1300 条；E²-TTT 另加完整 SQuAD／SWDE 的耗时。

| checkpoint | 完整规模 | 点估计 | 建议预留范围 |
|---|---:|---:|---:|
| LaCT Momentum 760M | 12000 | 156 小时，约 6.5 天 | 5–8 天 |
| E²-TTT MLP 1.3B | 10095 | 18.9 小时 | 16–24 小时 |
| E²-TTT SwiGLU 1.3B | 10095 | 15.3 小时 | 12–20 小时 |
| In-Place TTT 32K TTT | 9100 | 69.9 小时，约 2.9 天 | 2.5–4 天 |
| In-Place TTT 32K baseline | 9100 | 68.4 小时，约 2.8 天 | 2.5–4 天 |
| 五个 checkpoint 串行合计 | 50390 | **329 小时，约 13.7 天** | **约 11–18 天** |

LaCT 的 4K／8K／16K／32K 单样本实测约为 15.23／25.34／49.13／97.82 秒。E²-TTT MLP 的 RULER 约需 14.8 小时，SQuAD 2.6 小时、SWDE 1.4 小时；SwiGLU 分别约 13.4／1.8／0.1 小时。In-Place TTT 的 256K 档单独约需 46 小时／checkpoint，是其主要成本。

这些是少量真实样本的外推，建议范围不是统计置信区间。不同任务的输出长度、首次遇到张量形状的编译、数据处理及 GPU 共享均会影响总耗时。估算未包含 LaCT 两个可选 Muon checkpoint。

## 产物与源码

服务器项目 `artifacts/reproductions/gpu-validation_20261109/` 保存本轮 `plan.json` 及下列结果：

- `lact.json`、`lact_timing.json`：首次生成、预热与六项原生计分。
- `e2_ttt.json`、`e2_ttt_harness.json`、`e2_native_harness/`、`e2_harness_predictions.jsonl`：两个变体的生成及原生 evaluator 结果。
- `in_place_ttt.json`、`in_place_ttt_expandable.json`：首次运行与调整分配器后的结果；首次 OOM 记录保留于对应 `.log`。
- `time_estimates.json`：逐长度实测耗时、外推规模和各 benchmark 的估时。

同名 `.log` 保留运行输出。代码先发布 GitHub，服务器通过 Gitee 同步后从项目仓库执行。同步前的服务器 v3.1.1 改动已保存于 `pre-reproduction-sync-20261109-gpu0` Git stash；其代码内容已包含在同步目标中，仅文档修订时间不同，未覆盖任何未保存代码。
