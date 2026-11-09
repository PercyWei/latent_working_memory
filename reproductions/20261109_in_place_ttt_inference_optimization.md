# 20261109_In-Place TTT 推理优化

创建时间：20261109 21:16:13 Asia/Shanghai

最后修订时间：20261110 00:46:58 Asia/Shanghai

## 实现与代码边界

本轮已完成尾段缓存压紧、MLP 分块和重复分词消除。两个 checkpoint 的代表性生成检查通过，主要收益是降低显存；SDPA 路径的耗时基本持平，完整评估仍约需 69／68 小时，尚未达到半天目标。未启动每个 checkpoint 的正式 9,100 条评估。

- `in_place_ttt/upstream/` 保持官方 revision `be2324829b0e91c8fd10a74d4b43714fde6676e1` 原样，未修改 site-packages。
- `in_place_ttt/optimized/model.py` 复用原参数模块和 checkpoint keys。尾段缓存仅在视图保留了更大底层存储时复制逻辑尾段；attention KV 与 TTT 权重不裁剪。MLP 默认每 16,384 tokens 计算 gate/up，TTT 仍按原 1,024-token chunk 顺序先读后写，保留完整 chunk 立即更新、独立卷积零填充和原投影方向。
- `in_place_ttt/scripts/runtime.py` 的原始路径调用原 OpenCompass `generate`；优化路径编码一次并复用相同输入 IDs，保持特殊 token、生成参数、EOS、解码与原生评分。每题 cache 独立。
- `eval.sh` 默认 `--inference-mode original`；显式 `optimized --mlp-chunk-size 16384` 启用全部优化，设为 `0` 仅使用尾段压紧与单次分词。运行命令见 [README](in_place_ttt/README.md#ruler-评估入口)。

`plan.json.source` 保存初始 Git commit、dirty/status 与实现路径；`runtime.json` 记录实际模型／cache／MLP 类、优化开关、backend 和依赖；`execution.json.attempts` 逐次保存启动状态，预测通过 `execution_attempt` 关联来源。续评严格核对模式和运行配置。

## 验证范围与结果

使用 `zcy-memory` 单张 A800-SXM4-80GB、物理 GPU 0，显式设置 `CUDA_VISIBLE_DEVICES=0` 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，所有 GPU 实验顺序执行。沿用独立环境：PyTorch 2.8.0+cu128、Transformers 4.57.3、OpenCompass 0.5.5、FlashAttention 2.8.3。

主验证代码为 `f94e2231`，TTT 与 baseline 两次运行的 Git 状态均为 clean。计划在执行前提交；服务器从项目仓库运行。配对验证只加载一份 `OptimizedQwen3ForCausalLM`，关闭开关时委托父类 MLP 和原生生成路径，实际类名已记录；没有同时加载两套长 KV。每次先释放上一条路径的缓存，再测量下一条。

| 检查 | TTT checkpoint | baseline checkpoint |
|---|---|---|
| 真实输入 | 七长度 4K–256K，每长度 `niah_single_1`／`niah_multiquery` 各一条，共 14 条 | 同左，共 14 条 |
| 对照模式 | 原始、仅压紧尾段、压紧尾段并分块 MLP | 原始、分块 MLP |
| 输入 IDs／完整生成 IDs／文本／原生评分 | 两组优化各 14/14 一致 | 14/14 一致 |
| 真实样本最终 TTT 状态 | 两组优化各 14/14 逐位一致 | 不启用 TTT |
| 合成边界检查 | 60 次有限性、argmax、cache 长度检查通过 | 40 次检查通过 |
| 跨样本隔离 | 三种模式通过 A→B→A 检查 | 两种模式通过 |

生成上限保持 64，沿用 EOS，本次选定样本均生成 64 tokens。合成前缀为 1022／1024／2046／17406 tokens，各接四个固定 token，覆盖 TTT chunk 边界及真正执行 MLP 分块的长前缀。分词使用 CPU 计时，prefill／decode 使用 CUDA forward 事件，wall 包含分词与生成，不包含事后的 CPU 状态比较；每种模式先完整预热。

**数值限制：**TTT 的仅尾段优化在全部边界检查中状态逐位一致；分块 MLP 在 17406-token 合成前缀的五步检查中状态不逐位一致，logits 最大绝对差为 0.25，状态最大绝对差为 2.671875。后者是 `past_h/past_t/past_w` 合并统计，不能单独解释为 fast weights 的差异。baseline 在同一长合成前缀的五步检查中也存在 logits 差异，最大为 0.25，argmax 一致。尚未逐算子定位原因，不能据此保证任意输入的状态或生成结果一致。

## 显存与耗时

下表为每个长度两条真实样本中的峰值 **PyTorch allocated** 显存，以及两条样本的平均 wall 耗时。不是显卡总占用或刻意占满显存后的测量。

| checkpoint／路径 | 256K 峰值显存 | 256K 平均耗时 | 完整 9,100 条外推 |
|---|---:|---:|---:|
| TTT 原始 | 70.72 GiB | 127.82 秒 | 69.21 小时 |
| TTT 仅压紧尾段 | 63.28 GiB | 127.54 秒 | 69.09 小时 |
| TTT 压紧尾段＋分块 MLP | 56.84 GiB | 127.38 秒 | 69.08 小时 |
| baseline 原始 | 62.87 GiB | 126.91 秒 | 68.50 小时 |
| baseline 分块 MLP | 56.43 GiB | 126.57 秒 | 68.32 小时 |

TTT 的 256K 峰值 allocated 减少 13.88 GiB，约 19.6%；prefill 后尾段保留的底层存储从约 7.50 GiB 降到 0.054 GiB，完整 attention KV 仍约 35.99 GiB。两种 checkpoint 的耗时变化不足以认定明显加速。

| 长度 | TTT 原始／优化（秒） | baseline 原始／优化（秒） |
|---|---:|---:|
| 4K | 2.685／2.720 | 2.568／2.559 |
| 8K | 2.983／2.980 | 2.846／2.836 |
| 16K | 3.639／3.626 | 3.563／3.572 |
| 32K | 5.634／5.667 | 5.548／5.527 |
| 64K | 12.112／12.175 | 11.914／11.866 |
| 128K | 36.777／36.742 | 36.343／36.259 |
| 256K | 127.824／127.383 | 126.910／126.571 |

优化列均包含分块 MLP。外推按每长度 13 项任务×100 条计算，即 \(\widehat T=1300\sum_\ell\bar t_\ell\)。每长度仅测两条，未覆盖全部任务的生成长度分布，外推不能替代完整评估耗时。

TTT 优化路径按上述规模加权，prefill 约占 wall 的 87.2%，decode 约占 11.9%；256K 的 prefill 约占 93.9%。即使消除全部 decode 耗时，当前外推仍约 60.85 小时，因此本轮暂不实施 KV 预分配。

## Attention 后端与产物

4K 的轻量 profiler 确认原 SDPA 已调用 PyTorch FlashAttention：BF16、32 query heads／8 KV heads、无显式 mask、使用原生 GQA；prefill 和 decode 均不是 eager attention。

代码 `5c7a179a`（运行时 clean）的 `scripts/compare_attention.py` 固定尾段压紧与 16,384-token MLP 分块，对 TTT checkpoint 的三条 `niah_single_1` 输入比较 SDPA 与独立 `flash_attention_2` 接口。每个后端分别预热，生成上限与 EOS 不变，两者在三条样本上均生成 64 tokens。

| 长度 | SDPA wall | 独立 FA2 wall | 完整生成 IDs |
|---|---:|---:|---|
| 4K | 2.789 秒 | 3.326 秒 | 一致 |
| 128K | 36.739 秒 | 35.864 秒 | 不一致，第 16 个 token 开始分歧 |
| 256K | 127.224 秒 | 121.015 秒 | 不一致，第 8 个 token 开始分歧 |

三条输入 IDs 相同，但最终 TTT 状态均不逐位一致。虽然三条任务分数相同，128K／256K 的 token 与文本比较失败；该对照按预设判据以失败状态保存，不将独立 FA2 纳入正式优化路径。生产后端保持 SDPA，前述 69／68 小时估算也使用 SDPA。

后续 [128K／256K 分项 profiling 与仅 prefill cuDNN 对照](20261109_in_place_ttt_prefill_profiling.md) 已完成：原 attention 内核占真实 GPU 活动时间约 80.3%／89.4%；cuDNN 端到端更慢，且未通过全部生成 IDs 检查，保持现有后端。

产物位于本地及服务器项目的 `artifacts/reproductions/in_place_ttt/inference-optimization_20261109/`：`ttt-full/`、`baseline-full/` 保存最终计划、来源、逐样本生成、边界与隔离检查及计时汇总；4K attention 诊断保存在 `ttt-preflight/attention_profile.json`；`ttt-attention-backends/` 保留失败的后端对照及逐条差异。

主验证经 Gitee 同步，后端比较经 GitHub 同步，均从服务器项目仓库执行。本地 Ruff、Python AST、shell 语法和 diff 检查通过，上游源码无差异。本轮 GPU 进程已退出。仅保留小型 JSON／日志证据，未保存大张量、完整 profiler trace 或独立代码快照，未创建额外工作树。
