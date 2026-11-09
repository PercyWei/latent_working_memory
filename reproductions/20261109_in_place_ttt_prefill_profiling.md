# 20261109_In-Place TTT 长输入 prefill 分项计时与 cuDNN 对照

创建时间：20261109 23:24:48 Asia/Shanghai

最后修订时间：20261110 00:46:58 Asia/Shanghai

## 目标与固定条件

测清 128K／256K prefill 的主要耗时，并检验仅替换 prefill attention 内核能否降低端到端生成时间。使用 `zcy-memory` 单张 A800 的物理 GPU 0，显式设置 `CUDA_VISIBLE_DEVICES=0`；所有 GPU 项目顺序执行。

本轮使用 `qwen3-4b-32k-ttt` checkpoint。两组均固定现有尾段缓存压紧和 `mlp_chunk_size=16384`，保持模型权重、TTT 更新目标与顺序、卷积边界、RoPE、完整 KV、输入和评分流程不变。

| 组别 | prefill | decode |
|---|---|---|
| SDPA 对照 | 当前 SDPA 默认后端 | 当前 SDPA 默认后端 |
| cuDNN 候选 | 仅完整 prefill（`q_len == kv_len > 1`、无显式 mask）强制 `CUDNN_ATTENTION` | 当前 SDPA 默认后端 |

cuDNN 不支持实际输入或出现 OOM 时保存失败，不自动回退后仍将其标记为 cuDNN。当前正式评估默认不变。

## 执行协议与判据

1. 每组先用同一条 4K 输入预热，生成上限保持 64，并保留 EOS；核对完整生成 IDs。长输入首次执行计划构造计入端到端计时。
2. 先进行不启用 profiler 的配对生成：128K／256K 的 `niah_single_1`、`niah_multiquery` 各取第 0 条，共四对输入，平衡组别执行顺序。记录 CPU 分词、wall、prefill／decode、峰值 allocated／reserved 显存及实际后端调用信息。
3. 比较完整输入 IDs、完整生成 IDs、预测文本、原生分数、有限性和 cache 进度。记录最终 TTT 状态差异；生成历史已经分歧时，不将最终状态差异归因于同输入下的纯后端误差。
4. 完成端到端对照后，分别对每个长度、每个后端的 `niah_single_1` 做一次独立 prefill 与单步 decode 分项 profiling。单独比较同输入 prefill 的末位 logits、argmax 和 TTT 状态，检查 profiling 与未插桩路径的生成前缀一致性。
5. 单项失败仍保留记录并继续其他计划项。流程完成、生成一致性和候选可采用性分别判断；任务分数相同不能替代完整生成 IDs 一致。

分项同时记录 CUDA event span 和 profiler 的 CPU／CUDA 算子时间。顶层区分 attention、普通 MLP、TTT MLP、norm、RoPE 及其他开销；SDPA 是 attention 的子项，TTT read／write 是 TTT MLP 的子项，不能重复相加。CUDA event span 可能包含 CPU launch 间隙，不等同于纯 kernel busy 时间。活动期 SM clock／功率可通过外部采样辅助判断。

profiling 单独执行，其耗时不作为端到端提速证据。不保存完整 trace、完整 KV 或大张量；CPU 状态快照仅用于内存中的比较。

## 来源与产物

代码优先经 GitHub 同步，无法连接时使用 Gitee，从服务器项目仓库运行。执行前保存 plan、Git commit、dirty/status、入口、实际优化开关、PyTorch／cuDNN 版本及运行配置；每次执行先记录状态，退出时补全结果与失败原因。

产物统一保存到本地及服务器项目的 `artifacts/reproductions/in_place_ttt/prefill-profiling_20261109/ttt/`。保留逐样本生成、后端签名与调用次数、分项时间、资源和一致性结果；不计入正式 9,100 条 benchmark。

在服务器项目根目录、已激活 In-Place TTT 独立环境后执行：

```bash
method_dir="$PWD/reproductions/in_place_ttt"
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH="$method_dir/upstream:$method_dir/scripts:$method_dir${PYTHONPATH:+:$PYTHONPATH}" \
python -u reproductions/in_place_ttt/scripts/profile_prefill.py \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --output-dir artifacts/reproductions/in_place_ttt/prefill-profiling_20261109/ttt \
  --lengths 131072 262144
```

本轮另以每秒一次的 `nvidia-smi` 记录 GPU 0 的时钟、功率、显存与利用率，保存到上述系列目录的 `gpu-hardware.csv`。诊断进程退出后结束采样。

首轮 profiler 汇总混入 CUDA user annotation，并发生 CPU／CUDA 同名 component 覆盖；该轮分项汇总不用于瓶颈占比。修正后通过下面的命令仅补测 profiling，引用原执行的完整生成与端到端计时，并在新计划中保存其来源，不重复生成：

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
PYTHONPATH="$method_dir/upstream:$method_dir/scripts:$method_dir${PYTHONPATH:+:$PYTHONPATH}" \
python -u reproductions/in_place_ttt/scripts/profile_prefill.py \
  --checkpoint ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --generation-results artifacts/reproductions/in_place_ttt/prefill-profiling_20261109/ttt \
  --output-dir artifacts/reproductions/in_place_ttt/prefill-profiling_20261109/ttt-profile-corrected \
  --lengths 131072 262144
```

修正后的 GPU 总量仅累计非 annotation CUDA 活动。补测发现 CPU 关联的 device 时间仍混有加载等 overhead／fallback 跨度，不能可靠归因；采集中的 `component_kernel_inclusive` 不用于结论。分项 latency 使用独立 CUDA-event span，attention kernel 使用真实 CUDA 叶子记录。当前脚本仅保留 CPU 的 reported device 时间供审计，不再将其标为纯 kernel 时间。

## 执行结果

四对端到端生成已完成，使用代码 `6ab044b7`，执行时 Git 状态为 clean。以下均为未启用 profiler 的实测，每个长度取两条样本的平均值；生成上限保持 64，八次生成均实际完成 64 tokens。

| 长度 | SDPA wall | cuDNN wall | wall 变化 | SDPA prefill | cuDNN prefill |
|---|---:|---:|---:|---:|---:|
| 128K | 36.424 秒 | 38.860 秒 | +6.69% | 32.127 秒 | 34.550 秒 |
| 256K | 127.217 秒 | 137.155 秒 | +7.81% | 119.443 秒 | 129.406 秒 |

128K 的 SDPA／cuDNN 平均 decode 为 3.889／3.883 秒，256K 为 6.864／6.846 秒。两组峰值 allocated 显存相同，128K 约 32.37 GiB，256K 约 56.84 GiB；切换后端没有降低该指标。

- 4K 预热的完整生成 IDs 一致。四对长输入的输入 IDs 和原生分数均一致，有限性、cache 进度和 prefill／decode 后端路由检查符合协议。
- `niah_multiquery` 的 128K／256K 两对完整生成 IDs 与文本一致；`niah_single_1` 分别从第 16／第 8 个生成 token 开始分歧。任务分数相同不能替代生成一致性。
- 四对最终 TTT 状态均不逐位一致；生成历史已经分歧的两对，终态差异不能解释为同输入下的纯后端误差。

**cuDNN prefill 候选同时未达到生成一致性和提速要求，不采用，生产默认保持不变。**四条 SDPA 对照的输入 token 数、完整生成 IDs、文本和评分均与前一轮相同输入的 chunked 结果一致。

## 分项结果与判断

补测使用 `14796403`，执行时 Git 状态为 clean；仅采集 profiling，引用 `6ab044b7` 的端到端记录。环境为 PyTorch 2.8.0+cu128、Transformers 4.57.3、cuDNN 9.10.2。四次 profile 的实际 prefill 算子、decode Flash 算子、前两生成 IDs、有限性及 cache 进度检查均通过。

以下为非 annotation CUDA 活动及实际 attention kernel 时间，分母不包含 CUDA 用户标注区间：

| 长度 | SDPA GPU 活动合计 | Flash attention kernel | 占合计 | cuDNN attention kernel |
|---|---:|---:|---:|---:|
| 128K | 32.112 秒 | 25.784 秒 | 80.29% | 27.688 秒 |
| 256K | 119.890 秒 | 107.146 秒 | 89.37% | 116.404 秒 |

cuDNN 的 GPU 活动合计分别为 34.120／129.232 秒，实际 attention kernel 比 Flash 慢约 7.39%／8.64%；本次端到端变慢也体现在内核执行上，并非只有首次计划构建成本。

下面为原 SDPA 的 CUDA-event 分项跨度。它们包含 launch gap 和插桩开销，不能当作纯 kernel busy 时间，也不能与上表直接相加：

| 分项 | 128K | 256K |
|---|---:|---:|
| attention 模块整体 | 28.163 秒 | 111.594 秒 |
| 其中：SDPA 调用 | 25.784 秒 | 107.145 秒 |
| 其中：attention 其余计算 | 2.379 秒 | 4.449 秒 |
| 普通 MLP | 2.468 秒 | 5.015 秒 |
| TTT MLP 整体 | 0.970 秒 | 1.956 秒 |
| 其中：TTT read contract | 0.169 秒 | 0.345 秒 |
| 其中：TTT write contract | 0.221 秒 | 0.450 秒 |
| attention 外部 norm | 0.710 秒 | 1.278 秒 |

带“其中”的行是子项，不能与父项重复求和。attention 其余计算包含投影、规范化、RoPE、cache 等，不把跨度差全部解释为 QKV GEMM。GPU 内核名称中的普通 BF16 GEMM 同时服务多个模块，也不单独据此归入 MLP。

同输入 prefill 比较中，128K／256K 末位 logits 最大绝对差分别为 0.125／1.3125，argmax 相同；TTT `h/t/W` 合并最大绝对差分别为 7.5／64，均有限但不逐位一致。这些来自相同输入历史；它们与生成分歧后的最终状态差异分别记录。

活动期 GPU SM 时钟中位数约 1245 MHz，最高 1410 MHz，最高温度 67°C；功率限制、默认值和可设上限均为 400 W。本轮没有调整硬件设置。GPU 利用率采样不代表 Tensor Core 利用率。

主要 GPU 成本已定位到长输入 attention。普通 MLP 与 TTT 写入都不是获得数量级提速的主线；后续优先验证 `qa_1` 等任务的完全相同 token 前缀复用，同时保存对应 KV 和 TTT 状态。本轮未实施前缀复用，也未达到半天评估目标。

完整生成和端到端记录位于 `ttt/`，可信分项位于 `ttt-profile-corrected/`。`55ba95a8` 的补测在 JSON 协议核对阶段退出，未生成新测量，保留于 `ttt-profile-input-contract-failure/`；改用持久化 JSON 表示核对后完成补测。首轮有问题的聚合字段保留作审计，不用于上述结论。

全部 GPU 实验与监控已结束，未启动正式 9,100 条评估。Ruff、Python AST 与 diff 检查通过；上游源码、现有优化计算和正式评估入口均无改动。
