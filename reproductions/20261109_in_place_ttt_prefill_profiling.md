# 20261109_In-Place TTT 长输入 prefill 分项计时与 cuDNN 对照

创建时间：20261109 23:24:48 Asia/Shanghai

最后修订时间：20261109 23:31:40 Asia/Shanghai

## 目标与固定条件

测清 128K／256K prefill 的主要耗时，并检验仅替换 prefill attention 内核能否降低端到端生成时间。使用 `zcy-memory` 单张 A800 的物理 GPU 0，显式设置 `CUDA_VISIBLE_DEVICES=0`；所有 GPU 项目顺序执行。

本轮使用 `qwen3-4b-32k-ttt` checkpoint。两组均固定现有尾段缓存压紧和 `mlp_chunk_size=16384`，保持模型权重、TTT 更新目标与顺序、卷积边界、RoPE、完整 KV、输入和评分流程不变。

| 组别 | prefill | decode |
|---|---|---|
| SDPA 对照 | 当前 SDPA 默认后端 | 当前 SDPA 默认后端 |
| cuDNN 候选 | 仅完整 prefill（`q_len == kv_len > 1`、无显式 mask）强制 `CUDNN_ATTENTION` | 当前 SDPA 默认后端 |

cuDNN 不支持实际输入或出现 OOM 时保存失败，不自动回退后仍将其标记为 cuDNN。当前正式评估默认不变。

## 执行与判据

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

## 执行结果

待执行。后续依据端到端耗时、分项瓶颈和完整生成一致性决定优化方向。
