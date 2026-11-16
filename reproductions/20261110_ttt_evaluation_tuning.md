# 20261110_In-Place TTT 评估运行参数

创建时间：20261110 11:51:20 Asia/Shanghai

最后修订时间：20261116 11:15:42 Asia/Shanghai

在 `zcy-memory` 的单张 A800-SXM4-80GB、物理 GPU 0 上测量，显式设置 `CUDA_VISIBLE_DEVICES=0`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。下列均为单卡小样本外推，不是完整 benchmark 实测耗时；运行命令见 [README](in_place_ttt/README.md#ruler-评估)。

| 模型 | 每进程 batch | 同卡进程数 | 所测整卡峰值 | 完整评估外推 |
|---|---:|---:|---:|---:|
| In-Place TTT 32K TTT | 1 | 1 | 67.8 GiB | 68.92 小时 |
| In-Place TTT 32K baseline | 1 | 1 | 69.0 GiB | 68.62 小时 |

峰值来自 `nvidia-smi`，包含载入和预热。TTT 使用 `optimized --mlp-chunk-size 0`，仅压紧尾部缓存；baseline 使用 `original`。128K 双进程的整池吞吐反而下降约 10.0%／9.2%，因此使用单进程。

每模型七个长度各 1,300 条，共 9,100 条，逐长度使用真实单进程均值外推。TTT／baseline 建议预留约 59–80／58–79 小时；256K 本身约占 45.9／45.7 小时。范围是规划余量，不是统计置信区间，假定 GPU 独占且环境与生成设置不变。

原始 In-Place 测量及选参保存在 `artifacts/reproductions/in_place_ttt/evaluation-tuning_20261110/`。LaCT 与 E²-TTT 的当前设置直接见各自 README，混长实现未重新测量吞吐或 batch 上限。
