# 20260912_SQuAD 动态记忆三设置训练（16:18:42 UTC+08:00）

创建时间：20260912 16:10:22 UTC+08:00  
最后修订时间：20260912 16:18:42 UTC+08:00

## 1. 数据与初始化

- SQuAD 完整文章长度闭区间为 4096–8192 tokens，记忆容量 K=1024，最终源 token 数/记忆位置数为 4–8。
- 内部 train/dev 分别有 167/22 篇，官方 dev 派生的 test 有 27 篇满足范围。
- 初始化为 mixed 预训练第 20000 步：`artifacts/v1/pretrain-data-comparison-2048_20260911/train/pretrain-mixed-157k-20260911/checkpoints/pretrain-step-020000.pt`。
- checkpoint 的 `k_limit=4096`、写入及读取窗口为 4096，支持当前容量。

## 2. 正式运行

按下列顺序运行，三个设置均从同一预训练 checkpoint 开始：

| run name | 梯度传播 |
|---|---|
| dynamic-k1024-full-167_mixed-157k_20260912 | 完整 BPTT |
| dynamic-k1024-tokens1024-167_mixed-157k_20260912 | 每累计 1024 个源 tokens 截断 |
| dynamic-k1024-updates4-167_mixed-157k_20260912 | 每累计 4 次 memory 更新截断 |

- 仅使用物理 GPU 6、7，双卡文章并行，每卡 microbatch=1，有效文章 batch=2。
- 每轮打乱后遍历全部文章；严格训练 3 个 epoch，共 501 篇次、251 次参数更新。末步仅处理 1 篇，并按实际篇数归一化。
- 学习率 3e-5、weight decay 0.01、梯度裁剪 1.0、seed=42。
- 每次写入最多 1 个当前问题、1 个历史问题，每题最多访问 2 次。
- 固定 8 篇内部 dev 文章，执行五条件评估，生成上限 64 tokens。step 0、84、168、251 评估；84、168、251 保存 checkpoint。
- SwanLab project 为 `latent-working-memory-v1`，group 为 `dynamic-squad-k1024-4k8k_20260912`，job_type 为 `train`。
- 产物位于 `artifacts/v1/dynamic-squad-k1024-4k8k_20260912/`，训练放在 `train/`，命令、测试记录和调度状态放在 `plan/`。
- 顺序调度遇到失败立即停止，不启动后续设置。

## 3. 性能验证

测试运行仅写本地日志，不创建 SwanLab run。分别选择段落数最多、最长单段最大的两篇训练文章，验证多次更新与单次写入的开销。

完整 BPTT 单用语言模型层级检查点在最重文章上显存不足；加入完整 QA 读取重算后完成测试，保留整条记忆梯度链。最终性能结果以 `plan/profile-*.jsonl` 为准。


同一组最多更新次数的两篇文章共 150 次写入、298 次 QA，实测如下。显存为两卡中较高的 PyTorch 已分配峰值，不含模型加载耗时。

| 设置 | 完整 QA 读取重算 | 单步耗时 | 峰值显存 |
|---|---|---:|---:|
| full | 开启 | 58.4 秒 | 26.5 GiB |
| tokens1024 | 开启 | 57.1 秒 | 19.2 GiB |
| updates4 | 关闭 | 41.2 秒 | 51.3 GiB |

最长单段压力样本下，updates4 峰值为 51.5 GiB。它开启重算时最重样本耗时 56.3 秒，因此正式运行关闭重算；tokens1024 关闭重算时显存不足。三个正式配置均关闭语言模型层级重算，使用 BF16 和可扩展 CUDA 分配段。
