# 20260909_FineWeb 预训练链路与小样本试验记录（14:52:14 UTC+08:00）

创建时间：20260908 20:20:50 UTC+08:00

最后修订时间：20260909 14:52:14 UTC+08:00

## 1. 数据与运行环境

FineWeb `sample-10BT` 已经用户授权并下载完成，aria2 报告下载成功。完整数据下载使用 `hf-mirror.com` 的服务器直连，清空代理环境变量并显式设置直连参数。原始文件位于服务器项目 `/data/bywei/projects/latent_working_memory` 下的 `data/raw/HuggingFaceFW-fineweb/sample-10BT/`。

数据按仓库名 `HuggingFaceFW-fineweb` 和子集名 `sample-10BT` 组织。运行配置指定语料来源、子集、数据 seed 和文档预算；准备记录保留原始文档、来源位置及预处理参数。

本次代码快照与独立环境位于服务器项目的 `artifacts/v1/pretrain-code-20260908/`。环境为 Python 3.11.16、PyTorch 2.14.0+cu130、Transformers 4.57.6、PEFT 0.20.0。基座使用服务器已有的 `/data/bywei/models/meta-llama/Llama-2-7b-chat-hf`，读取采用 BF16，可训练模块参数保持 FP32。数据准备与模型加载使用本地文件和 Hugging Face 离线模式。

## 2. 真实文档准备

从已下载 Parquet 顺序读取 512 篇文档，经来源划分和自然边界切分生成以下数据。该样本规模用于工程与小样本质量验证。

| 划分 | 文档数 | AE/LM 样本对 |
|---|---:|---:|
| train | 464 | 6,972 |
| dev | 29 | 447 |
| test | 19 | 290 |

| 输入粒度 | 样本数 | tokens 范围 | 平均 tokens |
|---|---:|---:|---:|
| 段落 | 1,870 | 3–565 | 52.35 |
| 连续句子组 | 2,043 | 6–1,024 | 181.45 |
| 单句 | 2,044 | 1–221 | 24.96 |
| 相邻段落组合 | 1,752 | 7–730 | 106.68 |

数据保存在服务器项目的 `data/v1/fineweb-pilot-20260908/`。本地准备统计见 [`preparation.json`](../../artifacts/v1/data-preparation/fineweb-pilot-20260908/preparation.json)。

## 3. 工程验收

本地根项目与 v1 测试共 45 项通过；服务器 v1 测试共 39 项通过。真实 GPU smoke 使用物理 GPU 0、batch 1、每步 1 个样本，验证面板包含 4 个独立文档。

第 1 步训练保存 `pretrain-step-000001.pt` 后完成多容量评估；独立进程从该保存点恢复至第 2 步，再次保存并完成评估。累计采样访问由 1 增至 2，损失与梯度均为有限值，运行峰值显存约 13.26 GiB。两个训练步使用不同样本，其损失用于检查数值行为。

服务器 checkpoint 位于 `artifacts/v1/pretrain-smoke-20260908/checkpoints/`；本地日志与评估结果位于 [smoke 结果](../../artifacts/v1/validation/pretrain-smoke-20260908/)。

额外变长 batch 验证将 1-token 与 1,024-token 的真实输入放入同一批次，分别分配 1 与 512 个记忆位置，完成 AE/LM 前向、反向与参数更新。写入投影、记忆更新器、读取投影与读取 LoRA 的梯度均非零且有限，峰值显存约 14.51 GiB。记录见 [`stress-batch.json`](../../artifacts/v1/validation/pretrain-server-validation/stress-batch.json)。

自由生成由 Hugging Face 根据持续扩展的 attention mask 计算位置编号；回归测试验证 KV cache 逐步生成的 logits 与完整前缀读取一致。该检查修正了首次评估中位置编号固定的问题。以下质量结果使用修正后的评估路径，本地保留修正后的固定面板与周期验证结果。修正前生成结果的清理记录见 [产物整理记录](20260909_artifact_organization.md)。

## 4. 固定样本质量试验

固定 16 个来源不同的训练样本，包含 7 个连续句子组、3 个单句、3 个段落和 3 个相邻段落组合，输入长度为 7–283 tokens。沿用 2/4/8 压缩率课程，每次访问重新采样容量；batch 为 2，梯度累积为 4，学习率为 1e-4，完成 100 个 optimizer steps，每 50 步保存和评估。

训练使用物理 GPU 1，固定面板评估分配至物理 GPU 0、1。训练前后评估同一组 16 个训练样本，独立 dev 面板包含 16 个文档。每个样本覆盖所有合法容量，并比较正确记忆、空记忆和来自其他来源的同容量错误记忆；AE 自由生成检查使用每个面板的第一个样本。

运行目录为服务器项目的 `artifacts/v1/pretrain-overfit-20260908/`。完整准备、训练恢复与小样本运行命令保存在代码快照的 `prepare_and_smoke.sh` 和 `run_overfit.sh`。

累计采样 800 次文档访问，即每个固定样本访问 50 次，累计写入 60,150 tokens、监督 103,350 个目标 tokens（含 EOS 和重复访问）。训练步合计耗时 285.72 秒，中位数为 2.81 秒/步，训练进程峰值显存约 13.62 GiB。

下表使用原文目标 token 加权 NLL，EOS 单独统计。同一面板在所有合法容量下重新写入后汇总，训练面板有 48 个容量条件，dev 面板有 47 个容量条件。

| 面板与 checkpoint | AE 正确记忆 | AE 空记忆 | AE 错误记忆 | LM 正确记忆 | LM 空记忆 | LM 错误记忆 |
|---|---:|---:|---:|---:|---:|---:|
| 固定训练样本，初始化 | 3.1738 | 2.9832 | 3.1791 | 3.0166 | 2.8967 | 3.0139 |
| 固定训练样本，step 100 | 0.1613 | 2.2523 | 0.4169 | 0.0278 | 1.7418 | 0.3606 |
| 独立 dev，初始化 | 3.3502 | 3.2001 | 3.3356 | 2.9987 | 2.9495 | 2.9978 |
| 独立 dev，step 50 | 4.3465 | 3.1109 | 4.3607 | 3.9567 | 2.8517 | 3.9199 |
| 独立 dev，step 100 | 5.3355 | 3.4710 | 5.3831 | 4.7354 | 3.0729 | 4.7492 |

训练样本上的 NLL 大幅下降，正确记忆优于错误记忆和空记忆，说明完整读写路径能获得有效训练信号。错误记忆仍取得较低的训练 NLL，表明小样本记忆、读取 LoRA 的拟合及教师强制提供的目标前缀也是结果的解释因素。

自由生成检查选取固定训练面板中的一个 66-token 片段。初始化时三个容量均未完整重建；step 100 在 $K=9,17,33$ 下均逐 token 完整重建原文并输出 EOS，三次检查全部通过。独立 dev 面板的生成样本在三个容量下均未完整重建。该生成检查覆盖每个面板各一个文本，后续扩大面板评估整体成功率。

修正前后的 dev NLL 完全一致；修正改变了自由生成时的缓存位置推进。完整结果见本地 [小样本试验结果](../../artifacts/v1/experiments/pretrain-overfit-20260908/)，训练前后面板分别位于 `fixed-panel-step-000000/` 与 `fixed-panel-step-000100/`，独立 dev 的 step 50、100 汇总位于运行目录根部。

独立 dev 上的正确记忆 NLL 随训练上升，并劣于空记忆；当前 checkpoint 呈现明显过拟合，泛化验收尚未通过。该 checkpoint 定位为工程验证产物。

## 5. 下一轮预训练

1. 从已下载的 `sample-10BT` 扩大独立文档覆盖，先检查正文、标题和导航短行的比例，再形成正式 pilot 数据。当前自然边界切分保留网页短行，产生了 1-token 片段；完整句子训练需要进一步筛选这些片段。
2. 使用更大的文档池持续混合各粒度和容量，记录实际输入长度与容量覆盖，依据独立 dev 的正确记忆收益和自由重建结果选择 checkpoint。
3. 扩大自由生成面板，分别检查完整重建、局部错误和压缩率变化。容量上限的数值验收与不同容量下的实际保真度分别报告。

后续动态训练继承达到独立文档质量要求的预训练 checkpoint。

## 6. 数据目录与 SwanLab 接入

服务器数据目录统一为 `data/raw/HuggingFaceFW-fineweb/sample-10BT/`。配置、预处理契约与已有 checkpoint 已完成配套迁移，模型参数、优化器和随机状态保持一致。本地 11 份迁移前元数据备份按已记录的迁移规则转换后，与当前文件内容一致；这些备份已移入本机废纸篓。迁移脚本及验收结果保存在 `artifacts/v1/legacy/fineweb-layout-swanlab-20260908/`。

数据准备直接用 PyArrow 分批读取本地 Parquet，并在达到文档预算时关闭文件。新目录上的 8 篇真实文档准备正常完成，进程正常退出。

SwanLab 0.10.0 使用服务器已有登录，将实验记录到私有项目 `latent-working-memory`。已有试验见 [SwanLab 看板](https://swanlab.cn/@percyWeeeeei/latent-working-memory/runs/hzg2z87k/chart)。本次导入前 50 步后结束记录，再以同一实验 ID 续接至第 100 步，云端包含完整训练曲线和 0、50、100 步 dev 指标。历史数据以 optimizer step 对齐，时间轴对应本次导入时间。

看板记录训练 AE/LM NLL、梯度范数、吞吐、显存、输入长度、记忆容量和文档覆盖；评估包含正确／空／错误记忆对照、长度与粒度及压缩率分层 NLL、重建原文与预测。`gain_vs_no_memory` 和 `gain_vs_wrong_memory` 表示对照 NLL 减去正确记忆 NLL，正值表示正确记忆更好。

在线记录通过 `--swanlab-mode online` 开启，离线记录使用 `offline`，默认 `disabled`。训练在同一输出目录恢复时读取 `swanlab.json` 并续接原实验；独立评估可用 `--swanlab-run-id` 指定该实验。开启记录的 tiny Llama 训练及恢复与连续运行结果一致。验收结果见 [`verification.json`](../../artifacts/v1/legacy/fineweb-layout-swanlab-20260908/verification.json)。
