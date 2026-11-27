# 20260904_ICAE 上游来源与适配

创建时间：20260904 10:13:57 Asia/Shanghai

最后修订时间：20261127 10:24:41 Asia/Shanghai

## ICAE v2

源码固定为 [getao/icae 的 469a468](https://github.com/getao/icae/tree/469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd)。[source/icae](source/icae) 保留上游 `code/icae_v2` 下的全部 Python 与 shell 源码、README 与 CC0-1.0 许可证。项目从该目录导入 `ICAE` 模型，训练与评估共用它的分段规则和特殊 token 嵌入。

Writer 和训练用 Reader 通过 Transformers 的 `attn_implementation="flash_attention_2"` 加载基座。训练与评估入口均设置 `mean_resizing=False`，扩展词表时跳过协方差计算。评估时从 checkpoint 加载记忆槽和指令标记的嵌入，只在基座原词表中生成回答。

| 资源 | 官方发布 |
|---|---|
| 基座和 tokenizer | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/63a8b081895390a26e140280378bc85ec8bce07a)，Apache-2.0 |
| Writer LoRA 和记忆槽 | [sggetao/icae](https://huggingface.co/sggetao/icae/tree/c95c70d670c8a284550610b49b0f26d2a3b2d1e3)，模型卡标注 Apache-2.0 |

项目的预训练与指令微调入口沿用官方 v2 的训练目标，均从命令行接收基座、训练集和可选的验证集路径。仅在提供验证集时于训练结束后计算验证 loss。指令微调从预训练权重初始化，按基座 tokenizer 的聊天模板拼接记忆、`[FT]` 和问题，对答案及 EOS 计算损失。每张 GPU 每个 microbatch 处理一篇正文，默认按训练进程数设置梯度累积，使全局有效批量为 256。[数据与训练实现](source/icae/code/icae_v2/training_utils.py) · [训练示例](README.md#二原论文复现)

优化超参数参考[论文附录 A](https://arxiv.org/html/2307.06945v4#A1) 的 Llama 系列实验：预训练默认学习率 `1e-4`、更新 `200000` 步，指令微调默认学习率 `5e-5`、更新 `30000` 步，均使用 AdamW、BF16、300 步 warmup 和 `2.0` 的梯度裁剪范数。这些是项目采用的论文参考配置，并非作者完整披露的 Mistral v2 训练配置。

两个阶段均冻结基座和 Reader，更新 Writer LoRA 和记忆及控制标记的嵌入。Trainer 将这些可训练参数写入 `model.safetensors`，与公开 ICAE v2 权重使用相同的参数名。微调与评估共用权重加载逻辑，训练中间的 `checkpoint-<step>/` 同时保留 Trainer 的续训状态。[模型与 checkpoint 加载](source/icae/code/icae_v2/modeling_icae_multi_span.py)

[模型适配](evaluation/model.py)和[会话适配](evaluation/locomo_adapter.py)负责 LoCoMo 问答、完整对话的压缩与 reader 缓存复用。它沿用官方 LoRA rank 512，通过 `--fixed-mem-size` 和 `--mean-compression-rate` 设置每块记忆槽数与平均压缩率，默认分别为 128 和 4，分段仍由官方实现计算。模型与特殊 token ID 使用同一槽数配置，checkpoint 按实际权重键和形状加载。原预训练示例用 `[AE]` 展示文本重建，项目以冻结指令基座直接读取预训练的槽回答问题，报告零样本 QA。微调权重沿用官方 v2 推理的 `[FT]`。与上游 5,120-token 演示的另一处差异是编码完整历史，并对压缩后的 reader 输入执行位置窗口检查。

Reader 按基座 tokenizer 的聊天模板拼接记忆、问题和回答起始标记，停止标记取自同一 tokenizer。Qwen3-4B-Instruct-2507 使用公共 Reader 解码。官方 Mistral-7B-Instruct-v0.2 使用专用 Reader，同一正文下的多个问题共享只读的前缀 KV 缓存。[Reader 输入适配](formatting.py) · [公共 Reader](../../benchmarks/locomo/reader.py) · [Mistral Reader](evaluation/mistral_reader.py)

[独立看板](evaluation/dashboard.py)复用[公共 LoCoMo 组件](../../benchmarks/locomo/dashboard.py)展示 F1 与逐题回答，阶段和记忆用量由本方法维护，由 [dashboard.sh](scripts/dashboard.sh) 启动。
