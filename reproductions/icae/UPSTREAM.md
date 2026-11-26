# 20260904_ICAE 上游来源与适配

创建时间：20260904 10:13:57 Asia/Shanghai

最后修订时间：20261126 20:00:15 Asia/Shanghai

## ICAE v2

源码固定为 [getao/icae 的 469a468](https://github.com/getao/icae/tree/469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd)。[source/icae](source/icae) 保留上游 `code/icae_v2` 下的全部 Python 与 shell 源码、README 与 CC0-1.0 许可证。项目从该目录导入 `ICAE` 模型，训练与评估共用它的分段规则和特殊 token 嵌入。

本地构造器允许传入 Transformers 的 `mean_resizing` 选项。项目的评估入口设为 `False`，扩展词表时跳过 CPU 上的协方差计算。记忆槽和指令标记的嵌入从 checkpoint 加载，生成时只使用基座原词表。

| 资源 | 官方发布 |
|---|---|
| 基座和 tokenizer | [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/63a8b081895390a26e140280378bc85ec8bce07a)，Apache-2.0 |
| Writer LoRA 和记忆槽 | [sggetao/icae](https://huggingface.co/sggetao/icae/tree/c95c70d670c8a284550610b49b0f26d2a3b2d1e3)，模型卡标注 Apache-2.0 |

[模型适配](evaluation/model.py)和[会话适配](evaluation/locomo_adapter.py)负责 LoCoMo 问答、完整对话的压缩与 reader 缓存复用。它沿用官方 LoRA rank 512，通过 `--fixed-mem-size` 和 `--mean-compression-rate` 设置每块记忆槽数与平均压缩率，默认分别为 128 和 4，分段仍由官方实现计算。模型与特殊 token ID 使用同一槽数配置，checkpoint 按实际权重键和形状加载。原预训练示例用 `[AE]` 展示文本重建，项目以冻结指令基座直接读取预训练的槽回答问题，报告零样本 QA。微调权重沿用官方 v2 推理的 `[FT]`。与上游 5,120-token 演示的另一处差异是编码完整历史，并对压缩后的 reader 输入执行位置窗口检查。

[Reader 输入适配](formatting.py)通过[公共输入构造](../../src/latent_working_memory/prompts.py)取得聊天模板前后缀，维护分段分词与 `[FT]` 插入位置；正文起始标记与生成停止标记取自 tokenizer。Mistral 输入的 token ID 与原适配一致，Qwen3-Instruct 已完成模板检查；Qwen3 模型训练与完整评估尚未适配。

[独立看板](evaluation/dashboard.py)复用[公共 LoCoMo 组件](../../benchmarks/locomo/dashboard.py)展示 F1 与逐题回答，阶段和记忆用量由本方法维护，由 [dashboard.sh](scripts/dashboard.sh) 启动。
