# 20261124_AutoCompressors 源码来源与接入边界

创建时间：20261124 11:54:28 Asia/Shanghai

最后修订时间：20261124 11:54:28 Asia/Shanghai

| 组件 | 来源与固定版本 |
|---|---|
| 官方仓库 | [princeton-nlp/AutoCompressors](https://github.com/princeton-nlp/AutoCompressors/tree/80352a4233c4a70504c75bf95c5f3e6d2533a863)，提交记录在 [source/revision](source/revision) |
| 权重与 tokenizer | [AutoCompressor-Llama-2-7b-6k](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k/tree/e32c98b484e24e66cead4218a7fee1e0e5dd616f)，作者发布的完整权重 |
| 论文 | [Adapting Language Models to Compress Contexts](https://aclanthology.org/2023.emnlp-main.232/)，EMNLP 2023 |

[scripts/setup.sh](scripts/setup.sh) 从官方仓库检出固定提交。该提交没有仓库级 LICENSE，`modeling_flash_llama.py` 包含原作者的 Apache-2.0 声明。模型卡另有权重许可标记。

上游的 `auto_compressor.py` 提供摘要嵌入、分段前向、摘要累积和生成接口，`modeling_flash_llama.py` 提供官方 Llama 注意力和 KV cache。LoCoMo 适配位于 [evaluation/eval_locomo.py](evaluation/eval_locomo.py)，负责完整对话写入、只读问答前缀和长度预算。数据、评分与落盘使用共用的 [LoCoMo 评估入口](../soft_context_locomo.py)。论文训练及官方困惑度评估脚本保存在检出的 `source/AutoCompressors/run/`。
