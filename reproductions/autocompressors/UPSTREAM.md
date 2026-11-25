# 20261124_AutoCompressors 源码来源与接入边界

创建时间：20261124 11:54:28 Asia/Shanghai

最后修订时间：20261125 11:45:30 Asia/Shanghai

| 组件 | 来源与固定版本 |
|---|---|
| 官方仓库 | [princeton-nlp/AutoCompressors](https://github.com/princeton-nlp/AutoCompressors/tree/80352a4233c4a70504c75bf95c5f3e6d2533a863)，提交记录在 [source/revision](source/revision) |
| 权重与 tokenizer | [AutoCompressor-Llama-2-7b-6k](https://huggingface.co/princeton-nlp/AutoCompressor-Llama-2-7b-6k/tree/e32c98b484e24e66cead4218a7fee1e0e5dd616f)，作者发布的完整权重 |
| 论文 | [Adapting Language Models to Compress Contexts](https://aclanthology.org/2023.emnlp-main.232/)，EMNLP 2023 |

[source/AutoCompressors](source/AutoCompressors/) 随本项目保存上述固定提交的全部受版本控制文件，官方源码保持原样。[source/revision](source/revision) 记录其来源 commit，评估配置据此保存 `upstream_revision`。[公共环境安装器](../setup_eval_env.sh) 负责创建独立环境和安装依赖。

该提交没有仓库级 LICENSE；[modeling_flash_llama.py](source/AutoCompressors/modeling_flash_llama.py) 中的原始版权与 Apache-2.0 声明随源码保留，模型卡另有权重许可标记。

上游的 `auto_compressor.py` 提供摘要嵌入、分段前向、摘要累积和生成接口，`modeling_flash_llama.py` 提供官方 Llama 注意力和 KV cache。论文训练及官方困惑度评估脚本保存在 `source/AutoCompressors/run/`。

[模型加载](evaluation/model.py)负责 checkpoint 与 tokenizer；[LoCoMo 适配](evaluation/locomo_adapter.py)负责完整对话压缩、输入配置、记忆插入、长度预算和批量回答，模板包装复用[公共输入构造](../../src/latent_working_memory/prompts.py)。官方缓存会原地更新，适配为各题分配独立存储并复制只读前缀。[评估入口](evaluation/eval_locomo.py)调用[公共 LoCoMo 框架](../../benchmarks/locomo/README.md)，复用会话分片、逐题续评、F1 计算和结果保存；记忆用量由方法回调汇总。

[独立看板](evaluation/dashboard.py)读取本方法的评估产物，复用[公共 LoCoMo 组件](../../benchmarks/locomo/dashboard.py)展示 F1 与逐题回答，并补充摘要记忆用量。启动入口为 [dashboard.sh](scripts/dashboard.sh)。
