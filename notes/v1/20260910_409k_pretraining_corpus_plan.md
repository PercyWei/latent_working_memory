# 20260910_40.88 万条预训练数据构造计划（11:24:22 UTC+08:00）

创建时间：20260910 11:20:20 UTC+08:00  
最后修订时间：20260910 11:24:22 UTC+08:00

## 规模依据

本轮构造 408,800 条通过 Qwen 判定的样本，定位为第一轮规模验证。正式预训练的充分性由分长度、分压缩率的 AE 重建指标和 LM 相对完整上下文的损失差判断。

| 研究 | 公开的预训练规模 | 计量口径 |
| --- | --- | --- |
| ICAE（ICLR 2024） | Pile，200,000 updates，batch size 256，默认文本上限 512 tokens | 按表中 batch size 换算为 5,120 万次样本呈现；独立文本数量与重复次数需另行区分 |
| AutoCompressors（EMNLP 2023，官方发布模型） | OPT 模型使用 2B tokens；Llama-2-7B 使用 15B RedPajama tokens | 累计训练 tokens，且训练目标与参数范围与本框架有差异 |
| 500xCompressor（ACL 2025） | ArxivCorpus 训练池 2,353,924 条；500→16 和 500→1 分别训练 42,000、103,800 steps，表中 batch size 为 4 | 原始语料池大小、优化步数和 batch size 分别报告；原文文本长度上限为 500 tokens |

来源：[ICAE 正文与附录 A](https://proceedings.iclr.cc/paper_files/paper/2024/file/0b276510ec2d3f6613a8b60c41ff0438-Paper-Conference.pdf)、[AutoCompressors 官方模型表](https://github.com/princeton-nlp/AutoCompressors#pre-trained-models)、[500xCompressor 附录表 7、8](https://aclanthology.org/2025.acl-long.1219.pdf)。

文献采用的规模跨度较大。10 万条左右的单类任务数据适合本轮验证学习曲线和长度泛化，后续可按新增来源逐步扩充到百万级。扩充的依据是验证指标随累计训练 tokens 的改善情况及过拟合迹象。压缩率 2、4、8 在训练时采样，同一文本使用多个压缩率的重复呈现计入训练量。

## 本轮配额

| 数据组合 | train | dev | test | 合计 |
| --- | ---: | ---: | ---: | ---: |
| semantic AE | 98,000 | 2,100 | 2,100 | 102,200 |
| semantic LM | 98,000 | 2,100 | 2,100 | 102,200 |
| random AE | 98,000 | 2,100 | 2,100 | 102,200 |
| random LM | 98,000 | 2,100 | 2,100 | 102,200 |
| 总计 | 392,000 | 8,400 | 8,400 | 408,800 |

每种组合、每个 split 按压缩输入 X 的长度分为 32–64、65–128、129–256、257–512、513–1024、1025–2048、2049–4096 七个区间，分别保留 14,000 / 300 / 300 条 train / dev / test 样本。粒度作为统计属性。AE 和 LM 独立采样，semantic 与 random 共享来源去重聚类及 split。

AE 的 X 长度为 32–4096；LM 的 X、Y 各为 32–4096，|X|/(|X|+|Y|) 为 0.3–0.7。配额统计通过模型筛选后实际保留的样本。

semantic 训练集共 196,000 条。由区间边界可得其 X 总量范围约 1.14 亿至 2.28 亿 tokens；采用各区间中点作粗估约 1.71 亿。两种版本合计约 3.42 亿 X tokens（中点估计）。最终以实测统计为准。X tokens 只衡量压缩输入规模；LM 的 Y、AE 重建目标、memory 位置及训练重复次数单独计量。同源片段之间可能重叠，以上数值表示样本累计 tokens。

## 执行安排

代码快照：`09a57f1`。从现有 FineWeb sample-10BT 取 100,000 篇候选文档建立共同来源池，执行属性检查、去重与来源划分，然后依次构造 semantic 和 random。来源不足时由程序报告具体 task/split/长度区间缺口。

Qwen/Qwen3.8-27B 使用服务器已有权重，物理 GPU 1，BF16、tensor parallel 1；服务上下文 16,384，支持长 X/Y 的联合判定；请求并发 16，最大输出 1,024 tokens，超时 600 秒。网络连接使用直连。数据构造使用独立代码快照和现有 Python 环境。

运行目录：`/data/bywei/projects/latent_working_memory/artifacts/v1/data-preparation/independent-409k-20260910`。数据目录：`/data/bywei/projects/latent_working_memory/data/v1/fineweb-independent-409k-20260910`。运行目录保存 recipe、配置、代码快照、进程号、分阶段日志、评分缓存及 status.json。

主流程完成各版本的结构审计及跨版本来源、长度分布核对。成品质量抽查在完整数据集构造后独立执行。后续训练优先采用 semantic 数据；训练前需完成基座长上下文及训练长度配置的适配。
