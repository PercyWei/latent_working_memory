# 20260910_4k 独立 AE/LM 数据准备（11:05:54 UTC+08:00）

创建时间：20260910 11:02:22 UTC+08:00。最后修订时间：20260910 11:05:54 UTC+08:00。

## 数据契约

所有长度按训练基座 tokenizer 对最终原文切片独立分词计算，不含提示和特殊 token。AE 的 X 为 32–4096 tokens，重建目标为 X。LM 的 X 与 Y 分别为 32–4096 tokens，X/(X+Y) 为 0.3–0.7，父片段可达 8192 tokens。X 与 Y 来自同一原文的相邻字符区间，Y 的全部正文及训练端追加的 EOS 参与监督。

`PreparationConfig` 保存 `min_sample_tokens=32`、`max_sample_tokens=4096`、`lm_prefix_fraction=[0.3,0.7]`。比例边界使用精确分数计算，避免浮点取整把合法端点误判。给定 X 长度 n，Y 的合法长度为 `max(32,ceil(3n/7))` 至 `min(4096,floor(7n/3))`。

`length_bounds=[64,128,256,512,1024,2048,4096]` 沿用包含上端点的区间口径：32–64、65–128、129–256、257–512、513–1024、1025–2048、2049–4096。边界值归较短区间。每个 split 内的 semantic AE、semantic LM、random AE、random LM 分别分配等量长度区间配额；总数除以区间数的余数依次分配到前几个区间，各区间最多相差一条。少量样本的配额允许部分区间为零。

配额按最终通过模型判定并去重后的样本计数。AE 与 LM 数量由 `samples_per_task` 分别控制，两者最终数量相等；semantic/random 使用同一任务和长度配额。训练访问比例独立由训练采样器控制。

## 构造流程

1. 从已有 FineWeb sample-10BT 混合读取固定候选来源池，执行显式属性检查和文档去重，按重复簇固定来源 split。全部派生样本继承该 split。
2. semantic 为每篇文档建立 X 的候选区间。句子、段落、连续句子集以及可选主题标注提供候选；连续句子集按不同长度区间提出候选。候选粒度决定来源性质，最终接收由长度配额决定。
3. 根据各任务、长度桶的剩余数量选择当前任务和目标桶。AE、LM 使用独立随机序列选择 X；AE 构造重建任务，LM 从 X 后的原文独立选择满足长度和比例条件的 Y。每次候选仅构造所选任务，允许同一来源文档的不同片段服务于多个任务。
4. random 独立选择来源和 token 起点，按当前桶抽取 X 长度，再按合法范围抽取 Y 长度。semantic 的 X 起止及 X/Y 切点遵循句界，random 保留随机词句边界。原文切片重新分词后统一检查实际长度和比例。
5. Qwen 评价最终 X/Y。keep 样本经样本去重和来源隔离检查后进入配额；其余候选和已满桶的结果保存在决策日志。`candidates_per_document` 控制两类边界版本每篇文档的候选尝试预算，默认 64。
6. 构造审计检查实际 token 长度、比例、原文连续性、任务提示、tokenizer、来源和模型判定。来源池耗尽时输出各 split、任务和长度桶的缺额；完成审计后写入 `preparation.json`。random 完成时核对 semantic 的相同数据契约、均衡配额及实际入选分布。

## 粒度统计

来源粒度包括 `sentence`、`paragraph`、`sentence_group`、`topic_group`。`sentence_group` 表示主题不限的连续句集；`topic_group` 依据 `--topic-annotations` 提供的连续主题区间。未提供主题标注时，可用候选来自句子、段落和连续句子集。

semantic 的 `source_granularity` 记录 X 的构造来源。random 保持 `granularity=random`，额外通过 `source_granularity` 记录 X 所在原文区域：落在单句内的片段标为 sentence；多句片段依次依据主题标注、同段落和跨段落范围确定标签；分句器未识别到覆盖区间时记为 unsegmented。该标签描述原文位置，随机截断后的片段仍按 random 边界解释。

`audit.json` 的 `composition` 同时按 split/task 和 split/task/length_up_to 分组，记录各来源粒度的样本数、样本比例、输入 token 数及输入 token 比例。独立抽查面板也记录来源粒度，用于分层检查。

## 配置与入口

```bash
uv run python -m latent_working_memory.data_preparation \
  --config configs/v1/pretrain_a800.json \
  --recipe configs/data_preparation/fineweb.json \
  --dataset-dir data/raw/HuggingFaceFW-fineweb \
  --output-dir data/v1/fineweb-independent-4k \
  --score-cache data/v1/quality-cache/sample-scores.jsonl
```

`--stage sources` 建立共享池；`--stage semantic` 和 `--stage random` 分别执行构造、评分与检查。来源池和完成数据均使用新输出目录，运行记录保存当前配方。构造契约保留 tokenizer、来源和任务提示；长度、比例及候选规则保存在 preparation recipe 中。

数据准备与训练容量分别校验。默认训练配置的 memory 上限提高到 4096，压缩率仍为 2、4、8；4096-token 输入对应的三个容量为 2048、1024、512。训练实际上下文必须覆盖完整写入，以及 memory、目标、提示和特殊 token 的读取总长。Qwen 服务窗口按其自身 tokenizer 容纳完整 X/Y、评分提示和输出预算，服务部署时单独设置。

## 验证

测试覆盖 AE/LM 独立抽样、X/Y 各 4096 tokens、32-token 下限、0.3/0.7 比例端点、实际重新分词后的随机边界、拒绝候选后的均衡配额补足、来源隔离、粒度统计、CLI 与独立模型检查接口。长样本测试使用本地 tokenizer 和文本夹具验证构造契约。

本次本地 v1 的 77 项测试全部通过，Ruff 检查通过。验证使用本地文本、tokenizer、微型模型和 HTTP 服务夹具。正式 FineWeb 数据构造与 GPU 评分属于后续运行步骤。
