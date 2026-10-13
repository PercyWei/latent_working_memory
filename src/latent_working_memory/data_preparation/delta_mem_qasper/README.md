# 20261013_δ-Mem QASPER 固定划分

创建时间：20261013 12:35:14 Asia/Shanghai（UTC+08:00）

最后修订时间：20261013 17:06:30 Asia/Shanghai（UTC+08:00）

- 来源：[δ-Mem 发布的 QASPER 数据](https://huggingface.co/datasets/huaXiaKyrie/delta-mem-qasper-data)，上游 revision 随构造配置记录。
- 将同篇论文的 QA 聚合为一条 `text` / `qas` 记录，保存至 `data/delta-mem-qasper/`。
- 全部原始 QA 均参与划分；本步骤不分词、不做长度或答案过滤、不截断正文。

## 执行

```bash
uv run --frozen python -m latent_working_memory.data_preparation.delta_mem_qasper \
  --config configs/data_preparation/delta-mem-qasper.json
```

- [配置](../../../../configs/data_preparation/delta-mem-qasper.json)中的相对路径以配置所在目录为基准。
- `input_file` 指向原始 JSONL，`output_dir` 指定目标目录；目标已存在则拒绝覆盖。
- 临时目录内完成写入后整体移动到目标目录，原始文件不修改。

## 转换与划分

- 每行仅包含以下字段；`text`、`question`、`answer` 均为非空字符串，`qas` 为非空列表：

```json
{"text": "完整正文", "qas": [{"question": "问题", "answer": "参考答案"}]}
```

- 正文：移除各消息的 `History turn … / Paper chunk …` 固定前缀，用两个换行符按原顺序连接正文块。
- QA：移除问题的固定提问包装，答案原样保存；同篇论文聚合为一行，文档和文档内 QA 均保留上游首次出现顺序。
- 校验：检查外部消息结构、固定包装、正文块数量、非空内容和唯一 QA 标识；同一 `paper_id` 的正文必须一致，异常直接报错。
- 划分：全部 `paper_id` 排序后用 seed 42 打乱，前 `round(N × dev_fraction)` 篇给 dev，随后 `round(N × test_fraction)` 篇给 test，其余给 train；同篇全部 QA 属于同一划分。
- 默认比例为 train/dev/test = 80%/10%/10%；当前得到 604/76/76 篇论文，分别含 1763/242/214 个 QA。
- 上游仅发布 train，这里的 dev/test 是本实验留出集，**不是 QASPER 官方 dev/test**；长度过滤在使用固定划分时执行。

## 产物

| 文件 | 内容 |
|---|---|
| `train.jsonl` / `dev.jsonl` / `test.jsonl` | 每行一篇正文及其 QA 列表 |
| `preparation.json` | 来源/revision、配置、转换与划分规则、实际统计和来源映射 |
| `README.md` | 按本次实际统计生成的数据说明 |

- `statistics.<split>.document_sources` 按输出行顺序记录 `paper_id` 和 `question_ids`；每条 `question_ids` 与对应文档的 `qas` 顺序一致。
