# 20261126_QASPER 公共评估

创建时间：20261126 16:09:15 Asia/Shanghai

最后修订时间：20261127 20:33:44 Asia/Shanghai

QASPER 是针对 NLP 论文的全文问答数据集。本目录维护官方 v0.3 测试集的读取、按论文生成回答、本地 Answer F1 评分及结果展示。[原论文](https://aclanthology.org/2021.naacl-main.365/) · [官方数据（CC BY 4.0）](https://huggingface.co/datasets/allenai/qasper)

## 评估约定

| `--split` | 官方文件 | 论文数 | 问题数 |
|---|---|---:|---:|
| `test` | `qasper-test-v0.3.json` | 416 | 1,451 |

- **数据范围**：评估测试集的全部题目，包括带有图表证据标注的问题。模型只读取论文正文文本和当前问题。人工参考答案及其类型用于评分。[数据下载](../README.md#2-数据准备)
- **论文正文**：沿用作者 LED 的 `full_text` 输入，按原顺序依次读取章节标题和段落，以空行分隔。输入不额外拼接题名、摘要、图注或图像。[官方正文读取实现](https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/qasper_baselines/dataset_reader.py#L398-L417)
- **答案生成**：按论文分配问题。同一篇论文的正文只准备一次，优先取同篇的问题，不足一批时加入下一篇。各篇正文可并行准备，各题从对应的正文状态独立贪心作答，每题默认最多生成 128 个新 token。提示要求简洁回答，是非题回答 `Yes` 或 `No`，多个片段以逗号分隔，论文未提供答案时输出 `Unanswerable`。
- **续评方式**：沿用原数据、模型和生成设置。部分完成的论文只为剩余问题重新准备正文状态并作答。
- **运行进度**：终端每 30 秒显示整个划分已完成的题数、本次耗时和平均速度（`q/min`）。速度只计本次新完成并保存评分的题目。各卡的逐题进度、论文 ID 和题号见 `progress/gpu-<设备编号>.log`。

## 公共接口

| 接口 | 约定 |
|---|---|
| `load_qasper(split, data_dir)` | 按官方顺序返回论文、题目和人工标注。论文保留原始 ID 及完整正文，各题通过原始 `question_id` 对齐 |
| `load_adapter(settings)` | 根据方法配置加载模型，返回具有 `prepare_contexts` 和 `answer_batch` 接口的适配器 |
| `prepare_contexts(documents)` | 为本批首次出现的各篇论文并行准备正文状态和待回答问题。适配器接收的题目仅含推理所需信息 |
| `answer_batch(states, questions, complete)` | 从各题对应的状态并行回答，完成一题即回调 `complete`，空回答按官方规则计分 |
| `run_evaluation(...)` | 按论文调度生成，在本地计算官方 Answer F1，按原始题序保存结果并汇总 |

分词、模型位置窗口检查和具体读写由适配器维护。公共调度参数由 `add_evaluation_arguments(parser, output_root, checkpoint_layout=False)` 定义。按模型或 checkpoint 组织结果的方法传入 `checkpoint_layout=True`，并向 `run_evaluation` 传入 `checkpoint_name` 和评估设置前缀 `output_name`。Full-context 使用模型目录名作为 `checkpoint_name`，`output_name=""` 时默认结果目录名为 `qasper_test`。[数据与输入](data.py) · [评估入口](evaluate.py)

| 参数 | 含义与默认值 |
|---|---|
| `--benchmark-data` | 输入数据目录，默认 `data/benchmarks/qasper` |
| `--split` | `test`（默认） |
| `--output-root` | 输出根目录，默认值由接入方提供 |
| `--output-id` | 结果目录名；Full-context 默认 `qasper_test`，ICAE、AutoCompressors 和 SDCC 默认 `<压缩设置>_qasper_test` |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--max-new-tokens` | 每题最多生成的新 token 数，默认 `128` |
| `--resume` | 沿用原运行进度，默认关闭 |

## 问答评分

- **逐题分数**：预测与参考答案统一转小写，去除英文标点、冠词，按空白切分 token，根据重复词的匹配次数计算 F1。多个抽取片段先以逗号和空格连接成一个答案。若一题有多份人工标注，取最高 F1；并列时以原顺序第一份最高分标注确定答案类型。[官方评分器](https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py) · [本地评分函数](../vendor/qasper.py) · [代码许可证](../vendor/QASPER-LICENSE.txt)
- **汇总口径**：总体 Answer F1 对全部题目取算术平均。`extractive`、`abstractive`、`boolean` 和 `none` 各自统计题数与平均 F1，分别对应抽取、自由文本、是非和不可回答。四组在单次运行中互斥且覆盖全部题目；同一题可能有不同类型的人工标注，最高分标注随预测变化，因此不同运行的分组题数可能不同。不可回答同样计算 Answer F1。证据选择属于独立任务。

## 产物与指标

ICAE、AutoCompressors 和 SDCC 写入 `<output-root>/eval/<checkpoint_name>/<output-id>/`，默认 ID 为 `<压缩设置>_qasper_test`，例如 `seg2048_qasper_test`。Full-context 写入 `<output-root>/eval/<model_name>/<output-id>/`，`model_name` 取 `--model-path` 指向的目录名，默认 ID 为 `qasper_test`。显式指定 `--output-id` 时以该值作为完整目录名；在相同命令后加 `--resume` 即可续评，自定义 ID 须沿用原值。

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法与模型配置、生成设置、划分和数据身份 |
| `progress/` | 生成进度与各卡日志，供中断后恢复 |
| `predictions.jsonl` | 按原始题序保存，每行含 `question_id` 和字符串 `predicted_answer` |
| `details.jsonl` | 每题的论文 ID、问题 ID、问题、预测、人工参考答案、用于分组的答案类型和 F1 |
| `summary.json` | 评估配置与汇总指标。`metrics.score` 为总体 Answer F1，`metrics.groups` 保存各答案类型的题数及平均分。分数均以 0–1 保存 |

## 结果展示

### 跨运行比较

激活已安装 Streamlit 的环境后，在项目根目录运行以下任一命令。将 `path/to/...` 替换为实际路径。

```bash
bash benchmarks/qasper/scripts/dashboard.sh --run path/to/run-a path/to/run-b
bash benchmarks/qasper/scripts/dashboard.sh --run-file configs/reproductions/benchmarks/qasper.txt
```

- `--run DIR [DIR ...]` 直接传入一个或多个结果目录，可重复使用。
- `--run-file FILE` 读取 UTF-8 清单，每行一个结果目录，忽略空行与以 `#` 开头的注释行。

两种入口可单独或合用。结果目录须直接包含 `summary.json` 和 `details.jsonl`，相对路径以项目根目录为基准，重复目录会报错。

看板按方法、checkpoint／模型和评估设置分组，默认全选。方法按首次出现的顺序排列，合用时先读 `--run`，再按行读 `--run-file`。ICAE 优先按评估配置中的训练阶段排序，`pretrained` 排在 `finetuned` 前。同一方法内再按 checkpoint 名自然排序和评估参数数值排序。勾选会同步更新图表、统计和逐题对照。

比较 test 的总体及分答案类型 Answer F1，统计表保留各类题数，逐题区按论文和问题筛选。各运行须使用相同的数据与评分器，题目、所属论文和参考答案须一致。答案类型按各运行中取得最高分的参考答案确定，因此各类题数可能不同。

### 方法看板接入

[公共展示组件](dashboard.py) 的 `show_runs` 和 `load_runs` 均接收非空的 `(展示标签, Path 类型的结果目录)` 序列，标签须唯一。

| 接口 | 用途 |
|---|---|
| `show_runs(runs)` | 读取并校验结果，展示 Answer F1、题数和逐题回答 |
| `load_runs(runs)` | 返回校验后的运行列表，每项含 `label`、`directory`、`summary` 和按 `question_id` 索引的 `records` |
| `show_scores(runs, show_chart=True)` | 展示已加载运行的指标。设 `show_chart=False` 时仅展示统计表 |
| `show_answers(runs)` | 对照已加载运行的同题回答 |

新方法接入跨运行看板时，在[运行配置解析](../comparison_runs.py)中定义模型、checkpoint 和评估参数的读取规则。
