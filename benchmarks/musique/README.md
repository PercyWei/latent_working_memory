# 20261127_MuSiQue 公共评估

创建时间：20261127 15:58:00 Asia/Shanghai

最后修订时间：20261128 15:11:56 Asia/Shanghai

MuSiQue 是需要关联多篇文档完成 2–4 跳推理的问答数据集。本目录维护 MuSiQue-Ans v1.0 的读取、答案生成调度、本地 Answer F1／EM 评分及结果展示。[原论文](https://aclanthology.org/2022.tacl-1.31/) · [官方仓库](https://github.com/StonyBrookNLP/musique)

## 评估约定

| `--split` | 官方文件 | 问题数 |
|---|---|---:|
| `dev` | `musique_ans_v1.0_dev.jsonl` | 2,417 |

- **数据范围**：使用 MuSiQue-Ans 开发集的全部可回答问题。划分选择见[公共 benchmark 指南](../README.md)。通过 `bash benchmarks/scripts/download_benchmarks.sh --benchmark musique` 下载原始数据。
- **上下文**：按原始顺序拼接每题全部候选段落的标题和正文，以空行分隔，保留支持段落和干扰段落。dev 每题含 17–20 个段落。问题分解用于统计跳数，参考答案及别名用于评分。
- **答案生成**：每题独立准备正文状态并贪心作答，默认最多生成 128 个新 token，提示要求仅输出简洁答案。各题上下文准备和回答可按批并行执行。
- **续评方式**：沿用原数据、模型和生成设置，复用已完成并评分的逐题结果，补齐剩余问题。

## 公共接口

| 接口 | 约定 |
|---|---|
| `load_musique(split, data_dir)` | 按官方顺序读取问题、完整候选上下文、参考答案及跳数，保留原始题号 |
| `load_adapter(settings)` | 根据方法配置加载模型，返回实现 `DocumentAdapter` 的适配器 |
| `prepare_contexts(documents)` | 为本批每题的候选上下文准备独立状态。输入为 `(正文, 问题列表)`，每份正文对应当前一题，题目仅含推理所需信息 |
| `answer_batch(states, questions, complete)` | 从各题对应的状态并行回答，完成一题即回调 `complete`，空回答按官方规则计分 |
| `run_evaluation(...)` | 调度逐题生成，按原始题序保存官方 Answer F1／EM 结果，默认追加 Judge Accuracy |

分词、模型位置窗口检查和具体读写由适配器维护，公共调度复用 [DocumentAdapter](../documents.py)。公共参数由 `add_evaluation_arguments(parser, output_root, checkpoint_layout=False)` 定义。按 checkpoint 组织结果时传入 `checkpoint_layout=True`，并向 `run_evaluation` 传入 `checkpoint_name` 和表示当前设置的 `output_name`。[数据与输入](data.py) · [评估入口](evaluate.py)

| 参数 | 含义与默认值 |
|---|---|
| `--resume` | 沿用原运行进度，默认关闭 |
| `--benchmark-data` | 输入数据目录，默认 `data/benchmarks/musique` |
| `--split` | 仅支持 `dev`（默认） |
| `--output-root` | 输出根目录，默认值由接入方提供 |
| `--output-id` | 结果目录名，默认值见[产物与指标](#产物与指标) |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--max-new-tokens` | 每题最多生成的新 token 数，默认 `128` |
| `--no-llm-as-judge` | 跳过 Judge Accuracy，生成答案并计算官方 Answer F1／EM |

## 问答评分

- **逐题分数**：采用官方答案评分规则，预测与参考答案统一转小写，去除英文标点、冠词并归一化空白。EM 判断归一化后的字符串是否完全相同；F1 按空白切分 token，根据重复词的匹配次数计算。参考答案包含主答案和别名，EM 和 F1 分别取各参考答案中的最高分。[官方评分实现](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py) · [本地评分函数](../vendor/musique.py) · [许可证](../vendor/MUSIQUE-LICENSE.txt)
- **汇总口径**：总体 Answer F1 和 EM 对全部问题取算术平均，另按 2、3、4 跳分别统计题数及平均分。当前任务评估最终答案，不生成或评分支持段落预测。
- **Judge Accuracy** 默认在官方 Answer F1／EM 写出后计算。裁判核对完整多跳问题的最终答案，完整匹配官方主答案或任一别名即可得分。总体及分跳数均按题统计。[裁判配置与补评](../README.md#judge-accuracy)

## 产物与指标

结果目录由公共 `evaluation_directory` 统一构造。普通布局为 `<output-root>/eval/<output-id>/musique/dev/`，默认 ID 为 `<output_name>_<YYYYMMDD-HHMM>`，续评时指定原 ID。checkpoint 布局为 `<output-root>/eval/<checkpoint_name>/<output-id>/`，默认 ID 为 `<output_name>_musique_dev`，无前缀时为 `musique_dev`。显式指定的 ID 用作完整目录名。[目录规则](../output.py)

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法与模型配置、生成设置、划分和数据身份 |
| `progress/` | 逐题进度与各卡日志，供中断后恢复 |
| `predictions.jsonl` | 按原始题序保存，每行含 `question_id` 和字符串 `predicted_answer` |
| `details.jsonl` | 每题的 `question_id`、`question`、`hops`、`predicted_answer`、`references`、`score`（F1）和 `exact_match`（EM） |
| `summary.json` | 评估配置及汇总；`metrics.score` 为总体 F1，`metrics.exact_match` 为总体 EM，`metrics.groups` 按 `hops` 保存题数及两项分数。分数均以 0–1 保存 |

## 结果展示

### 跨运行比较

激活已安装 Streamlit 的环境后，在项目根目录运行以下任一命令。将 `path/to/...` 替换为实际路径。

```bash
bash benchmarks/musique/scripts/dashboard.sh --run path/to/run-a path/to/run-b
bash benchmarks/musique/scripts/dashboard.sh --run-file path/to/runs.txt
```

- `--run DIR [DIR ...]` 接收一个或多个结果目录，可重复指定。
- `--run-file FILE` 读取 UTF-8 清单，每行一个结果目录，忽略空行及以 `#` 开头的整行注释。

两种入口可单独或合用。结果目录须直接包含 `summary.json` 和 `details.jsonl`，相对路径以项目根目录为基准，重复目录会报错。

看板按方法、checkpoint／模型和评估设置分组，默认全选。方法按首次出现的顺序排列，合用时先读 `--run`，再按行读 `--run-file`。ICAE 优先按评估配置中的训练阶段排序，`pretrained` 排在 `finetuned` 前。同一方法内再按 checkpoint 名自然排序和评估参数数值排序。勾选会同步更新图表、统计和逐题对照。

主图分别比较 MuSiQue dev 总体和各跳数的 Answer F1、Answer EM、Judge Accuracy。逐题区可按跳数筛选并查看裁判的判定理由。参与比较的运行须使用相同数据和评分器，逐题结果按题号核对问题、跳数及参考答案。

### 方法看板接入

[展示组件](dashboard.py) 的 `runs` 输入为 `(标签, Path 类型的结果目录)` 序列。

| 接口 | 用途 |
|---|---|
| `show_runs(runs)` | 加载并校验结果，展示指标表格与逐题回答 |
| `load_runs(runs)` | 返回经校验的运行列表，供 `show_scores(loaded)` 和 `show_answers(loaded)` 分开渲染 |
| `comparison_panels(metrics)` | 返回 F1 与 EM 两组主图指标 |

新方法接入公共比较时，在 [comparison_runs.py](../comparison_runs.py) 定义模型与评估参数的读取规则。

[checkpoint 看板入口](../checkpoint_dashboard.py) 按 `MuSiQue · dev` 展示模型或 checkpoint 目录下的已评分设置。
