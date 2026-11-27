# 20261126_QASPER 公共评估

创建时间：20261126 16:09:15 Asia/Shanghai

最后修订时间：20261127 15:28:01 Asia/Shanghai

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

分词、模型位置窗口检查和具体读写由适配器维护。公共调度参数由 `add_evaluation_arguments(parser, output_root, checkpoint_layout=False)` 定义。按 checkpoint 组织结果的方法传入 `checkpoint_layout=True`，并向 `run_evaluation` 传入 `checkpoint_name` 和表示压缩设置的 `output_name`。[数据与输入](data.py) · [评估入口](evaluate.py)

| 参数 | 含义与默认值 |
|---|---|
| `--benchmark-data` | 输入数据目录，默认 `data/benchmarks/qasper` |
| `--split` | `test`（默认） |
| `--output-root` | 输出根目录，默认值由接入方提供 |
| `--output-id` | 结果目录名；ICAE、AutoCompressors 和 SDCC 默认 `<压缩设置>_qasper_test`，其他方法默认 `<output_name>_<YYYYMMDD-HHMM>` |
| `--batch-size-per-gpu` | 每卡批处理数，默认 `1` |
| `--max-new-tokens` | 每题最多生成的新 token 数，默认 `128` |
| `--resume` | 沿用原运行进度，默认关闭 |

## 问答评分

- **逐题分数**：预测与参考答案统一转小写，去除英文标点、冠词，按空白切分 token，根据重复词的匹配次数计算 F1。多个抽取片段先以逗号和空格连接成一个答案。若一题有多份人工标注，取最高 F1；并列时以原顺序第一份最高分标注确定答案类型。[官方评分器](https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py) · [本地评分函数](../vendor/qasper.py) · [代码许可证](../vendor/QASPER-LICENSE.txt)
- **汇总口径**：总体 Answer F1 对全部题目取算术平均。`extractive`、`abstractive`、`boolean` 和 `none` 各自统计题数与平均 F1，分别对应抽取、自由文本、是非和不可回答。同一题的人工标注可能属于不同类型，因此各组题数可随预测变化。证据选择属于独立任务。

## 产物与指标

ICAE、AutoCompressors 和 SDCC 写入 `<output-root>/eval/<checkpoint_name>/<output-id>/`。省略 `--output-id` 时使用 `<压缩设置>_qasper_test`，例如 `seg2048_qasper_test`；显式指定时以该值作为完整目录名。这三种方法在相同命令后加 `--resume` 即可续评，自定义 ID 须沿用原值。Full-context 写入 `<output-root>/eval/<output-id>/qasper/test/`，续评时指定原 ID。

| 文件 | 内容 |
|---|---|
| `evaluation_config.json` | 方法与模型配置、生成设置、划分和数据身份 |
| `progress/` | 生成进度与各卡日志，供中断后恢复 |
| `predictions.jsonl` | 按原始题序保存，每行含 `question_id` 和字符串 `predicted_answer` |
| `details.jsonl` | 每题的论文 ID、问题 ID、问题、预测、人工参考答案、用于分组的答案类型和 F1 |
| `summary.json` | 评估配置与汇总指标。`metrics.score` 为总体 Answer F1，`metrics.groups` 保存各答案类型的题数及平均分。分数均以 0–1 保存 |

## 结果展示

### 跨运行比较

公共看板脚本使用当前环境的 Python 启动 Streamlit。启动前激活已安装看板依赖的环境。`--run` 后按“展示标签、已评分的划分目录”成对传参，可连续指定多组，也可重复使用 `--run`。[公共看板脚本](scripts/dashboard.sh)

```bash
source .venv/bin/activate
bash benchmarks/qasper/scripts/dashboard.sh \
  --run 方法A-运行1 artifacts/method-a/eval/run-1/qasper/test \
        方法A-运行2 artifacts/method-a/eval/run-2/qasper/test \
        方法B-运行1 artifacts/method-b/eval/run-1/qasper/test
```

每个目录直接包含 `summary.json` 和 `details.jsonl`，展示标签须非空且各不相同，含空格时加引号。页面展示全部传入运行的总体和各答案类型的 Answer F1 及题数，按论文和原始 `question_id` 并排展示参考答案、模型回答及逐题分数。

参与比较的运行须使用相同数据指纹，逐题结果覆盖 test 的完整题目集合。看板读取已保存的分数，支持不同方法、模型及输入设置的运行并列展示。[展示接口](dashboard.py)

### 方法看板接入

方法维护自己的 Streamlit 页面和启动脚本，通过 `show_runs(runs)` 嵌入 QASPER 的指标与逐题回答对照。页面入口负责设置页面标题和选择结果目录。[公共展示组件](dashboard.py)

| 接口 | 用法 |
|---|---|
| `show_runs(runs)` | 接收非空的 `(展示标签, Path 类型的划分目录)` 序列，标签须各不相同。读取已评分产物，核对 QASPER Answer F1 指标、划分和数据指纹，展示总体与分答案类型结果及同题回答 |
| `load_runs(runs)` | 接收相同的标签与目录序列，核对指标、划分和数据指纹，返回包含 `label`、`directory`、`summary` 和 `records` 的运行字典列表；`records` 按原始 `question_id` 索引 |
| `show_scores(runs)` | 接收已加载的运行对象列表，展示总体与分答案类型的 Answer F1 和题数 |
| `show_answers(runs)` | 接收同样的运行对象列表，按论文和问题展示多运行回答、参考答案及逐题分数 |

`show_runs` 复用 `load_runs` 完成加载与校验；自定义布局可先调用 `load_runs`，再分别调用 `show_scores` 和 `show_answers`，分开展示概览与逐题内容。在方法的 Streamlit 页面中可直接调用下面的代码，查看一次运行的测试集结果。传入多对标签与目录即可比较多个运行。

```python
from pathlib import Path

from benchmarks.qasper.dashboard import show_runs

run_dir = Path("artifacts/my-method/eval/run-1")
show_runs([("My method · test", run_dir / "qasper" / "test")])
```

ICAE、AutoCompressors 和 SDCC 使用 `benchmarks.checkpoint_dashboard.run_dashboard(method, title, render_memory, argv=None)`。`--output-dir` 接收一个或多个 `<output-root>/eval/<checkpoint_name>/`，可连续列出目录或重复该参数；侧栏默认选中全部 checkpoint。入口根据 `summary.json` 中的任务信息汇集 QASPER test，跨 checkpoint 比较全部运行的总体和各答案类型 Answer F1 及题数。底部任务选择器用于查看逐题回答与分数。`render_memory(runs)` 只负责 LoCoMo 的记忆统计。[checkpoint 看板入口](../checkpoint_dashboard.py) · [AutoCompressors 接入示例](../../reproductions/autocompressors/evaluation/dashboard.py)

多 checkpoint 的标签为 `checkpoint_name/output_id`，单 checkpoint 为 `output_id`。先按压缩参数数值升序，再按 checkpoint 名和 `output_id` 自然排序；看板图表中相同参数使用同色系，不同 checkpoint 以深浅区分。[多目录命令与任务分组](../locomo/README.md#方法看板接入)

按运行 ID 保存结果的方法使用 `benchmarks.locomo.dashboard.run_dashboard(method, title, render_results, argv=None, qasper=False)`，通过 `qasper=True` 启用 QASPER，`--output-dir` 指向包含 `qasper/test/` 的运行目录。入口选择一个已评分任务，核对方法、benchmark 和划分后调用 `show_runs` 展示 QASPER；`render_results` 回调负责 LoCoMo 的方法专属页面。[公共方法入口](../locomo/dashboard.py)

方法环境需安装 Streamlit 等看板依赖，启动脚本使用该环境的 Python。例如在项目根目录查看 AutoCompressors checkpoint 的已评分设置：

```bash
bash reproductions/autocompressors/scripts/dashboard.sh \
  --output-dir artifacts/reproductions/autocompressors/eval/AutoCompressor-Llama-2-7b-6k
```
