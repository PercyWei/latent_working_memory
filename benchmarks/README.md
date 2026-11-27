# 20261014_公共 Benchmark 数据准备

创建时间：20261014 11:04:16 Asia/Shanghai（UTC+08:00）

最后修订时间：20261127 20:46:57 Asia/Shanghai

本文介绍 RULER、LoCoMo、LongMemEval、QASPER 和 MuSiQue 的数据格式、评估属性、规模与长度统计及准备方式。以下命令均从项目根目录执行。

## 1. 数据概览

### 数据格式

| Benchmark | 文件格式与记录单位 | 正文 | 题目 | 答案 |
|---|---|---|---|---|
| RULER v1 | JSONL，每行一条生成任务 | `input` 中的背景文本 | 同一 `input` 中的任务指令和问题，无独立问题字段 | `outputs`，一个或多个目标答案 |
| LoCoMo | JSON 数组，每条是一段含多道题的多日对话 | `conversation` 中各 `session_n` 的 `speaker`、`text`、可选 `blip_caption`，以及 `session_n_date_time` | `qa[].question` | 常规题用 `answer`，对抗题用 `adversarial_answer` |
| LongMemEval | JSON 数组，每条是一道题及其会话历史 | `haystack_sessions` 中的 `role`、`content`，结合 `haystack_dates` 按日期排列 | `question` 与提问日期 `question_date` | `answer` |
| QASPER v0.3 | JSON 对象，以论文 ID 为键，每篇论文含多道题 | `full_text` 中的 `section_name` 与 `paragraphs`，按章节原顺序拼接 | `qas[].question` | `qas[].answers`，每题可有多份人工标注 |
| MuSiQue-Ans v1.0 | JSONL，每行一道题及其候选段落 | `paragraphs` 中全部段落的 `title`、`paragraph_text`；dev 每题 17–20 段，保留原始顺序 | `question` | `answer` 及别名 `answer_aliases` |

### 评估属性

| Benchmark | 任务类别 | 官方评估指标 |
|---|---|---|
| RULER v1 | 4 大类、13 项任务：检索（NIAH，8 项）、多跳追踪（VT，1 项）、聚合（CWE/FWE，2 项）、问答（QA，2 项） | **[子串匹配分数](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/eval/synthetic/constants.py)**：忽略大小写；QA 命中任一参考答案计 1，否则计 0；其余任务计算命中的目标答案数占目标答案总数的比例。 |
| LoCoMo | 5 类题型：多跳、时间推理、开放域知识、单跳、对抗 | **[答案 F1](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py)**：归一化及词干化后计算词级 F1；多跳题对每个参考子答案取最佳匹配 F1，再求平均。<br>**对抗题判分**：回答包含 `no information available` 或 `not mentioned` 计 1，否则计 0（忽略大小写）。 |
| LongMemEval | 6 类题型：单会话用户信息、单会话用户偏好、单会话助手信息、多会话、时间推理、知识更新；另以 `_abs` 标记拒答题 | **[正确性判定](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)**：本项目使用 `gpt-6-luna`，按官方题型规则，根据问题与参考答案判断回答；正确计 1，否则计 0。 |
| QASPER v0.3 | 4 种答案类型：抽取片段、自由文本、是非、不可回答 | **[Answer F1](https://github.com/allenai/qasper-led-baseline/blob/afd0fb96bf78ce8cd8157639c6f6a6995e4f9089/scripts/evaluator.py)**：预测与数据集参考答案归一化后计算词级 F1，多份参考答案取最高分。<br>**Evidence F1**：预测与数据集参考证据按条目精确匹配计算 F1，多组参考证据取最高分。 |
| MuSiQue-Ans v1.0 | 按推理跳数分 3 组：2 跳、3 跳、4 跳 | **[Answer F1／EM](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/evaluate_v1.0.py)**：答案归一化后，F1 按词级匹配计算，EM 在字符串完全相同时计 1，否则计 0；两项分别取主答案及别名中的最高分。<br>**[Support F1](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/support.py)**：按预测与参考支持段落索引集合的交集计算 F1。 |

### 规模与长度

长度使用 [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) tokenizer 统计，单位为 tokens，不加特殊 token、不截断。RULER 统计完整题面，其余统计正文（不含问题和聊天模板），正文按表中单位计数，同一对话或论文只计一次；P50、P95 采用线性插值。

RULER 使用默认设置（`chat`、seed 42、每档每任务 100 条）；`length` 包含生成预算，表中为实际题面长度。LongMemEval S 与 Oracle 对应同一批题目。

| Benchmark | 统计单位 | 样本数 | 题目数 | 每样本题数 | 最短–最长 | P50 | P95 |
|---|---|---:|---:|---|---:|---:|---:|
| **RULER v1** | | | | | | | |
| length 8,192 | 题面 | 1,300 | 1,300 | 1 | 5,544–8,100 | 7,910.5 | 8,066 |
| length 16,384 | 题面 | 1,300 | 1,300 | 1 | 12,646–16,286 | 16,120 | 16,250 |
| length 32,768 | 题面 | 1,300 | 1,300 | 1 | 28,345–32,676 | 32,450.5 | 32,642 |
| **LoCoMo** | | | | | | | |
|  | 对话 | 10 | 1,986 | 105–260（均值 198.6） | 13,479–25,580 | 23,548 | 25,557.05 |
| **LongMemEval** | | | | | | | |
| S | 会话历史 | 500 | 500 | 1 | 99,394–110,731 | 106,311 | 107,960.35 |
| Oracle | 会话历史 | 500 | 500 | 1 | 146–22,448 | 5,805.5 | 12,047.9 |
| **QASPER** | | | | | | | |
| test | 论文 | 416 | 1,451 | 1–10（均值 3.49） | 819–29,828 | 4,441.5 | 8,439 |
| **MuSiQue-Ans** | | | | | | | |
| dev | 候选段落集 | 2,417 | 2,417 | 1 | 1,079–5,315 | 2,480 | 3,615.4 |

### 长度分布

每个区间包含 4,096 个整数长度，统计值为数量（占比）。

<details>
<summary>RULER v1</summary>

| 长度区间（tokens） | length 8,192 | length 16,384 | length 32,768 |
|---|---:|---:|---:|
| 4,097–8,192 | 1,300（100.00%） | 0（0.00%） | 0（0.00%） |
| 12,289–16,384 | 0（0.00%） | 1,300（100.00%） | 0（0.00%） |
| 24,577–28,672 | 0（0.00%） | 0（0.00%） | 1（0.08%） |
| 28,673–32,768 | 0（0.00%） | 0（0.00%） | 1,299（99.92%） |

</details>

<details>
<summary>LoCoMo</summary>

| 长度区间（tokens） | LoCoMo |
|---|---:|
| 12,289–16,384 | 1（10.00%） |
| 16,385–20,480 | 2（20.00%） |
| 20,481–24,576 | 4（40.00%） |
| 24,577–28,672 | 3（30.00%） |

</details>

<details>
<summary>LongMemEval</summary>

| 长度区间（tokens） | S | Oracle |
|---|---:|---:|
| 1–4,096 | 0（0.00%） | 170（34.00%） |
| 4,097–8,192 | 0（0.00%） | 255（51.00%） |
| 8,193–12,288 | 0（0.00%） | 50（10.00%） |
| 12,289–16,384 | 0（0.00%） | 16（3.20%） |
| 16,385–20,480 | 0（0.00%） | 8（1.60%） |
| 20,481–24,576 | 0（0.00%） | 1（0.20%） |
| 98,305–102,400 | 7（1.40%） | 0（0.00%） |
| 102,401–106,496 | 273（54.60%） | 0（0.00%） |
| 106,497–110,592 | 219（43.80%） | 0（0.00%） |
| 110,593–114,688 | 1（0.20%） | 0（0.00%） |

</details>

<details>
<summary>QASPER</summary>

| 长度区间（tokens） | test |
|---|---:|
| 1–4,096 | 185（44.47%） |
| 4,097–8,192 | 207（49.76%） |
| 8,193–12,288 | 17（4.09%） |
| 12,289–16,384 | 5（1.20%） |
| 24,577–28,672 | 1（0.24%） |
| 28,673–32,768 | 1（0.24%） |

</details>

<details>
<summary>MuSiQue-Ans</summary>

| 长度区间（tokens） | dev |
|---|---:|
| 1–4,096 | 2,387（98.76%） |
| 4,097–8,192 | 30（1.24%） |

</details>

## 2. 数据准备

数据下载和 RULER 样本生成由用户手动运行，无需 GPU。下载依赖系统 `curl`、`tar` 及项目环境：

```bash
uv sync --frozen
bash benchmarks/scripts/download_benchmarks.sh --benchmark all
```

**原始数据下载：[download_benchmarks.sh](scripts/download_benchmarks.sh)**

下载入口覆盖所选 benchmark 的同路径文件。RULER 下载素材后还需生成样本；其余 benchmark 可直接读取原始 JSON 或 JSONL 评估。

- **RULER v1**：采用固定提交的 [NVIDIA/RULER v1 源码](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a)。
  - 下载源码及 essay、SQuAD、HotpotQA、CWE 词表，保存到 `--ruler-source-dir`。
  - 下载 NLTK 的 `punkt`、`punkt_tab`，保存到 `data/raw/nltk_data/tokenizers/`。
- **LoCoMo**：下载[官方 `locomo10.json`](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)。
- **LongMemEval-S / Oracle**：下载作者[清洗版](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)的 S 与 Oracle JSON。
- **QASPER v0.3**：从[测试集压缩包](https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz)提取 `qasper-test-v0.3.json`。
- **MuSiQue-Ans v1.0**：从固定提交的 [`voidful/MuSiQue` 第三方镜像](https://huggingface.co/datasets/voidful/MuSiQue/tree/a7d9f9adf6191604fc67cde318ee1a86fcf7babc)下载原始 `musique_ans_v1.0_dev.jsonl`；[官方来源](https://github.com/StonyBrookNLP/musique#data)。
  - test 未公开答案标签，因此使用完整 dev 的 2,417 道题作为当前评估划分。

| 参数 | 填法与默认值 |
|---|---|
| `--benchmark` | 支持 `all`（默认）、`ruler`、`locomo`、`longmemeval`、`qasper` 或 `musique` |
| `--data-root` | LoCoMo、LongMemEval、QASPER 和 MuSiQue 的数据父目录，默认 `data/benchmarks` |
| `--ruler-source-dir` | RULER 官方源码及下载素材目录，默认 `data/raw/ruler-v1-source` |

`HF_ENDPOINT` 可统一覆盖 HotpotQA、LongMemEval 和 MuSiQue 的下载地址；未设置时，HotpotQA 使用 `https://hf-mirror.com`，其余两者使用 `https://huggingface.co`。

通过 ModelScope 分发或还原数据时，保留下列完整目录：

```text
data/
├── raw/
│   ├── ruler-v1-source/          # RULER 源码与已下载素材
│   └── nltk_data/                # punkt、punkt_tab 及其目录结构
└── benchmarks/
    ├── locomo/
    │   └── locomo10.json
    ├── longmemeval/
    │   ├── longmemeval_s_cleaned.json
    │   └── longmemeval_oracle.json
    ├── qasper/
    │   └── qasper-test-v0.3.json
    └── musique/
        └── musique_ans_v1.0_dev.jsonl
```

**RULER 样本生成：[generate_ruler.sh](scripts/generate_ruler.sh)**

使用上述固定版本的源码与素材，仅加载 tokenizer 构造样本。源码目录不在默认位置时，传入对应的 `--ruler-source-dir`。样本保存为 `<output-dir>/<length>/<task>/test.jsonl`。

```bash
bash benchmarks/scripts/generate_ruler.sh \
  --model-path "$HOME/models/Qwen3-4B-Instruct-2507" \
  --lengths 8192 16384 32768 \
  --num-samples 100 \
  --seed 42
```

| 参数 | 填法与默认值 |
|---|---|
| `--ruler-source-dir` | 已下载的 RULER 源码及素材目录，默认 `data/raw/ruler-v1-source` |
| `--output-dir` | 默认 `data/benchmarks/ruler-v1/<model-name>`；显式指定时直接使用该目录，不附加模型名 |
| `--model-path` | 与训练一致的 tokenizer 来源；本地路径或 HF repo ID，默认 `~/models/Qwen3-4B-Instruct-2507` |
| `--prompt-format` | `chat`（默认）或 `base`，输入协议见下文 |
| `--tasks` | 空格分隔的 RULER 任务名称；默认全部 13 项 |
| `--lengths` | 空格分隔的长度列表，默认 `8192 16384 32768` |
| `--tokens-to-generate` | 可选，统一覆盖各任务的生成预算；默认沿用官方任务预算 |
| `--num-samples` | 每个长度、每项任务的样本数，默认 `100` |
| `--seed` | 样本生成的随机种子，默认 `42` |

- **输出目录**：`<model-name>` 取模型路径末级目录名或 HF 仓库名。已有长度目录拒绝覆盖；修改设置或失败重试时，移走对应目录或指定新的 `--output-dir`。
- **长度预算**：`<length>` 包括完整输入与生成预算。`chat` 为聊天模板预留空间，`base` 为 BOS 等特殊 token 预留空间。默认生成预算为 NIAH 128、VT 30、CWE 120、FWE 50、QA 32 tokens。
- **输入格式**：保留官方 `input`、`outputs`、`answer_prefix` 等字段。聊天评估只使用 `input`；`base` 评估使用 `input + answer_prefix`。多值／多 query 任务保持一条输入、一次回答。
- **生成器**：`base` 的 NIAH 使用[本地 NVIDIA 生成器副本](vendor/ruler_niah.py)，其余任务使用下载的官方生成器。

## 3. 评估入口

RULER 与 LoCoMo 的评估用法见各方法文档：

| 方法 | 运行说明 |
|---|---|
| v4 | [Benchmark 评估与看板](../src/latent_working_memory/v4/benchmarks/README.md) |
| v4.1 | [独立评估与结果看板](../src/latent_working_memory/v4_1/scripts/README.md#4-独立评估) |
| Full-context 基线 | [基线运行说明](../src/latent_working_memory/full_context/README.md) |

### 公共 LoCoMo 评估

Full-context、ICAE v2 和 v3.1.1 的公共接口与使用说明见 [LoCoMo 文档](locomo/README.md)。

### 公共 LongMemEval 评估

Full-context、ICAE v2、AutoCompressors、SDCC、LaCT、E²-TTT 和 In-Place TTT 均提供 `scripts/eval_longmemeval.sh`。[使用说明](longmemeval/README.md) · [裁判配置](longmemeval/README.md#问答评分)

### 公共 QASPER 评估

Full-context、ICAE v2、AutoCompressors 和 SDCC 均通过 `scripts/eval_qasper.sh` 评估 test。[公共协议与续评参数](qasper/README.md) · [Full-context](../src/latent_working_memory/full_context/README.md) · [ICAE](../reproductions/icae/README.md) · [AutoCompressors](../reproductions/autocompressors/README.md) · [SDCC](../reproductions/sdcc/README.md)

### 公共 MuSiQue 评估

Full-context、ICAE v2、AutoCompressors 和 SDCC 均通过 `scripts/eval_musique.sh` 评估 MuSiQue-Ans dev，汇总 Answer F1 和 EM。[公共协议与续评参数](musique/README.md) · [Full-context](../src/latent_working_memory/full_context/README.md) · [ICAE](../reproductions/icae/README.md) · [AutoCompressors](../reproductions/autocompressors/README.md) · [SDCC](../reproductions/sdcc/README.md)
