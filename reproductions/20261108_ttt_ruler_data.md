# 20261108_TTT baselines 的 RULER 数据构造

创建时间：20261108 18:59:02 Asia/Shanghai

最后修订时间：20261108 19:43:55 Asia/Shanghai

本文规定 LaCT、E²-TTT 和 In-Place TTT 本次评估使用的 RULER 数据规模与构造方式。各方法按自身 tokenizer 计长，并采用下表规定的任务、长度档位和样本数；不要求生成文本、needle 位置或逐字符内容与论文相同。以下为准备方案，尚无模型评估结果。

## 任务与规模

| 方法 | 任务 | 总长度档位（tokens） | 每任务、每长度样本数 | 唯一样本总数 | 生成预算 |
|---|---|---|---:|---:|---:|
| LaCT 760M（Momentum 与两个 Muon 变体共用） | S-NIAH-1/2/3、MK-NIAH-1、MQ-NIAH、MV-NIAH，共 6 项 | 4096、8192、16384、32768 | 500 | 12000 | 128 |
| E²-TTT 1.3B | S-NIAH-1、S-NIAH-2，共 2 项 | 512、1024、2048、4096、8192、16384 | 500 | 6000 | 128 |
| In-Place TTT Qwen3-4B | RULER v1 全部 13 项 | 4096、8192、16384、32768、65536、131072、262144 | 100 | 9100 | 64 |

LaCT 的六项任务对应 `niah_single_1`、`niah_single_2`、`niah_single_3`、`niah_multikey_1`、`niah_multiquery`、`niah_multivalue`。其论文明确列出任务与长度，但未明确每项的样本数；**500 是本次沿用 NVIDIA RULER 默认值的选择，不是论文已确认的样本数**。官方默认见 [NVIDIA RULER 配置](https://github.com/NVIDIA/RULER/blob/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a/scripts/config_tasks.sh)。E²-TTT 的范围来自其 `upstream/eval.sh` 与 RULER loader；In-Place TTT 的范围和 100 条设置来自 `upstream/eval_config/ruler_*.py`，64-token 生成预算来自 `upstream/eval_config/models.py`。

LaCT 的以下三个 checkpoint 已核对 `tokenizer.json`、`tokenizer_config.json` 和 `special_tokens_map.json` 内容一致，共用一份 12000 条数据：

- `lact-momentum-rope-prenorm-nheads4-chunk2048-760m`
- `lact-muon-nope-postnorm-nheads2-chunk2048-760m`
- `lact-muon-nope-prenorm-nheads2-chunk2048-760m`

E²-TTT 的 MLP、SwiGLU 两个 1.3B checkpoint 使用相同 tokenizer 和 prompt 协议，共用一份 6000 条数据，分别运行模型评估。In-Place TTT 第三方 32K TTT 与 baseline 权重同样共用一份 9100 条数据。三份数据合计 27100 条；模型重复评估不增加唯一样本数。

In-Place TTT 的 64K、128K、256K 输入对当前第三方 32K 权重属于长度外推。即使完成这些评估，也不等于复现原论文经过 128K 阶段继续预训练及位置扩展后的模型。256K 数据可以提前准备，当前评估入口的长度限制仍需落实后才能运行。

## 素材与 prompt

复用项目已有 `data/raw/ruler-v1-source/` 中的 NVIDIA RULER v1 源码与素材，以及 `data/raw/nltk_data/` 中的 NLTK 资源。下方命令约定已下载的 checkpoint/tokenizer 位于 `~/rivermind-data/models/reproductions/<method>/`；实际目录不同时调整 `--model-path`。已有 Qwen3 Instruct 聊天协议数据不直接作为本次 base 模型的复现输入。

三种设置均使用 base 文本续写，不添加聊天角色或 thinking 模板。NVIDIA 生成器将任务尾部提示单独写入 `answer_prefix`；实际输入为：

```python
prompt = row["input"] + row.get("answer_prefix", "")
```

拼接后再进行 tokenizer 编码；`outputs` 只用于计分，不传入模型。不能沿用项目现有聊天协议中省略 `answer_prefix` 的规则，也不能重复拼接前缀。

长度档位表示**实际编码的完整 prompt 加预留生成预算的总上限**，不是只计算 haystack，也不要求每条样本恰好填满上限。生成器应计入完整任务模板及 `answer_prefix`，并按 tokenizer 的默认 special tokens 预留 BOS 等开销。LaCT 的公开 tokenizer 默认添加一个 BOS；NVIDIA HF 生成器的 `.tokenize()` 计长不包含该 BOS，需要在最终编码预算中考虑。base 模板没有额外聊天包装，不能减去 Qwen 聊天模板的开销。

## 构造命令

以下命令从项目根目录运行，仅使用 CPU 和 tokenizer，随机种子统一为 42。

LaCT 使用 Momentum 目录的 tokenizer 构造，供其三个 760M checkpoint 使用：
```bash
bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh \
  --model-path ~/rivermind-data/models/reproductions/lact/lact-momentum-rope-prenorm-nheads4-chunk2048-760m \
  --prompt-format base \
  --tasks niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multiquery niah_multivalue \
  --lengths 4096 8192 16384 32768 \
  --num-samples 500 \
  --seed 42 \
  --output-dir data/benchmarks/ruler-v1/reproductions/lact-760m
```
E²-TTT 使用 MLP checkpoint 的 tokenizer 构造，供其两个 1.3B checkpoint 使用：
```bash
bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh \
  --model-path ~/rivermind-data/models/reproductions/e2_ttt/mlp-1.3b-15b \
  --prompt-format base \
  --tasks niah_single_1 niah_single_2 \
  --lengths 512 1024 2048 4096 8192 16384 \
  --num-samples 500 \
  --seed 42 \
  --output-dir data/benchmarks/ruler-v1/reproductions/e2-ttt-1.3b
```
In-Place TTT 使用第三方 TTT checkpoint 的 tokenizer 构造，供其 TTT 和 baseline 使用：
```bash
bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh \
  --model-path ~/rivermind-data/models/reproductions/in_place_ttt/qwen3-4b-32k-ttt \
  --prompt-format base \
  --lengths 4096 8192 16384 32768 65536 131072 262144 \
  --num-samples 100 \
  --seed 42 \
  --tokens-to-generate 64 \
  --output-dir data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b
```
base NIAH 使用本地 NVIDIA 生成器副本，修正短长度下背景最少 500 词导致无法缩短及超长无限重试的问题；其余任务继续调用已有素材目录中的生成器，任务定义与 JSONL 格式保持不变。

LaCT 与 E²-TTT 不指定 `--tokens-to-generate`，沿用官方 NIAH 的 128-token 预算；In-Place TTT 显式使用其原评估配置的 64-token 预算，生成数据时同步预留该预算。

输出采用 NVIDIA 原生结构：`<output-dir>/<length>/<task>/test.jsonl`，保留 `input`、`answer_prefix`、`outputs` 及原始元数据。

## 接入评估的边界

E²-TTT 官方 harness 与 In-Place TTT 的 OpenCompass 配置当前会现场生成样本；它们不会因为上述目录存在就自动读取本地文件。后续需显式接入这些 JSONL，并核对实际读取路径、任务集合、样本数、长度预算及前缀处理，才能在结果中记录“使用本次预构造数据”。现有原生评估入口可继续作为对照，但不能声称已使用这些本地样本。

本地 CPU 验证：使用实际 checkpoint 的 tokenizer 构造 LaCT 六项各 1 条（4K）、E²-TTT 两项各 2 条（512），In-Place TTT 十三项各 1 条（4K）及额外 1 条 S-NIAH-1（256K），合计 24 条。完整 prompt 编码后加预留输出预算均未超过目标长度；shell 和 Python 语法检查通过。验证样本位于临时目录，不计入上述正式规模；尚未全量构造或运行模型评估。
