# 20261014_RULER 与 LoCoMo 评估

创建时间：20261014 11:04:16 Asia/Shanghai（UTC+08:00）

最后修订时间：20261014 11:16:01 Asia/Shanghai（UTC+08:00）

## 1. 数据位置与评估协议

| Benchmark | 本地数据 | 默认评估范围 | 评分 |
|---|---|---|---|
| RULER v1 | `data/benchmarks/ruler-v1/<length>/<task>/test.jsonl` | 8192、16384、32768 tokens；完整 13 项任务 | 官方必答项召回／任一参考答案命中，按任务等权汇总 |
| LoCoMo | `data/benchmarks/locomo/locomo10.json` | 全部会话的 QA 类别 1–4 | 官方分类别 F1 与按题平均 F1 |

- 以下指令从项目根目录执行；数据下载和 RULER 样本生成由用户手动运行。
- RULER 保留官方 `input`、`outputs` 等字段；LoCoMo 保留官方完整 JSON，不转换为训练用的 `{text,qas}`。
- 每题独立初始化记忆，按 checkpoint 的窗口和写入设置流式读取，再执行 greedy 生成。参考答案只用于评分。
- 输入只应用一次 Qwen chat template，关闭 thinking；不套用训练数据的长度筛选，不截断 benchmark 内容。
- RULER 按任务使用官方生成预算：NIAH 128、VT 30、CWE 120、FWE 50、QA 32 tokens。LoCoMo 为 50 tokens。
- LoCoMo 按 session 时间顺序保留日期、说话人、正文和 `blip_caption`；不同问题不继承之前生成的答案。类别 1–4 分别为 multi-hop、temporal、open-domain、single-hop；不评估 adversarial 类别 5。

## 2. 手动准备数据

先安装项目锁定依赖；RULER 准备程序使用本项目的 Python 环境，无需安装其推理服务框架。

```bash
uv sync --frozen
```

### RULER v1

固定采用 [NVIDIA/RULER](https://github.com/NVIDIA/RULER/tree/c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a) 的 v1 生成代码。系统需安装 `git-lfs` 和 `curl`；`english_words.json` 由 Git LFS 管理，用于 CWE 任务。下载源代码和所需的 essay、SQuAD、HotpotQA 素材：

```bash
mkdir -p data/raw data/benchmarks/ruler-v1
git clone --no-checkout https://github.com/NVIDIA/RULER.git data/raw/ruler-v1-source
git -C data/raw/ruler-v1-source lfs install --local
git -C data/raw/ruler-v1-source checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
git -C data/raw/ruler-v1-source lfs pull --include="scripts/data/synthetic/json/english_words.json"
uv run --frozen python -m nltk.downloader punkt punkt_tab

benchmark_project_dir="$PWD"
(
  cd data/raw/ruler-v1-source/scripts/data/synthetic/json
  uv run --frozen --project "$benchmark_project_dir" python download_paulgraham_essay.py
  curl -fL https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -o squad.json
  curl -fL https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json -o hotpotqa.json
)
```

使用训练时相同的 tokenizer 生成固定样本，默认每个长度、每项任务 100 条，seed 为 42：

```bash
benchmark_model_path="$HOME/models/Qwen3-4B-Instruct-2507"
benchmark_template_overhead="$(uv run --frozen python - "$benchmark_model_path" <<'PY'
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
chat_tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": ""}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
)
# 官方 --prepare_for_ns 会再次减去 base 模板本身的 token 数。
print(len(chat_tokens) - len(tokenizer.tokenize("{task_template}")))
PY
)"

for length in 8192 16384 32768; do
  for task in niah_single_1 niah_single_2 niah_single_3 \
    niah_multikey_1 niah_multikey_2 niah_multikey_3 \
    niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2; do
    uv run --frozen python data/raw/ruler-v1-source/scripts/data/prepare.py \
      --save_dir "data/benchmarks/ruler-v1/$length" \
      --benchmark synthetic --task "$task" --subset test \
      --tokenizer_path "$benchmark_model_path" --tokenizer_type hf \
      --max_seq_length "$((length - benchmark_template_overhead))" \
      --model_template_type base --prepare_for_ns \
      --num_samples 100 --random_seed 42
  done
done
```

- `<length>` 表示完整输入与该任务生成预算的总长度上限；上述指令预留 chat template 开销，官方生成器另行预留任务的输出预算。
- 官方生成器将 `answer_prefix` 单独保存；评估只读取 `input`，不把它拼回 prompt。多值／多 query 任务保持一条输入、一次回答。
- 所有比较方法应使用同一批生成样本；更换 tokenizer 或生成设置后，应重新生成到独立目录，并以 `--data-dir` 选择。

### LoCoMo

直接下载 [官方正式数据](https://github.com/snap-research/locomo/tree/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376)，无需额外构造样本：

```bash
mkdir -p data/benchmarks/locomo
curl -fL \
  https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json \
  -o data/benchmarks/locomo/locomo10.json
```

## 3. 运行评估与查看结果

将 `global_step_N` 替换为最终 checkpoint 的实际目录：

```bash
bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --benchmark ruler --gpus 0 \
  --checkpoint artifacts/v4/<run-id>/train/conditioned-m32/checkpoints/global_step_N \
  --lengths 8192 16384 32768

bash src/latent_working_memory/v4/scripts/run.sh \
  --evaluate-only --benchmark locomo --gpus 0 \
  --checkpoint artifacts/v4/<run-id>/train/conditioned-m32/checkpoints/global_step_N
```

- `--data-dir` 可覆盖表中的 benchmark 根目录；RULER 指向包含各长度子目录的目录，LoCoMo 指向包含 `locomo10.json` 的目录。
- 模型、revision、tokenizer 和 memory 权重取自 checkpoint；只需选择一张 GPU。
- 默认产物位于原实验的 `eval/<method-dir>/ruler/` 或 `eval/<method-dir>/locomo/`，保存逐题预测与汇总。重跑时用 `--output-dir` 指定新目录。
- 在线结果追加到已完成训练的原 SwanLab run，按 benchmark 展示任务／长度或 QA 类别分数。`--tracking disabled` 仅保存本地结果。
- 复用项目的 checkpoint 加载、流式生成、资源记录和发布流程，评分沿用上述固定官方源码；不计算答案 NLL。
