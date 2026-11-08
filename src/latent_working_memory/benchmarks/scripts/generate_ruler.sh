#!/usr/bin/env bash
set -euo pipefail

LWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$LWM_ROOT"

usage() {
  cat <<'HELP'
用法：bash src/latent_working_memory/benchmarks/scripts/generate_ruler.sh [参数]

从已下载的 RULER v1 素材构造指定评估任务，默认全部 13 项。

  --ruler-source-dir PATH  RULER 源码与素材，默认 data/raw/ruler-v1-source
  --output-dir PATH        评估数据目录，默认 data/benchmarks/ruler-v1/<model-name>
  --model-path PATH        tokenizer 来源，默认 ~/models/Qwen3-4B-Instruct-2507
  --prompt-format FORMAT   chat（默认）或 base；base 使用 input + answer_prefix
  --tasks NAME [NAME ...]  任务名称，默认全部 13 项
  --lengths N [N ...]      输入与生成预算的总长度，默认 8192 16384 32768
  --tokens-to-generate N   统一生成预算；默认使用各任务的官方预算
  --num-samples N          每个长度、每项任务的样本数，默认 100
  --seed N                 随机种子，默认 42
  -h, --help               显示帮助

先下载 RULER 素材与 data/raw/nltk_data 中的 NLTK 资源；仅需 tokenizer 与 CPU。
可执行 bash src/latent_working_memory/benchmarks/scripts/download_benchmarks.sh --benchmark ruler，或从 ModelScope 还原上述目录。
相对路径以项目根目录为基准；已有长度目录不覆盖，另用 --output-dir 保存新设置。
model-name 取 --model-path 的最后一级目录名或 HF 仓库名。
HELP
}

fail() { printf '错误：%s\n' "$*" >&2; exit 2; }
require_value() {
  [[ $# -ge 2 && -n $2 && $2 != --* ]] || fail "$1 缺少参数值"
}
expand_user_path() {
  case "$1" in
    '~') printf '%s\n' "$HOME" ;;
    '~/'*) printf '%s/%s\n' "$HOME" "${1#\~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

ruler_source_dir=data/raw/ruler-v1-source
output_dir=""
model_path="$HOME/models/Qwen3-4B-Instruct-2507"
lengths=(8192 16384 32768)
num_samples=100
seed=42
prompt_format=chat
tasks=(all)
tokens_to_generate=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ruler-source-dir) require_value "$@"; ruler_source_dir="$2"; shift 2 ;;
    --output-dir) require_value "$@"; output_dir="$2"; shift 2 ;;
    --model-path) require_value "$@"; model_path="$2"; shift 2 ;;
    --prompt-format) require_value "$@"; prompt_format="$2"; shift 2 ;;
    --tokens-to-generate) require_value "$@"; tokens_to_generate="$2"; shift 2 ;;
    --num-samples) require_value "$@"; num_samples="$2"; shift 2 ;;
    --seed) require_value "$@"; seed="$2"; shift 2 ;;
    --tasks)
      shift
      tasks=()
      while [[ $# -gt 0 && $1 != --* ]]; do tasks+=("$1"); shift; done
      [[ ${#tasks[@]} -gt 0 ]] || fail '--tasks 至少需要一个任务'
      ;;
    --lengths)
      shift
      lengths=()
      while [[ $# -gt 0 && $1 != --* ]]; do lengths+=("$1"); shift; done
      [[ ${#lengths[@]} -gt 0 ]] || fail '--lengths 至少需要一个长度'
      ;;
    -h|--help) usage; exit 0 ;;
    *) fail "未知参数：$1；使用 --help 查看说明" ;;
  esac
done

[[ $num_samples =~ ^[1-9][0-9]*$ ]] || fail '--num-samples 必须为正整数'
[[ $seed =~ ^(0|[1-9][0-9]*)$ ]] || fail '--seed 必须为非负整数'
[[ $prompt_format == chat || $prompt_format == base ]] || fail '--prompt-format 必须为 chat 或 base'
[[ -z $tokens_to_generate || $tokens_to_generate =~ ^[1-9][0-9]*$ ]] || fail '--tokens-to-generate 必须为正整数'
for ((i=0; i<${#lengths[@]}; i++)); do
  [[ ${lengths[i]} =~ ^[1-9][0-9]*$ ]] || fail '--lengths 必须为正整数列表'
  for ((j=0; j<i; j++)); do
    [[ ${lengths[i]} != "${lengths[j]}" ]] || fail '--lengths 不能重复'
  done
done
ruler_source_dir="$(expand_user_path "$ruler_source_dir")"
model_path="$(expand_user_path "$model_path")"
if [[ -z $output_dir ]]; then
  output_dir="data/benchmarks/ruler-v1/$(basename "$model_path")"
fi
output_dir="$(expand_user_path "$output_dir")"
[[ $ruler_source_dir == /* ]] || ruler_source_dir="$LWM_ROOT/$ruler_source_dir"
[[ $output_dir == /* ]] || output_dir="$LWM_ROOT/$output_dir"
[[ -f "$ruler_source_dir/scripts/synthetic.yaml" ]] || fail 'RULER 源码不存在；请先执行 bash src/latent_working_memory/benchmarks/scripts/download_benchmarks.sh --benchmark ruler'
for length in "${lengths[@]}"; do
  [[ ! -e "$output_dir/$length" ]] || fail "RULER 长度目录已存在：$output_dir/$length；请更换 --output-dir"
done
mkdir -p "$output_dir"

export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="$LWM_ROOT/data/raw/nltk_data"
uv run --frozen python - \
  "$ruler_source_dir" "$output_dir" "$model_path" \
  "$num_samples" "$seed" "$prompt_format" "$tokens_to_generate" \
  "${#lengths[@]}" "${lengths[@]}" "${tasks[@]}" <<'PY'
from pathlib import Path
import subprocess
import sys

from transformers import AutoTokenizer
import yaml

from latent_working_memory.benchmarks.data import RULER_TASKS
from latent_working_memory.benchmarks import vendor

source = Path(sys.argv[1]) / "scripts" / "data"
destination = Path(sys.argv[2])
model_path = sys.argv[3]
num_samples, seed = map(int, sys.argv[4:6])
prompt_format = sys.argv[6]
generation_budget = int(sys.argv[7]) if sys.argv[7] else None
num_lengths = int(sys.argv[8])
lengths = [int(value) for value in sys.argv[9:9 + num_lengths]]
tasks = sys.argv[9 + num_lengths:]
if tasks == ["all"]:
    tasks = list(RULER_TASKS)
if len(tasks) != len(set(tasks)) or any(task not in RULER_TASKS for task in tasks):
    raise ValueError(f"--tasks must contain distinct names from {RULER_TASKS}")
sys.path.insert(0, str(source))
from synthetic.constants import TASKS
from template import Templates

with (source.parent / "synthetic.yaml").open() as stream:
    task_settings = yaml.safe_load(stream)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if prompt_format == "chat":
    chat_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    base_tokens = len(tokenizer.tokenize(Templates["base"]))
    overhead = len(chat_tokens) - base_tokens
else:
    # 官方生成器以 tokenize 计长；为实际编码时添加的 BOS 等特殊 token 预留空间。
    base_tokens = 0
    overhead = tokenizer.num_special_tokens_to_add(pair=False)

for length in lengths:
    for task in tasks:
        config = dict(task_settings[task])
        config.update(TASKS[config["task"]])
        budget = generation_budget if generation_budget is not None else config["tokens_to_generate"]
        if length <= overhead + base_tokens + budget:
            raise ValueError(f"{task}: length {length} leaves no input budget")
        task_template = Templates["base"].format(task_template=config["template"])
        task_template += config.get("answer_prefix", "")
        generator = source / "synthetic" / f"{config['task']}.py"
        if prompt_format == "base" and config["task"] == "niah":
            generator = Path(vendor.__file__).parent / "ruler_niah.py"
        command = [
            sys.executable, str(generator),
            "--save_dir", str(destination / str(length)), "--save_name", task,
            "--subset", "test", "--tokenizer_path", model_path, "--tokenizer_type", "hf",
            "--max_seq_length", str(length - overhead),
            "--tokens_to_generate", str(budget),
            "--num_samples", str(num_samples), "--random_seed", str(seed),
            "--model_template_token", str(base_tokens),
            "--template", task_template,
        ]
        if prompt_format == "base" and config["task"] == "niah":
            command.extend(("--ruler-source-dir", str(source.parent.parent)))
        for name, value in config["args"].items():
            command.extend((f"--{name}", str(value)))
        if config["task"] == "qa":
            command.extend(("--pre_samples", "0"))
        # 直接调用官方生成器，让失败返回非零退出码；避免 prepare.py 吞掉子进程异常。
        print(f"RULER {length} / {task}", flush=True)
        subprocess.run(command, check=True)
PY
printf 'RULER 数据已生成：%s\n' "$output_dir"
