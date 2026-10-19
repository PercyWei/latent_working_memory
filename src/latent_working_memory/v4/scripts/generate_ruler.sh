#!/usr/bin/env bash
set -euo pipefail

LWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$LWM_ROOT"

usage() {
  cat <<'HELP'
用法：bash src/latent_working_memory/v4/scripts/generate_ruler.sh [参数]

从已下载的 RULER v1 素材构造全部 13 项评估任务。

  --ruler-source-dir PATH  RULER 源码与素材，默认 data/raw/ruler-v1-source
  --output-dir PATH        评估数据目录，默认 data/benchmarks/ruler-v1/<model-name>
  --model-path PATH        tokenizer 来源，默认 ~/models/Qwen3-4B-Instruct-2507
  --lengths N [N ...]      输入与生成预算的总长度，默认 8192 16384 32768
  --num-samples N          每个长度、每项任务的样本数，默认 100
  --seed N                 随机种子，默认 42
  -h, --help               显示帮助

先下载 RULER 素材与 data/raw/nltk_data 中的 NLTK 资源；仅需 tokenizer 与 CPU。
可执行 download_benchmarks.sh --benchmark ruler，或从 ModelScope 还原上述目录。
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ruler-source-dir) require_value "$@"; ruler_source_dir="$2"; shift 2 ;;
    --output-dir) require_value "$@"; output_dir="$2"; shift 2 ;;
    --model-path) require_value "$@"; model_path="$2"; shift 2 ;;
    --num-samples) require_value "$@"; num_samples="$2"; shift 2 ;;
    --seed) require_value "$@"; seed="$2"; shift 2 ;;
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
[[ -f "$ruler_source_dir/scripts/synthetic.yaml" ]] || fail 'RULER 源码不存在；请先执行 download_benchmarks.sh --benchmark ruler'
for length in "${lengths[@]}"; do
  [[ ! -e "$output_dir/$length" ]] || fail "RULER 长度目录已存在：$output_dir/$length；请更换 --output-dir"
done
mkdir -p "$output_dir"

export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="$LWM_ROOT/data/raw/nltk_data"
uv run --frozen python - \
  "$ruler_source_dir" "$output_dir" "$model_path" \
  "$num_samples" "$seed" "${lengths[@]}" <<'PY'
from pathlib import Path
import subprocess
import sys

from transformers import AutoTokenizer
import yaml

from latent_working_memory.benchmarks.data import RULER_TASKS

source = Path(sys.argv[1]) / "scripts" / "data"
destination = Path(sys.argv[2])
model_path = sys.argv[3]
num_samples, seed = map(int, sys.argv[4:6])
lengths = [int(value) for value in sys.argv[6:]]
sys.path.insert(0, str(source))
from synthetic.constants import TASKS
from template import Templates

with (source.parent / "synthetic.yaml").open() as stream:
    task_settings = yaml.safe_load(stream)
tokenizer = AutoTokenizer.from_pretrained(model_path)
chat_tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": ""}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False,
)
base_tokens = len(tokenizer.tokenize(Templates["base"]))
overhead = len(chat_tokens) - base_tokens

for length in lengths:
    for task in RULER_TASKS:
        config = dict(task_settings[task])
        config.update(TASKS[config["task"]])
        if length <= len(chat_tokens) + config["tokens_to_generate"]:
            raise ValueError(f"{task}: length {length} leaves no input budget")
        task_template = Templates["base"].format(task_template=config["template"])
        task_template += config.get("answer_prefix", "")
        command = [
            sys.executable, str(source / "synthetic" / f"{config['task']}.py"),
            "--save_dir", str(destination / str(length)), "--save_name", task,
            "--subset", "test", "--tokenizer_path", model_path, "--tokenizer_type", "hf",
            "--max_seq_length", str(length - overhead),
            "--tokens_to_generate", str(config["tokens_to_generate"]),
            "--num_samples", str(num_samples), "--random_seed", str(seed),
            "--model_template_token", str(base_tokens),
            "--template", task_template,
        ]
        for name, value in config["args"].items():
            command.extend((f"--{name}", str(value)))
        if config["task"] == "qa":
            command.extend(("--pre_samples", "0"))
        # 直接调用官方生成器，让失败返回非零退出码；避免 prepare.py 吞掉子进程异常。
        print(f"RULER {length} / {task}", flush=True)
        subprocess.run(command, check=True)
PY
printf 'RULER 数据已生成：%s\n' "$output_dir"
