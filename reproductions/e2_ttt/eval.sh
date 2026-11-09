#!/usr/bin/env bash
set -euo pipefail

eval_gpu="${CUDA_VISIBLE_DEVICES:-0}"
if [[ $eval_gpu != 0 && $eval_gpu != 1 ]]; then
  printf 'E²-TTT evaluation requires one physical GPU: CUDA_VISIBLE_DEVICES=0 or 1.\n' >&2
  exit 2
fi
if [[ ${WORLD_SIZE:-1} != 1 ]]; then
  printf 'E²-TTT evaluation requires a single process.\n' >&2
  exit 2
fi
method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$eval_gpu"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$method_dir/upstream:$method_dir/upstream/lm-evaluation-harness${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec python -u "$method_dir/scripts/evaluate.py" "$@"
