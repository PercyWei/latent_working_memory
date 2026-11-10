#!/usr/bin/env bash
set -euo pipefail

eval_gpu="${CUDA_VISIBLE_DEVICES:-0}"
if [[ $eval_gpu != 0 && $eval_gpu != 1 && $eval_gpu != 0,1 && $eval_gpu != 1,0 ]]; then
  printf 'CUDA_VISIBLE_DEVICES must be 0, 1, 0,1 or 1,0.\n' >&2
  exit 2
fi
if [[ ${WORLD_SIZE:-1} != 1 || ${LOCAL_RANK+x} || ${RANK+x} ]]; then
  printf 'E²-TTT evaluation requires one launcher process.\n' >&2
  exit 2
fi
method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$eval_gpu"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$method_dir:$method_dir/upstream:$method_dir/upstream/lm-evaluation-harness${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
if [[ $CUDA_VISIBLE_DEVICES == *,* ]]; then
  exec bash "$method_dir/eval_parallel.sh" "$@"
fi
exec python -u "$method_dir/scripts/evaluate.py" "$@"
