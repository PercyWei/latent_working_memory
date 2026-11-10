#!/usr/bin/env bash
set -euo pipefail

eval_gpu="${CUDA_VISIBLE_DEVICES:-0}"
if [[ $eval_gpu != 0 && $eval_gpu != 1 ]]; then
  printf 'E²-TTT parallel evaluation requires one physical GPU: CUDA_VISIBLE_DEVICES=0 or 1.\n' >&2
  exit 2
fi
if [[ ${WORLD_SIZE:-1} != 1 ]]; then
  printf 'E²-TTT parallel evaluation requires one launcher process.\n' >&2
  exit 2
fi
method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="$eval_gpu"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
cd "$project_root"
exec python -u "$method_dir/scripts/evaluate_parallel.py" "$@"
