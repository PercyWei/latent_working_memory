#!/usr/bin/env bash
set -euo pipefail

if [[ ${CUDA_VISIBLE_DEVICES:-1} != 1 ]]; then
  printf 'E²-TTT evaluation requires CUDA_VISIBLE_DEVICES=1.\n' >&2
  exit 2
fi
if [[ ${WORLD_SIZE:-1} != 1 ]]; then
  printf 'E²-TTT evaluation requires a single process.\n' >&2
  exit 2
fi
method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$method_dir/upstream:$method_dir/upstream/lm-evaluation-harness${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec python -u "$method_dir/scripts/evaluate.py" "$@"
