#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
if [[ ${CUDA_VISIBLE_DEVICES:-1} != 1 ]]; then
  printf 'LaCT 复现仅允许物理 GPU 1；CUDA_VISIBLE_DEVICES 必须为 1。\n' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$method_dir/upstream/lact_llm:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec python -u "$method_dir/scripts/evaluate.py" "$@"
