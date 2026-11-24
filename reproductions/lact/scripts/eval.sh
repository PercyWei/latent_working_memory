#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
case ${CUDA_VISIBLE_DEVICES-0} in
  0|1|0,1|1,0) export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0} ;;
  *)
    printf 'CUDA_VISIBLE_DEVICES 必须为 0、1、0,1 或 1,0。\n' >&2
    exit 2
    ;;
esac
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$project_root:$project_root/reproductions:$method_dir:$method_dir/source/lact/lact_llm:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec python -u "$method_dir/evaluation/evaluate_parallel.py" "$@"
