#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
case ${CUDA_VISIBLE_DEVICES-0} in
  0|1) export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0} ;;
  *)
    printf 'LaCT 复现仅允许单卡；CUDA_VISIBLE_DEVICES 必须为 0 或 1。\n' >&2
    exit 2
    ;;
esac
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$method_dir:$method_dir/upstream/lact_llm:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec python -u "$method_dir/scripts/evaluate.py" "$@"
