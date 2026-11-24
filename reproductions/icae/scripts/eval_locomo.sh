#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
case "$CUDA_VISIBLE_DEVICES" in
  ''|*,*) printf 'LoCoMo 每次评估使用一张 GPU，请通过 CUDA_VISIBLE_DEVICES 选择一张卡。\n' >&2; exit 2 ;;
esac
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec python -u -m reproductions.icae.evaluation.eval_locomo "$@"
