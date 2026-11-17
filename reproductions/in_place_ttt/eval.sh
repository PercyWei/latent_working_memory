#!/usr/bin/env bash
set -euo pipefail

METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"
EVAL_GPU="${CUDA_VISIBLE_DEVICES-0}"
if [[ $EVAL_GPU != 0 && $EVAL_GPU != 1 && $EVAL_GPU != 0,1 && $EVAL_GPU != 1,0 ]]; then
  printf '%s\n' 'CUDA_VISIBLE_DEVICES must be 0, 1, 0,1 or 1,0.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT/reproductions:$METHOD_DIR/source/in_place_ttt:$METHOD_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec python -u -m in_place_ttt.scripts.evaluate_parallel "$@"
