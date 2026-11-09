#!/usr/bin/env bash
set -euo pipefail

METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"
EVAL_GPU="${CUDA_VISIBLE_DEVICES-0}"
if [[ $EVAL_GPU != 0 && $EVAL_GPU != 1 ]]; then
  printf '%s\n' 'In-Place TTT evaluation requires a single physical GPU: CUDA_VISIBLE_DEVICES=0 or 1.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$METHOD_DIR/upstream:$METHOD_DIR/scripts:$METHOD_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec python -u "$METHOD_DIR/scripts/evaluate.py" "$@"
