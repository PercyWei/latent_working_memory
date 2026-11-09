#!/usr/bin/env bash
set -euo pipefail

METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$METHOD_DIR/../.." && pwd)"
if [[ ${CUDA_VISIBLE_DEVICES-1} != 1 ]]; then
  printf '%s\n' 'In-Place TTT evaluation requires physical GPU 1: CUDA_VISIBLE_DEVICES=1.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$METHOD_DIR/upstream:$METHOD_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec python -u "$METHOD_DIR/scripts/evaluate.py" "$@"
