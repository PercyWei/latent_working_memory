#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec python -u -m reproductions.sdcc.evaluation.eval_longmemeval "$@"
