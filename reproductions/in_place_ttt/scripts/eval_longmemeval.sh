#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS-1}"
export PYTHONPATH="$project_root:$project_root/src:$method_dir/source/in_place_ttt${PYTHONPATH:+:$PYTHONPATH}"
cd "$project_root"
exec python -u -m reproductions.in_place_ttt.evaluation.eval_longmemeval "$@"
