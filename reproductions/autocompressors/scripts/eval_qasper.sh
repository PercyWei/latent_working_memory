#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project_root="$(cd "$method_dir/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$project_root:$project_root/src:$method_dir/source/AutoCompressors${PYTHONPATH:+:$PYTHONPATH}"

cd "$project_root"
exec python -u -m reproductions.autocompressors.evaluation.eval_qasper "$@"
