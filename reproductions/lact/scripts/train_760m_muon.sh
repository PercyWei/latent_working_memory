#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$method_dir/../.." && pwd)"
: "${DATA_FILES:?Set DATA_FILES to the Long-Data-Collections Arrow file pattern}"
: "${TOKENIZER_PATH:?Set TOKENIZER_PATH to the original Mixstra tokenizer directory}"
output_dir="${OUTPUT_DIR:-$repo_root/artifacts/reproductions/lact/760m-muon-paper/train}"
output_dir=$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$output_dir")

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$method_dir/upstream/lact_llm:$method_dir/upstream_flame${PYTHONPATH:+:$PYTHONPATH}"
cd "$method_dir/upstream_flame"

exec torchrun --standalone --nproc_per_node=1 -m flame.train \
  --job.config_file "$method_dir/configs/760m_muon_paper_single_gpu.toml" \
  --job.dump_folder "$output_dir" \
  --model.tokenizer_path "$TOKENIZER_PATH" \
  --training.data_files "$DATA_FILES" \
  "$@"
