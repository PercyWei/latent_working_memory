#!/usr/bin/env bash
# GPU 网站任务入口；uv 使用仓库根目录 .venv 和冻结的 uv.lock。
set -euo pipefail

LWM_REPO_DIR="${LWM_REPO_DIR:-/dfs/data/latent_working_memory}"
cd "$LWM_REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec uv run --frozen python -m latent_working_memory.v3.gpu_job "$@"
