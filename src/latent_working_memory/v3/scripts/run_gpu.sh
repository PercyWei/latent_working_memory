#!/usr/bin/env bash
set -euo pipefail

LWM_REPO_DIR="/data/zhangdw12/percyw/latent_working_memory"
cd "$LWM_REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec uv run --frozen python -m latent_working_memory.v3.gpu_job "$@"
