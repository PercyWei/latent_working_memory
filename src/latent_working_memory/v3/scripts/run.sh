#!/usr/bin/env bash
set -euo pipefail

LWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$LWM_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec uv run --frozen python -m latent_working_memory.v3.gpu_job "$@"
