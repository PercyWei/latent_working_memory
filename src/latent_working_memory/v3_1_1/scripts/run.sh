#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"

case "${1:-}" in
  prepare-pre) module="latent_working_memory.v3_1_1.prepare_pre"; shift ;;
  prepare-post) module="latent_working_memory.v3_1_1.prepare_post"; shift ;;
  prepare-length) module="latent_working_memory.v3_1_1.prepare_length"; shift ;;
  *) module="latent_working_memory.v3_1_1.training.run" ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$project_root"
exec uv run --frozen python -m "$module" "$@"
