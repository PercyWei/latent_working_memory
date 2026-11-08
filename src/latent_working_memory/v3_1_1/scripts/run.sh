#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  prepare-pre) module="latent_working_memory.v3_1_1.prepare_pre"; shift ;;
  prepare-post) module="latent_working_memory.v3_1_1.prepare_post"; shift ;;
  *) module="latent_working_memory.v3_1_1.training.run" ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$project_root"
exec uv run --frozen python -m "$module" "$@"
