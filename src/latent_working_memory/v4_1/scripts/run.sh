#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  prepare) module="latent_working_memory.v4_1.teacher.prepare"; shift ;;
  *) module="latent_working_memory.v4_1.training.run" ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$project_root"
exec uv run --frozen python -m "$module" "$@"
