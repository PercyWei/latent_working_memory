#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$project_root"
exec uv run --frozen python -m latent_working_memory.v3_1_1.preparation.pretrain "$@"
