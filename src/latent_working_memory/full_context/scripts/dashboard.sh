#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$project_root"

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec uv run --frozen python -m latent_working_memory.full_context.dashboard "$@"
fi

export STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS:-true}"
exec uv run --frozen streamlit run src/latent_working_memory/full_context/dashboard.py -- "$@"
