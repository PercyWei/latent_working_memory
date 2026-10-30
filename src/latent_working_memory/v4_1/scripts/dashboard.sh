#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$project_root"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec uv run --frozen python -m latent_working_memory.v4_1.dashboard "$@"
fi

export STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS:-true}"
exec uv run --frozen streamlit run src/latent_working_memory/v4_1/dashboard.py -- "$@"
