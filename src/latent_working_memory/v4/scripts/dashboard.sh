#!/usr/bin/env bash
set -euo pipefail

LWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$LWM_ROOT"

exec uv run --frozen streamlit run src/latent_working_memory/v4/dashboard.py -- "$@"
