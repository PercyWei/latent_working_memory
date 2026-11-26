#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd -- "$project_root"
export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS:-true}"

exec uv run --frozen streamlit run \
  src/latent_working_memory/v3_2/exploration/show_data_statistics.py -- "$@"
