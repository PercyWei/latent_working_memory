#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd -- "$project_root"
export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec python -m reproductions.e2_ttt.evaluation.dashboard "$@"
fi

export STREAMLIT_SERVER_HEADLESS="${STREAMLIT_SERVER_HEADLESS:-true}"
exec python -m streamlit run reproductions/e2_ttt/evaluation/dashboard.py -- "$@"
