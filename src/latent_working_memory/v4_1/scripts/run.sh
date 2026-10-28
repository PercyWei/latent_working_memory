#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash src/latent_working_memory/v4_1/scripts/run.sh ACTION [module arguments]

Actions: prepare, train, evaluate, report
The script runs from the repository root with uv run --frozen.
Set CUDA_VISIBLE_DEVICES=0 or 1 to select the GPU before running GPU actions.
Use ACTION --help to inspect that module's arguments.
EOF
}

if [[ $# -eq 0 || "$1" == "--help" ]]; then
  usage
  exit 0
fi

action="$1"
shift
case "$action" in
  prepare) module="latent_working_memory.v4_1.teacher.prepare" ;;
  train) module="latent_working_memory.v4_1.training.train" ;;
  evaluate) module="latent_working_memory.v4_1.evaluation.evaluate" ;;
  report) module="latent_working_memory.v4_1.evaluation.report" ;;
  *) printf 'Unknown action: %s\n' "$action" >&2; usage >&2; exit 2 ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$project_root"
exec uv run --frozen python -m "$module" "$@"
