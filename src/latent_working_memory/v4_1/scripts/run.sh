#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash src/latent_working_memory/v4_1/scripts/run.sh ACTION [--gpu 0|1] [module arguments]

Actions: prepare, train, evaluate, report
The script runs from the repository root with uv run --frozen.
--gpu selects a physical GPU; the Python process sees it as cuda:0.
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

gpu=0
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      gpu="${2:?--gpu requires 0 or 1}"
      shift 2
      ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ "$gpu" != "0" && "$gpu" != "1" ]]; then
  printf 'Only physical GPU 0 or 1 is allowed: %s\n' "$gpu" >&2
  exit 2
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd -- "$project_root"
export CUDA_VISIBLE_DEVICES="$gpu"
if [[ ${#args[@]} -eq 0 ]]; then
  exec uv run --frozen python -m "$module"
fi
exec uv run --frozen python -m "$module" "${args[@]}"
