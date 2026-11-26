#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1}"
export TOKENIZERS_PARALLELISM=false
module=reproductions.icae.source.icae.code.icae_v2.instruction_finetune
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  exec python -m "$module" "$@"
fi
exec python -m torch.distributed.run --standalone --nproc-per-node=gpu --module "$module" "$@"
