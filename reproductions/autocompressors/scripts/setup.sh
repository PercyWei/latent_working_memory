#!/usr/bin/env bash
set -euo pipefail

method_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$method_dir/source/AutoCompressors"
revision="$(cat "$method_dir/source/revision")"
env_dir="$method_dir/.venv"

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --no-checkout https://github.com/princeton-nlp/AutoCompressors.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$revision" ]]; then
  printf '已有官方检出不在所需 revision %s，请检查 %s\n' "$revision" "$source_dir" >&2
  exit 1
fi
if [[ ! -d "$env_dir" ]]; then
  uv venv --python 3.11 "$env_dir"
fi
uv pip install --python "$env_dir/bin/python" -r "$method_dir/requirements-eval.txt"
uv pip check --python "$env_dir/bin/python"
