#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  printf '用法：bash scripts/sync_cuda_env.sh [wheel目录]\n' >&2
  exit 2
fi

if [[ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
  printf '此入口用于 Linux x86_64 的 torch2.6-cuda12.4 环境。\n' >&2
  exit 2
fi

wheel_dir="${1:-$HOME/wheels}"
if [[ "$wheel_dir" != /* ]]; then
  wheel_dir="$PWD/$wheel_dir"
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"

sync_args=(--frozen --inexact --no-group macos)
local_wheels=()
for package_wheel in \
  'flash-attn|flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' \
  'flashinfer-python|flashinfer_python-0.2.5+cu124torch2.6-cp38-abi3-linux_x86_64.whl'; do
  package="${package_wheel%%|*}"
  wheel="$wheel_dir/${package_wheel#*|}"
  if [[ -f "$wheel" ]]; then
    printf '%s：使用本地 wheel %s\n' "$package" "$wheel"
    sync_args+=(--no-install-package "$package")
    local_wheels+=("$wheel")
  else
    printf '%s：未找到本地 wheel，使用锁定依赖源\n' "$package"
  fi
done

uv sync "${sync_args[@]}"
if [[ ${#local_wheels[@]} -gt 0 ]]; then
  uv pip install --python "$project_root/.venv/bin/python" --offline --no-deps \
    "${local_wheels[@]}"
fi
