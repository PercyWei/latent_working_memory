#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  printf '用法：bash prepare_cuda_wheels.sh [wheel目录]\n' >&2
  exit 2
fi

download_dir="${XDG_CACHE_HOME:-$HOME/.cache}/latent-working-memory/wheels"
if [[ $# -eq 1 ]]; then
  wheel_dirs=("$1")
  download_dir="$1"
else
  wheel_dirs=("$download_dir" "$HOME/wheels")
fi
for index in "${!wheel_dirs[@]}"; do
  if [[ "${wheel_dirs[$index]}" != /* ]]; then
    wheel_dirs[$index]="$PWD/${wheel_dirs[$index]}"
  fi
done
if [[ "$download_dir" != /* ]]; then
  download_dir="$PWD/$download_dir"
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_root"
mkdir -p -- "$project_root/.wheels"
for wheel_url in \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' \
  'https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.5/flashinfer_python-0.2.5+cu124torch2.6-cp38-abi3-linux_x86_64.whl'; do
  wheel_name="${wheel_url##*/}"
  target="$project_root/.wheels/$wheel_name"
  if [[ -f "$target" ]]; then
    printf '已准备 wheel：%s\n' "$target"
    continue
  fi
  wheel=""
  for wheel_dir in "${wheel_dirs[@]}"; do
    candidate="$wheel_dir/$wheel_name"
    if [[ -f "$candidate" ]]; then
      wheel="$candidate"
      break
    fi
  done
  if [[ -z "$wheel" ]]; then
    mkdir -p -- "$download_dir"
    wheel="$download_dir/$wheel_name"
    partial_file="$wheel.part"
    trap 'rm -f -- "$partial_file"' EXIT
    printf '下载 wheel：%s\n' "$wheel_name"
    curl --fail --location --retry 3 --output "$partial_file" "$wheel_url"
    mv -- "$partial_file" "$wheel"
    trap - EXIT
  fi
  if [[ ! -f "$target" ]]; then
    ln -sfn -- "$wheel" "$target"
  fi
  printf '准备本地 wheel：%s -> %s\n' "$target" "$wheel"
done

printf 'wheel 已准备，可运行 uv sync。\n'
