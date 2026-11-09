#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
method="${1:?用法：bash reproductions/setup_eval_env.sh lact|e2_ttt|in_place_ttt}"
case "$method" in
  lact|in_place_ttt) python_minor=3.11 ;;
  e2_ttt) python_minor=3.12 ;;
  *) printf '未知方法：%s\n' "$method" >&2; exit 2 ;;
esac

environment_dir="$HOME/rivermind-data/envs/ttt-$method"
wheel_dir="$HOME/rivermind-data/wheels/ttt"
export UV_CACHE_DIR="$HOME/rivermind-data/cache/uv"
export UV_PYTHON_DOWNLOADS=never
if [[ ! -e $environment_dir ]]; then
  uv --no-config venv --python "$python_minor" --seed "$environment_dir"
fi
"$environment_dir/bin/python" -c \
  'import sys; assert f"{sys.version_info.major}.{sys.version_info.minor}" == sys.argv[1]' \
  "$python_minor"

install_args=(-r "$project_root/reproductions/$method/requirements-eval.txt")
if [[ $method == e2_ttt ]]; then
  install_args+=(-e "$project_root/reproductions/e2_ttt/upstream"
                -e "$project_root/reproductions/e2_ttt/upstream/lm-evaluation-harness")
fi
uv --no-config pip install --python "$environment_dir/bin/python" \
  --index-url https://pypi.org/simple --find-links "$wheel_dir" \
  --only-binary flash-attn "${install_args[@]}"
uv --no-config pip check --python "$environment_dir/bin/python"
printf '环境已安装：%s\n' "$environment_dir"
