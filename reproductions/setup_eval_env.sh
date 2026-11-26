#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
用法：bash reproductions/setup_eval_env.sh <方法> [--env-root <目录>]

方法：icae、autocompressors、sdcc、lact、in_place_ttt、e2_ttt
--env-root  环境根目录，默认 ~；环境保存在 <env-root>/envs/<环境名>

uv 配置默认读取 ${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml，可用 UV_CONFIG_FILE 指定。
EOF
}
if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ $1 == --help || $1 == -h ]]; then
  usage
  exit 0
fi
method="$1"
shift
env_root="$HOME"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-root)
      if [[ $# -lt 2 || -z $2 || $2 == -* ]]; then
        printf '%s\n' '--env-root 需要目录参数。' >&2
        exit 2
      fi
      env_root="$2"
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) printf '未知参数：%s\n' "$1" >&2; exit 2 ;;
  esac
done
environment_name="$method"
case "$method" in
  lact|in_place_ttt|autocompressors|sdcc) python_minor=3.11 ;;
  e2_ttt) python_minor=3.12 ;;
  icae) python_minor=3.11; environment_name=icae-v2 ;;
  *) printf '未知方法：%s\n' "$method" >&2; exit 2 ;;
esac

uv_config_file="${UV_CONFIG_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml}"
if [[ ! -f $uv_config_file ]]; then
  printf 'uv 用户配置不存在：%s；请创建该文件或通过 UV_CONFIG_FILE 指定。\n' "$uv_config_file" >&2
  exit 2
fi

environment_dir="$env_root/envs/$environment_name"
wheel_dir="${XDG_CACHE_HOME:-$HOME/.cache}/latent-working-memory/wheels"
export UV_PYTHON_DOWNLOADS=never
if [[ ! -e $environment_dir ]]; then
  uv --config-file "$uv_config_file" venv --python "$python_minor" --seed "$environment_dir"
fi
"$environment_dir/bin/python" -c \
  'import sys; assert f"{sys.version_info.major}.{sys.version_info.minor}" == sys.argv[1]' \
  "$python_minor"

requirements_file="$project_root/reproductions/$method/requirements-eval.txt"
install_args=(-r "$requirements_file")
while IFS= read -r wheel_url; do
  wheel_name="${wheel_url##*/}"
  wheel_name="${wheel_name//%2B/+}"
  wheel_file="$wheel_dir/$wheel_name"
  if [[ ! -f $wheel_file && -f "$HOME/wheels/$wheel_name" ]]; then
    wheel_file="$HOME/wheels/$wheel_name"
  fi
  if [[ ! -f $wheel_file ]]; then
    mkdir -p "$wheel_dir"
    printf '下载 wheel：%s\n' "$wheel_name"
    partial_file="$(mktemp "$wheel_file.part.XXXXXX")"
    trap 'rm -f -- "$partial_file"' EXIT
    curl --fail --location --retry 3 --output "$partial_file" "$wheel_url"
    mv -- "$partial_file" "$wheel_file"
    trap - EXIT
  fi
  printf '使用 wheel：%s\n' "$wheel_file"
  install_args+=("$wheel_file")
done < <(sed -n 's/^# Official prebuilt wheel: //p' "$requirements_file")

if [[ $method == e2_ttt ]]; then
  install_args+=(-e "$project_root/reproductions/e2_ttt/source/e2_ttt"
                -e "$project_root/reproductions/e2_ttt/source/e2_ttt/lm-evaluation-harness")
fi
uv --config-file "$uv_config_file" pip install --python "$environment_dir/bin/python" \
  --only-binary flash-attn "${install_args[@]}"
uv --config-file "$uv_config_file" pip check --python "$environment_dir/bin/python"
printf '环境已安装：%s\n' "$environment_dir"
