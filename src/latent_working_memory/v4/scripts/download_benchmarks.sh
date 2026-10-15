#!/usr/bin/env bash
set -euo pipefail

LWM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$LWM_ROOT"

fail() { printf '错误：%s\n' "$*" >&2; exit 2; }
require_value() {
  [[ $# -ge 2 && -n $2 && $2 != --* ]] || fail "$1 缺少参数值"
}
expand_user_path() {
  case "$1" in
    '~') printf '%s\n' "$HOME" ;;
    '~/'*) printf '%s/%s\n' "$HOME" "${1#\~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

benchmark=all
data_root=data/benchmarks
ruler_source_dir=data/raw/ruler-v1-source
hf_endpoint="${HF_ENDPOINT:-https://hf-mirror.com}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) require_value "$@"; benchmark="$2"; shift 2 ;;
    --data-root) require_value "$@"; data_root="$2"; shift 2 ;;
    --ruler-source-dir) require_value "$@"; ruler_source_dir="$2"; shift 2 ;;
    *) fail "未知参数：$1" ;;
  esac
done

case "$benchmark" in all|ruler|locomo) ;; *) fail '--benchmark 只能为 all、ruler 或 locomo' ;; esac
data_root="$(expand_user_path "$data_root")"
ruler_source_dir="$(expand_user_path "$ruler_source_dir")"
[[ $data_root == /* ]] || data_root="$LWM_ROOT/$data_root"
[[ $ruler_source_dir == /* ]] || ruler_source_dir="$LWM_ROOT/$ruler_source_dir"

download_file() {
  local url="$1" destination="$2"
  if [[ -f $destination ]]; then
    printf '复用：%s\n' "$destination"
  else
    curl -fL --retry 3 "$url" -o "$destination.tmp"
    mv "$destination.tmp" "$destination"
  fi
}

if [[ $benchmark == all || $benchmark == ruler ]]; then
  mkdir -p "$(dirname "$ruler_source_dir")"
  if [[ ! -d $ruler_source_dir ]]; then
    git clone --no-checkout https://github.com/NVIDIA/RULER.git "$ruler_source_dir"
  fi
  git -C "$ruler_source_dir" lfs install --local
  GIT_LFS_SKIP_SMUDGE=1 git -C "$ruler_source_dir" checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
  git -C "$ruler_source_dir" lfs pull --include="scripts/data/synthetic/json/english_words.json"
  # 仅为 NLTK 下载允许使用现有代理，调用方可通过同名环境变量覆盖。
  NLTK_ALLOW_PROXIED_URLOPEN="${NLTK_ALLOW_PROXIED_URLOPEN:-1}" \
    uv run --frozen python - <<'PY'
import nltk

for package in ("punkt", "punkt_tab"):
    nltk.download(package, raise_on_error=True)
PY
  (
    cd "$ruler_source_dir/scripts/data/synthetic/json"
    if [[ ! -f PaulGrahamEssays.json ]]; then
      uv run --frozen --project "$LWM_ROOT" python download_paulgraham_essay.py
    fi
    download_file https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json squad.json
    download_file "${hf_endpoint%/}/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json" hotpotqa.json
  )
  printf 'RULER 素材已下载：%s\n' "$ruler_source_dir"
fi

if [[ $benchmark == all || $benchmark == locomo ]]; then
  mkdir -p "$data_root/locomo"
  download_file https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json "$data_root/locomo/locomo10.json"
  printf 'LoCoMo 数据已准备：%s/locomo/locomo10.json\n' "$data_root"
fi
