#!/usr/bin/env bash
set -euo pipefail

LWM_REPO_DIR="$(
  .venv/bin/python - .env <<'PY'
from pathlib import Path
import sys

from dotenv import dotenv_values

repo_dir = dotenv_values(sys.argv[1], interpolate=False).get("LWM_REPO_DIR")
if not repo_dir or not Path(repo_dir).is_absolute():
    raise ValueError("LWM_REPO_DIR in project .env must be a nonempty absolute path")
print(repo_dir)
PY
)"
cd "$LWM_REPO_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec uv run --frozen python -m latent_working_memory.v3.gpu_job "$@"
