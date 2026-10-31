#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(git -C "$script_dir" rev-parse --show-toplevel)
cd "$project_root"
module=$(.venv/bin/python - "$script_dir" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1]).parent.relative_to(Path.cwd())
parts = path.parts[1:] if path.parts[0] == "src" else path.parts
print(".".join((*parts, "run")))
PY
)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec .venv/bin/python -m "$module" "$@"
