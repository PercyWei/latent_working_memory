"""记录当前仓库状态及 LaCT 实际模型类与推理代码路径。"""

import inspect
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REVISION = "a648340f9798f173227a0626fde66a6e9b65879a"
UPSTREAM_MODEL = "reproductions/lact/upstream/lact_llm/lact_model/modeling_lact.py"
UPSTREAM_LAYER = "reproductions/lact/upstream/lact_llm/lact_model/layer_lact_swiglu.py"
UPSTREAM_TTT = "reproductions/lact/upstream/lact_llm/lact_model/ttt_operation.py"
INCREMENTAL_MODEL = "reproductions/lact/incremental/model.py"
INCREMENTAL_TTT = "reproductions/lact/incremental/ttt.py"


def capture_source(entrypoint, model_classes):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    implementations = {}
    for mode, model_class in model_classes.items():
        model_path = str(Path(inspect.getfile(model_class)).resolve().relative_to(PROJECT_ROOT))
        paths = [model_path]
        if mode == "incremental":
            forward = f"{INCREMENTAL_MODEL}:IncrementalLaCTForCausalLM.forward"
            paths.extend((INCREMENTAL_MODEL, INCREMENTAL_TTT))
        else:
            forward = f"{UPSTREAM_MODEL}:LaCTForCausalLM.forward"
        paths.extend((UPSTREAM_MODEL, UPSTREAM_LAYER, UPSTREAM_TTT))
        implementations[mode] = {
            "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
            "forward": forward,
            "use_cache": mode == "incremental",
            "code_path": list(dict.fromkeys(paths)),
        }
    return {
        "local_git_commit": commit,
        "local_git_dirty": bool(status),
        "local_git_status": status,
        "entrypoint": str(Path(entrypoint).resolve().relative_to(PROJECT_ROOT)),
        "implementations": implementations,
    }
