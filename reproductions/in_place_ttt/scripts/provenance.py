"""记录当前 Git 状态及 In-Place TTT 模型、缓存和 MLP 实现来源。"""

import inspect
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REVISION = "be2324829b0e91c8fd10a74d4b43714fde6676e1"
UPSTREAM_MODEL = "reproductions/in_place_ttt/upstream/inference_model/hf_qwen3/modeling_qwen3.py"
OPTIMIZED_MODEL = "reproductions/in_place_ttt/optimized/model.py"
RUNTIME = "reproductions/in_place_ttt/scripts/runtime.py"


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
        implementation_path = OPTIMIZED_MODEL if mode == "optimized" else UPSTREAM_MODEL
        implementations[mode] = {
            "model_class": f"{model_class.__module__}.{model_class.__qualname__}",
            "model_code": model_path,
            "cache_code": implementation_path,
            "mlp_code": implementation_path,
            "upstream_model_code": UPSTREAM_MODEL,
            "generation_code": (
                f"{RUNTIME}:generate_one; OpenCompass HuggingFaceBaseModel.generate"
                if mode == "original"
                else f"{RUNTIME}:generate_one"
            ),
            "code_path": list(
                dict.fromkeys((model_path, implementation_path, UPSTREAM_MODEL, RUNTIME))
            ),
        }
    return {
        "local_git_commit": commit,
        "local_git_dirty": bool(status),
        "local_git_status": status,
        "upstream_revision": UPSTREAM_REVISION,
        "entrypoint": str(Path(entrypoint).resolve().relative_to(PROJECT_ROOT)),
        "implementations": implementations,
    }
