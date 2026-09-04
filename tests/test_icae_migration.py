from __future__ import annotations

import ast
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ICAE_ROOT = ROOT / "reproductions" / "icae"


def test_required_icae_sources_are_present() -> None:
    required = {
        "src/icae/llama_icae_modeling.py",
        "src/icae/llama_icae_learning.py",
        "src/icae/base/modeling_llama_icae.py",
        "src/icae/utils/stable_trainer.py",
        "vendor/peft/src/peft/tuners/lora.py",
        "LICENSE",
        "UPSTREAM.md",
    }
    missing = [path for path in sorted(required) if not (ICAE_ROOT / path).is_file()]
    assert not missing, f"Missing migrated ICAE files: {missing}"


def test_migrated_python_sources_parse() -> None:
    for source in ICAE_ROOT.rglob("*.py"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_no_models_or_datasets_were_migrated() -> None:
    forbidden_suffixes = {".pt", ".pth", ".bin", ".safetensors", ".jsonl"}
    forbidden = [
        path.relative_to(ICAE_ROOT).as_posix()
        for path in ICAE_ROOT.rglob("*")
        if path.is_file() and path.suffix in forbidden_suffixes
    ]
    assert not forbidden, f"Unexpected model/data artifacts: {forbidden}"
