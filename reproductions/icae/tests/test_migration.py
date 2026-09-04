from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "icae"


def test_all_migrated_sources_parse() -> None:
    for source in SOURCE_ROOT.rglob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_default_memory_size_is_128() -> None:
    model_source = (SOURCE_ROOT / "llama_icae_modeling.py").read_text(encoding="utf-8")
    assert "default=128" in model_source


def test_transformers_version_is_pinned() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"transformers==4.31.0"' in pyproject
    assert '"torch==2.0.1"' in pyproject
