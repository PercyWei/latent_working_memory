from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CDIC_ROOT = ROOT / "reproductions" / "cdic"


def test_required_cdic_sources_are_present() -> None:
    required = {
        "README.md",
        "UPSTREAM.md",
        "ASSUMPTIONS.md",
        "configs/paper.yaml",
        "src/cdic_repro/config.py",
        "src/cdic_repro/checkpoint.py",
        "src/cdic_repro/credit.py",
        "src/cdic_repro/engine.py",
        "src/cdic_repro/icae_adapter.py",
        "src/cdic_repro/memory_state.py",
        "src/cdic_repro/model_protocol.py",
        "src/cdic_repro/retrieval.py",
        "src/cdic_repro/run_dialogue.py",
        "src/cdic_repro/trace.py",
        "src/cdic_repro/writeback.py",
    }
    missing = [path for path in sorted(required) if not (CDIC_ROOT / path).is_file()]
    assert not missing, f"Missing C-DIC reproduction files: {missing}"


def test_cdic_python_sources_parse() -> None:
    for source in CDIC_ROOT.rglob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_cdic_tree_contains_no_downloaded_artifacts() -> None:
    forbidden_suffixes = {".pt", ".pth", ".bin", ".safetensors", ".jsonl"}
    forbidden = [
        path.relative_to(CDIC_ROOT).as_posix()
        for path in CDIC_ROOT.rglob("*")
        if path.is_file() and path.suffix in forbidden_suffixes
    ]
    assert not forbidden, f"Unexpected model/data artifacts: {forbidden}"
