from __future__ import annotations

import ast
from pathlib import Path


def test_gradient_checkpoint_branch_forwards_enable_lora() -> None:
    source_path = (
        Path(__file__).resolve().parents[1] / "src" / "icae" / "base" / "modeling_llama_icae.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    custom_forward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "custom_forward"
    )

    assert any(
        isinstance(node, ast.Call)
        and any(keyword.arg == "enable_lora" for keyword in node.keywords)
        for node in ast.walk(custom_forward)
    )
