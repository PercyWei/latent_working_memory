from __future__ import annotations

import sys
from pathlib import Path

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("cdic")
    group.addoption(
        "--cdic-gpu-config",
        action="store",
        default=None,
        help="Path to the JSON configuration for the real C-DIC GPU smoke test.",
    )
    group.addoption(
        "--cdic-msc-eval-config",
        action="store",
        default=None,
        help="Path to the JSON configuration for held-out MSC GPU evaluation.",
    )
