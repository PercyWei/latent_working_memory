from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

SOURCE_ROOT = Path(__file__).parents[1] / "src"
PACKAGE_ROOT = SOURCE_ROOT / "cdic_repro"

# 第一阶段尚未替换 cdic_repro 顶层的旧 ICAE 导入，因此测试只装载新的独立子包。
cdic_repro = ModuleType("cdic_repro")
cdic_repro.__path__ = [str(PACKAGE_ROOT)]
sys.modules["cdic_repro"] = cdic_repro
