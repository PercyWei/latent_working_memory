"""将复现结果与论文参照渲染为独立 HTML 图表。"""

import json
from pathlib import Path


def write_comparison(directory, data):
    template = Path(__file__).with_name("comparison_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output = directory / "comparison.html"
    output.write_text(template.replace("__REPORT_DATA__", payload), encoding="utf-8")
    return output
