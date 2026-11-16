"""按全局样本 ID 合并 In-Place TTT 各轮记录并执行原生 OpenCompass 评分。"""

import argparse
import json
from pathlib import Path

from in_place_ttt.scripts.records import collect_records, load_samples, merge_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.eval_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    protocol = json.loads((directory / "evaluation_plan.json").read_text())["protocol"]
    expected = load_samples(protocol)
    records, sources = collect_records(directory, protocol, expected)
    merge_records(directory, protocol, expected, records, sources, output)


if __name__ == "__main__":
    main()
