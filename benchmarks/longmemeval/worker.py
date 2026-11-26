"""在一张 GPU 上生成分配给本进程的 LongMemEval 回答。"""

import argparse
from importlib import import_module
import json
from pathlib import Path

from benchmarks.data import load_samples
from benchmarks.protocol import data_identity
from .parallel import generate_assigned, history_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.directory / "evaluation_config.json").read_text(encoding="utf-8"))
    benchmark, data_dir = plan["benchmark"], plan["data"]["directory"]
    samples = load_samples(benchmark, data_dir)
    if data_identity(samples, data_dir, benchmark) != plan["data"]:
        raise ValueError("LongMemEval data changed after the generation run was planned")
    assigned = {i: samples[i] for i in args.indices}
    del samples
    groups = history_groups(assigned, args.indices)
    factory = plan["adapter"]
    load_adapter = getattr(import_module(factory["module"]), factory["factory"])
    generate_assigned(args.directory, assigned, groups, args.batch_size, load_adapter, plan["settings"])


if __name__ == "__main__":
    main()
