"""在一张 GPU 上生成分配给本进程的 LongMemEval 回答。"""

import argparse
from importlib import import_module
import json
from pathlib import Path

from benchmarks.data import load_samples
from benchmarks.io import write_json
from benchmarks.progress import QuestionTimer
from benchmarks.protocol import data_identity
from .generate import generate_batches


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
    assigned = [samples[i] for i in args.indices]
    del samples
    factory = plan["adapter"]
    timer = QuestionTimer()
    adapter = getattr(import_module(factory["module"]), factory["factory"])(plan["settings"])
    completed = 0
    for batch in generate_batches(adapter, assigned, args.batch_size):
        for row in batch:
            index = args.indices[completed]
            write_json(args.directory / "progress" / f"{index}.json", row)
            completed += 1
        print(f"{benchmark} | {completed}/{len(assigned)} | {batch[-1]['question_id']} | "
              f"{timer.format_rate(completed)}", flush=True)


if __name__ == "__main__":
    main()
