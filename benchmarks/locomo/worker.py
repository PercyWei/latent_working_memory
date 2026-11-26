"""单张 GPU 的公共 LoCoMo worker。"""

import argparse
from importlib import import_module
import json
from pathlib import Path

from benchmarks.data import load_samples
from benchmarks.progress import QuestionTimer
from benchmarks.protocol import data_identity
from benchmarks.scoring import score_sample
from .engine import evaluate_conversations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads((args.directory / "evaluation_config.json").read_text())
    work = json.loads((args.directory / "work.json").read_text())
    samples = load_samples("locomo", plan["data"]["directory"])
    if data_identity(samples, plan["data"]["directory"], "locomo") != plan["data"]:
        raise ValueError("LoCoMo data changed after the evaluation was planned")
    by_id = {sample.id: sample for sample in samples}
    pending = [by_id[sample_id] for sample_id in work["workers"][args.rank]]
    factory = plan["adapter"]
    load_adapter = getattr(import_module(factory["module"]), factory["factory"])
    timer = QuestionTimer()
    adapter = load_adapter(plan["settings"])
    completed = 0
    path = args.directory / "progress" / f"worker-{args.rank}.jsonl"
    with path.open("a") as stream:
        def commit(sample, answer, statistics):
            nonlocal completed
            row = {
                **answer, "id": sample.id, "question": sample.metadata["question"],
                "metadata": sample.metadata, "references": list(sample.references),
                "score": score_sample("locomo", sample, answer["prediction"]),
            }
            if statistics:
                row["context"] = statistics
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            completed += 1
            print(f"LoCoMo worker={args.rank} {completed}/{len(pending)} {sample.id} | "
                  f"{timer.format_rate(completed)}", flush=True)
        evaluate_conversations(pending, adapter, work["batch_size_per_gpu"], commit)


if __name__ == "__main__":
    main()
