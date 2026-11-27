"""使用原生完整上下文和 greedy 解码，离线评估 RULER。"""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import torch
from transformers import AutoConfig

from benchmarks.comparison import comparison_data
from benchmarks.data import load_samples
from benchmarks.output import add_output_arguments, evaluation_directory
from benchmarks.protocol import DEFAULT_RULER_LENGTHS, data_identity, protocol
from benchmarks.scoring import summarize_scores
from latent_working_memory.cli import length_batch_size, positive_int
from latent_working_memory.devices import visible_devices
from latent_working_memory.v4.io import write_json

from .parallel import evaluate_parallel, evaluation_log, load_progress, shard_samples
from .model import ATTENTION_IMPLEMENTATIONS, WEIGHT_DTYPES, validate_compute


METHOD = "full-context"


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _matching(actual, expected, path):
    if actual != expected:
        changed = sorted(
            name
            for name in actual.keys() | expected.keys()
            if actual.get(name) != expected.get(name)
        )
        raise ValueError(
            f"baseline evaluation configuration differs at {path}: {', '.join(changed)}"
        )


def _print_scores(summary):
    metrics = summary["metrics"]
    for row in metrics["by_length"]:
        print(f"RULER {row['length']} | accuracy={row['score']:.4f} | samples={row['samples']}")
    print(
        f"{summary['benchmark']} overall | {metrics['metric']}={metrics['score']:.4f} "
        f"| samples={summary['samples']}",
        flush=True,
    )


def _prepare_evaluation(args, model_path, output):
    settings_path = output / "eval.json"
    if args.resume:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        data_dir = Path(settings["data"]["directory"])
        seed = settings["seed"]
        model_config = dict(settings["model"])
        if args.model_path is None:
            model_path = Path(model_config["model_name_or_path"])
        lengths = settings["protocol"]["lengths"]
    else:
        if args.benchmark_data is None:
            raise ValueError("specify --benchmark-data when creating an evaluation")
        data_dir = args.benchmark_data
        seed = 42 if args.seed is None else args.seed
        model_config = {
            "backbone_dtype": WEIGHT_DTYPES[args.dtype or "bf16"],
            "attention_implementation": args.attention_implementation or "flash_attention_2",
            "mixed_precision": args.mixed_precision,
        }
        lengths = args.lengths or DEFAULT_RULER_LENGTHS
    unplanned = args.batch_size_by_length.keys() - set(lengths)
    if unplanned:
        raise ValueError(f"--batch-size-by-length includes unselected lengths: {sorted(unplanned)}")
    model_config["model_name_or_path"] = str(model_path)
    validate_compute(model_config)
    samples = load_samples(args.benchmark, data_dir, tuple(lengths))
    model_window = AutoConfig.from_pretrained(model_path).max_position_embeddings
    snapshot = {
        "method": METHOD,
        "seed": seed,
        "model": model_config,
        "benchmark": args.benchmark,
        "split": "test",
        "model_window": model_window,
        "data": data_identity(samples, data_dir, args.benchmark),
        "protocol": protocol(
            args.benchmark,
            lengths,
            "full prompt remains in the native KV cache; no memory compression",
            use_qa_system=True,
        ),
    }
    if args.resume:
        _matching(settings, snapshot, settings_path)
    return samples, snapshot


def _evaluate(samples, snapshot, devices, output, shards, execution):
    records, resources = evaluate_parallel(samples, snapshot, devices, output, shards, execution)
    benchmark = snapshot["benchmark"]
    summary = {
        **snapshot,
        "comparison_data": comparison_data(samples, benchmark),
        "samples": len(records),
        "metrics": summarize_scores(benchmark, records),
        "input_tokens": sum(row["prompt_tokens"] for row in records),
        "generated_tokens": sum(row["generated_tokens"] for row in records),
        "resources": {
            **resources,
            **execution,
            "batch_sizes": sorted({row["eval_batch_size"] for row in records}),
            "batch_unit": "questions",
            "decode_batch_sizes": sorted({row["decode_batch_size"] for row in records}),
        },
    }
    _print_scores(summary)
    return summary


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("launch the evaluation script directly; it starts one worker per GPU")
    model_path = args.model_path or Path("~/models/Qwen3-4B-Instruct-2507")
    model_path = model_path.expanduser().resolve()
    output = evaluation_directory(args, args.benchmark, output_name="", checkpoint_name=model_path.name)
    if args.resume:
        if not (output / "eval.json").is_file():
            raise ValueError(f"--resume requires an existing evaluation plan: {output / 'eval.json'}")
    else:
        samples, snapshot = _prepare_evaluation(args, model_path, output)
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.resume:
            samples, snapshot = _prepare_evaluation(args, model_path, output)
        else:
            write_json(output / "eval.json", snapshot)
        summary_path = output / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _matching({key: summary[key] for key in snapshot}, snapshot, summary_path)
            with (output / "details.jsonl").open(encoding="utf-8") as stream:
                completed_ids = [json.loads(line)["id"] for line in stream]
            if completed_ids != [sample.id for sample in samples]:
                raise ValueError("completed evaluation details differ from the planned sample order")
            print(f"{args.benchmark} reusing cached summary: {summary_path}", flush=True)
            _print_scores(summary)
            return summary
        completed = set(load_progress(output, samples, repair=True))
        pending = [sample for sample in samples if sample.id not in completed]
        execution = {"batch_size_per_gpu": args.batch_size_per_gpu,
                     "batch_size_by_length": args.batch_size_by_length}
        status = {
            "method": METHOD,
            "benchmark": args.benchmark,
            "status": "running",
            "gpus": "",
            "cached_samples": len(completed),
            "scheduled_samples": len(pending),
            "device": "cuda",
            "execution": execution,
            "command": [sys.executable, *sys.argv],
            "started_at": _now(),
        }
        with evaluation_log(output / "eval.log"):
            try:
                devices, shards = [], []
                if pending:
                    devices = visible_devices()
                    available = torch.cuda.device_count()
                    if available != len(devices):
                        raise RuntimeError(
                            f"CUDA_VISIBLE_DEVICES={','.join(devices)} selects {len(devices)} GPU(s), "
                            f"but PyTorch detects {available}"
                        )
                    shards = shard_samples(pending, len(devices))
                    devices = devices[:len(shards)]
                status["gpus"] = ",".join(devices)
                write_json(output / "status.json", status)
                print(f"{args.benchmark} | {len(completed)} cached | {len(pending)} pending", flush=True)
                summary = _evaluate(samples, snapshot, devices, output, shards, execution)
                status.update(status="finished", compute_status="finished", finished_at=_now(), return_code=0)
                write_json(output / "status.json", status)
                write_json(summary_path, summary)
            except BaseException as error:
                return_code = 130 if isinstance(error, KeyboardInterrupt) else 1
                if isinstance(error, SystemExit) and isinstance(error.code, int):
                    return_code = error.code
                status.update(
                    status="failed",
                    compute_status="failed",
                    error=f"{type(error).__name__}: {error}",
                    finished_at=_now(),
                    return_code=return_code,
                )
                write_json(output / "status.json", status)
                print(status["error"], file=sys.stderr, flush=True)
                raise
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    epilog="使用 CUDA_VISIBLE_DEVICES 指定物理 GPU，例如 0 或 0,1。")
    parser.set_defaults(benchmark="ruler")
    parser.add_argument("--resume", action="store_true", help="沿用原评估计划，只补评未完成的题目")
    parser.add_argument("--benchmark-data", type=Path, help="新建评估时必填，续跑沿用原数据")
    parser.add_argument("--model-path", type=Path,
                        help="基座模型目录，默认 ~/models/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--dtype", choices=tuple(WEIGHT_DTYPES), help="基座权重精度，首次默认 bf16",
    )
    parser.add_argument(
        "--attention-implementation", choices=ATTENTION_IMPLEMENTATIONS, help="注意力实现，首次默认 flash_attention_2",
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="为 FP32 权重启用 PyTorch BF16 autocast，首次默认关闭",
    )
    parser.add_argument("--batch-size-per-gpu", type=positive_int, default=1,
                        help="每卡评估批量大小，默认 1")
    parser.add_argument("--seed", type=int, help="首次默认 42，复用时沿用已存配置")
    root = Path(__file__).resolve().parents[3]
    add_output_arguments(parser, root / "artifacts/baselines/full_context", checkpoint_layout=True)
    parser.add_argument("--lengths", type=positive_int, nargs="+", help="首次默认 8192 16384 32768")
    parser.add_argument("--batch-size-by-length", type=length_batch_size, nargs="+", default=[],
                       metavar="LENGTH=BATCH", help="按 RULER 长度档位覆盖每卡批量，如 8192=4 32768=1")
    args = parser.parse_args(argv)
    if args.resume:
        fixed = ("benchmark_data", "dtype", "attention_implementation", "seed", "lengths")
        ignored = [name for name in fixed if getattr(args, name, None) is not None]
        if args.mixed_precision:
            ignored.append("mixed_precision")
        if ignored:
            flags = ", ".join("--" + name.replace("_", "-") for name in ignored)
            print(f"[resume] ignored options; using the saved plan: {flags}", file=sys.stderr)
    elif args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    pairs = args.batch_size_by_length
    args.batch_size_by_length = dict(pairs)
    if len(args.batch_size_by_length) != len(pairs):
        parser.error("--batch-size-by-length must not repeat a length")
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
