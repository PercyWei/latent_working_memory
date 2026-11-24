"""使用原生完整上下文和 greedy 解码，离线评估 RULER／LoCoMo。"""

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
from benchmarks.protocol import DEFAULT_RULER_LENGTHS, data_identity, protocol
from benchmarks.scoring import summarize_scores
from latent_working_memory.cli import length_batch_size, positive_int
from latent_working_memory.devices import visible_devices
from latent_working_memory.v4.io import write_json

from .parallel import evaluate_parallel, evaluation_log, load_progress, shard_samples


METHOD = "full-context"
WEIGHT_DTYPES = {"fp32": "float32", "bf16": "bfloat16"}
ATTENTION_IMPLEMENTATIONS = ("sdpa", "flash_attention_2")


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
    if summary["benchmark"] == "ruler":
        for row in metrics["by_length"]:
            print(f"RULER {row['length']} | accuracy={row['score']:.4f} | samples={row['samples']}")
    else:
        for row in metrics["groups"]:
            print(
                f"LoCoMo {row['category_name']} | f1={row['score']:.4f} | samples={row['samples']}"
            )
    print(
        f"{summary['benchmark']} overall | {metrics['metric']}={metrics['score']:.4f} "
        f"| samples={summary['samples']}",
        flush=True,
    )


def _prepare_evaluation(args, model_path, output):
    settings_path = output / "eval.json"
    if args.resume:
        if not settings_path.is_file():
            raise ValueError(f"--resume requires an existing evaluation plan: {settings_path}")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        data_dir = Path(settings["data"]["directory"])
        seed = settings["seed"]
        model_config = dict(settings["model"])
        if args.model_path is None:
            model_path = Path(model_config["model_name_or_path"])
        lengths = settings["protocol"].get("lengths", ())
    else:
        if settings_path.exists():
            raise ValueError(f"evaluation plan already exists: {settings_path}; use --resume")
        if any(path.name != ".evaluation.lock" for path in output.iterdir()):
            raise ValueError(f"evaluation output is not empty and has no eval.json: {output}")
        if args.benchmark_data is None:
            raise ValueError("specify --benchmark-data when creating an evaluation")
        data_dir = args.benchmark_data
        seed = 42 if args.seed is None else args.seed
        model_config = {
            "backbone_dtype": WEIGHT_DTYPES[args.dtype or "fp32"],
            "attention_implementation": args.attention_implementation or "sdpa",
            "mixed_precision": args.mixed_precision,
        }
        lengths = (args.lengths or DEFAULT_RULER_LENGTHS) if args.benchmark == "ruler" else ()
    if args.benchmark == "ruler":
        unplanned = args.batch_size_by_length.keys() - set(lengths)
        if unplanned:
            raise ValueError(f"--batch-size-by-length includes unselected lengths: {sorted(unplanned)}")
    model_config["model_name_or_path"] = str(model_path)
    if model_config["backbone_dtype"] not in WEIGHT_DTYPES.values():
        raise ValueError(f"{settings_path}: unsupported model.backbone_dtype")
    if model_config["attention_implementation"] not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError("attention implementation must be sdpa or flash_attention_2")
    amp = model_config.get("mixed_precision")
    if type(amp) is not bool:
        raise ValueError(f"{settings_path}: model.mixed_precision must be true or false")
    if amp and model_config["backbone_dtype"] != "float32":
        raise ValueError("--mixed-precision requires --dtype fp32")
    if (
        model_config["attention_implementation"] == "flash_attention_2"
        and model_config["backbone_dtype"] == "float32"
        and not amp
    ):
        raise ValueError("flash_attention_2 with --dtype fp32 requires --mixed-precision")
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
    if args.benchmark == "locomo":
        snapshot["protocol"]["memory_sharing"] = (
            "prefill each conversation's shared chat-token prefix once; answer questions from independent KV branches"
        )
    if args.resume:
        _matching(settings, snapshot, settings_path)
    else:
        write_json(settings_path, snapshot)
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
            "batch_unit": "conversations" if benchmark == "locomo" else "questions",
            "decode_batch_sizes": sorted({row["decode_batch_size"] for row in records}),
        },
    }
    _print_scores(summary)
    return summary


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("launch the evaluation script directly; it starts one worker per GPU")
    model_path = args.model_path or Path("~/models/Qwen3-4B-Instruct-2507")
    model_path = Path(os.path.abspath(model_path.expanduser()))
    output = args.output_root.expanduser().resolve() / model_path.name / args.benchmark
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        samples, snapshot = _prepare_evaluation(args, model_path, output)
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
        execution = {"batch_size_per_gpu": args.batch_size_per_gpu}
        if args.benchmark == "ruler":
            execution["batch_size_by_length"] = args.batch_size_by_length
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
                    shards = shard_samples(pending, args.benchmark, len(devices))
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
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--resume", action="store_true", help="沿用原评估计划，只补评未完成的题目")
    common.add_argument("--benchmark-data", type=Path, help="新建评估时必填，续跑沿用原数据")
    common.add_argument("--model-path", type=Path,
                        help="基座模型目录，默认 ~/models/Qwen3-4B-Instruct-2507")
    common.add_argument(
        "--dtype", choices=tuple(WEIGHT_DTYPES), help="基座权重精度，首次默认 fp32",
    )
    common.add_argument(
        "--attention-implementation", choices=ATTENTION_IMPLEMENTATIONS, help="首次默认 sdpa",
    )
    common.add_argument(
        "--mixed-precision", action="store_true",
        help="为 FP32 权重启用 PyTorch BF16 autocast，首次默认关闭",
    )
    common.add_argument("--batch-size-per-gpu", type=positive_int, default=1,
                        help="每卡并行题数上限，默认 1。LoCoMo 每批题目来自同一段对话")
    common.add_argument("--seed", type=int, help="首次默认 42，复用时沿用已存配置")
    common.add_argument("--output-root", type=Path, default=Path("artifacts/baselines/base"),
                        help="结果根目录，其下按模型目录名和任务名保存")
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="benchmark", required=True)
    device_help = "使用 CUDA_VISIBLE_DEVICES 指定物理 GPU，例如 0 或 0,1。"
    tasks.add_parser("locomo", parents=[common], description="评估 LoCoMo", epilog=device_help)
    ruler = tasks.add_parser("ruler", parents=[common], description="评估 RULER", epilog=device_help)
    ruler.add_argument("--lengths", type=positive_int, nargs="+", help="首次默认 8192 16384 32768")
    ruler.add_argument("--batch-size-by-length", type=length_batch_size, nargs="+", default=[],
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
    if args.benchmark == "ruler":
        pairs = args.batch_size_by_length
        args.batch_size_by_length = dict(pairs)
        if len(args.batch_size_by_length) != len(pairs):
            parser.error("--batch-size-by-length must not repeat a length")
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
