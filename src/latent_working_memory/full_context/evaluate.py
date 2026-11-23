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

from latent_working_memory.benchmarks.comparison import comparison_data
from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import DEFAULT_RULER_LENGTHS, data_identity, protocol
from latent_working_memory.benchmarks.scoring import summarize_scores
from latent_working_memory.cli import length_batch_size, positive_int
from latent_working_memory.devices import visible_devices
from latent_working_memory.v4.io import write_json

from .parallel import evaluate_parallel, evaluation_log, shard_samples


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
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else None
    data_dir = args.benchmark_data
    if data_dir is None:
        if settings is None:
            raise ValueError(f"{settings_path} is missing; specify --benchmark-data")
        data_dir = Path(settings["data"]["directory"])
    seed = 42
    model_config = {
        "backbone_dtype": "float32",
        "attention_implementation": "sdpa",
        "mixed_precision": False,
    }
    lengths = ()
    if settings is not None:
        seed = settings["seed"]
        model_config = dict(settings["model"])
    if args.benchmark == "ruler":
        lengths = args.lengths
        if lengths is None:
            lengths = DEFAULT_RULER_LENGTHS if settings is None else settings["protocol"]["lengths"]
        unplanned = args.batch_size_by_length.keys() - set(lengths)
        if unplanned:
            raise ValueError(f"--batch-size-by-length includes unselected lengths: {sorted(unplanned)}")
    model_config["model_name_or_path"] = str(model_path)
    if args.dtype is not None:
        model_config["backbone_dtype"] = WEIGHT_DTYPES[args.dtype]
    if args.attention_implementation is not None:
        model_config["attention_implementation"] = args.attention_implementation
    if args.mixed_precision:
        model_config["mixed_precision"] = True
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
        "seed": seed if args.seed is None else args.seed,
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
    if settings is not None:
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
            "device": f"CUDA (physical GPUs: {','.join(devices)})",
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
    model_path = Path(os.path.abspath(args.model_path.expanduser()))
    output = args.output_root.expanduser().resolve() / model_path.name / args.benchmark
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        samples, snapshot = _prepare_evaluation(args, model_path, output)
        summary_path = output / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _matching({key: summary[key] for key in snapshot}, snapshot, summary_path)
            print(f"{args.benchmark} reusing cached summary: {summary_path}", flush=True)
            _print_scores(summary)
            return summary
        devices = visible_devices()
        available = torch.cuda.device_count()
        if available != len(devices):
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES={','.join(devices)} selects {len(devices)} GPU(s), "
                f"but PyTorch detects {available}"
            )
        shards = shard_samples(samples, args.benchmark, len(devices))
        devices = devices[:len(shards)]
        execution = {"batch_size_per_gpu": args.batch_size_per_gpu}
        if args.benchmark == "ruler":
            execution["batch_size_by_length"] = args.batch_size_by_length
        write_json(output / "eval.json", snapshot)
        status = {
            "method": METHOD,
            "benchmark": args.benchmark,
            "status": "running",
            "gpus": ",".join(devices),
            "device": "cuda",
            "execution": execution,
            "command": [sys.executable, *sys.argv],
            "started_at": _now(),
        }
        write_json(output / "status.json", status)
        with evaluation_log(output / "eval.log"):
            try:
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
    common.add_argument("--benchmark-data", type=Path, help="首次必填，复用时沿用已存的数据目录")
    common.add_argument("--model-path", type=Path, default=Path("~/models/Qwen3-4B-Instruct-2507"),
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
                        help="默认 1。LoCoMo 限制每批预填充会话数和每批解码题数，RULER 按题数计")
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
    if args.seed is not None and args.seed < 0:
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
