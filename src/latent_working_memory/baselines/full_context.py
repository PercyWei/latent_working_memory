"""使用原生完整上下文和 greedy 解码，离线评估 RULER／LoCoMo。"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from latent_working_memory.benchmarks.data import load_samples
from latent_working_memory.benchmarks.protocol import (
    DEFAULT_RULER_LENGTHS,
    MODEL_FIELDS,
    data_identity,
    encode_prompt,
    protocol,
)
from latent_working_memory.benchmarks.scoring import score_sample, summarize_scores
from latent_working_memory.v4.io import write_json


METHOD = "full-context"


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


class _Log:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    @property
    def encoding(self):
        return self.console.encoding

    def isatty(self):
        return False

    def write(self, value):
        self.stream.write(value)
        return self.console.write(value)

    def flush(self):
        self.stream.flush()
        self.console.flush()


def _common_config(args, path):
    requested = {
        "model_name_or_path": str(args.model_path.expanduser().resolve())
        if args.model_path is not None
        else None,
        "backbone_dtype": args.backbone_dtype,
        "attention_implementation": args.attention_implementation,
    }
    if path.is_file():
        common = json.loads(path.read_text(encoding="utf-8"))
        if set(common) != {"seed", "model"} or set(common["model"]) != set(MODEL_FIELDS):
            raise ValueError("baseline config.json requires seed and the three model settings")
        if args.seed is not None and args.seed != common["seed"]:
            raise ValueError("seed differs from the saved baseline configuration; use a new run-id")
        for name, value in requested.items():
            if value is not None and value != common["model"][name]:
                raise ValueError(f"{name} differs from saved configuration; use a new run-id")
    else:
        common = {
            "seed": 42 if args.seed is None else args.seed,
            "model": {
                "model_name_or_path": requested["model_name_or_path"]
                or os.path.expanduser("~/models/Qwen3-4B-Instruct-2507"),
                "backbone_dtype": requested["backbone_dtype"] or "bfloat16",
                "attention_implementation": requested["attention_implementation"] or "sdpa",
            },
        }
    if type(common["seed"]) is not int or common["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if common["model"]["backbone_dtype"] not in {"float32", "bfloat16"}:
        raise ValueError("backbone_dtype must be float32 or bfloat16")
    if common["model"]["attention_implementation"] not in {"eager", "sdpa"}:
        raise ValueError("attention_implementation must be eager or sdpa")
    if not path.is_file():
        write_json(path, common)
    return common


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


def _evaluate(args, plan, output):
    with (plan / ".config.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        common = _common_config(args, plan / "config.json")
    settings_path = plan / f"{args.benchmark}.json"
    settings = (
        json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else None
    )
    data_dir = args.benchmark_data
    if data_dir is None:
        if settings is None:
            raise ValueError(f"{settings_path} is missing; specify --benchmark-data")
        data_dir = Path(settings["data"]["directory"])
    lengths = args.lengths
    if lengths is None:
        lengths = (
            tuple(settings["protocol"]["lengths"])
            if settings is not None and args.benchmark == "ruler"
            else DEFAULT_RULER_LENGTHS
        )
    samples = load_samples(args.benchmark, data_dir, tuple(lengths))
    model_config = common["model"]
    model_path = model_config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("benchmark generation requires tokenizer.eos_token_id")
    model_window = AutoConfig.from_pretrained(model_path).max_position_embeddings
    computation = {
        "benchmark": args.benchmark,
        "split": "test",
        "model_window": model_window,
        "data": data_identity(samples, data_dir, args.benchmark),
        "protocol": protocol(
            args.benchmark,
            lengths,
            "full prompt remains in the native KV cache; no memory compression",
        ),
    }
    snapshot = {"method": METHOD, **computation, **common}
    if settings is not None:
        _matching(settings, computation, settings_path)
    for name in ("eval.json", "summary.json"):
        path = output / name
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            _matching({name: saved[name] for name in snapshot}, snapshot, path)
    if settings is None:
        write_json(settings_path, computation)
    if (output / "summary.json").is_file():
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        print(f"{args.benchmark} reusing cached summary: {output / 'summary.json'}", flush=True)
        _print_scores(summary)
        return summary, "cached"
    write_json(output / "eval.json", snapshot)
    set_seed(common["seed"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, model_config["backbone_dtype"]),
            attn_implementation=model_config["attention_implementation"],
        )
        .to(device)
        .eval()
    )
    records = []
    input_tokens, generated_tokens, generation_seconds = 0, 0, 0.0
    peak_allocated, peak_reserved = 0, 0
    started = time.perf_counter()
    with (output / "details.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(samples, 1):
            ids = encode_prompt(sample, tokenizer, model_window, args.benchmark)
            generation = GenerationConfig(
                max_new_tokens=sample.max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                use_cache=True,
            )
            inputs = torch.tensor([ids], dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            generated_at = time.perf_counter()
            with torch.no_grad():
                tokens = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=generation,
                    use_model_defaults=False,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
            elapsed = time.perf_counter() - generated_at
            generated = tokens[0, len(ids) :].tolist()
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            record = {
                "id": sample.id,
                "metadata": sample.metadata,
                "references": list(sample.references),
                "prediction": prediction,
                "score": score_sample(args.benchmark, sample, prediction),
                "prompt_tokens": len(ids),
                "generated_tokens": len(generated),
                "generated_token_ids": generated,
                "generation_budget": sample.max_new_tokens,
                "finish_reason": "eos"
                if generated[-1] == tokenizer.eos_token_id
                else "max_new_tokens",
                "generation_seconds": elapsed,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
            input_tokens += len(ids)
            generated_tokens += len(generated)
            generation_seconds += elapsed
            print(
                f"{args.benchmark} {index}/{len(samples)} | score {record['score']:.4f} "
                f"| elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )
    summary = {
        **snapshot,
        "samples": len(records),
        "metrics": summarize_scores(args.benchmark, records),
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "resources": {
            "evaluation_seconds": time.perf_counter() - started,
            "generation_seconds": generation_seconds,
            "scope": "evaluation excludes model/data loading; generation includes prefill and decode",
            **(
                {
                    "peak_memory_allocated_bytes": peak_allocated,
                    "peak_memory_reserved_bytes": peak_reserved,
                }
                if device.type == "cuda"
                else {}
            ),
        },
    }
    _matching(
        json.loads((plan / "config.json").read_text(encoding="utf-8")), common, plan / "config.json"
    )
    write_json(output / "summary.json", summary)
    _print_scores(summary)
    return summary, "finished"


def run(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("full-context baseline requires one process and one GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    run_id = args.run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    directory = args.output_root.expanduser().resolve() / METHOD / run_id
    plan = directory / "plan"
    output = directory / "eval" / args.benchmark
    plan.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status = {
            "method": METHOD,
            "benchmark": args.benchmark,
            "status": "running",
            "gpus": args.gpus,
            "device": args.device,
            "command": [sys.executable, *sys.argv],
            "started_at": _now(),
        }
        write_json(output / "status.json", status)
        with (output / "eval.log").open("a", encoding="utf-8") as log:
            with redirect_stdout(_Log(sys.stdout, log)), redirect_stderr(_Log(sys.stderr, log)):
                try:
                    summary, computation = _evaluate(args, plan, output)
                except BaseException as error:
                    status.update(
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                        finished_at=_now(),
                        return_code=1,
                    )
                    write_json(output / "status.json", status)
                    print(status["error"], file=sys.stderr, flush=True)
                    raise
                status.update(
                    status="finished", compute_status=computation, finished_at=_now(), return_code=0
                )
                write_json(output / "status.json", status)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("ruler", "locomo"), required=True)
    parser.add_argument("--benchmark-data", type=Path, help="首次必填；后续沿用同一 run 的评估配置")
    parser.add_argument(
        "--lengths", type=int, nargs="+", help="仅用于 RULER，首次默认 8192 16384 32768"
    )
    parser.add_argument("--model-path", type=Path, help="首次默认 ~/models/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--backbone-dtype", choices=("bfloat16", "float32"), help="首次默认 bfloat16"
    )
    parser.add_argument(
        "--attention-implementation", choices=("sdpa", "eager"), help="首次默认 sdpa"
    )
    parser.add_argument("--seed", type=int, help="首次默认 42；已有公共配置不可覆盖")
    parser.add_argument("--gpus", default="0", help="单张物理 GPU 编号，默认 0")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-id", help="默认上海时间 YYYYMMDD-HHMMSS；复用缓存须指定原编号")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/baselines"))
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9]+", args.gpus):
        parser.error("--gpus requires exactly one nonnegative physical GPU index")
    args.gpus = str(int(args.gpus))
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.run_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must contain only letters, digits, '-' and '_'")
    if args.benchmark == "locomo" and args.lengths is not None:
        parser.error("--lengths is only supported by RULER")
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
