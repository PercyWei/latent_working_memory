"""固定内存优化，配对比较 SDPA 与外部 FlashAttention 2；不改变生产默认。"""

import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import time

from opencompass.utils.text_postprocessors import extract_non_reasoning_content
from provenance import capture_source
from runtime import MODEL_CLASSES, load_model, runtime_info
import torch

from validate_optimization import (
    EVALUATORS,
    PROJECT_ROOT,
    append_json,
    compare_states,
    compare_tokens,
    measure,
    select_mode,
    timestamp,
    write_json,
)


BACKENDS = ("sdpa", "flash_attention_2")
TASK = "niah_single_1"


def run(args, output, source, logger):
    plan = {
        "created_at": timestamp(),
        "purpose": "paired attention backend experiment; not formal benchmark",
        "checkpoint": str(args.checkpoint),
        "ruler_data": str(args.ruler_data),
        "lengths": args.lengths,
        "task": TASK,
        "sample_position": 0,
        "backends": list(BACKENDS),
        "compact_tails": True,
        "mlp_chunk_size": args.mlp_chunk_size,
        "generation": {"max_new_tokens": 64, "do_sample": False, "preserve_eos": True},
        "warmup": "每个后端各一次完整4K生成，不计入正式测量",
        "source": source,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "timing": "单样本wall、CUDA forward事件及显存；小样本不能代表完整benchmark",
    }
    write_json(output / "plan.json", plan)
    wrapper = load_model(args.checkpoint, "optimized", args.mlp_chunk_size, max(args.lengths))
    plan["checkpoint_config"] = wrapper.model.config.to_dict()
    select_mode(wrapper, "chunked", args.mlp_chunk_size)
    plan["runtime_by_backend"] = {}
    for backend in BACKENDS:
        wrapper.model.set_attn_implementation(backend)
        plan["runtime_by_backend"][backend] = runtime_info(wrapper)
        plan["runtime_by_backend"][backend].update(
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)
        )
        plan["runtime_by_backend"][backend]["packages"]["flash-attn"] = importlib.metadata.version(
            "flash-attn"
        )
    write_json(output / "plan.json", plan)
    samples = {}
    for length in dict.fromkeys((4096, *args.lengths)):
        path = args.ruler_data / str(length) / TASK / "test.jsonl"
        with path.open(encoding="utf-8") as stream:
            samples[length] = json.loads(next(stream))
    warmup_prompt = samples[4096]["input"] + samples[4096]["answer_prefix"]
    for backend in BACKENDS:
        wrapper.model.set_attn_implementation(backend)
        logger.info("4K warmup backend=%s", backend)
        result, state = measure(wrapper, warmup_prompt, 4096)
        append_json(output / "warmup.jsonl", {"backend": backend, **result})
        del state
    records = []
    for index, length in enumerate(args.lengths):
        row = samples[length]
        prompt = row["input"] + row["answer_prefix"]
        measured, states = {}, {}
        order = BACKENDS if index % 2 == 0 else tuple(reversed(BACKENDS))
        for backend in order:
            wrapper.model.set_attn_implementation(backend)
            logger.info("Generating length=%s backend=%s", length, backend)
            measured[backend], states[backend] = measure(wrapper, prompt, length)
            measured[backend]["native_score"] = EVALUATORS[TASK]().score(
                [extract_non_reasoning_content(measured[backend]["raw_prediction"])],
                [row["outputs"]],
            )
            append_json(
                output / "measurement.jsonl",
                {"length": length, "backend": backend, **measured[backend]},
            )
        reference, actual = (measured[backend] for backend in BACKENDS)
        comparison = {
            **compare_tokens(reference["token_ids"], actual["token_ids"]),
            "input_ids_equal": torch.equal(
                states[BACKENDS[0]]["input_ids"], states[BACKENDS[1]]["input_ids"]
            ),
            "text_equal": reference["raw_prediction"] == actual["raw_prediction"],
            "native_score_equal": reference["native_score"] == actual["native_score"],
            "state": compare_states(states[BACKENDS[0]]["ttt"], states[BACKENDS[1]]["ttt"]),
        }
        passed = (
            comparison["equal"]
            and comparison["input_ids_equal"]
            and comparison["text_equal"]
            and comparison["native_score_equal"]
            and comparison["state"]["finite"]
            and all(
                r["last_logits_finite"]
                and r["ttt_state_finite"]
                and r["prefill_cache"]["consistent"]
                and r["final_cache"]["consistent"]
                for r in measured.values()
            )
        )
        record = {
            "length": length,
            "task": TASK,
            "sample_position": 0,
            "source_index": row["index"],
            "references": row["outputs"],
            "order": list(order),
            "backends": measured,
            "comparison": comparison,
            "passed": passed,
            "sdpa_seconds_over_fa2_seconds": reference["seconds"] / actual["seconds"],
        }
        append_json(output / "generation.jsonl", record)
        records.append(record)
        logger.info("length=%s passed=%s comparison=%s", length, passed, comparison)
        del states
    summary = {
        "finished_at": timestamp(),
        "passed": all(row["passed"] for row in records),
        "samples": len(records),
        "formal_benchmark_completed": False,
        "plan_path": str(output / "plan.json"),
        "records_path": str(output / "generation.jsonl"),
        "note": "仅对选定NIAH样本比较后端；未通过严格token IDs检查的后端不纳入正式评估。",
    }
    write_json(output / "summary.json", summary)
    if not summary["passed"]:
        raise RuntimeError("Attention backend comparison failed; inspect saved records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ruler-data",
        type=Path,
        default=PROJECT_ROOT / "data/benchmarks/ruler-v1/reproductions/in-place-ttt-qwen3-4b",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        choices=(4096, 8192, 16384, 32768, 65536, 131072, 262144),
        default=[4096, 131072, 262144],
    )
    parser.add_argument("--mlp-chunk-size", type=int, default=16384)
    args = parser.parse_args()
    if len(set(args.lengths)) != len(args.lengths) or args.mlp_chunk_size <= 0:
        parser.error("lengths must be unique and mlp-chunk-size must be positive")
    if os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0") not in {"0", "1"}:
        parser.error("Select a single physical GPU 0 or 1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    source = capture_source(__file__, {"optimized": MODEL_CLASSES["optimized"]})
    args.checkpoint, args.ruler_data = (
        args.checkpoint.expanduser().resolve(),
        args.ruler_data.expanduser().resolve(),
    )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "comparison.log")],
    )
    execution = {"started_at": timestamp(), "source": source, "status": "running"}
    write_json(output / "execution.json", execution)
    started = time.perf_counter()
    try:
        run(args, output, source, logging.getLogger(__name__))
        execution["status"] = "completed"
    except BaseException as error:
        execution.update(status="failed", error=f"{type(error).__name__}: {error}")
        logging.exception("Attention backend comparison failed")
        raise
    finally:
        execution.update(finished_at=timestamp(), seconds=time.perf_counter() - started)
        write_json(output / "execution.json", execution)


if __name__ == "__main__":
    main()
