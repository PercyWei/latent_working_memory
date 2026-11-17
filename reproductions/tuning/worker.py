"""Run In-Place TTT batch-one samples for a single GPU tuning worker."""

import argparse
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import platform
import time
import traceback

from in_place_ttt.scripts.runtime import generate_one, load_model
import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--rank", type=int, required=True)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
rank = args.rank
output_dir = Path(plan["output_dir"])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_prompts():
    assigned = [plan["samples"][index] for index in plan["assignments"][rank]]
    sources = {}
    for sample in assigned:
        sources.setdefault(sample["source"], set()).add(sample["row_index"])
    rows = {}
    for source, indices in sources.items():
        with Path(source).open() as stream:
            for index, line in enumerate(stream):
                if index in indices:
                    rows[source, index] = json.loads(line)
                if index >= max(indices):
                    break
    prompts = []
    for sample in assigned:
        row = rows[sample["source"], sample["row_index"]]
        prompt = row["input"] + row["answer_prefix"]
        prompts.append((sample["id"], prompt))
    return prompts


def generate(loaded, sample_id, prompt):
    started = time.perf_counter()
    generated = generate_one(loaded, prompt, max_out_len=64, max_total_tokens=plan["length"])
    input_tokens = generated["input_tokens"]
    token_ids = generated["token_ids"]
    prediction = generated["raw_prediction"]
    torch.cuda.synchronize(0)
    return {
        "id": sample_id, "token_ids": token_ids, "input_tokens": input_tokens,
        "generated_tokens": len(token_ids), "prediction": prediction,
        "seconds": time.perf_counter() - started,
    }


def execute():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Tuning requires physical GPU 0: CUDA_VISIBLE_DEVICES=0")
    prompts = read_prompts()
    loaded = load_model(
        str(Path(plan["checkpoint"]).expanduser()),
        plan["inference_mode"], plan["mlp_chunk_size"], plan["length"],
    )
    model = loaded.model
    runtime = {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_source": inspect.getfile(type(model)),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "transformers": version("transformers"),
        "device_name": torch.cuda.get_device_name(0),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    warmup = generate(loaded, *prompts[0])
    write_json(output_dir / f"ready-{rank}.json", {
        "rank": rank, "pid": os.getpid(), "runtime": runtime, "warmup": warmup,
    })
    deadline = time.monotonic() + 1800
    while not (output_dir / "go.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Tuning controller did not release the start barrier")
        time.sleep(0.02)
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    samples = []
    with (output_dir / f"samples-{rank}.jsonl").open("x") as stream:
        for sample_id, prompt in prompts:
            result = generate(loaded, sample_id, prompt)
            samples.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    torch.cuda.synchronize(0)
    write_json(output_dir / f"worker-{rank}.json", {
        "rank": rank, "pid": os.getpid(), "runtime": runtime, "samples": samples,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "timing_interpretation": "Includes CPU encoding, generation, decoding and per-sample JSONL writes; excludes model load and warmup.",
    })


if __name__ == "__main__":
    try:
        execute()
    except Exception as error:
        write_json(output_dir / f"error-{rank}.json", {
            "rank": rank, "pid": os.getpid(), "error": repr(error),
            "traceback": traceback.format_exc(),
            "cuda_out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError),
        })
        raise
