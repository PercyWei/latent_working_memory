"""Run independent batch-one evaluation samples for a single GPU tuning worker."""

import argparse
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import platform
import time
import traceback

import torch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--rank", type=int, required=True)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
rank = args.rank
output_dir = Path(plan["output_dir"])
method = plan["method"]


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
        if plan["dataset"] == "ruler":
            with Path(source).open() as stream:
                for index, line in enumerate(stream):
                    if index in indices:
                        rows[source, index] = json.loads(line)
                    if index >= max(indices):
                        break
        else:
            dataset = Dataset.from_parquet(source)
            for index in indices:
                rows[source, index] = dataset[index]
    prompts = []
    for sample in assigned:
        row = rows[sample["source"], sample["row_index"]]
        prompt = row["input"] + row["answer_prefix"] if plan["dataset"] == "ruler" else row["text"].strip()
        prompts.append((sample["id"], prompt))
    return prompts


def load():
    checkpoint = str(Path(plan["checkpoint"]).expanduser())
    if method == "lact":
        model, _ = load_checkpoint(checkpoint, plan["inference_mode"])
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        return model, tokenizer
    if method == "in_place_ttt":
        wrapper = load_model(
            checkpoint, plan["inference_mode"], plan["mlp_chunk_size"], plan["length"]
        )
        return wrapper, wrapper.tokenizer
    lm = HFLM(
        pretrained=checkpoint, tokenizer=checkpoint, dtype="bfloat16", device="cuda:0",
        batch_size=1, max_length=plan["length"], truncation=False, trust_remote_code=False,
    )
    return lm, lm.tokenizer


def generate(loaded, tokenizer, sample_id, prompt):
    started = time.perf_counter()
    if method == "lact":
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        input_tokens = encoded["input_ids"].shape[1]
        if input_tokens + 128 > plan["length"]:
            raise ValueError(f"{sample_id}: prompt plus output exceeds token budget")
        inputs = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = loaded.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=128,
                use_cache=plan["inference_mode"] == "incremental", logits_to_keep=1,
                return_dict_in_generate=False, output_scores=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        token_ids = generated[0, input_tokens:].detach().cpu().tolist()
        prediction = tokenizer.decode(token_ids, skip_special_tokens=True)
    elif method == "in_place_ttt":
        generated = generate_one(loaded, prompt, max_out_len=64, max_total_tokens=plan["length"])
        input_tokens = generated["input_tokens"]
        token_ids = generated["token_ids"]
        prediction = generated["raw_prediction"]
    else:
        recall = plan["dataset"] != "ruler"
        budget = 48 if recall else 128
        input_limit = 2000 if recall else plan["length"] - budget
        encoded, mask = loaded.tok_batch_encode(
            [prompt], left_truncate_len=input_limit if recall else None, truncation=False,
        )
        input_tokens = encoded.shape[1]
        if input_tokens > input_limit:
            raise ValueError(f"{sample_id}: prompt plus output exceeds token budget")
        stop = handle_stop_sequences(
            ["\n"] if recall else [],
            eos=loaded.tok_decode(loaded.eot_token_id, skip_special_tokens=False),
        )
        generated = loaded._model_generate(
            context=encoded.to(loaded.device), attention_mask=mask.to(loaded.device),
            max_length=input_tokens + budget, stop=stop, do_sample=False, temperature=0.0,
        )
        token_ids = generated[0, input_tokens:].detach().cpu().tolist()
        prediction = postprocess_generated_text(
            generation=loaded.tok_decode(token_ids), stop=stop,
            think_end_token=loaded.think_end_token,
        )
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
    loaded, tokenizer = load()
    model = loaded if method == "lact" else loaded.model
    runtime = {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_source": inspect.getfile(type(model)),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "transformers": version("transformers"),
        "device_name": torch.cuda.get_device_name(0),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    warmup = generate(loaded, tokenizer, *prompts[0])
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
            result = generate(loaded, tokenizer, sample_id, prompt)
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
        if method == "lact":
            from evaluate import load_checkpoint
            from transformers import AutoTokenizer
        elif method == "in_place_ttt":
            from runtime import generate_one, load_model
        elif method == "e2_ttt":
            import e2_ttt  # noqa: F401 -- register the original model classes
            from datasets import Dataset
            from lm_eval.models.huggingface import HFLM
            from lm_eval.models.utils import handle_stop_sequences, postprocess_generated_text
        else:
            raise ValueError(f"Unknown method: {method}")
        execute()
    except Exception as error:
        write_json(output_dir / f"error-{rank}.json", {
            "rank": rank, "pid": os.getpid(), "error": repr(error),
            "traceback": traceback.format_exc(),
            "cuda_out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError),
        })
        raise
