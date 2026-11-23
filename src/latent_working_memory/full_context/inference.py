"""在指定 CUDA 设备加载基座，按任务批量生成并写入逐题明细。"""

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

from latent_working_memory.benchmarks.scoring import score_sample

from .locomo import evaluate_locomo
from .ruler import evaluate_ruler


def evaluate_samples(samples, snapshot, device: torch.device, physical_gpu, progress_path, execution):
    """独立评估一个 GPU 分片，记录实测耗时与峰值显存。"""
    benchmark = snapshot["benchmark"]
    model_config = snapshot["model"]
    model_path = model_config["model_name_or_path"]
    torch.cuda.set_device(device)
    set_seed(snapshot["seed"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("benchmark generation requires tokenizer.eos_token_id")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, model_config["backbone_dtype"]),
        attn_implementation=model_config["attention_implementation"],
        device_map=device,
    ).eval()
    generation = GenerationConfig(
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    pending = {sample.id for sample in samples}
    started = time.perf_counter()
    with (
        progress_path.open("a", encoding="utf-8") as stream,
        torch.no_grad(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=model_config["mixed_precision"],
        ),
    ):
        def commit(sample, prompt_tokens, generated, elapsed, batch_size, decode_batch_size):
            if sample.id not in pending:
                raise ValueError(f"unexpected or duplicate evaluation record on GPU {physical_gpu}: {sample.id}")
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            record = {
                "id": sample.id,
                "metadata": sample.metadata,
                "references": list(sample.references),
                "prediction": prediction,
                "score": score_sample(benchmark, sample, prediction),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": len(generated),
                "generated_token_ids": generated,
                "generation_budget": sample.max_new_tokens,
                "finish_reason": "eos" if generated[-1] == tokenizer.eos_token_id else "max_new_tokens",
                "generation_seconds": elapsed,
                "eval_batch_size": batch_size,
                "decode_batch_size": decode_batch_size,
            }
            entry = {
                "record": record,
                "physical_gpu": physical_gpu,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            pending.remove(sample.id)
            print(
                f"{device} | {benchmark} {len(samples) - len(pending)}/{len(samples)} | {sample.id} "
                f"| decode batch {decode_batch_size} | score {record['score']:.4f} "
                f"| elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )

        batch_size = execution["batch_size_per_gpu"]
        if benchmark == "locomo":
            evaluate_locomo(samples, tokenizer, model, generation, snapshot["model_window"],
                            device, batch_size, commit)
        else:
            evaluate_ruler(samples, tokenizer, model, generation, snapshot["model_window"],
                           device, batch_size, execution["batch_size_by_length"], commit)
    if pending:
        raise ValueError(f"missing {len(pending)} evaluation records from GPU {physical_gpu}")
    return {"evaluation_seconds": time.perf_counter() - started}
