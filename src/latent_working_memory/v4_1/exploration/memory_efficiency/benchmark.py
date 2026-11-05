"""从各自 Git worktree 运行基线和优化，再离线比较完整训练步骤。"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import gc
import inspect
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, set_seed

from latent_working_memory.runtime import initialize_device
from latent_working_memory.v4_1.config import load_config
from latent_working_memory.v4_1.data import compression_boundaries, dataset_identity, load_episodes
from latent_working_memory.v4_1.evaluation.records import write_json
from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.teacher.cache import TeacherCache, build_cache_identity
from latent_working_memory.v4_1.teacher.replay import load_teacher
from latent_working_memory.v4_1.teacher.runtime import RuntimeTeacher
from latent_working_memory.v4_1.training.engine import GRAD_NORM_EVERY, TrainingEngine, training_steps


def _repository():
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True,
    ).strip()).resolve()
    sources = {
        name: str(Path(inspect.getfile(value)).resolve())
        for name, value in (
            ("benchmark", _repository), ("student", OnlineMemoryLM),
            ("teacher", RuntimeTeacher), ("teacher_loader", load_teacher),
            ("engine", TrainingEngine),
        )
    }
    if any(not Path(path).is_relative_to(root) for path in sources.values()):
        raise ValueError(f"imports must resolve inside the current Git worktree: {sources}")
    return {
        "root": str(root),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": subprocess.check_output(["git", "status", "--short"], text=True).splitlines(),
        "sources": sources,
    }


def _sample_groups(episodes, size, seed):
    # 每文档选一个真实 QA，使长文压力组覆盖不同文档。
    documents = {}
    for episode in episodes:
        documents.setdefault(episode.document_id, episode)
    candidates = list(documents.values())
    if len(candidates) < size:
        raise ValueError(f"benchmark needs {size} distinct training documents")
    indices = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed)).tolist()
    return {
        "representative": [candidates[index] for index in indices[:size]],
        "long": sorted(candidates, key=lambda item: (-len(item.context_ids), item.id))[:size],
    }


def _sample_plan(episodes, config, world_size):
    return [{
        "id": episode.id, "document_id": episode.document_id, "rank": index % world_size,
        "context_tokens": len(episode.context_ids), "question_tokens": len(episode.question_ids),
        "target_tokens": len(episode.target_ids),
        "write_events": len(compression_boundaries(
            len(episode.context_ids), config.model.pending_size, config.model.recent_size,
        )),
    } for index, episode in enumerate(episodes)]


def _step(engine, episodes, teacher, generator, seed, step, device, diagnostics):
    # 固定每次参照抽样；Adam 状态和模型参数按 warmup、repeat 顺序推进。
    generator.manual_seed(seed)
    set_seed(seed)
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before_allocated = torch.cuda.memory_allocated(device)
    before_reserved = torch.cuda.memory_reserved(device)
    started = time.perf_counter()
    metrics = engine.train_batch(
        episodes, teacher, generator, step,
        inner_diagnostics=diagnostics,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    resources = torch.tensor([
        elapsed, torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device),
        torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device),
        before_allocated, before_reserved,
    ], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(resources, op=dist.ReduceOp.MAX)
    seconds, peak_allocated, peak_reserved, allocated, reserved, before_allocated, before_reserved = (
        resources.tolist()
    )
    return {
        "metrics": metrics,
        "resources": {
            "seconds": seconds, "samples_per_second": metrics["samples"] / seconds,
            "peak_allocated_bytes": int(peak_allocated), "peak_reserved_bytes": int(peak_reserved),
            "allocated_after_step_bytes": int(allocated), "reserved_after_step_bytes": int(reserved),
            "allocated_before_step_bytes": int(before_allocated),
            "reserved_before_step_bytes": int(before_reserved),
        },
    }


def run(args):
    if args.repeats < 3:
        raise ValueError("benchmark requires at least three measured repeats")
    device = initialize_device("cuda")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    overrides = {"training": {
        "micro_batch_size": args.micro_batch_size_per_gpu,
        "gradient_accumulation_steps": args.gradient_accumulation_steps, "tracking": "disabled",
    }, "teacher": {"batch_size": args.teacher_batch_size_per_gpu}}
    if args.model_name_or_path is not None:
        overrides["model"] = {"model_name_or_path": args.model_name_or_path}
    for name in ("dataset_dir", "teacher_cache_dir"):
        if getattr(args, name) is not None:
            overrides["training"][name] = str(Path(getattr(args, name)).expanduser().resolve())
    config = load_config(args.config, overrides)
    if (config.model.inner_objective, config.training.outer_write_obj) != (
        "output_only_mass_calibrated", "distill",
    ):
        raise ValueError("this benchmark requires OOMC inner updates and distill outer loss")
    config = replace(config, training=replace(
        config.training,
        dataset_dir=str(Path(config.training.dataset_dir).expanduser().resolve()),
        teacher_cache_dir=str(Path(config.training.teacher_cache_dir).expanduser().resolve()),
    ))
    output = Path(args.output_dir).expanduser().resolve()
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    episodes, _, statistics_ = load_episodes(
        config.training.dataset_dir, tokenizer, split="train",
        min_text_tokens=config.model.pending_size + config.model.recent_size,
        max_text_tokens=config.training.max_train_data_tokens,
        model_window=AutoConfig.from_pretrained(config.model.model_name_or_path).max_position_embeddings,
        max_samples=config.training.max_train_samples, seed=config.seed,
    )
    global_batch = world_size * args.micro_batch_size_per_gpu * args.gradient_accumulation_steps
    config = replace(config, training=replace(
        config.training, max_steps=training_steps(config.training, len(episodes), global_batch),
    ))
    groups = _sample_groups(episodes, global_batch, config.seed)
    teacher_cache = TeacherCache(
        config.training.teacher_cache_dir, expected_identity=build_cache_identity(config, tokenizer),
    )
    cases = [{"name": f"{group}_{mode}", "sample_group": group,
              "step_argument": GRAD_NORM_EVERY if mode == "diagnostic" else 1,
              "diagnostics": mode == "diagnostic"}
             for group in ("representative", "long") for mode in ("regular", "diagnostic")]
    plan = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "repository": _repository(), "config": asdict(config), "world_size": world_size,
        "global_batch_size": global_batch, "warmup_steps_per_case": 1, "repeats": args.repeats,
        "teacher_backbone_sharing": True,
        "gradient_capture": "FP32 parameter.grad after global mean and clip, before next zero_grad",
        "parameter_capture": "FP32 trainable parameters after AdamW update",
        "repeat_protocol": "reset trainable parameters and Adam per case; fixed teacher RNG per call; "
                           "one warmup followed by consecutive Adam updates on the same samples",
        "timing": "complete engine.train_batch including teacher, diagnostics, reduction and AdamW; "
                  "artifact CPU copies and file writes excluded",
        "dataset": statistics_, "teacher_identity": teacher_cache.identity,
        "sample_groups": {
            name: {"identity": dataset_identity(values), "samples": _sample_plan(values, config, world_size)}
            for name, values in groups.items()
        },
        "cases": cases,
    }
    status = [None]
    if rank == 0:
        try:
            if output.exists() and any(output.iterdir()):
                raise ValueError(f"benchmark output directory must be empty: {output}")
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "plan.json", plan)
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    if dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])

    model = OnlineMemoryLM(config.model).to(device)
    teacher = RuntimeTeacher(load_teacher(config.model, device, model.backbone), config, teacher_cache)
    initial_state = model.trainable_state_dict()
    for case in cases:
        model.load_trainable_state_dict(initial_state)
        for parameter in model.parameters():
            parameter.grad = None
        engine = TrainingEngine(model, config.training)
        set_seed(config.seed + rank)
        query_seed = config.seed + 1 + rank
        generator = torch.Generator().manual_seed(query_seed)
        samples = groups[case["sample_group"]]
        local_samples = samples[rank::world_size]
        case_dir = output / case["name"]
        case_dir.mkdir(exist_ok=True)
        torch.cuda.empty_cache()
        warmup = _step(
            engine, local_samples, teacher, generator, query_seed, case["step_argument"], device,
            [] if case["diagnostics"] else None,
        )
        if rank == 0:
            write_json(case_dir / "warmup.json", warmup)
        for repeat in range(1, args.repeats + 1):
            diagnostics = [] if case["diagnostics"] else None
            record = _step(
                engine, local_samples, teacher, generator, query_seed, case["step_argument"], device,
                diagnostics,
            )
            record.update(repeat=repeat, optimizer_updates=repeat + 1,
                          step_argument=case["step_argument"])
            tensor_path = case_dir / f"repeat-{repeat:02d}_rank-{rank:02d}.pt"
            torch.save({
                "gradients": {
                    name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                    for name, parameter in model.named_parameters() if parameter.requires_grad
                },
                "parameters": model.trainable_state_dict(),
            }, tensor_path)
            write_json(tensor_path.with_suffix(".json"), {
                "rank": rank, "sample_ids": [episode.id for episode in local_samples],
                "inner_diagnostics": diagnostics,
            })
            if rank == 0:
                record["resources"]["context_tokens_per_second"] = (
                    sum(len(episode.context_ids) for episode in samples) / record["resources"]["seconds"]
                )
                write_json(case_dir / f"repeat-{repeat:02d}.json", record)
                print(json.dumps({"case": case["name"], **record}, ensure_ascii=False), flush=True)
        del engine
        gc.collect()
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        write_json(output / "complete.json", {"cases": len(cases), "repeats": args.repeats})


def _tensor_comparison(reference, actual, atol, rtol):
    if reference.keys() != actual.keys():
        raise ValueError("trainable parameter names differ")
    squared_error = squared_reference = squared_actual = dot = 0.0
    maximum_error, worst_parameter, all_close = 0.0, None, True
    none_mismatches = []
    for name, expected in reference.items():
        observed = actual[name]
        if expected is None or observed is None:
            if (expected is None) != (observed is None):
                none_mismatches.append(name)
            continue
        if expected.shape != observed.shape or expected.dtype != torch.float32 or observed.dtype != torch.float32:
            raise ValueError(f"expected matching FP32 tensor shapes for {name}")
        expected, observed = expected.double(), observed.double()
        difference = observed - expected
        absolute = difference.abs()
        error = float(absolute.max())
        if error > maximum_error:
            maximum_error, worst_parameter = error, name
        all_close &= bool(torch.all(absolute <= atol + rtol * expected.abs()))
        squared_error += float(difference.square().sum())
        squared_reference += float(expected.square().sum())
        squared_actual += float(observed.square().sum())
        dot += float((expected * observed).sum())
    denominator = math.sqrt(squared_reference * squared_actual)
    return {
        "allclose": all_close and not none_mismatches, "max_abs": maximum_error,
        "worst_parameter": worst_parameter, "l2_error": math.sqrt(squared_error),
        "relative_l2": math.sqrt(squared_error / squared_reference) if squared_reference else None,
        "reference_l2": math.sqrt(squared_reference), "actual_l2": math.sqrt(squared_actual),
        "cosine": dot / denominator if denominator else None, "none_mismatches": none_mismatches,
    }


def _json_comparison(reference, actual, atol, rtol):
    differences = []
    maximum_error = 0.0
    worst_path = None

    def visit(expected, observed, path):
        nonlocal maximum_error, worst_path
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if not isinstance(observed, (int, float)) or isinstance(observed, bool):
                differences.append(path)
                return
            error = abs(observed - expected)
            if error > maximum_error:
                maximum_error, worst_path = error, path
            if not math.isfinite(observed) or error > atol + rtol * abs(expected):
                differences.append(path)
        elif isinstance(expected, dict) and isinstance(observed, dict):
            if expected.keys() != observed.keys():
                differences.append(path + ".keys")
            for key in expected.keys() & observed.keys():
                visit(expected[key], observed[key], f"{path}.{key}")
        elif isinstance(expected, list) and isinstance(observed, list):
            if len(expected) != len(observed):
                differences.append(path + ".length")
            for index, (left, right) in enumerate(zip(expected, observed)):
                visit(left, right, f"{path}[{index}]")
        elif expected != observed:
            differences.append(path)

    visit(reference, actual, "root")
    return {"allclose": not differences, "max_abs": maximum_error,
            "worst_path": worst_path, "mismatches": differences}


def compare(args):
    if not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        raise ValueError("comparison tolerances must be finite and non-negative")
    baseline, optimized = Path(args.baseline).resolve(), Path(args.optimized).resolve()
    plans = [json.loads((root / "plan.json").read_text()) for root in (baseline, optimized)]
    for root in (baseline, optimized):
        if not (root / "complete.json").is_file():
            raise ValueError(f"benchmark is incomplete: {root}")
    for key in ("config", "world_size", "global_batch_size", "warmup_steps_per_case", "repeats",
                "sample_groups", "cases", "teacher_identity", "repeat_protocol"):
        values = [plan[key] for plan in plans]
        if key == "config":
            # worktree 和本地产物路径不参与数值设置的一致性判定。
            values = [{**value, "training": {
                name: item for name, item in value["training"].items()
                if name not in {"output_root", "output_dir", "run_id"}
            }} for value in values]
        if values[0] != values[1]:
            raise ValueError(f"benchmark plans differ in {key}")
    result = {"baseline": str(baseline), "optimized": str(optimized),
              "baseline_commit": plans[0]["repository"]["commit"],
              "optimized_commit": plans[1]["repository"]["commit"],
              "atol": args.atol, "rtol": args.rtol, "cases": [], "allclose": True}
    for case in plans[0]["cases"]:
        name = case["name"]
        case_result = {"name": name, "repeats": []}
        timing, memory, reserved = [[], []], [[], []], [[], []]
        for repeat in range(1, plans[0]["repeats"] + 1):
            records = [json.loads((root / name / f"repeat-{repeat:02d}.json").read_text())
                       for root in (baseline, optimized)]
            metrics = _json_comparison(records[0]["metrics"], records[1]["metrics"], args.atol, args.rtol)
            row = {"repeat": repeat, "metrics": metrics, "ranks": []}
            result["allclose"] &= metrics["allclose"]
            for rank in range(plans[0]["world_size"]):
                paths = [root / name / f"repeat-{repeat:02d}_rank-{rank:02d}.pt"
                         for root in (baseline, optimized)]
                tensors = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
                rank_result = {"rank": rank}
                for field in ("gradients", "parameters"):
                    rank_result[field] = _tensor_comparison(
                        tensors[0][field], tensors[1][field], args.atol, args.rtol,
                    )
                    result["allclose"] &= rank_result[field]["allclose"]
                del tensors
                diagnostics = [json.loads(path.with_suffix(".json").read_text()) for path in paths]
                rank_result["diagnostics"] = _json_comparison(
                    diagnostics[0], diagnostics[1], args.atol, args.rtol,
                )
                result["allclose"] &= rank_result["diagnostics"]["allclose"]
                row["ranks"].append(rank_result)
            for index, record in enumerate(records):
                timing[index].append(record["resources"]["seconds"])
                memory[index].append(record["resources"]["peak_allocated_bytes"])
                reserved[index].append(record["resources"]["peak_reserved_bytes"])
            case_result["repeats"].append(row)
        seconds = [statistics.median(values) for values in timing]
        allocated = [max(values) for values in memory]
        maximum_reserved = [max(values) for values in reserved]
        case_result["performance"] = {
            "baseline_median_seconds": seconds[0], "optimized_median_seconds": seconds[1],
            "speedup": seconds[0] / seconds[1], "baseline_peak_allocated_bytes": allocated[0],
            "optimized_peak_allocated_bytes": allocated[1],
            "peak_allocated_reduction_fraction": 1 - allocated[1] / allocated[0],
            "baseline_peak_reserved_bytes": maximum_reserved[0],
            "optimized_peak_reserved_bytes": maximum_reserved[1],
            "baseline_samples_per_second": plans[0]["global_batch_size"] / seconds[0],
            "optimized_samples_per_second": plans[0]["global_batch_size"] / seconds[1],
            "baseline_seconds": timing[0], "optimized_seconds": timing[1],
        }
        result["cases"].append(case_result)
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "comparison.json", result)
    print(json.dumps({"allclose": result["allclose"], "comparison": str(destination / "comparison.json"),
                      "performance": [{"name": item["name"], **item["performance"]}
                                      for item in result["cases"]]}, ensure_ascii=False), flush=True)
    return 0 if result["allclose"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="在当前 Git worktree 运行独立完整步骤 benchmark")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--model-name-or-path")
    run_parser.add_argument("--dataset-dir")
    run_parser.add_argument("--teacher-cache-dir")
    run_parser.add_argument("--micro-batch-size-per-gpu", type=int, default=2)
    run_parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    run_parser.add_argument("--teacher-batch-size-per-gpu", type=int, default=64)
    run_parser.add_argument("--repeats", type=int, default=3)
    compare_parser = commands.add_parser("compare", help="离线比较所有梯度、更新参数、指标与诊断")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--optimized", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--atol", type=float, default=1e-5)
    compare_parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    try:
        if args.command == "run":
            run(args)
            return 0
        return compare(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
