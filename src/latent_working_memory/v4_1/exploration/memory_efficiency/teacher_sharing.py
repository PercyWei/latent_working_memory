"""逐项比较共享基座和独立加载的 teacher，并定位实际文档参照的首次差异。"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, set_seed
from transformers.cache_utils import DynamicCache

from latent_working_memory.v4_1.config import load_config
from latent_working_memory.v4_1.data import compression_boundaries, load_episodes
from latent_working_memory.v4_1.evaluation.records import write_json
from latent_working_memory.v4_1.modeling.model import OnlineMemoryLM
from latent_working_memory.v4_1.teacher.cache import TeacherCache, build_cache_identity, content_identity
from latent_working_memory.v4_1.teacher.replay import load_teacher
from latent_working_memory.v4_1.teacher.runtime import RuntimeTeacher


def _tensor_metadata(tensor):
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
        "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset(),
        "contiguous": tensor.is_contiguous(), "requires_grad": tensor.requires_grad,
        "is_meta": tensor.is_meta,
    }


def _compare_tensor(shared, independent):
    left, right = _tensor_metadata(shared), _tensor_metadata(independent)
    result = {"shared": left, "independent": right, "same_metadata": left == right,
              "same_values": False, "max_abs": None, "first_mismatch_index": None}
    if shared.shape != independent.shape:
        return result
    if shared.is_meta or independent.is_meta:
        return result
    if shared.device != independent.device:
        independent = independent.to(shared.device)
    result["same_values"] = torch.equal(shared, independent)
    result["max_abs"] = 0.0
    if result["same_values"]:
        return result
    shared, independent = shared.detach().reshape(-1), independent.detach().reshape(-1)
    # 大词表矩阵只在不相等时分块统计，不额外常驻整份 FP64 权重。
    for start in range(0, shared.numel(), 1024 * 1024):
        actual = shared[start:start + 1024 * 1024]
        expected = independent[start:start + 1024 * 1024]
        result["max_abs"] = max(
            result["max_abs"], float((actual.double() - expected.double()).abs().max()),
        )
        if result["first_mismatch_index"] is None:
            positions = (actual != expected).nonzero()
            if positions.numel():
                flat_index = start + int(positions[0, 0])
                index, remaining = [], flat_index
                for size in reversed(left["shape"]):
                    index.append(remaining % size)
                    remaining //= size
                result["first_mismatch_index"] = list(reversed(index))
                result["first_shared_value"] = actual[int(positions[0, 0])].item()
                result["first_independent_value"] = expected[int(positions[0, 0])].item()
    return result


def _numeric_attribute(value):
    if isinstance(value, (bool, int, float)) or value is None:
        return True
    return isinstance(value, (tuple, list)) and all(_numeric_attribute(item) for item in value)


def _public_attributes(model):
    return {
        name: {key: value for key, value in vars(module).items()
               if not key.startswith("_") and _numeric_attribute(value)}
        for name, module in model.named_modules()
    }


def _module_configs(model):
    configs = {}
    for name, module in model.named_modules():
        if hasattr(module, "config"):
            configs[name] = module.config.to_dict()
            configs[name]["_attn_implementation"] = module.config._attn_implementation
    return configs


def _different_items(shared, independent):
    return {name: {"shared": shared.get(name), "independent": independent.get(name)}
            for name in sorted(shared.keys() | independent.keys())
            if shared.get(name) != independent.get(name)}


def _compare_named_tensors(shared, independent):
    return {
        "only_shared": sorted(shared.keys() - independent.keys()),
        "only_independent": sorted(independent.keys() - shared.keys()),
        "tensors": {name: _compare_tensor(shared[name], independent[name])
                    for name in sorted(shared.keys() & independent.keys())},
    }


def run(args):
    if args.device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("run this single-GPU diagnostic with CUDA_VISIBLE_DEVICES=0")
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    overrides = {"training": {"dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
                              "teacher_cache_dir": str(Path(args.teacher_cache_dir).expanduser().resolve())}}
    if args.model_name_or_path is not None:
        overrides["model"] = {"model_name_or_path": args.model_name_or_path}
    config = load_config(args.config, overrides)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"diagnostic output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path)
    episodes, _, _ = load_episodes(
        config.training.dataset_dir, tokenizer, split="train",
        min_text_tokens=config.model.pending_size + config.model.recent_size,
        max_text_tokens=config.training.max_train_data_tokens,
        model_window=AutoConfig.from_pretrained(config.model.model_name_or_path).max_position_embeddings,
        max_samples=config.training.max_train_samples, seed=config.seed,
    )
    if args.sample_id is not None:
        episode = next(episode for episode in episodes if episode.id == args.sample_id)
    else:
        documents = {}
        for candidate in episodes:
            documents.setdefault(candidate.document_id, candidate)
        candidates = list(documents.values())
        first = int(torch.randperm(len(candidates), generator=torch.Generator().manual_seed(config.seed))[0])
        episode = candidates[first]
    cache = TeacherCache(config.training.teacher_cache_dir,
                         expected_identity=build_cache_identity(config, tokenizer))
    boundaries = compression_boundaries(
        len(episode.context_ids), config.model.pending_size, config.model.recent_size,
    )
    write_json(output / "plan.json", {
        "config": asdict(config), "query_seed": args.query_seed, "device": str(device),
        "sample_id": episode.id, "document_id": episode.document_id,
        "context_tokens": len(episode.context_ids), "boundaries": boundaries,
    })
    student = OnlineMemoryLM(config.model).to(device)
    shared = load_teacher(config.model, device, student.backbone)
    independent = load_teacher(config.model, device)
    shared_config, independent_config = shared.config.to_dict(), independent.config.to_dict()
    shared_config["_attn_implementation"] = shared.config._attn_implementation
    independent_config["_attn_implementation"] = independent.config._attn_implementation
    attributes = _different_items(_public_attributes(shared), _public_attributes(independent))
    construction = {
        "parameters": _compare_named_tensors(dict(shared.named_parameters()),
                                             dict(independent.named_parameters())),
        "buffers": _compare_named_tensors(dict(shared.named_buffers()),
                                          dict(independent.named_buffers())),
        "rotary_original_inv_freq": _compare_tensor(shared.model.rotary_emb.original_inv_freq,
                                                    independent.model.rotary_emb.original_inv_freq),
        "rotary_attention_scaling": {
            "shared": shared.model.rotary_emb.attention_scaling,
            "independent": independent.model.rotary_emb.attention_scaling,
        },
        "shared_config": shared_config, "independent_config": independent_config,
        "config_differences": _different_items(shared_config, independent_config),
        "module_config_differences": _different_items(_module_configs(shared),
                                                     _module_configs(independent)),
        "module_class_differences": _different_items(
            {name: type(module).__qualname__ for name, module in shared.named_modules()},
            {name: type(module).__qualname__ for name, module in independent.named_modules()},
        ),
        "public_attribute_differences": attributes,
    }
    write_json(output / "construction.json", construction)
    runtimes = [RuntimeTeacher(model, config, cache) for model in (shared, independent)]
    prefixes = [DynamicCache(config=model.config) for model in (shared, independent)]
    generators = [torch.Generator().manual_seed(args.query_seed) for _ in runtimes]
    identity = content_identity(episode.context_ids)
    first_difference, boundary_reports = None, []
    layered_fields = ("queries", "outputs", "log_mass", "scales_o", "scales_z")
    for boundary in boundaries:
        references = [runtime._reference(episode, prefix, identity, *boundary, generator)
                      for runtime, prefix, generator in zip(runtimes, prefixes, generators)]
        report = {"boundary": boundary, "fields": {}, "distill_branches": [],
                  "same_generator_state": torch.equal(generators[0].get_state(), generators[1].get_state())}
        differences = []
        for field in (*layered_fields, "position_indices"):
            comparison = _compare_tensor(references[0][field], references[1][field])
            report["fields"][field] = comparison
            if not comparison["same_values"] or not comparison["same_metadata"]:
                index = comparison["first_mismatch_index"]
                differences.append({"field": field, "layer_index": index[0] if index and field in layered_fields else None,
                                    "comparison": comparison})
        branches = [reference.get("distill_branches", []) for reference in references]
        report["same_branch_count"] = len(branches[0]) == len(branches[1])
        for index, (left, right) in enumerate(zip(*branches)):
            comparison = _compare_tensor(left[2], right[2])
            branch = {"branch_index": index, "same_input_ids": left[0] == right[0],
                      "same_first_logit_index": left[1] == right[1], "log_probs": comparison}
            report["distill_branches"].append(branch)
            if not all((branch["same_input_ids"], branch["same_first_logit_index"],
                        comparison["same_values"], comparison["same_metadata"])):
                differences.append({"field": f"distill_branches.{index}", "layer_index": None,
                                    "comparison": branch})
        if differences and first_difference is None:
            first_difference = {"boundary": boundary, **min(
                differences, key=lambda item: item["layer_index"] if item["layer_index"] is not None else float("inf"),
            )}
        boundary_reports.append(report)
        write_json(output / f"write_{boundary[0]}.json", report)
        print(json.dumps({"write_index": boundary[0], "different_fields": [item["field"] for item in differences]},
                         ensure_ascii=False), flush=True)
        del references
    write_json(output / "summary.json", {"first_reference_difference": first_difference,
                                          "boundaries": len(boundary_reports),
                                          "config_differences": construction["config_differences"],
                                          "public_attribute_differences": attributes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--teacher-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--query-seed", type=int, default=43)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
