"""冻结 PPO checkpoint，采集自主决策点的隐藏向量与真实 pending 长度。"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import torch
from transformers import set_seed

from latent_working_memory.runtime import initialize_device
from ..cli import nonnegative_int, positive_int
from ..config import config_from_dict
from ..io import write_json
from ..modeling.compressor import MemoryModel
from ..rollout import execute_batch
from ..training.checkpoint import checkpoint_file, read_checkpoint


def _now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S Asia/Shanghai")


def _inputs(checkpoint_path, documents_file):
    checkpoint_path = checkpoint_file(checkpoint_path)
    state = read_checkpoint(checkpoint_path)
    config = config_from_dict(state["config"])
    if not config.uses_boundary_policy or config.training.objective.strategy != "ppo":
        raise ValueError("pending-length features require a PPO boundary-policy checkpoint")
    documents_file = Path(documents_file).expanduser().resolve()
    inputs = torch.load(documents_file, map_location="cpu", weights_only=False)
    if set(inputs) != {"checkpoint", "tokenization", "data_identity", "documents"}:
        raise ValueError("documents file does not match the pending-length input contract")
    identities = {split: state["binding"][split]["identity"] for split in ("train", "dev")}
    if (inputs["checkpoint"] != str(checkpoint_path)
        or inputs["tokenization"] != state["binding"]["tokenization"]
        or inputs["data_identity"] != identities):
        raise ValueError("documents file differs from the checkpoint binding")
    documents = inputs["documents"]
    if any(set(document) != {"id", "split", "input_ids"}
           or document["split"] not in {"train", "dev"} for document in documents):
        raise ValueError("document rows must contain id, split and input_ids")
    counts = {split: sum(row["split"] == split for row in documents) for split in identities}
    if any(counts[split] != identities[split]["documents"] for split in counts):
        raise ValueError("document counts differ from the checkpoint binding")
    if len({document["id"] for document in documents}) != len(documents):
        raise ValueError("train/dev must contain distinct documents")
    step = state["cursor"]["step"]
    transition = config.training.objective.policy_transition
    temperature = (1.0 if transition is None
                   else transition.temperature(step, config.training.loop.max_steps))
    metadata = {
        "checkpoint": str(checkpoint_path), "checkpoint_step": step,
        "documents_file": str(documents_file), "tokenization": inputs["tokenization"],
        "seed": config.seed, "temperature": temperature,
        "recent_size": config.recent_size(training=True),
        "pending_stride": config.model.structure.pending_stride,
        "pending_size": config.model.structure.pending_size,
        "data_identity": identities, "total_documents": counts,
        "mode": "sampled", "paths_per_document": 1,
        "sampling_seed": "blake2b(seed:pending-length-probe:split:document_id), 8 bytes",
        "feature_source": "event_features[1:-1][decision_mask]",
        "pending_definition": "Pending before the arriving unit; excludes recent.",
        "after_first_compress_definition": "A previous autonomous action compressed; "
                                           "the first compress action itself is before.",
        "read_tokens_definition": "Cumulative source tokens including the arriving unit.",
        "qa_scoring": False,
    }
    return state, config, documents, metadata


def _uniforms(document, metadata):
    text = f"{metadata['seed']}:pending-length-probe:{document['split']}:{document['id']}"
    seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
    generator = torch.Generator().manual_seed(seed)
    prefix = max(0, len(document["input_ids"]) - metadata["recent_size"])
    count = max(0, (prefix - 1) // metadata["pending_stride"])
    return torch.rand(count, generator=generator).tolist()


def _document_record(document, path, metadata):
    mask = torch.tensor(path.decision_mask, dtype=torch.bool, device=path.event_features.device)
    features = path.event_features[1:-1][mask].detach().cpu()
    prefix = len(document["input_ids"]) - len(path.recent_ids)
    stride = metadata["pending_stride"]
    units = [min(stride, prefix - offset) for offset in range(0, prefix, stride)]
    pending = units[0] if units else 0
    read = pending
    seen_compress = False
    lengths, after, reads, actions = [], [], [], []
    for unit, action, allowed in zip(units[1:], path.actions, path.decision_mask, strict=True):
        read += unit
        if allowed != (pending >= metadata["pending_size"]) or (action and not allowed):
            raise ValueError("autonomous action mask disagrees with reconstructed pending")
        if allowed:
            lengths.append(pending)
            after.append(seen_compress)
            reads.append(read)
            actions.append(action)
        pending = unit if action else pending + unit
        seen_compress |= action
    trace = path.boundary_trace
    if (len(features) != len(lengths) or lengths != trace["pending_tokens_before"]
        or [index for index, action in enumerate(actions) if action] != trace["compress_indices"]):
        raise ValueError("hidden features, pending lengths and boundary trace are misaligned")
    if not torch.isfinite(features).all():
        raise ValueError("hidden features contain non-finite values")
    return {
        "id": document["id"], "split": document["split"], "features": features,
        "pending_tokens_before": torch.tensor(lengths, dtype=torch.int32),
        "after_first_compress": torch.tensor(after, dtype=torch.bool),
        "read_tokens": torch.tensor(reads, dtype=torch.int32),
        "compress_actions": torch.tensor(actions, dtype=torch.bool),
    }


def collect(checkpoint, documents_file, output_dir, batch_size=2, shard_index=0, num_shards=1):
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("feature workers use one GPU without torchrun")
    folder = Path(output_dir).expanduser().resolve() / "features"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"shard_{shard_index}.pt"
    if destination.exists():
        raise FileExistsError(f"feature shard already exists: {destination}")
    progress_path = folder / f"progress_{shard_index}.json"
    progress = {
        "status": "running", "started_at": _now(), "updated_at": _now(),
        "shard_index": shard_index, "num_shards": num_shards,
        "batch_size": batch_size, "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "completed_documents": 0, "decisions": 0,
    }
    write_json(progress_path, progress)
    started = time.monotonic()
    try:
        state, config, documents, metadata = _inputs(checkpoint, documents_file)
        selected = documents[shard_index::num_shards]
        if not selected:
            raise ValueError("feature shard has no documents")
        metadata.update(shard_index=shard_index, num_shards=num_shards, batch_size=batch_size)
        progress.update(total_documents=len(selected), checkpoint=metadata["checkpoint"])
        write_json(progress_path, progress)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for frozen Writer feature collection")
        device = initialize_device("cuda")
        set_seed(config.seed)
        model = MemoryModel(config.model_name_or_path, asdict(config.model),
                            asdict(config.compute), device, with_value_head=True)
        model.load_trainable_state_dict(state["model"])
        model.requires_grad_(False)
        model.eval()
        del state
        records = []
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            with torch.no_grad(), model.forward_context():
                paths = execute_batch(
                    model, [document["input_ids"] for document in batch],
                    metadata["pending_stride"],
                    uniforms=[_uniforms(document, metadata) for document in batch],
                    temperature=metadata["temperature"], pending_size=metadata["pending_size"],
                    recent_size=metadata["recent_size"], collect_features=True,
                    collect_boundary=True,
                )
            for document, path in zip(batch, paths, strict=True):
                record = _document_record(document, path, metadata)
                records.append(record)
                progress["decisions"] += len(record["features"])
            del paths, path
            progress.update(completed_documents=len(records), updated_at=_now(),
                            seconds=time.monotonic() - started)
            write_json(progress_path, progress)
            print(f"shard={shard_index} documents={len(records)}/{len(selected)} "
                  f"decisions={progress['decisions']} seconds={progress['seconds']:.1f}",
                  flush=True)
        metadata.update(feature_dtype=str(records[0]["features"].dtype),
                        feature_width=model.width, collected_at=_now())
        temporary = destination.with_suffix(".pt.tmp")
        torch.save({"metadata": metadata, "documents": records}, temporary)
        temporary.replace(destination)
        progress.update(status="complete", updated_at=_now(), seconds=time.monotonic() - started)
        write_json(progress_path, progress)
    except Exception as error:
        progress.update(status="failed", updated_at=_now(),
                        error=f"{type(error).__name__}: {error}")
        write_json(progress_path, progress)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--documents-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--shard-index", type=nonnegative_int, default=0)
    parser.add_argument("--num-shards", type=positive_int, default=1)
    args = parser.parse_args(argv)
    collect(args.checkpoint, args.documents_file, args.output_dir,
            args.batch_size, args.shard_index, args.num_shards)


if __name__ == "__main__":
    main()
