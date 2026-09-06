from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import torch

from cdic_repro.distributed import DistributedContext, initialize_distributed
from cdic_repro.icae_adapter import IcaeV1TrainingAdapter, torch_cosine_similarity
from cdic_repro.msc import MscEpisode, load_msc_episodes, summarize_msc_episodes
from cdic_repro.training import CdicTrainingEngine, EpisodeTrainingResult
from cdic_repro.training_checkpoint import (
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)
from cdic_repro.training_config import CdicMscTrainingConfig, load_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C-DIC on the official MSC episodes")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Validate and summarize MSC data without loading the model.",
    )
    arguments = parser.parse_args()

    config = _apply_cli_overrides(
        load_training_config(arguments.config),
        seed=arguments.seed,
        output_dir=arguments.output_dir,
        resume_from=arguments.resume_from,
    )
    episodes = load_msc_episodes(
        config.data.root,
        session_id=config.data.session_id,
        split=config.data.split,
        max_episodes=config.data.max_episodes,
        max_turns_per_episode=config.data.max_turns_per_episode,
        strict_pairs=config.data.strict_pairs,
    )
    data_summary = summarize_msc_episodes(episodes)
    if arguments.validate_data_only:
        print(json.dumps(data_summary, indent=2, sort_keys=True))
        return
    run_training(config, episodes=episodes, data_summary=data_summary)


def run_training(
    config: CdicMscTrainingConfig,
    *,
    episodes: tuple[MscEpisode, ...],
    data_summary: dict[str, float | int],
) -> None:
    distributed = initialize_distributed(
        primary_device=config.model.device,
        devices=config.model.devices,
    )
    try:
        _run_training_worker(
            config,
            episodes=episodes,
            data_summary=data_summary,
            distributed=distributed,
        )
    finally:
        distributed.close()


def _run_training_worker(
    config: CdicMscTrainingConfig,
    *,
    episodes: tuple[MscEpisode, ...],
    data_summary: dict[str, float | int],
    distributed: DistributedContext,
) -> None:
    output_dir = config.training.output_dir
    _prepare_output_directory(output_dir, resume_from=config.training.resume_from)
    distributed.barrier()

    local_model_config = replace(config.model, device=distributed.device, devices=())
    adapter = IcaeV1TrainingAdapter.load(local_model_config)
    trainable_parameters = tuple(adapter.trainable_parameters())
    distributed.broadcast_parameters(trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        foreach=False,
    )
    engine = CdicTrainingEngine(
        model=adapter,
        similarity=torch_cosine_similarity,
        retrieval_config=config.retrieval,
    )
    progress = TrainingProgress()
    if config.training.resume_from is not None:
        progress = load_training_checkpoint(
            config.training.resume_from,
            model=adapter,
            optimizer=optimizer,
            expected_config_fingerprint=config.fingerprint(),
            rank=distributed.rank,
        )

    if distributed.is_main:
        _write_json(output_dir / "config.resolved.json", config.to_dict())
        _write_json(output_dir / "data_summary.json", data_summary)
        _write_json(
            output_dir / "trainable_parameters.json",
            adapter.trainable_parameter_report(),
        )
    distributed.barrier()

    suffix = "" if distributed.world_size == 1 else f".rank{distributed.rank:02d}"
    metrics_path = output_dir / f"metrics{suffix}.jsonl"
    traces_path = output_dir / f"memory_trace{suffix}.jsonl"
    if config.training.resume_from is not None:
        _truncate_jsonl_after_step(metrics_path, max_step=progress.global_step)
        _truncate_jsonl_after_step(traces_path, max_step=progress.global_step)
    mode = "a" if config.training.resume_from is not None else "w"
    global_step = progress.global_step
    with (
        metrics_path.open(mode, encoding="utf-8") as metrics_file,
        traces_path.open(mode, encoding="utf-8") as traces_file,
    ):
        for epoch in range(progress.epoch, config.training.epochs):
            order = _episode_order(
                len(episodes),
                seed=config.training.seed,
                epoch=epoch,
                shuffle=config.training.shuffle,
            )
            start_position = progress.next_episode_position if epoch == progress.epoch else 0
            for batch_start in range(start_position, len(order), distributed.world_size):
                position = batch_start + distributed.rank
                active_workers = min(distributed.world_size, len(order) - batch_start)
                optimizer.zero_grad(set_to_none=True)
                _synchronize_cuda_devices((distributed.device,))
                _reset_peak_memory((distributed.device,))
                started_at = time.perf_counter()
                result = (
                    engine.train_episode(episodes[order[position]])
                    if position < len(order)
                    else None
                )
                local_gradient_coverage = adapter.gradient_coverage_report()
                distributed.average_gradients(
                    trainable_parameters,
                    active_workers=active_workers,
                )
                grad_norm = _gradient_norm_and_clip(
                    trainable_parameters,
                    max_norm=config.training.max_grad_norm,
                )
                optimizer.step()
                _synchronize_cuda_devices((distributed.device,))
                duration_seconds = time.perf_counter() - started_at
                peak_memory_by_device = _peak_memory_by_device((distributed.device,))
                peak_memory_bytes = sum(peak_memory_by_device.values())
                global_step += 1
                if result is not None:
                    metric_record = {
                        "rank": distributed.rank,
                        "world_size": distributed.world_size,
                        "global_batch_size": active_workers,
                        "epoch": epoch,
                        "episode_position": position,
                        "global_step": global_step,
                        "gradient_norm": grad_norm,
                        "local_gradient_coverage": local_gradient_coverage,
                        "duration_seconds": duration_seconds,
                        "peak_memory_bytes": peak_memory_bytes,
                        "peak_memory_bytes_by_device": peak_memory_by_device,
                        **result.to_summary_dict(),
                    }
                    metrics_file.write(json.dumps(metric_record, ensure_ascii=False) + "\n")
                    for record in result.records:
                        traces_file.write(
                            json.dumps(
                                {
                                    "rank": distributed.rank,
                                    "world_size": distributed.world_size,
                                    "epoch": epoch,
                                    "episode_position": position,
                                    "global_step": global_step,
                                    "episode_id": result.episode_id,
                                    **record.to_dict(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                metrics_file.flush()
                traces_file.flush()

                if result is not None and global_step % config.training.log_every_steps == 0:
                    _print_progress(
                        distributed.rank,
                        epoch,
                        position,
                        len(order),
                        global_step,
                        result,
                        grad_norm,
                        duration_seconds,
                        peak_memory_bytes,
                    )
                next_progress = _next_progress(
                    epoch=epoch,
                    position=batch_start,
                    epoch_size=len(order),
                    world_size=distributed.world_size,
                    global_step=global_step,
                )
                if global_step % config.training.save_every_steps == 0:
                    rng_states = distributed.gather_rng_states()
                    checkpoint_dir = output_dir / "checkpoints"
                    if distributed.is_main:
                        save_training_checkpoint(
                            checkpoint_dir / f"step-{global_step:06d}.pt",
                            model=adapter,
                            optimizer=optimizer,
                            progress=next_progress,
                            config_fingerprint=config.fingerprint(),
                            rng_states=rng_states,
                        )
                        _prune_step_checkpoints(
                            checkpoint_dir,
                            keep_last=config.training.keep_last_checkpoints,
                        )
                    distributed.barrier()
            progress = TrainingProgress(
                epoch=epoch + 1, next_episode_position=0, global_step=global_step
            )

    last_step_was_saved = global_step % config.training.save_every_steps == 0
    if config.training.save_final_checkpoint and not last_step_was_saved:
        rng_states = distributed.gather_rng_states()
        if distributed.is_main:
            save_training_checkpoint(
                output_dir / "checkpoints" / "final.pt",
                model=adapter,
                optimizer=optimizer,
                progress=TrainingProgress(
                    epoch=config.training.epochs,
                    next_episode_position=0,
                    global_step=global_step,
                ),
                config_fingerprint=config.fingerprint(),
                rng_states=rng_states,
            )
        distributed.barrier()


def _prepare_output_directory(output_dir: Path, *, resume_from: Path | None) -> None:
    if resume_from is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"training output directory is not empty: {output_dir}; use a new path or --resume-from"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _apply_cli_overrides(
    config: CdicMscTrainingConfig,
    *,
    seed: int | None,
    output_dir: Path | None,
    resume_from: Path | None,
) -> CdicMscTrainingConfig:
    model = config.model if seed is None else replace(config.model, seed=seed)
    training = replace(
        config.training,
        seed=config.training.seed if seed is None else seed,
        output_dir=config.training.output_dir if output_dir is None else output_dir,
        resume_from=config.training.resume_from if resume_from is None else resume_from,
    )
    return replace(config, model=model, training=training)


def _episode_order(size: int, *, seed: int, epoch: int, shuffle: bool) -> list[int]:
    order = list(range(size))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    return order


def _prune_step_checkpoints(checkpoint_dir: Path, *, keep_last: int | None) -> None:
    if keep_last is None:
        return
    checkpoints = sorted(checkpoint_dir.glob("step-*.pt"))
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


def _next_progress(
    *,
    epoch: int,
    position: int,
    epoch_size: int,
    world_size: int = 1,
    global_step: int,
) -> TrainingProgress:
    next_position = min(position + world_size, epoch_size)
    if next_position == epoch_size:
        return TrainingProgress(epoch=epoch + 1, next_episode_position=0, global_step=global_step)
    return TrainingProgress(
        epoch=epoch,
        next_episode_position=next_position,
        global_step=global_step,
    )


def _synchronize_cuda_devices(devices: tuple[str, ...]) -> None:
    if not torch.cuda.is_available():
        return
    for device in devices:
        torch.cuda.synchronize(torch.device(device))


def _reset_peak_memory(devices: tuple[str, ...]) -> None:
    if not torch.cuda.is_available():
        return
    for device in devices:
        torch.cuda.reset_peak_memory_stats(torch.device(device))


def _peak_memory_by_device(devices: tuple[str, ...]) -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    return {
        device: int(torch.cuda.max_memory_allocated(torch.device(device)))
        for device in devices
    }


def _gradient_norm_and_clip(
    parameters: tuple[object, ...],
    *,
    max_norm: float | None,
) -> float:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    total_squared_norm = sum(
        float(gradient.detach().float().pow(2).sum().item()) for gradient in gradients
    )
    total_norm = math.sqrt(total_squared_norm)
    if max_norm is not None and total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        for gradient in gradients:
            gradient.mul_(scale)
    return total_norm


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _truncate_jsonl_after_step(path: Path, *, max_step: int) -> None:
    if not path.exists():
        return
    retained: list[str] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or "global_step" not in record:
                raise ValueError(f"invalid training log record at {path}:{line_number}")
            if int(record["global_step"]) <= max_step:
                retained.append(json.dumps(record, ensure_ascii=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    temporary.replace(path)


def _print_progress(
    rank: int,
    epoch: int,
    position: int,
    epoch_size: int,
    global_step: int,
    result: EpisodeTrainingResult,
    grad_norm: float | None,
    duration_seconds: float,
    peak_memory_bytes: int | None,
) -> None:
    print(
        json.dumps(
            {
                "rank": rank,
                "epoch": epoch,
                "episode": position + 1,
                "episodes_in_epoch": epoch_size,
                "global_step": global_step,
                "mean_loss": result.mean_loss,
                "backward_turns": result.backward_turns,
                "memory_states": result.final_memory_states,
                "gradient_norm": grad_norm,
                "duration_seconds": duration_seconds,
                "peak_memory_bytes": peak_memory_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
