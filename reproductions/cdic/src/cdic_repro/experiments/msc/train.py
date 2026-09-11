"""MSC 训练配置、单 episode 训练逻辑与完整训练入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from cdic_repro.config import RetrievalConfig, RetrievedStateOrder
from cdic_repro.credit import build_compression_gradient_plan, build_credit_plan
from cdic_repro.experiments.checkpoint import (
    TrainingProgress,
    load_cdic_checkpoint,
    save_cdic_checkpoint,
)
from cdic_repro.experiments.distributed import DistributedContext, initialize_distributed
from cdic_repro.experiments.msc import MSC_SWANLAB_TAGS
from cdic_repro.experiments.msc.data import (
    MscEpisode,
    load_msc_episodes,
    summarize_msc_episodes,
)
from cdic_repro.experiments.tracking import (
    CDIC_SWANLAB_PROJECT,
    SWANLAB_MODES,
    swanlab_run,
)
from cdic_repro.icae.adapter import (
    IcaeV1AdapterConfig,
    IcaeV1TrainingAdapter,
    torch_cosine_similarity,
)
from cdic_repro.memory_state import MemoryBank
from cdic_repro.model_protocol import CdicTrainingAdapter
from cdic_repro.retrieval import SimilarityFunction, retrieve
from cdic_repro.trace import TurnTrace
from cdic_repro.writeback import NewStatePayload, apply_write_back


@dataclass(frozen=True, slots=True)
class MscDataConfig:
    root: Path
    split: str = "train"
    session_id: int = 4
    max_episodes: int | None = None
    max_turns_per_episode: int | None = None
    strict_pairs: bool = False

    def __post_init__(self) -> None:
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("data.split must be train, valid, or test")
        if not 2 <= self.session_id <= 5:
            raise ValueError("data.session_id must be between 2 and 5")
        if self.split == "train" and self.session_id == 5:
            raise ValueError("official MSC session 5 has no training split")
        if self.max_episodes is not None and self.max_episodes < 1:
            raise ValueError("data.max_episodes must be positive")
        if self.max_turns_per_episode is not None and self.max_turns_per_episode < 1:
            raise ValueError("data.max_turns_per_episode must be positive")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    output_dir: Path
    epochs: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 42
    shuffle: bool = True
    max_grad_norm: float | None = None
    save_every_steps: int = 50
    keep_last_checkpoints: int | None = 2
    save_final_checkpoint: bool = True
    log_every_steps: int = 1
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("training.epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("training.weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("training.max_grad_norm must be positive")
        if self.save_every_steps < 1:
            raise ValueError("training.save_every_steps must be positive")
        if self.keep_last_checkpoints is not None and self.keep_last_checkpoints < 1:
            raise ValueError("training.keep_last_checkpoints must be positive")
        if self.log_every_steps < 1:
            raise ValueError("training.log_every_steps must be positive")

    @property
    def final_checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoints" / "final.pt"


@dataclass(frozen=True, slots=True)
class MscTrainingConfig:
    model: IcaeV1AdapterConfig
    data: MscDataConfig
    retrieval: RetrievalConfig
    training: OptimizationConfig

    def to_dict(self) -> dict[str, object]:
        return _serialize_paths(asdict(self))

    def fingerprint(self) -> str:
        serialized = self.to_dict()
        model = dict(serialized["model"])  # type: ignore[arg-type]
        if model.get("gradient_window_size") == 1:
            model.pop("gradient_window_size")
        serialized["model"] = model
        training = dict(serialized["training"])  # type: ignore[arg-type]
        training["resume_from"] = None
        serialized["training"] = training
        payload = json.dumps(serialized, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingTurnRecord:
    turn_id: str
    query: str
    response: str
    loss: float
    loss_tokens: int
    backward_applied: bool
    trace: TurnTrace

    def to_dict(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "query": self.query,
            "response": self.response,
            "loss": self.loss,
            "loss_tokens": self.loss_tokens,
            "backward_applied": self.backward_applied,
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EpisodeTrainingResult:
    episode_id: str
    mean_loss: float
    turns: int
    loss_tokens: int
    backward_turns: int
    final_memory_states: int
    records: tuple[TrainingTurnRecord, ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "mean_loss": self.mean_loss,
            "turns": self.turns,
            "loss_tokens": self.loss_tokens,
            "backward_turns": self.backward_turns,
            "final_memory_states": self.final_memory_states,
        }


class MscTrainingEngine:
    """在一个 MSC episode 内执行 teacher forcing 和检索感知的有限梯度回传。"""

    def __init__(
        self,
        model: CdicTrainingAdapter,
        similarity: SimilarityFunction,
        retrieval_config: RetrievalConfig | None = None,
    ) -> None:
        self.model = model
        self.similarity = similarity
        self.retrieval_config = retrieval_config or RetrievalConfig()

    def train_episode(self, episode: MscEpisode) -> EpisodeTrainingResult:
        memory = MemoryBank()
        loss_scale = 1.0 / len(episode.turns)
        records: list[TrainingTurnRecord] = []
        backward_turns = 0
        total_loss = 0.0
        total_loss_tokens = 0

        for turn_number, turn in enumerate(episode.turns, start=1):
            retrieval = retrieve(
                memory,
                query_key=self.model.encode_query(turn.query),
                turn=turn_number,
                similarity=self.similarity,
                config=self.retrieval_config,
            )
            retrieved_states = memory.select(retrieval.selected_state_ids)
            credit = build_credit_plan(retrieval)
            training_loss = self.model.response_loss(
                retrieved_states,
                turn.query,
                turn.response,
                credit,
            )
            loss_value = float(training_loss.value)
            backward_applied = training_loss.value.requires_grad
            gradient_plan = build_compression_gradient_plan(
                retrieved_states,
                credit,
                gradient_window_size=self.model.gradient_window_size,
            )
            if backward_applied:
                (training_loss.value * loss_scale).backward(
                    retain_graph=gradient_plan.retained_state_id is not None,
                )
                backward_turns += 1

            compressed = self.model.compress_gold(
                retrieved_states,
                turn.query,
                turn.response,
                credit,
            )
            write_back = apply_write_back(
                memory,
                retrieval=retrieval,
                payload=NewStatePayload(
                    latent=compressed.latent,
                    retrieval_key=compressed.retrieval_key,
                    provenance=compressed.provenance,
                    graph_connected=True,
                    gradient_depth=compressed.gradient_depth,
                ),
                turn=turn_number,
            )
            records.append(
                TrainingTurnRecord(
                    turn_id=turn.turn_id,
                    query=turn.query,
                    response=turn.response,
                    loss=loss_value,
                    loss_tokens=training_loss.token_count,
                    backward_applied=backward_applied,
                    trace=TurnTrace(
                        turn=turn_number,
                        query_id=turn.turn_id,
                        retrieval=retrieval,
                        credit=credit,
                        write_back=write_back,
                    ),
                )
            )
            total_loss += loss_value
            total_loss_tokens += training_loss.token_count

        return EpisodeTrainingResult(
            episode_id=episode.episode_id,
            mean_loss=total_loss / len(episode.turns),
            turns=len(episode.turns),
            loss_tokens=total_loss_tokens,
            backward_turns=backward_turns,
            final_memory_states=len(memory),
            records=tuple(records),
        )


def load_training_config(path: Path) -> MscTrainingConfig:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise TypeError("training config must contain a JSON object")

    model = _mapping(payload, "model")
    data = _mapping(payload, "data")
    retrieval = _mapping(payload, "retrieval")
    training = _mapping(payload, "training")
    return MscTrainingConfig(
        model=IcaeV1AdapterConfig(
            model_path=Path(_required_string(model, "model_path")),
            checkpoint_path=Path(_required_string(model, "checkpoint_path")),
            device=str(model.get("device", "cuda:0")),
            devices=_device_tuple(model.get("devices")),
            dtype=str(model.get("dtype", "bfloat16")),
            memory_size=int(model.get("memory_size", 128)),
            max_turn_tokens=int(model.get("max_turn_tokens", 512)),
            max_new_tokens=int(model.get("max_new_tokens", 128)),
            lora_alpha=int(model.get("lora_alpha", 32)),
            lora_dropout=float(model.get("lora_dropout", 0.05)),
            lora_rank=int(model.get("lora_rank", 128)),
            seed=int(training.get("seed", 42)),
            use_ft_markers=bool(model.get("use_ft_markers", True)),
            turn_template=str(
                model.get("turn_template", "<s>[INST] {query} [/INST] {response} </s>")
            ),
            gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
            gradient_window_size=int(model.get("gradient_window_size", 1)),
        ),
        data=MscDataConfig(
            root=Path(_required_string(data, "root")),
            split=str(data.get("split", "train")),
            session_id=int(data.get("session_id", 4)),
            max_episodes=_optional_int(data.get("max_episodes")),
            max_turns_per_episode=_optional_int(data.get("max_turns_per_episode")),
            strict_pairs=bool(data.get("strict_pairs", False)),
        ),
        retrieval=RetrievalConfig(
            threshold=float(retrieval.get("threshold", 0.8)),
            decay=float(retrieval.get("decay", 0.05)),
            retrieved_state_order=RetrievedStateOrder(
                str(retrieval.get("retrieved_state_order", "score_desc"))
            ),
            max_retrieved=_optional_int(retrieval.get("max_retrieved")),
        ),
        training=OptimizationConfig(
            output_dir=Path(_required_string(training, "output_dir")),
            epochs=int(training.get("epochs", 2)),
            learning_rate=float(training.get("learning_rate", 2e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            seed=int(training.get("seed", 42)),
            shuffle=bool(training.get("shuffle", True)),
            max_grad_norm=_optional_float(training.get("max_grad_norm")),
            save_every_steps=int(training.get("save_every_steps", 50)),
            keep_last_checkpoints=_optional_int(training.get("keep_last_checkpoints", 2)),
            save_final_checkpoint=bool(training.get("save_final_checkpoint", True)),
            log_every_steps=int(training.get("log_every_steps", 1)),
            resume_from=_optional_path(training.get("resume_from")),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C-DIC on the official MSC episodes")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--swanlab-mode", choices=SWANLAB_MODES, default="disabled")
    parser.add_argument("--swanlab-project", default=CDIC_SWANLAB_PROJECT)
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
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
    run_training(
        config,
        episodes=episodes,
        data_summary=data_summary,
        swanlab_mode=arguments.swanlab_mode,
        swanlab_project=arguments.swanlab_project,
        swanlab_group=arguments.swanlab_group,
        swanlab_tags=MSC_SWANLAB_TAGS + tuple(arguments.swanlab_tag),
    )


def run_training(
    config: MscTrainingConfig,
    episodes: tuple[MscEpisode, ...],
    data_summary: dict[str, float | int],
    swanlab_mode: str = "disabled",
    swanlab_project: str = CDIC_SWANLAB_PROJECT,
    swanlab_group: str | None = None,
    swanlab_tags: tuple[str, ...] = MSC_SWANLAB_TAGS,
) -> None:
    if swanlab_mode != "disabled" and not swanlab_group:
        raise ValueError("enabled SwanLab runs require a group")
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
            swanlab_mode=swanlab_mode,
            swanlab_project=swanlab_project,
            swanlab_group=swanlab_group,
            swanlab_tags=swanlab_tags,
        )
    finally:
        distributed.close()


def _run_training_worker(
    config: MscTrainingConfig,
    episodes: tuple[MscEpisode, ...],
    data_summary: dict[str, float | int],
    distributed: DistributedContext,
    swanlab_mode: str,
    swanlab_project: str,
    swanlab_group: str | None,
    swanlab_tags: tuple[str, ...],
) -> None:
    output_dir = config.training.output_dir
    _prepare_output_directory(output_dir, resume_from=config.training.resume_from)
    distributed.barrier()

    local_model_config = replace(config.model, device=distributed.device, devices=())
    adapter = IcaeV1TrainingAdapter.load(local_model_config)
    trainable_parameters = tuple(
        parameter for parameter in adapter.model.parameters() if parameter.requires_grad
    )
    distributed.broadcast_parameters(trainable_parameters)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        foreach=False,
    )
    engine = MscTrainingEngine(
        model=adapter,
        similarity=torch_cosine_similarity,
        retrieval_config=config.retrieval,
    )
    progress = TrainingProgress()
    if config.training.resume_from is not None:
        progress = load_cdic_checkpoint(
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
        swanlab_run(
            output_dir,
            config.to_dict() | {"data_summary": data_summary},
            mode=swanlab_mode if distributed.is_main else "disabled",
            project=swanlab_project,
            group=swanlab_group,
            tags=swanlab_tags,
            job_type="train",
        ) as tracking,
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
                if (
                    swanlab_mode != "disabled"
                    and global_step % config.training.log_every_steps == 0
                ):
                    tracking_records = distributed.gather_objects(
                        _training_tracking_record(
                            result,
                            duration_seconds=duration_seconds,
                            peak_memory_bytes=peak_memory_bytes,
                        )
                    )
                    if tracking is not None:
                        tracking.log(
                            _aggregate_training_metrics(tracking_records, grad_norm),
                            step=global_step,
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
                        save_cdic_checkpoint(
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
            save_cdic_checkpoint(
                config.training.final_checkpoint_path,
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


def _prepare_output_directory(output_dir: Path, resume_from: Path | None) -> None:
    if resume_from is None and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"training output directory is not empty: {output_dir}; use a new path or --resume-from"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _apply_cli_overrides(
    config: MscTrainingConfig,
    seed: int | None,
    output_dir: Path | None,
    resume_from: Path | None,
) -> MscTrainingConfig:
    model = config.model if seed is None else replace(config.model, seed=seed)
    training = replace(
        config.training,
        seed=config.training.seed if seed is None else seed,
        output_dir=config.training.output_dir if output_dir is None else output_dir,
        resume_from=config.training.resume_from if resume_from is None else resume_from,
    )
    return replace(config, model=model, training=training)


def _episode_order(size: int, seed: int, epoch: int, shuffle: bool) -> list[int]:
    order = list(range(size))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)
    return order


def _prune_step_checkpoints(checkpoint_dir: Path, keep_last: int | None) -> None:
    if keep_last is None:
        return
    checkpoints = sorted(checkpoint_dir.glob("step-*.pt"))
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


def _next_progress(
    epoch: int,
    position: int,
    epoch_size: int,
    global_step: int,
    world_size: int = 1,
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


def _training_tracking_record(
    result: EpisodeTrainingResult | None,
    duration_seconds: float,
    peak_memory_bytes: int,
) -> dict[str, float | int] | None:
    if result is None:
        return None
    retrieval_records = [record for record in result.records if record.trace.turn > 1]
    return {
        "loss_sum": result.mean_loss * result.turns,
        "turns": result.turns,
        "backward_turns": result.backward_turns,
        "retrieval_turns": len(retrieval_records),
        "on_topic_turns": sum(record.trace.retrieval.on_topic for record in retrieval_records),
        "selected_states": sum(
            len(record.trace.retrieval.selected_state_ids) for record in retrieval_records
        ),
        "final_memory_states": result.final_memory_states,
        "duration_seconds": duration_seconds,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _aggregate_training_metrics(
    gathered: tuple[object, ...],
    gradient_norm: float,
) -> dict[str, float]:
    records = [record for record in gathered if record is not None]
    if not records or not all(isinstance(record, dict) for record in records):
        raise TypeError("distributed SwanLab records must be mappings")
    turns = sum(int(record["turns"]) for record in records)
    metrics = {
        "train/mean_turn_nll": sum(float(record["loss_sum"]) for record in records) / turns,
        "train/gradient_norm": gradient_norm,
        "train/backward_fraction": sum(
            int(record["backward_turns"]) for record in records
        )
        / turns,
        "memory/mean_final_states": sum(
            int(record["final_memory_states"]) for record in records
        )
        / len(records),
        "resources/step_seconds": max(
            float(record["duration_seconds"]) for record in records
        ),
        "resources/peak_memory_gib": max(
            int(record["peak_memory_bytes"]) for record in records
        )
        / 1024**3,
    }
    retrieval_turns = sum(int(record["retrieval_turns"]) for record in records)
    if retrieval_turns:
        metrics.update(
            {
                "retrieval/on_topic_rate": sum(
                    int(record["on_topic_turns"]) for record in records
                )
                / retrieval_turns,
                "retrieval/mean_selected_states": sum(
                    int(record["selected_states"]) for record in records
                )
                / retrieval_turns,
            }
        )
    return metrics


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _truncate_jsonl_after_step(path: Path, max_step: int) -> None:
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


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"training config field {key!r} must be an object")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"training config field {key!r} must be a non-empty string")
    return value


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("resume_from must be null or a non-empty path")
    return Path(value)


def _device_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("model.devices must be a non-empty list of device strings")
    return tuple(value)


def _serialize_paths(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_paths(item) for item in value]
    if isinstance(value, RetrievedStateOrder):
        return value.value
    return value


if __name__ == "__main__":
    main()
