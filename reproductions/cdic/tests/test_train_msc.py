from __future__ import annotations

from pathlib import Path

from cdic_repro.experiments.train_msc import (
    _apply_cli_overrides,
    _episode_order,
    _next_progress,
    _prune_step_checkpoints,
    _truncate_jsonl_after_step,
)
from cdic_repro.experiments.training_config import load_training_config


def test_episode_order_is_deterministic_per_epoch() -> None:
    assert _episode_order(6, seed=42, epoch=0, shuffle=True) == _episode_order(
        6, seed=42, epoch=0, shuffle=True
    )
    assert _episode_order(6, seed=42, epoch=0, shuffle=True) != _episode_order(
        6, seed=42, epoch=1, shuffle=True
    )
    assert _episode_order(4, seed=42, epoch=0, shuffle=False) == [0, 1, 2, 3]


def test_next_progress_advances_epoch_after_last_episode() -> None:
    assert (
        _next_progress(epoch=0, position=1, epoch_size=3, global_step=2).next_episode_position == 2
    )
    completed = _next_progress(epoch=0, position=2, epoch_size=3, global_step=3)
    assert completed.epoch == 1
    assert completed.next_episode_position == 0

    distributed = _next_progress(
        epoch=0,
        position=2,
        epoch_size=7,
        world_size=2,
        global_step=2,
    )
    assert distributed.epoch == 0
    assert distributed.next_episode_position == 4


def test_checkpoint_retention_keeps_latest_steps(tmp_path: Path) -> None:
    for step in (1, 2, 10):
        (tmp_path / f"step-{step:06d}.pt").write_text("checkpoint", encoding="utf-8")

    _prune_step_checkpoints(tmp_path, keep_last=2)

    assert [path.name for path in sorted(tmp_path.glob("step-*.pt"))] == [
        "step-000002.pt",
        "step-000010.pt",
    ]


def test_resume_truncates_uncheckpointed_log_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"global_step": 1, "loss": 2.0}\n{"global_step": 2, "loss": 1.0}\n',
        encoding="utf-8",
    )

    _truncate_jsonl_after_step(path, max_step=1)

    assert path.read_text(encoding="utf-8") == '{"global_step": 1, "loss": 2.0}\n'


def test_cli_seed_and_output_overrides_update_model_and_training(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
{
  "model": {"model_path": "/model", "checkpoint_path": "/checkpoint"},
  "data": {"root": "/data"},
  "retrieval": {},
  "training": {"output_dir": "/old", "seed": 42}
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = _apply_cli_overrides(
        load_training_config(config_path),
        seed=43,
        output_dir=Path("/new"),
        resume_from=Path("/resume.pt"),
    )

    assert config.model.seed == 43
    assert config.training.seed == 43
    assert config.training.output_dir == Path("/new")
    assert config.training.resume_from == Path("/resume.pt")
