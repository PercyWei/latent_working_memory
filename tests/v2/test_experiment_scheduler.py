"""正式配置快照、双卡调度、失败停队与续训入口。"""

import argparse
import json
from pathlib import Path

import pytest

from latent_working_memory.v2.pretrain import experiment
from latent_working_memory.v2.pretrain.train import read_experiment


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs/v2/pretrain"
CONFIG_NAMES = [
    "qwen3-4b_pooling_ae-warmup",
    "qwen3-4b_pooling_ae-lm-warmup",
    "qwen3-4b_pooling_ae_dynamic",
    "qwen3-4b_pooling_ae-lm_dynamic",
    "qwen3-4b_pooling_ae-static",
    "qwen3-4b_pooling_ae-lm-static",
]


def arguments(tmp_path):
    return argparse.Namespace(
        experiments=[CONFIG_ROOT / name / "experiment.json" for name in CONFIG_NAMES],
        output_dir=tmp_path,
        run_date="20260917",
        group="test-reconstruction",
        model_path=tmp_path / "model",
        resume=False,
        plan_only=True,
        min_free_gib=75,
    )


def test_six_configs_keep_fixed_batch_and_cover_both_stage_lengths(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    records = experiment.prepare_runs(args)
    monkeypatch.setattr(experiment, "free_memory", lambda: pytest.fail("plan-only queried GPUs"))
    experiment.run_queue(args, [[4, 5], [6, 7]], records)
    expected_stages = [(1, 2), (1, 2), (0, 3), (0, 3), (3, 0), (3, 0)]
    for i, (record, stages) in enumerate(zip(records, expected_stages, strict=True)):
        command = record["command"]
        assert "torch.distributed.run" in command and "--nproc_per_node=2" in command
        assert "--stop-after-steps" not in command
        model, selection, cfg = read_experiment(Path(command[command.index("--experiment") + 1]))
        assert model.model_name_or_path == str(args.model_path)
        assert Path(selection.dataset_dir).is_absolute()
        assert (cfg.warmup_epochs, cfg.multiround_epochs) == stages
        assert cfg.global_batch_size == 32 and cfg.micro_batch_size == (16 if i >= 4 else 8)
        assert 32 // (2 * cfg.micro_batch_size) == (1 if i >= 4 else 2)
        assert cfg.lm_ratio == (0.5 if i % 2 else 0)
        maximum_encoder = 4096 if cfg.warmup_epochs else 2048
        assert cfg.micro_batch_encoder_tokens >= cfg.micro_batch_size * maximum_encoder
        assert cfg.micro_batch_decoder_tokens >= cfg.micro_batch_size * 4616
    args.resume = True
    assert experiment.prepare_runs(args) == records
    args.group = "different"
    with pytest.raises(ValueError, match="same experiment list"):
        experiment.prepare_runs(args)


def test_resume_uses_latest_checkpoint_and_skips_complete_run(tmp_path):
    record = {"output": str(tmp_path), "command": ["train"], "state": "failed"}
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for step in (10, 100):
        (ckpts / f"step-{step:06d}.pt").touch()
    assert experiment.resume_command(record) == ["train", "--resume", str(ckpts / "step-000100.pt")]
    (tmp_path / "training-result.json").write_text(json.dumps({"complete": True}))
    assert experiment.resume_command(record) is None
    assert record["state"] == "complete"


def test_queue_uses_pairs_and_stops_dispatch_after_failure(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    records = experiment.prepare_runs(args)[:3]
    args.plan_only = False
    launches = []

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(launches)
            self.command = command
            self.returncode = None
            self.polls = 0
            launches.append(kwargs)

        def poll(self):
            self.polls += 1
            if self.pid == 100:
                self.returncode = 1
            elif self.polls == 2:
                output = Path(self.command[self.command.index("--output-dir") + 1])
                output.mkdir(parents=True)
                (output / "training-result.json").write_text(json.dumps({"complete": True}))
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(experiment.subprocess, "Popen", Process)
    monkeypatch.setattr(experiment, "free_memory", lambda: {4: 79, 5: 79, 6: 79, 7: 79})
    monkeypatch.setattr(experiment.time, "sleep", lambda _: None)
    with pytest.raises(SystemExit) as error:
        experiment.run_queue(args, [[4, 5], [6, 7]], records)
    assert error.value.code == 1
    assert [x["env"]["CUDA_VISIBLE_DEVICES"] for x in launches] == ["4,5", "6,7"]
    assert all(x["start_new_session"] for x in launches)
    status = json.loads((tmp_path / "status.json").read_text())
    assert [r["state"] for r in status["runs"]] == ["failed", "complete", "queued"]


def test_interrupt_signals_only_owned_torchrun_groups(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    records = experiment.prepare_runs(args)[:2]
    args.plan_only = False
    processes, signals = [], []

    class Process:
        def __init__(self, *a, **kw):
            self.pid = 200 + len(processes)
            self.returncode = None
            processes.append(self)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment.subprocess, "Popen", Process)
    monkeypatch.setattr(experiment, "free_memory", lambda: {4: 79, 5: 79, 6: 79, 7: 79})
    monkeypatch.setattr(experiment.time, "sleep", interrupt)
    monkeypatch.setattr(experiment.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        experiment.run_queue(args, [[4, 5], [6, 7]], records)
    assert [pid for pid, _ in signals] == [200, 201]
    assert all(r["state"] == "interrupted" for r in records)
