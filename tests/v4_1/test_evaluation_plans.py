"""评估计划固定协议、checkpoint 追加与模型配置校验。"""

from dataclasses import asdict, replace
import json

import pytest

from latent_working_memory.v4_1.config import ExperimentConfig, TrainingConfig
from latent_working_memory.v4_1.plans import prepare_evaluation_plan


@pytest.fixture
def saved_plan(tmp_path):
    run_dir = tmp_path / "run"
    config = ExperimentConfig(training=TrainingConfig(output_dir=str(run_dir / "train")))
    checkpoints = [str(run_dir / f"train/checkpoints/global_step_{step}/state.pt") for step in (2, 10)]
    expected = prepare_evaluation_plan(run_dir, config, "test", "test", {
        "data": tmp_path / "data", "checkpoints": checkpoints,
    })
    path = run_dir / "plan/eval-test/test.json"
    saved = json.loads(path.read_text())
    saved["model"].pop("inner_objective")
    saved["model"].pop("gradient_checkpointing")
    path.write_text(json.dumps(saved), encoding="utf-8")
    return run_dir, config, path, expected


def test_plan_resume_preserves_protocol_and_appends_checkpoints(saved_plan):
    run_dir, config, path, expected = saved_plan
    original = path.read_bytes()
    with pytest.raises(ValueError, match="plan already exists"):
        prepare_evaluation_plan(run_dir, config, "test", "test", {})
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_evaluation_plan(run_dir, config, "test", "test", {}, overwrite=True, resume=True)
    with pytest.raises(ValueError, match="plan does not exist"):
        prepare_evaluation_plan(run_dir, config, "test", "ruler", {}, resume=True)
    actual = prepare_evaluation_plan(run_dir, config, "test", "test", {
        "data": run_dir / "other-data", "checkpoints": expected["checkpoints"][:1],
        "max_samples": 1, "max_new_tokens": 3, "lengths": [32768],
    }, resume=True)
    assert actual == expected
    assert actual["model"] == asdict(config.model)
    assert path.read_bytes() == original
    earlier = str(run_dir / "train/checkpoints/global_step_1/state.pt")
    added = str(run_dir / "train/checkpoints/global_step_17/state.pt")
    expanded = prepare_evaluation_plan(run_dir, config, "test", "test", {
        "checkpoints": [added, earlier, *expected["checkpoints"]], "max_new_tokens": 3,
    }, resume=True)
    assert expanded == {**expected, "checkpoints": [earlier, *expected["checkpoints"], added]}
    original = path.read_bytes()
    assert prepare_evaluation_plan(run_dir, config, "test", "test", {
        "checkpoints": expected["checkpoints"][:1],
    }, resume=True) == expanded
    assert path.read_bytes() == original


@pytest.mark.parametrize(("field", "value", "error", "message"), [
    ("inner_objective", "output_only_mass_calibrated", ValueError, "settings differ"),
    ("unknown_model_field", True, TypeError, "unexpected keyword"),
    ("inner_steps", 0, ValueError, "inner_steps must be a positive integer"),
])
def test_plan_defaults_preserve_strict_validation(saved_plan, field, value, error, message):
    run_dir, config, path, _ = saved_plan
    saved = json.loads(path.read_text())
    saved["model"][field] = value
    path.write_text(json.dumps(saved), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(error, match=message):
        prepare_evaluation_plan(run_dir, config, "test", "test", {}, resume=True)
    assert path.read_bytes() == original


def test_omitted_method_cannot_reuse_new_inner_method(saved_plan):
    run_dir, config, path, _ = saved_plan
    changed = replace(config, model=replace(config.model, inner_objective="output_only_mass_calibrated"))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        prepare_evaluation_plan(run_dir, changed, "test", "test", {}, resume=True)
    assert path.read_bytes() == original
