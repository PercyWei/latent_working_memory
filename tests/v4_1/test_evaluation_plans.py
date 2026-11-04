"""旧评估计划省略默认字段时按有效配置复用，真实差异仍严格拒绝。"""

from dataclasses import asdict, replace
import json

import pytest

from latent_working_memory.v4_1.config import ExperimentConfig, TrainingConfig
from latent_working_memory.v4_1.plans import prepare_evaluation_plan


@pytest.fixture
def saved_plan(tmp_path):
    run_dir = tmp_path / "run"
    config = ExperimentConfig(training=TrainingConfig(output_dir=str(run_dir / "train")))
    expected = prepare_evaluation_plan(run_dir, config, "test", "test", {"data": tmp_path / "data"})
    path = run_dir / "plan/eval-test/test.json"
    saved = json.loads(path.read_text())
    saved["model"].pop("inner_objective")
    saved["model"].pop("gradient_checkpointing")
    path.write_text(json.dumps(saved), encoding="utf-8")
    return run_dir, config, path, expected


def test_plan_reuses_effective_model_defaults(saved_plan):
    run_dir, config, path, expected = saved_plan
    actual = prepare_evaluation_plan(run_dir, config, "test", "test", {})
    assert actual == expected
    assert actual["model"] == asdict(config.model)
    assert json.loads(path.read_text()) == expected


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
        prepare_evaluation_plan(run_dir, config, "test", "test", {})
    assert path.read_bytes() == original


def test_omitted_method_cannot_reuse_new_inner_method(saved_plan):
    run_dir, config, path, _ = saved_plan
    changed = replace(config, model=replace(config.model, inner_objective="output_only_mass_calibrated"))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        prepare_evaluation_plan(run_dir, changed, "test", "test", {})
    assert path.read_bytes() == original
