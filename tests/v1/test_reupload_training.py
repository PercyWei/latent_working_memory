import json
from contextlib import contextmanager
from dataclasses import replace

import swanlab
import torch

from latent_working_memory.data_preparation.pipeline import prepare_fineweb
from latent_working_memory.v1 import training
from latent_working_memory.v1.reupload_training import prepare_replay, replay_training


def normalize(value):
    if isinstance(value, swanlab.Text):
        return {"text": value.content, "caption": value.caption}
    if isinstance(value, swanlab.echarts.Table):
        return value.html_content
    if isinstance(value, (swanlab.echarts.Line, swanlab.echarts.Bar)):
        return json.loads(value.dump_options_with_quotes())
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


class RecordingRun:
    def __init__(self):
        self.events = []

    def log(self, values, step):
        self.events.append((step, {key: normalize(value) for key, value in values.items()}))


def test_live_training_and_saved_replay_emit_identical_records(
    tmp_path, tiny_config, tokenizer, preparation_records, preparation_recipe, components, monkeypatch,
):
    config = replace(tiny_config, split_fractions=(0.6, 0.2, 0.2), eval_every=1)
    preparation_recipe = replace(preparation_recipe, samples_per_task=(16, 16, 16),
                                 candidates_per_document=16)
    data = tmp_path / "data"
    prepare_fineweb(preparation_records, tokenizer, config, data, preparation_recipe)
    live = RecordingRun()
    configs = []

    @contextmanager
    def tracking(output_dir, config, *args, **kwargs):
        configs.append(config)
        yield live

    monkeypatch.setattr(training, "swanlab_run", tracking)
    monkeypatch.setattr(training, "load_backbone", lambda *args: (tokenizer, components[0]))
    output = tmp_path / "train"
    training.run_pretraining(config, data / "semantic", output, torch.device("cpu"),
                             max_steps=2, save_every=1)
    prepared = prepare_replay(output)
    replay = RecordingRun()
    replay_training(replay, prepared)
    assert prepared["config"] == json.loads(json.dumps(configs[0]))
    assert replay.events == live.events
    assert [step for step, _ in replay.events] == [0, 1, 1, 2, 2]
    # Future reports already exist, but each event contains only its own scalar data.
    first = replay.events[0][1]
    assert isinstance(first["dev/overview/ae/nll/dev/memory"], float)
    assert "dev/overview/ae/nll" not in first


def test_replay_preserves_inherited_token_counters(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"batch_size": 2, "gradient_accumulation_steps": 2}))
    (tmp_path / "provenance.json").write_text(json.dumps({
        "run_identity": {"world_size": 2, "evaluation_preparations": {"semantic": "data"}},
        "preparation": {}, "inherited_steps": 5000,
    }))
    rows = [{"step": step, "input_tokens": 20, "target_tokens": 30} for step in (5001, 5002)]
    (tmp_path / "train-from-005000-test.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (tmp_path / "resources-from-005000-test.json").write_text(json.dumps({
        "completed_steps": 5002, "cumulative_input_tokens": 1000, "cumulative_target_tokens": 2000,
    }))
    (tmp_path / "semantic").mkdir()
    (tmp_path / "semantic/dev-step-005002.json").write_text(json.dumps({"training_input_tokens": 1000}))
    prepared = prepare_replay(tmp_path)
    assert prepared["initial_input_tokens"] == 960
    assert prepared["initial_target_tokens"] == 1940
    assert prepared["config"]["global_batch_size"] == 8
    assert prepared["records"][0]["step"] == 5001
