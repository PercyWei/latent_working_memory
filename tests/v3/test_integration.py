from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latent_working_memory.v3.config import (
    ExperimentConfig,
    ModelConfig,
    ObjectiveConfig,
    TrainingConfig,
)
from latent_working_memory.v3.engine import TokenMemoryEngine
from latent_working_memory.v3.objective import TokenMemoryTask, memory_change_score
from latent_working_memory.v3.pretrain_data import PretrainExample
from latent_working_memory.v3.runtime import (
    dataset_identity,
    load_checkpoint,
    load_initialization,
    make_run,
    read_checkpoint,
    save_checkpoint,
)
from .test_model import build_model
from .test_objective import TinyTokenizer, trajectory


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def engine_config(micro_batch_size=1):
    return TrainingConfig(
        dataset_dir="unused",
        output_dir="unused",
        micro_batch_size_per_gpu=micro_batch_size,
        gradient_accumulation_steps=2,
        learning_rate=0.003,
        weight_decay=0.0,
        gradient_clip=100.0,
    )


def tiny_task(model_type, method, stage, **options):
    config = ObjectiveConfig(
        method=method,
        stage=stage,
        segment_tokens=3 if method == "icae_multi" else 12,
        ac_min_segment_tokens=3,
        ac_max_segment_tokens=3,
        ac_bptt_steps=2,
        **options,
    )
    return TokenMemoryTask(
        build_model(
            model_type, attention_implementation="sdpa" if model_type == "qwen3" else "eager"
        ),
        TinyTokenizer(),
        config,
    )


def pretraining_examples():
    text = (3, 4, 5, 6, 7, 8)
    return (
        PretrainExample("ae", "pretrain-doc", "pretrain-cluster", "ae", text, text),
        PretrainExample("lm", "pretrain-doc", "pretrain-cluster", "continuation", text, (9, 10)),
    )


def parameter_snapshot(model, requires_grad):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad == requires_grad
    }


def assert_frozen(model, before):
    parameters = dict(model.named_parameters())
    for name, value in before.items():
        assert not parameters[name].requires_grad
        assert parameters[name].grad is None
        torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)


def assert_writer_updated(model, before):
    after = parameter_snapshot(model, True)
    assert not torch.equal(after["codec.memory_embeddings"], before["codec.memory_embeddings"])
    assert any(
        not torch.equal(value, before[name]) for name, value in after.items() if "lora_B" in name
    )
    assert all(torch.isfinite(value).all() for value in after.values())


@pytest.mark.parametrize("model_type", ["llama", "qwen3"])
@pytest.mark.parametrize("micro_batch_size", [1, 2])
@pytest.mark.parametrize(
    "method,stage",
    [
        ("icae_single", "pretrain"),
        ("icae_single", "qa"),
        ("icae_multi", "pretrain"),
        ("icae_multi", "qa"),
        ("autocompressors", "lm"),
        ("memory_change", "pretrain"),
        ("memory_change", "warmup"),
        ("memory_change", "policy"),
        ("information_loss", "pretrain"),
        ("information_loss", "warmup"),
        ("information_loss", "policy"),
    ],
)
def test_actual_writer_reader_and_engine_train_every_method_stage(
    model_type, method, stage, micro_batch_size
):
    model = tiny_task(model_type, method, stage)
    engine = TokenMemoryEngine(model, engine_config(micro_batch_size), "cpu")
    engine.initialize()
    batch = (
        pretraining_examples()
        if stage in {"pretrain", "lm"}
        else (trajectory(n=2), replace(trajectory(n=3), trajectory_id="other"))
    )
    frozen, writer = parameter_snapshot(model, False), parameter_snapshot(model, True)

    calls = {"writer": [], "reader": []}

    def observe(module, args, kwargs):
        reading = module is model.codec.decoder.model
        calls["reader" if reading else "writer"].append(kwargs["inputs_embeds"].shape[0])

    hooks = [
        backbone.register_forward_pre_hook(observe, with_kwargs=True)
        for backbone in (
            model.codec.language_model.get_base_model().model,
            model.codec.decoder.model,
        )
    ]
    try:
        metrics = engine.step(batch, epoch=1)
    finally:
        for hook in hooks:
            hook.remove()

    assert max(calls["writer"]) == micro_batch_size
    if micro_batch_size == 2:
        assert max(calls["reader"]) >= 2
    if stage not in {"pretrain", "lm"}:
        assert max(calls["reader"]) <= micro_batch_size * model.cfg.qa_batch_size
    assert metrics["samples"] == 2
    assert math.isfinite(metrics["loss"]) and metrics["loss"] > 0
    assert math.isfinite(metrics["grad_norm"]) and metrics["grad_norm"] > 0
    assert_writer_updated(model, writer)
    assert_frozen(model, frozen)
    after_training = parameter_snapshot(model, True)
    evaluation = engine.eval_batch(batch, epoch=1)
    assert math.isfinite(evaluation["loss"])
    assert evaluation["grad_norm"] is None
    assert model.training and model.codec.language_model.training and model.codec.decoder.training
    for name, value in parameter_snapshot(model, True).items():
        torch.testing.assert_close(value, after_training[name], rtol=0, atol=0)
    assert_frozen(model, frozen)


def calibrated_dynamic_batch(method):
    """Use real scores to put the two ranks on opposite sides of one threshold."""
    model = tiny_task("qwen3", method, "policy")
    first = trajectory(n=2)
    if method == "memory_change":
        second = replace(
            first,
            trajectory_id="different-text",
            segments=(first.segments[0], replace(first.segments[1], input_ids=(30, 31, 32))),
        )
        with torch.no_grad():
            old = model.codec.compress(model.ids(first.segments[0].input_ids))
            scores = [
                float(
                    memory_change_score(
                        old,
                        model.codec.compress(model.ids(row.segments[1].input_ids), [old]),
                        model.cfg.rms_epsilon,
                    )
                )
                for row in (first, second)
            ]
        assert abs(scores[0] - scores[1]) > 1e-6
        model.cfg = replace(model.cfg, threshold_i=sum(scores) / 2)
    else:
        # Selecting one positive and one negative gain is possible because these are
        # normalized next-token distributions from two different real memory states.
        with torch.no_grad():
            old = model.codec.compress(model.ids(first.segments[0].input_ids))
            rewritten = model.codec.compress(model.ids(first.segments[1].input_ids), [old])
            appended = model.codec.compress(model.ids(first.segments[1].input_ids))
            prompt = model.prompt_ids(first.qas["gate0"].question)
            answers = [model.ids((value,)) for value in range(41)]
            losses = [
                model.codec.answer_nll([memory] * 41, [prompt] * 41, answers)
                for memory in (rewritten, torch.cat((old, appended)))
            ]
            gains = losses[0] - losses[1]
        low, high = int(gains.argmin()), int(gains.argmax())
        assert float(gains[low]) < 0 < float(gains[high])
        model.cfg = replace(model.cfg, threshold_d=1e8, threshold_g=float(gains[high]) / 2)

        def with_gate_answer(record, token, identifier):
            qas = dict(record.qas)
            qas["gate0"] = replace(qas["gate0"], answer=str(token), answer_ids=(token,))
            return replace(record, trajectory_id=identifier, qas=qas)

        first = with_gate_answer(first, low, "negative-gain")
        second = with_gate_answer(first, high, "positive-gain")
    with torch.no_grad():
        actions = [model.build_memory(row)[1][-1]["action"] for row in (first, second)]
    assert set(actions) == {"append", "overwrite"}
    return model, (first, second)


def serial_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack([model(row)["loss"] for row in batch]).mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], 100.0
    )
    optimizer.step()
    return float(loss.detach()), float(norm)


def distributed_gist_worker(rank, rendezvous, method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        model, batch = calibrated_dynamic_batch(method)
        reference = deepcopy(model)
        frozen = parameter_snapshot(model, False)
        engine = TokenMemoryEngine(model, engine_config(), "cpu")
        engine.initialize()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in reference.parameters() if parameter.requires_grad],
            lr=engine_config().learning_rate,
            weight_decay=0.0,
        )
        for index, rows in enumerate((batch, batch[:1])):
            expected_loss, expected_norm = serial_step(reference, optimizer, rows)
            observed = engine.step(rows)
            assert observed["loss"] == pytest.approx(expected_loss, rel=1e-5, abs=1e-6)
            assert observed["grad_norm"] == pytest.approx(expected_norm, rel=1e-4, abs=1e-6)
            assert observed["samples"] == len(rows)
            if index == 0:
                assert observed["appends"] == observed["overwrites"] == 0.5
            for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
            assert_frozen(model, frozen)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
def test_actual_gist_ddp_opposite_actions_and_empty_rank_match_serial(tmp_path, method):
    mp.spawn(
        distributed_gist_worker,
        args=(str(tmp_path / "rendezvous"), method),
        nprocs=2,
        join=True,
    )


def experiment_config(model, output, init_checkpoint=None):
    return ExperimentConfig(
        ModelConfig(
            model_name_or_path="tiny-qwen3",
            dtype="float32",
            memory_slots=3,
            lora_rank=2,
            lora_alpha=4,
            lora_target_modules=("q_proj", "v_proj"),
        ),
        model.cfg,
        TrainingConfig(
            dataset_dir="synthetic-fixture",
            output_dir=str(output),
            micro_batch_size_per_gpu=1,
            gradient_accumulation_steps=2,
            learning_rate=0.003,
            weight_decay=0.0,
            gradient_clip=100.0,
            init_checkpoint=init_checkpoint,
        ),
    )


def assert_same_writer(left, right):
    left_state, right_state = parameter_snapshot(left, True), parameter_snapshot(right, True)
    assert left_state.keys() == right_state.keys()
    for name, value in left_state.items():
        torch.testing.assert_close(value, right_state[name], rtol=0, atol=0)


def test_actual_writer_checkpoint_resume_and_pretrain_warmup_policy_chain(tmp_path):
    model = tiny_task("qwen3", "memory_change", "pretrain")
    config = experiment_config(model, tmp_path / "pretrain")
    engine = TokenMemoryEngine(model, config.training, "cpu")
    engine.initialize()
    examples = pretraining_examples()
    splits = {"train": examples, "dev": (), "test": ()}
    statistics = {"source_data": {name: dataset_identity(rows) for name, rows in splits.items()}}
    run = make_run(config, splits, statistics, "cpu", 1, resolved_model_revision="tiny-fixture")
    engine.step(examples)
    checkpoint = tmp_path / "pretrain.pt"
    cursor = {"epoch": 1, "sample_offset": 0, "step": 1, "sample_visits": 2}
    save_checkpoint(checkpoint, engine, run, cursor)
    saved = read_checkpoint(checkpoint)
    assert set(saved["trainable"]) == {"memory_embeddings", "adapter"}

    resumed = tiny_task("qwen3", "memory_change", "pretrain")
    resumed_engine = TokenMemoryEngine(resumed, config.training, "cpu")
    resumed_engine.initialize()
    assert load_checkpoint(checkpoint, resumed_engine, run) == cursor
    assert_same_writer(model, resumed)
    assert all(state["step"].item() == 1 for state in resumed_engine.optimizer.state.values())

    for stage in ("warmup", "policy"):
        current = tiny_task("qwen3", "information_loss", stage)
        config = experiment_config(current, tmp_path / stage, str(checkpoint))
        initialization = load_initialization(checkpoint, current, config)
        assert_same_writer(model, current)
        assert initialization["stage"] == ("pretrain" if stage == "warmup" else "warmup")
        assert initialization["pretraining_sources"] == {
            "document_ids": ["pretrain-doc"],
            "dedup_clusters": ["pretrain-cluster"],
        }
        batch = (trajectory(n=2), replace(trajectory(n=3), trajectory_id="other"))
        splits = {"train": batch, "dev": (), "test": ()}
        run = make_run(
            config,
            splits,
            {"source_data": {name: dataset_identity(rows) for name, rows in splits.items()}},
            "cpu",
            1,
            resolved_model_revision="tiny-fixture",
            initialization=initialization,
        )
        current_engine = TokenMemoryEngine(current, config.training, "cpu")
        current_engine.initialize()
        assert not current_engine.optimizer.state
        frozen, writer = parameter_snapshot(current, False), parameter_snapshot(current, True)
        current_engine.step(batch)
        assert_writer_updated(current, writer)
        assert_frozen(current, frozen)
        checkpoint = tmp_path / f"{stage}.pt"
        save_checkpoint(checkpoint, current_engine, run, cursor)
        model = current


@pytest.mark.parametrize("method", ["memory_change", "information_loss"])
def test_actual_microbatch_opposite_policy_actions_match_serial(method):
    model, batch = calibrated_dynamic_batch(method)
    reference = deepcopy(model)
    engine = TokenMemoryEngine(model, engine_config(2), "cpu")
    engine.initialize()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in reference.parameters() if parameter.requires_grad],
        lr=engine_config().learning_rate,
        weight_decay=0.0,
    )
    expected_loss, expected_norm = serial_step(reference, optimizer, batch)
    actual = engine.step(batch)
    assert actual["appends"] == actual["overwrites"] == 0.5
    assert actual["loss"] == pytest.approx(expected_loss, rel=1e-5, abs=1e-6)
    assert actual["grad_norm"] == pytest.approx(expected_norm, rel=1e-4, abs=1e-6)
    for observed, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(observed, expected, rtol=1e-4, atol=2e-6)
