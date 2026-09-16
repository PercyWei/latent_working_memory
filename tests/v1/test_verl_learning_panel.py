from dataclasses import replace
import json
from tensordict import TensorDict
from verl.utils.tensordict_utils import assign_non_tensor
from verl.workers.engine.utils import postprocess_batch_func

from latent_working_memory.data_preparation.pretrain.text_samples import TextSample
from benchmark_verl_learning import panel, panel_description, grouped_nll


def test_real_data_panel_is_replayable(tmp_path, tokenizer, tiny_config):
    config = replace(
        tiny_config,
        max_input_tokens=2048,
        max_continuation_tokens=2048,
        write_context_tokens=4096,
        read_context_tokens=4096,
        input_length_bounds=(128, 512, 1024, 2048),
    )
    for variant in ("semantic", "random"):
        path = tmp_path / variant
        path.mkdir()
        rows = []
        for task in ("ae", "continuation"):
            for length in (80, 256, 768, 1536):
                text = " ".join(["First"] * length)
                continuation = "First sentence" if task == "continuation" else None
                identifier = f"{variant}/{task}/{length}"
                sample = TextSample(
                    identifier,
                    identifier,
                    identifier,
                    identifier,
                    task,
                    text,
                    continuation,
                    [0, len(text)],
                    [len(text), len(text) + len(continuation)] if continuation else None,
                    "pysbd_conservative" if variant == "semantic" else "random_token",
                    length,
                    2 if continuation else length,
                )
                rows.append(json.dumps(sample.to_record()))
        (path / "train.jsonl").write_text("\n".join(rows) + "\n")
    first = panel(tmp_path, "train", 1, tokenizer, config, 42)
    second = panel(tmp_path, "train", 1, tokenizer, config, 42)
    assert panel_description(first) == panel_description(second)
    assert len(first) == len({e.episode.episode_id for e in first}) == 16
    assert {e.input_length for e in first} == {80, 256, 768, 1536}
    rows = [
        {
            "boundary_variant": e.episode.sources[0].provenance["boundary_variant"],
            "input_tokens": e.input_length,
            "ae_nll": 1.0 if e.ae else None,
            "lm_nll": 3.0 if e.lm else None,
        }
        for e in first
    ]
    scores = grouped_nll(rows)
    assert scores["all"] == {"nll": 2.0, "samples": 16}
    assert scores["ae"] == {"nll": 1.0, "samples": 8}
    data = TensorDict({}, batch_size=[])
    assign_non_tensor(data, use_dynamic_bsz=False)
    outputs = [
        {"model_output": {}, "loss": 0.5, "metrics": {"records": rows[:8]}},
        {"model_output": {}, "loss": 0.5, "metrics": {"records": rows[8:]}},
    ]
    merged = postprocess_batch_func(outputs, None, data)
    assert grouped_nll(merged["metrics"]["records"]) == scores
