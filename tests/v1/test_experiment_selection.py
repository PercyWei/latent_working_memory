import torch
from transformers import LlamaConfig, LlamaForCausalLM
from latent_working_memory.v1.pretrain.training import run_pretraining
from latent_working_memory.v1.pretrain.evaluate import main as evaluate_main
from latent_working_memory.v1.checkpoint import load_model_checkpoint
from dataclasses import replace
import json
import re


from latent_working_memory.data_preparation.pretrain.pipeline import prepare_fineweb


def test_selection_train_resume_and_evaluate(
    parquet_source,
    tmp_path,
    tokenizer,
    tiny_config,
    preparation_records,
    preparation_recipe,
    epoch_selection,
):
    model = tmp_path / "model"
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).save_pretrained(model)
    tokenizer.save_pretrained(model)
    cfg = replace(
        tiny_config,
        model_name_or_path=str(model),
        eval_generation_examples=0,
        input_length_weights=None,
        split_fractions=(0.6, 0.2, 0.2),
    )
    raw = tmp_path / "raw"
    # Keep fragments from distinct fixture documents distinct after Parquet shuffling.
    preparation_records = [
        dict(row, text=re.sub(r"\b\w+\b", lambda m: m[0] + str(i), row["text"]))
        for i, row in enumerate(preparation_records)
    ]
    recipe = replace(preparation_recipe, samples_per_task=(16, 16, 16), candidates_per_document=16)
    prepare_fineweb(parquet_source(preparation_records), tokenizer, cfg, raw, recipe)
    path = epoch_selection(raw, cfg, {v: v for v in ("semantic", "random")})
    spec = json.loads(path.read_text())
    run = tmp_path / "train"
    first = run_pretraining(cfg, path, run, torch.device("cpu"), epochs=2, stop_after_steps=1)
    assert json.loads((run / "data-selection.json").read_text()) == spec
    result = run_pretraining(
        cfg,
        run / "data-selection.json",
        run,
        torch.device("cpu"),
        epochs=2,
        stop_after_steps=2,
        resume=first.final_checkpoint,
    )
    assert load_model_checkpoint(result.final_checkpoint).progress["next_step"] == 2
    out = tmp_path / "evaluation"
    evaluate_main(
        [
            "--checkpoint",
            str(result.final_checkpoint),
            "--data-selection",
            str(path),
            "--output-dir",
            str(out),
            "--split",
            "test",
            "--device",
            "cpu",
            "--generation-examples",
            "0",
        ]
    )
    assert all((out / v / "test-step-000002.json").exists() for v in spec["sources"])
    assert json.loads((out / "data-selection.json").read_text()) == spec

    single = tmp_path / "single-evaluation"
    evaluate_main(
        [
            "--checkpoint",
            str(result.final_checkpoint),
            "--data-selection",
            str(path),
            "--evaluation-source",
            "semantic",
            "--output-dir",
            str(single),
            "--split",
            "test",
            "--device",
            "cpu",
            "--generation-examples",
            "0",
        ]
    )
    assert (single / "test-step-000002.json").exists()
    assert not (single / "random").exists()
