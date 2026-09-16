from dataclasses import asdict
import json
import sys

import pytest
import torch
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments, set_seed

from latent_working_memory.v2.gmsa_checkpoint import load_weights
from latent_working_memory.v2.gmsa_config import GMSAConfig
from latent_working_memory.v2.gmsa import GMSA
from latent_working_memory.v2.static.data import StaticCollator, StaticDataset
from latent_working_memory.v2.static.evaluate import main as evaluate_main
from latent_working_memory.v2.static.train import StaticTrainer, main as train_main


def data_file(tmp_path):
    path = tmp_path / "qa.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"input": "red sky blue water", "prompt": "What color ?", "answer": ["red"]},
                {"input": "blue water red sky", "prompt": "What color ?", "answer": ["blue"]},
            ]
        )
        + "\n"
    )
    return path


def test_dataset_never_silently_truncates(tiny_base, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    data = data_file(tmp_path)
    with pytest.raises(ValueError, match="no truncation"):
        StaticDataset(data, tokenizer, "autoencoding", 2, 64)
    ae = StaticDataset(data, tokenizer, "autoencoding", 64, 64)
    assert ae[0]["labels"].tolist() == [*ae[0]["context_ids"].tolist(), tokenizer.eos_token_id]


class StopAfterFirst(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_training_stop = True
        return control


def test_trainer_exact_resume(tiny_base, tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(tiny_base)
    dataset = StaticDataset(data_file(tmp_path), tokenizer, "autoencoding", 64, 64)
    config = GMSAConfig(
        str(tiny_base),
        encoder_layers=2,
        alignment_layers=1,
        compression_ratios=(2, 4),
        lora_rank=2,
        lora_dropout=0.1,
    )

    def trainer(path, callbacks):
        set_seed(42)
        model = GMSA(config)
        args = TrainingArguments(
            output_dir=str(path),
            use_cpu=True,
            max_steps=2,
            learning_rate=0.001,
            per_device_train_batch_size=1,
            save_steps=1,
            report_to=[],
            remove_unused_columns=False,
            label_names=["labels"],
            prediction_loss_only=True,
            disable_tqdm=True,
            seed=42,
        )
        return StaticTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            eval_dataset=dataset,
            processing_class=tokenizer,
            data_collator=StaticCollator(0),
            callbacks=callbacks,
        )

    full = trainer(tmp_path / "full", [])
    full.train()
    partial = trainer(tmp_path / "resumed", [StopAfterFirst()])
    partial.train()
    resumed = trainer(tmp_path / "resumed", [])
    resumed.train(resume_from_checkpoint=str(tmp_path / "resumed/checkpoint-1"))
    assert resumed.state.global_step == 2
    for name, tensor in full.model.state_dict().items():
        torch.testing.assert_close(tensor, resumed.model.state_dict()[name], rtol=0, atol=0)
    metrics = resumed.evaluate()
    assert "eval_r2_loss" in metrics and "eval_r4_loss" in metrics


def test_cli_ae_qa_evaluate(tiny_base, tmp_path, monkeypatch):
    data = data_file(tmp_path)
    model_path = tmp_path / "model.json"
    config = GMSAConfig(
        str(tiny_base),
        encoder_layers=2,
        alignment_layers=1,
        compression_ratios=(2, 4),
        lora_rank=2,
        lora_dropout=0,
    )
    model_path.write_text(json.dumps(asdict(config)))
    for stage in ("autoencoding", "finetune"):
        settings = tmp_path / f"{stage}.json"
        settings.write_text(
            json.dumps(
                dict(
                    stage=stage,
                    max_context_tokens=64,
                    max_target_tokens=64,
                    trainer=dict(
                        use_cpu=True,
                        max_steps=2,
                        save_steps=1,
                        eval_strategy="steps",
                        eval_steps=1,
                        logging_steps=1,
                        per_device_train_batch_size=1,
                        per_device_eval_batch_size=2,
                        disable_tqdm=True,
                    ),
                )
            )
        )
        argv = [
            "train",
            "--model-config",
            str(model_path),
            "--training-config",
            str(settings),
            "--train-file",
            str(data),
            "--eval-file",
            str(data),
            "--output-dir",
            str(tmp_path / stage),
        ]
        if stage == "finetune":
            argv += ["--initialize-from", str(tmp_path / "autoencoding/final")]
        monkeypatch.setattr(sys, "argv", argv)
        train_main()
        model = GMSA(config)
        load_weights(model, tmp_path / stage / "final")
        assert (tmp_path / stage / "checkpoint-2/optimizer.pt").exists()
        # Resume keeps the run's initialization provenance without reinitializing its weights.
        resume_argv = (
            argv[: argv.index("--initialize-from")] if "--initialize-from" in argv else argv
        )
        monkeypatch.setattr(
            sys, "argv", resume_argv + ["--resume", str(tmp_path / stage / "checkpoint-1")]
        )
        train_main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--checkpoint",
            str(tmp_path / "finetune/final"),
            "--data",
            str(data),
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--stage",
            "finetune",
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
        ],
    )
    evaluate_main()
    summary = json.loads((tmp_path / "evaluation/summary.json").read_text())
    assert len(summary) == 8
    assert all(value["reads"] == 2 for value in summary.values())
