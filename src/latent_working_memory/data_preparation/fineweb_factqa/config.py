"""FactQA 构造参数及单次运行目录。"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from latent_working_memory.data_preparation.fineweb_source import split_fractions
from latent_working_memory.data_preparation.segmentation import SegmentationConfig


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        "source_dir",
        "source_batch_size",
        "source_seed",
        "selection_seed",
        "split_counts",
        "window",
        "batch_split_counts",
        "qa",
        "annotation",
        "prompts_dir",
    }
    if not isinstance(config, dict):
        raise ValueError("FactQA configuration must be a JSON object")
    missing, unknown = fields - config.keys(), config.keys() - fields
    if missing or unknown:
        raise ValueError(
            f"invalid FactQA configuration fields: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    for name in ("source_dir", "prompts_dir"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty directory")
    if type(config["source_batch_size"]) is not int or config["source_batch_size"] <= 0:
        raise ValueError("source_batch_size must be a positive integer")
    for name in ("source_seed", "selection_seed"):
        if type(config[name]) is not int or config[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    split_fractions(config["split_counts"])
    if "continuation_tokens" in config["window"]:
        raise ValueError("FactQA window does not configure continuation_tokens")
    SegmentationConfig(**config["window"], continuation_tokens=0)
    batches = config["batch_split_counts"]
    if set(batches) != {"train", "dev", "test"} or any(
        type(value) is not int or value < 0 or (config["split_counts"][split] > 0 and value == 0)
        for split, value in batches.items()
    ):
        raise ValueError("batch_split_counts requires a positive count for each requested split")
    if type(config["qa"]["max_answer_chars"]) is not int or config["qa"]["max_answer_chars"] <= 0:
        raise ValueError("qa.max_answer_chars must be a positive integer")
    return config


def run_config(
    config: dict,
    output_root: Path = Path("data"),
    artifacts_root: Path = Path("artifacts/fineweb-factqa"),
    run_id: str | None = None,
    previous_datasets=(),
) -> dict:
    if run_id is None:
        run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError(
            "run_id must start with a letter or digit and use letters, digits, '.', '_' or '-'"
        )
    window = SegmentationConfig(**config["window"], continuation_tokens=0)
    name = (
        f"fineweb-factqa-k{window.capacity}"
        f"-seg{window.min_segment_ratio:g}to{window.max_segment_ratio:g}x"
        f"_train{config['split_counts']['train']}_{run_id}"
    )
    return {
        **config,
        "run_id": run_id,
        "dataset_dir": str(Path(output_root) / name),
        "artifacts_dir": str(Path(artifacts_root) / name),
        "previous_datasets": [str(directory) for directory in previous_datasets],
    }


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="FactQA construction JSON")
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/fineweb-factqa"))
    parser.add_argument("--run-id", help="Run identifier; defaults to the current Shanghai date")
    parser.add_argument("--previous-datasets", type=Path, nargs="+", default=[], metavar="DIR")
