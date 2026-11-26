"""在清洗版 LongMemEval-S 或 Oracle 上生成回答，供独立评分。"""

import argparse
from collections.abc import Callable
from importlib import import_module
import json
from pathlib import Path

from benchmarks.data import load_samples
from benchmarks.documents import DocumentAdapter
from benchmarks.io import run_lock, write_json
from benchmarks.protocol import data_identity
from .data import LONGMEMEVAL_FILES
from .parallel import generate_predictions


def read_predictions(path, samples):
    """读取官方格式的预测，文件中的题目必须按原始数据顺序构成一个前缀。"""
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                if (i >= len(samples) or not isinstance(row, dict)
                        or set(row) != {"question_id", "hypothesis"}
                        or row["question_id"] != samples[i].id
                        or not isinstance(row["hypothesis"], str)):
                    raise ValueError(f"{path}:{i + 1}: invalid prediction or question order")
                rows.append(row)
    return rows


def run_generation(
    benchmark, data_dir, output_root, output_id, method, settings,
    load_adapter: Callable[[dict], DocumentAdapter], batch_size=1, resume=False, devices=None,
):
    """返回官方 predictions.jsonl 的路径。settings 记录模型身份、输入和解码设置。"""
    if benchmark not in LONGMEMEVAL_FILES:
        raise ValueError(f"unsupported LongMemEval setting: {benchmark}")
    if not output_id.strip() or output_id in {".", ".."} or any(c in output_id for c in "/\\"):
        raise ValueError("output_id must be a single nonempty directory name")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not isinstance(settings, dict):
        raise ValueError("settings must be a JSON object")
    settings = json.loads(json.dumps(settings, allow_nan=False))
    samples = load_samples(benchmark, data_dir)
    config = {
        "method": method,
        "benchmark": benchmark,
        "settings": settings,
        "adapter": {"module": load_adapter.__module__, "factory": load_adapter.__name__},
        "data": data_identity(samples, data_dir, benchmark),
    }
    directory = Path(output_root).expanduser().resolve() / "eval" / output_id / benchmark
    config_path = directory / "evaluation_config.json"
    predictions = directory / "predictions.jsonl"
    if resume:
        if not config_path.is_file():
            raise FileNotFoundError(f"resume requires {config_path}")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory) as lock:
        if resume:
            if json.loads(config_path.read_text()) != config:
                raise ValueError("saved generation settings or data differ")
        else:
            write_json(config_path, config)
        rows = read_predictions(predictions, samples)
        generate_predictions(directory, samples, rows, devices, batch_size, lock, load_adapter, settings)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split", choices=LONGMEMEVAL_FILES, required=True,
                        help="选择要生成回答的 LongMemEval 设置（S 或 Oracle）")
    parser.add_argument("--benchmark-data", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/benchmarks/longmemeval")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--adapter", required=True, metavar="MODULE:FACTORY",
                        help="方法提供的模型加载函数")
    parser.add_argument("--settings", type=Path, required=True, help="方法的推理配置 JSON 文件")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    module, separator, factory = args.adapter.partition(":")
    if not (module and separator and factory):
        parser.error("--adapter requires MODULE:FACTORY")
    run_generation(
        args.split, args.benchmark_data, args.output_root, args.output_id, args.method,
        json.loads(args.settings.read_text()), getattr(import_module(module), factory),
        args.batch_size, args.resume,
    )


if __name__ == "__main__":
    main()
