"""固定同类评估的计算协议，复用已完成结果并允许中断后重算。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from latent_working_memory.v4.io import write_json


def _computation_config(value):
    names = ["split", "training_run", "model", "tokenizer", "model_window", "protocol"]
    names.extend(
        ["benchmark", "data"]
        if "benchmark" in value
        else ["test_file", "dataset", "max_new_tokens", "eos_token_id"]
    )
    return {name: value[name] for name in names}


def _require_matching(actual, expected, path):
    if actual != expected:
        changed = sorted(
            name for name in actual.keys() | expected.keys()
            if actual.get(name) != expected.get(name)
        )
        raise ValueError(f"evaluation configuration differs at {path}: {', '.join(changed)}")


class EvaluationHistory:
    def __init__(self, output_dir, evaluation_config):
        self.output_dir = Path(output_dir)
        self.root = self.output_dir.parent
        self.computation_config = _computation_config(evaluation_config)
        self.checkpoint = evaluation_config["checkpoint"]

    @contextmanager
    def lock(self):
        """同一 checkpoint 的计算与发布串行执行。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / ".evaluation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def cached_summary(self):
        """校验计算协议；summary.json 是原子写入的完成标记。"""
        shared_path = self.root / "eval.json"
        config_path = self.output_dir / "eval.json"
        summary_path = self.output_dir / "summary.json"
        expected = {**self.computation_config, "checkpoint": self.checkpoint}
        with (self.root / ".evaluations.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if shared_path.exists():
                    _require_matching(
                        json.loads(shared_path.read_text(encoding="utf-8")),
                        self.computation_config,
                        shared_path,
                    )
                else:
                    for path in self.root.glob("global_step_*/summary.json"):
                        summary = json.loads(path.read_text(encoding="utf-8"))
                        _require_matching(
                            _computation_config(summary), self.computation_config, path
                        )
                if config_path.exists():
                    _require_matching(
                        json.loads(config_path.read_text(encoding="utf-8")), expected, config_path
                    )
                summary = None
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    _require_matching(
                        {**_computation_config(summary), "checkpoint": summary["checkpoint"]},
                        expected,
                        summary_path,
                    )
                if not shared_path.exists():
                    write_json(shared_path, self.computation_config)
                if not config_path.exists():
                    write_json(config_path, expected)
                return summary
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
