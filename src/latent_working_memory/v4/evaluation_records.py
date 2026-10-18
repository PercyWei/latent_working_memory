"""固定同类评估的计算协议，复用已完成结果并允许中断后重算。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from latent_working_memory.v4.config import load_common_config
from latent_working_memory.v4.io import write_json


def evaluation_config_path(output_dir):
    """从规范的 checkpoint 结果目录定位 plan 下同类评估的共享配置。"""
    directory = Path(output_dir)
    return (
        directory.parents[3] / "plan" / directory.parents[1].name
        / f"{directory.parent.name}.json"
    )


def load_evaluation_settings(output_dir):
    """沿用同类评估设置；seed 始终读取公共配置。"""
    path = evaluation_config_path(output_dir)
    evaluation = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    seed = load_common_config(path.with_name("config.json"))["seed"]
    return evaluation, seed


def _computation_config(value):
    names = ["split", "training_run", "seed", "model", "model_window", "protocol"]
    names.extend(
        ["benchmark", "data"]
        if "benchmark" in value
        else ["test_file", "dataset", "max_test_samples", "max_new_tokens", "eos_token_id"]
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
        self.output_dir = Path(output_dir).resolve()
        self.root = self.output_dir.parent
        self.publication_path = self.root / "swanlab.json"
        self.computation_config = _computation_config(evaluation_config)
        self.shared_config = {
            name: value for name, value in self.computation_config.items() if name != "seed"
        }
        self.checkpoint = evaluation_config["checkpoint"]

    def validate_summary(self, summary=None):
        """校验已启动及完成评估的实际协议，防止跨 checkpoint 混用 seed。"""
        common_path = evaluation_config_path(self.output_dir).with_name("config.json")
        _require_matching(
            {"seed": self.computation_config["seed"]},
            {"seed": load_common_config(common_path)["seed"]},
            common_path,
        )
        records = {
            path: json.loads(path.read_text(encoding="utf-8"))
            for name in ("eval.json", "summary.json")
            for path in sorted(self.root.glob(f"global_step_*/{name}"))
        }
        if summary is not None:
            records[self.output_dir / "summary.json"] = summary
        steps = {}
        for path, value in records.items():
            _require_matching(_computation_config(value), self.computation_config, path)
            checkpoint = value["checkpoint"]
            step = checkpoint["step"]
            if step in steps and steps[step] != checkpoint["path"]:
                raise ValueError("different checkpoints cannot share the same optimizer step")
            steps[step] = checkpoint["path"]

    @contextmanager
    def publication_lock(self):
        """同类评估的完整报告串行发布。"""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".publication.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def report_publication(self):
        return (
            json.loads(self.publication_path.read_text(encoding="utf-8"))
            if self.publication_path.exists() else None
        )

    def published_summaries(self):
        """只读取上一份成功报告的成员，离线结果须显式选择后才发布。"""
        publication = self.report_publication()
        if publication is None:
            return []
        return [
            (path.parent, json.loads(path.read_text(encoding="utf-8")))
            for path in map(Path, publication["summaries"])
        ]

    def save_summary(self, summary):
        """在同类结果锁内校验并提交完成标记，冲突结果不覆盖原缓存。"""
        with (self.root / ".evaluations.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.validate_summary(summary)
            write_json(self.output_dir / "summary.json", summary)

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
        shared_path = evaluation_config_path(self.output_dir)
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / "eval.json"
        summary_path = self.output_dir / "summary.json"
        expected = {**self.computation_config, "checkpoint": self.checkpoint}
        with (self.root / ".evaluations.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if shared_path.exists():
                    _require_matching(
                        json.loads(shared_path.read_text(encoding="utf-8")),
                        self.shared_config,
                        shared_path,
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
                self.validate_summary(summary)
                if not shared_path.exists():
                    write_json(shared_path, self.shared_config)
                if not config_path.exists():
                    write_json(config_path, expected)
                return summary
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
