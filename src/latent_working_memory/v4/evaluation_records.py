"""同一评估计划的设置校验、结果缓存与 RULER 长度扩展。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from latent_working_memory.v4.config import load_common_config
from latent_working_memory.v4.io import write_json


def validate_evaluation_id(value):
    if not value.strip() or value in {".", ".."} or Path(value).name != value or "\\" in value:
        raise ValueError("eval-id must be a nonempty directory name")
    return value


def default_evaluation_id(training_dir):
    return f"{Path(training_dir).parents[1].name}-eval"


def evaluation_config_path(output_dir):
    """结果目录：eval/<method>/<eval-id>/<task>/global_step_N。"""
    directory = Path(output_dir)
    return (
        directory.parents[4] / "plan" / directory.parents[2].name
        / directory.parents[1].name / f"{directory.parent.name}.json"
    )


def evaluation_common_config_path(output_dir):
    return evaluation_config_path(output_dir).parent.parent / "config.json"


def load_evaluation_settings(output_dir):
    """各任务独立保存设置，seed 始终读取所属训练 run 的公共配置。"""
    path = evaluation_config_path(output_dir)
    evaluation = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    seed = load_common_config(evaluation_common_config_path(output_dir))["seed"]
    return evaluation, seed


def normalize_ruler_lengths(lengths):
    values = list(lengths)
    if not values or any(type(length) is not int or length < 1 for length in values):
        raise ValueError("RULER lengths must be positive integers")
    if len(values) != len(set(values)):
        raise ValueError("RULER lengths must be unique")
    return sorted(values)


def require_ruler_extension(previous, requested):
    previous = normalize_ruler_lengths(previous)
    requested = normalize_ruler_lengths(requested)
    if not set(previous).issubset(requested):
        raise ValueError("RULER lengths may only stay the same or expand within an eval-id")
    return requested


def _computation_config(value):
    names = ["eval_id", "split", "training_run", "seed", "model", "model_window", "protocol"]
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
        raise ValueError(f"evaluation settings differ at {path}: {', '.join(changed)}")


def matching_evaluation_config(actual, expected, path):
    """校验固定设置；RULER 另校验重叠长度的样本身份，长度覆盖可不同。"""
    actual_config = _computation_config(actual)
    expected_config = _computation_config(expected)
    if actual.get("benchmark") == expected.get("benchmark") == "ruler":
        for config in (actual_config, expected_config):
            config["protocol"] = {
                name: value for name, value in config["protocol"].items() if name != "lengths"
            }
            config["data"] = {
                name: config["data"][name] for name in ("directory", "source_commit")
            }
        actual_lengths = {row["length"]: row for row in actual["data"]["by_length"]}
        expected_lengths = {row["length"]: row for row in expected["data"]["by_length"]}
        for length in actual_lengths.keys() & expected_lengths.keys():
            _require_matching(actual_lengths[length], expected_lengths[length], path)
    _require_matching(actual_config, expected_config, path)


def load_completed_evaluations(directory):
    """读取同一计划、同一任务的已有结果；各 checkpoint 可覆盖不同 RULER 长度。"""
    directory = Path(directory).expanduser().resolve()
    records = []
    expected = None
    lengths = {}
    steps = set()
    for path in directory.glob("global_step_*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = summary["checkpoint"]
        checkpoint_path = Path(checkpoint["path"])
        training_dir = Path(summary["training_run"])
        step = checkpoint["step"]
        if (
            type(step) is not int or step < 1
            or path.parent.name != f"global_step_{step}"
            or checkpoint_path.name != path.parent.name
            or checkpoint_path.parent.name != "checkpoints"
            or checkpoint_path.parent.parent != training_dir
            or training_dir.name != directory.parent.parent.name
            or summary["eval_id"] != directory.parent.name
            or summary["split"] != "test"
            or summary.get("benchmark", "test") != directory.name
        ):
            raise ValueError(f"summary does not match its checkpoint, eval-id and task: {path}")
        if step in steps:
            raise ValueError(f"duplicate checkpoint step at {path}")
        steps.add(step)
        if expected is None:
            expected = summary
        else:
            matching_evaluation_config(summary, expected, path)
        if summary.get("benchmark") == "ruler":
            for row in summary["data"]["by_length"]:
                if row["length"] in lengths:
                    _require_matching(row, lengths[row["length"]], path)
                lengths[row["length"]] = row
        records.append((path.parent, summary))
    return sorted(records, key=lambda pair: pair[1]["checkpoint"]["step"])


def print_evaluation_summary(summary):
    step = summary["checkpoint"]["step"]
    metrics = summary["metrics"]
    if "benchmark" in summary:
        values = f"{metrics['metric']} {metrics['score']:.4f}"
        if summary["benchmark"] == "ruler":
            values += f" | lengths={summary['protocol']['lengths']}"
    else:
        values = f"NLL {metrics['nll']:.4f} | EM {metrics['em']:.4f} | F1 {metrics['f1']:.4f}"
    print(f"[evaluation] {summary.get('benchmark', 'test')} step={step} | {values}", flush=True)


class EvaluationHistory:
    def __init__(self, output_dir, evaluation_config):
        self.output_dir = Path(output_dir).resolve()
        self.root = self.output_dir.parent
        self.computation_config = _computation_config(evaluation_config)
        self.shared_config = {
            name: value for name, value in self.computation_config.items() if name != "seed"
        }
        self.checkpoint = evaluation_config["checkpoint"]
        self.ruler = evaluation_config.get("benchmark") == "ruler"

    def validate_summary(self, summary=None):
        """同计划同任务固定设置一致，RULER 的旧覆盖范围保留有效。"""
        common_path = evaluation_common_config_path(self.output_dir)
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
            matching_evaluation_config(value, self.computation_config, path)
            checkpoint = value["checkpoint"]
            step = checkpoint["step"]
            if step in steps and steps[step] != checkpoint["path"]:
                raise ValueError("different checkpoints cannot share the same optimizer step")
            steps[step] = checkpoint["path"]

    def save_summary(self, summary):
        """summary.json 记录实际完成范围，在锁内原子提交。"""
        with (self.root / ".evaluations.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            matching_evaluation_config(summary, self.computation_config, self.output_dir)
            _require_matching(summary["checkpoint"], self.checkpoint, self.output_dir)
            if self.ruler and not set(summary["protocol"]["lengths"]).issubset(
                self.computation_config["protocol"]["lengths"]
            ):
                raise ValueError("completed RULER lengths exceed the requested range")
            self.validate_summary(summary)
            write_json(self.output_dir / "summary.json", summary)

    @contextmanager
    def lock(self):
        """同一 checkpoint 的计算串行执行。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / ".evaluation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def cached_summary(self):
        """校验已有结果并保存目标配置；RULER 返回的汇总可尚未覆盖全部目标长度。"""
        shared_path = evaluation_config_path(self.output_dir)
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / "eval.json"
        summary_path = self.output_dir / "summary.json"
        expected = {**self.computation_config, "checkpoint": self.checkpoint}
        with (self.root / ".evaluations.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if shared_path.exists():
                saved = json.loads(shared_path.read_text(encoding="utf-8"))
                matching_evaluation_config(
                    {**saved, "seed": expected["seed"]}, expected, shared_path
                )
                if self.ruler:
                    require_ruler_extension(saved["protocol"]["lengths"], expected["protocol"]["lengths"])
            if config_path.exists():
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                matching_evaluation_config(saved, expected, config_path)
                _require_matching(saved["checkpoint"], self.checkpoint, config_path)
            summary = None
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                matching_evaluation_config(summary, expected, summary_path)
                _require_matching(summary["checkpoint"], self.checkpoint, summary_path)
            self.validate_summary(summary)
            write_json(shared_path, self.shared_config)
            write_json(config_path, expected)
            return summary
