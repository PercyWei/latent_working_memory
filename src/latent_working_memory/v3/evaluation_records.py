"""从 checkpoint 结果校验评估协议，集中保存累计 SwanLab 发布记录。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from latent_working_memory.v3.config import load_common_config


def _save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluation_log_path(output_dir):
    """评估日志保存在对应的 checkpoint 结果目录中。"""
    return Path(output_dir).resolve() / "eval.log"


class EvaluationHistory:
    def __init__(self, output_dir, evaluation_config):
        self.output_dir = Path(output_dir).resolve()
        self.root = self.output_dir.parent.parent
        self.path = self.root / "swanlab.json"
        self.evaluation_config = evaluation_config

    def validate_summaries(self, summaries):
        common_path = (
            self.output_dir.parents[4] / "plan" / self.output_dir.parents[2].name / "config.json"
        )
        if load_common_config(common_path)["seed"] != self.evaluation_config["seed"]:
            raise ValueError("evaluation seed differs from the common seed")
        snapshots = set(self.root.glob("*/global_step_*/eval.json"))
        if (self.output_dir / "eval.json").is_file():
            snapshots.add(self.output_dir / "eval.json")
        for path in snapshots:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if snapshot["seed"] != self.evaluation_config["seed"]:
                raise ValueError("saved evaluation snapshot seed differs from the common seed")
        method, protocol = None, None
        steps = {}
        for summary in summaries:
            metadata = summary["metadata"]
            if metadata["evaluation_config"] != self.evaluation_config:
                raise ValueError(
                    "checkpoint comparisons require the saved evaluation configuration; "
                    "dataset, seed, sample limit and generation limit cannot change"
                )
            current_protocol = {
                "split": summary["split"],
                "dataset_signature": summary["dataset_signature"],
                "qa_prompt": summary["objective"]["qa_prompt"],
                "gate_signature": summary["qa_sampling"]["gate_signature"],
                "max_new_tokens": summary["max_new_tokens"],
            }
            if method is not None and method != summary["method"]:
                raise ValueError("evaluation history must belong to one method run")
            if protocol is not None and protocol != current_protocol:
                raise ValueError(
                    "checkpoint comparisons require the same task pool, gate selections and QA prompt"
                )
            method, protocol = summary["method"], current_protocol
            checkpoint, step = metadata["checkpoint"], metadata["global_step"]
            if step in steps and steps[step] != checkpoint:
                raise ValueError("different checkpoints cannot share the same global optimizer step")
            steps[step] = checkpoint

    def _saved_summaries(self):
        return {
            path: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.root.glob("*/global_step_*/summary.json"))
        }

    @contextmanager
    def _results_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".results.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def cached_summary(self, checkpoint):
        with self._results_lock():
            summaries = self._saved_summaries()
            self.validate_summaries(summaries.values())
            summary = summaries.get(self.output_dir / "summary.json")
            if summary is not None:
                if summary["metadata"]["checkpoint"] != str(Path(checkpoint).resolve()):
                    raise ValueError("checkpoint output already contains a different evaluation")
                return summary
            self.output_dir.mkdir(parents=True, exist_ok=True)
            _save_json(self.output_dir / "eval.json", self.evaluation_config)
            return None

    def validate_configuration(self):
        """在恢复训练或修改方法级配置前检查已有评估快照及结果协议。"""
        with self._results_lock():
            self.validate_summaries(self._saved_summaries().values())

    def validate_summary(self, summary):
        """校验当前结果与本地其他 checkpoint 的协议及累计 step，不保存重复索引。"""
        summaries = self._saved_summaries()
        summaries[self.output_dir / "summary.json"] = summary
        self.validate_summaries(summaries.values())

    def save_summary(self, summary):
        """在结果提交边界串行校验，失败的候选结果不进入完整结果集合。"""
        with self._results_lock():
            self.validate_summary(summary)
            _save_json(self.output_dir / "summary.json", summary)

    def report_publication(self):
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None

    def published_summaries(self):
        """仅加载已成功发布集合；未发布的本地结果须显式选中后才加入报告。"""
        publication = self.report_publication()
        if publication is None:
            return []
        return [
            (path.parent, json.loads(path.read_text(encoding="utf-8")))
            for path in map(Path, publication["summaries"])
        ]

    @contextmanager
    def publication_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".publication.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def record_report_publication(self, publication):
        """上传及云端核验成功后，原子替换完整报告的唯一发布记录。"""
        _save_json(self.path, publication)
