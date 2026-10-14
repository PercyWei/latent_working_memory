"""按 checkpoint 保存评估配置、结果索引与 SwanLab 发布记录。"""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path


def _save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class EvaluationHistory:
    def __init__(self, output_dir, evaluation_config):
        self.output_dir = Path(output_dir).resolve()
        self.root = self.output_dir.parent.parent
        self.path = self.root / "evaluations.json"
        self.evaluation_config = evaluation_config

    @contextmanager
    def _index(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".evaluations.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            index = (
                json.loads(self.path.read_text(encoding="utf-8"))
                if self.path.exists()
                else {
                    "evaluation_config": self.evaluation_config,
                    "method": None,
                    "protocol": None,
                    "checkpoints": [],
                }
            )
            if index["evaluation_config"] != self.evaluation_config:
                if index["checkpoints"]:
                    raise ValueError(
                        "checkpoint comparisons require the saved evaluation configuration; "
                        "dataset, seed, sample limit and generation limit cannot change"
                    )
                index["evaluation_config"] = self.evaluation_config
            yield index
            _save_json(self.path, index)

    def cached_summary(self, checkpoint):
        with self._index():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = self.output_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    summary["metadata"]["checkpoint"] != str(Path(checkpoint).resolve())
                    or summary["metadata"]["evaluation_config"] != self.evaluation_config
                ):
                    raise ValueError("checkpoint output already contains a different evaluation")
                return summary
            _save_json(self.output_dir / "eval.json", self.evaluation_config)
            return None

    def validate_configuration(self):
        """在恢复训练或修改方法级配置前检查已有评估历史。"""
        with self._index():
            pass

    def record(self, summary):
        protocol = {
            "split": summary["split"],
            "dataset_signature": summary["dataset_signature"],
            "qa_prompt": summary["objective"]["qa_prompt"],
            "gate_signature": summary["qa_sampling"]["gate_signature"],
            "max_new_tokens": summary["max_new_tokens"],
        }
        metadata = summary["metadata"]
        with self._index() as index:
            publication = self.publication()
            entry = {
                "checkpoint": metadata["checkpoint"],
                "stage": summary["stage"],
                "stage_step": metadata["stage_step"],
                "global_step": metadata["global_step"],
                "summary": str(self.output_dir / "summary.json"),
                "publication_step": None if publication is None else publication["publication_step"],
            }
            if index["method"] is not None and index["method"] != summary["method"]:
                raise ValueError("evaluation history must belong to one method run")
            if index["protocol"] is not None and index["protocol"] != protocol:
                raise ValueError(
                    "checkpoint comparisons require the same task pool, gate selections and QA prompt"
                )
            index["method"], index["protocol"] = summary["method"], protocol
            entries = [
                row for row in index["checkpoints"] if row["checkpoint"] != entry["checkpoint"]
            ]
            if any(row["global_step"] == entry["global_step"] for row in entries):
                raise ValueError("different checkpoints cannot share the same global optimizer step")
            entries.append(entry)
            index["checkpoints"] = sorted(entries, key=lambda row: row["global_step"])

    def publication(self):
        path = self.output_dir / "swanlab-publication.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def report_publication(self):
        with self._index() as index:
            return index.get("publication")

    def published_summaries(self):
        """读取已有发布记录的本地结果，返回各自的输出目录和 summary。"""
        with self._index() as index:
            summaries = []
            for entry in index["checkpoints"]:
                path = Path(entry["summary"])
                if (path.parent / "swanlab-publication.json").is_file():
                    summaries.append((path.parent, json.loads(path.read_text(encoding="utf-8"))))
            return summaries

    def last_publication_step(self):
        with self._index() as index:
            return max(
                (
                    row["publication_step"]
                    for row in index["checkpoints"]
                    if row["publication_step"] is not None
                ),
                default=-1,
            )

    @contextmanager
    def publication_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".publication.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def record_report_publication(self, publication, checkpoints):
        """同一份累计报告的发布状态与各 checkpoint 记录一并登记。"""
        with self._index() as index:
            entries = {row["checkpoint"]: row for row in index["checkpoints"]}
            for checkpoint in checkpoints:
                entry = entries[checkpoint["checkpoint"]]
                _save_json(Path(entry["summary"]).parent / "swanlab-publication.json", checkpoint)
                entry["publication_step"] = publication["publication_step"]
            index["publication"] = publication
