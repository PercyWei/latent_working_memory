"""一个 checkpoint 使用全部 worker；checkpoint 之间由生命周期入口串行调度。"""

from pathlib import Path
import sys
import time

from .. import method
from ..execution.processes import run_workers
from ..execution.runtime import file_digest
from ..execution.state import (
    checkpoint_file,
    commit_results,
    completed_result,
    prepare_work,
    read_checkpoint,
    read_progress,
)


def evaluate_checkpoint(checkpoint, settings, output_dir, device, workers):
    """调用方持有任务锁；同一 eval-id 只允许重新分配尚未完成的样本。"""
    checkpoint = checkpoint_file(checkpoint)
    output_dir = Path(output_dir)
    state = read_checkpoint(checkpoint)
    config = state["config"]
    del state
    samples, identity, _ = method.load_samples(
        config,
        "test",
        data_dir=settings["data"],
        max_samples=settings["max_samples"],
    )
    if not samples:
        raise ValueError("evaluation requires at least one selected sample")
    sample_ids = [sample.id for sample in samples]
    request = {
        **settings,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "data_identity": identity,
        "protocol": method.evaluation_protocol(config, settings),
    }
    completed = completed_result(output_dir, request, sample_ids)
    if completed is not None:
        print(
            f"evaluation cached checkpoint={checkpoint.parent.name} samples={len(samples)}",
            flush=True,
        )
        return completed

    work, entries = prepare_work(output_dir, request, sample_ids, workers)
    print(
        f"evaluation checkpoint={checkpoint.parent.name} samples={len(samples)} "
        f"cached={len(entries)} pending={work['pending_samples']} "
        f"workers={len(work['workers'])}",
        flush=True,
    )
    started = time.perf_counter()
    commands = [
        [
            sys.executable,
            "-u",
            "-m",
            f"{__package__}.worker",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--rank",
            str(rank),
            "--device",
            device,
        ]
        for rank in range(len(work["workers"]))
    ]
    if commands:
        run_workers(commands, output_dir / "workers.log")
    entries = read_progress(output_dir, sample_ids)
    records = [entries[sample_id]["record"] for sample_id in sample_ids]
    resources = {
        "sample_seconds": sum(entry["resources"]["sample_seconds"] for entry in entries.values()),
        "final_attempt_wall_seconds": time.perf_counter() - started,
    }
    return commit_results(
        output_dir, request, sample_ids, entries, method.summarize(records), resources
    )
