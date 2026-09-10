from __future__ import annotations

import json
import os
from pathlib import Path

FILES = ("train", "dev", "test", "documents", "sample-decisions")


def save_progress(directory: Path, handles: dict, contract: dict, next_source: int, counts: dict):
    offsets = {}
    for name, handle in handles.items():
        handle.flush()
        os.fsync(handle.fileno())
        offsets[name] = handle.tell()
    state = {
        "contract": contract,
        "next_source": next_source,
        "offsets": offsets,
        "counts": counts,
    }
    temporary = directory / "progress.tmp"
    with temporary.open("w") as handle:
        json.dump(state, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(directory / "progress.json")


def load_progress(directory: Path, contract: dict) -> dict:
    state = json.loads((directory / "progress.json").read_text())
    if state["contract"] != contract:
        raise ValueError("resume configuration or construction protocol differs")
    # A crash can leave rows beyond the last committed window. Only that tail is rolled back.
    for name, offset in state["offsets"].items():
        path = directory / f"{name}.jsonl"
        if path.stat().st_size < offset:
            raise ValueError(f"resume data is shorter than its committed offset: {path}")
    for name, offset in state["offsets"].items():
        with (directory / f"{name}.jsonl").open("r+b") as handle:
            handle.truncate(offset)
    return state
