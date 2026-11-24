"""统一 LoCoMo 产物与逐题恢复，正文统计按会话计一次。"""

import json
from pathlib import Path

from benchmarks.scoring import summarize_scores


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_progress(directory, samples, repair=False):
    expected = {sample.id: sample for sample in samples}
    records = {}
    for path in sorted(directory.glob("progress/*.jsonl")):
        with path.open("r+b" if repair else "rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    if repair:
                        stream.truncate(offset)
                    break
                row = json.loads(line)
                if row["id"] not in expected or row["id"] in records:
                    raise ValueError(f"unexpected or duplicate question in {path}: {row['id']}")
                sample = expected[row["id"]]
                if row["references"] != list(sample.references) or row["metadata"] != sample.metadata:
                    raise ValueError(f"saved question differs: {sample.id}")
                records[row["id"]] = row
    return records


def summarize(plan, records):
    contexts = {}
    for row in records:
        cid = row["metadata"]["conversation_id"]
        stats = row["context"]
        if contexts.setdefault(cid, stats) != stats:
            raise ValueError(f"inconsistent conversation statistics: {cid}")
    return {**plan, "metrics": summarize_scores("locomo", records), "conversations": contexts}


def file_identity(path):
    path = Path(path).expanduser().absolute()
    files = [path] if path.is_file() else sorted(
        child for child in path.iterdir()
        if child.suffix in {".json", ".model", ".safetensors", ".bin", ".pt"}
    )
    return {str(child): child.stat().st_mtime_ns for child in files}
