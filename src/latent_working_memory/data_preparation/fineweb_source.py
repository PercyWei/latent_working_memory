"""FineWeb 本地分片、划分配额与实际已用原文的公共契约。"""

import json
import os
from pathlib import Path
import tempfile


USED_SOURCES_FILE = "used-sources.jsonl"
SPLITS = ("train", "dev", "test")


def source_files(source_dir) -> list[Path]:
    paths = sorted(path for path in Path(source_dir).glob("*.parquet") if path.is_file())
    if not paths:
        raise ValueError(f"no FineWeb Parquet files in source_dir: {source_dir}")
    return paths


def split_fractions(split_counts) -> tuple[float, float, float]:
    if (
        not isinstance(split_counts, dict)
        or set(split_counts) != set(SPLITS)
        or any(type(value) is not int or value < 0 for value in split_counts.values())
        or split_counts["train"] == 0
    ):
        raise ValueError("split_counts require positive train and nonnegative dev/test integers")
    total = sum(split_counts.values())
    return tuple(split_counts[split] / total for split in SPLITS)


def _unique_sources(references):
    unique = {}
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {
            "document_id",
            "dedup_cluster",
            "source",
        }:
            raise ValueError("used source requires document_id, dedup_cluster and source")
        if any(
            not isinstance(reference[key], str) or not reference[key].strip()
            for key in ("document_id", "dedup_cluster")
        ):
            raise ValueError("used source identifiers must be nonempty strings")
        source = reference["source"]
        if (
            not isinstance(source, dict)
            or set(source) != {"file", "row_group", "row_index"}
            or not isinstance(source["file"], str)
            or not source["file"].strip()
            or any(
                type(source[key]) is not int or source[key] < 0
                for key in ("row_group", "row_index")
            )
        ):
            raise ValueError("used source requires a file and nonnegative row_group/row_index")
        identity = reference["document_id"]
        if identity in unique and unique[identity]["source"] != source:
            raise ValueError(f"previous document has inconsistent source locations: {identity}")
        unique.setdefault(identity, reference)
    return [unique[identity] for identity in sorted(unique)]


def load_previous_sources(directories) -> list[dict]:
    """仅读取显式目录的已用原文；兼容已发布数据中内嵌的 used_sources。"""
    references = []
    for directory in directories:
        root = Path(directory)
        ledger = root / USED_SOURCES_FILE
        if ledger.exists():
            with ledger.open(encoding="utf-8") as stream:
                references.extend(json.loads(line) for line in stream)
        else:
            metadata = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
            if "used_sources_file" in metadata:
                raise FileNotFoundError(f"missing used source ledger: {ledger}")
            references.extend(metadata["used_sources"])
    return _unique_sources(references)


def write_used_sources(directory, references):
    """原子替换本次实际使用来源，按文档 ID 去重排序。"""
    references = _unique_sources(references)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".used-sources-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for reference in references:
                stream.write(json.dumps(reference, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / USED_SOURCES_FILE)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
