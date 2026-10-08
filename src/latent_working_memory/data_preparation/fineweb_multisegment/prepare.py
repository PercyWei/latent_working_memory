"""从 FineWeb 构造固定字符分段的正文、续文候选及来源记录。"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import accumulate
import json
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.fineweb_multisegment.records import MultisegmentSample
from latent_working_memory.data_preparation.fineweb_multisegment.sources import collect_documents
from latent_working_memory.data_preparation.fineweb_source import (
    USED_SOURCES_FILE,
    load_previous_sources,
    source_files,
    write_used_sources,
)
from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.segmentation import TOKEN_ESTIMATION_RULE


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    split: str
    dedup_cluster: str
    source: dict
    windows: list[tuple[int, list[int], int]]


def build_samples(documents, config, split):
    """输出配额选择时保存的非重叠窗口计划，字符分段与 FactQA 共用规则。"""
    for document in (d for d in documents if d.split == split):
        for start, parts, candidate_end in document.windows:
            cuts = [0, *accumulate(config.window.reserved_chars(part) for part in parts)]
            end = start + cuts[-1]
            text = document.text[start:end]
            yield MultisegmentSample(
                trajectory_id=f"{document.document_id}:{start}:{end}",
                document_id=document.document_id,
                dedup_cluster=document.dedup_cluster,
                split=split,
                source=document.source,
                window_char_span=[start, end],
                text=text,
                segments=[
                    {"segment_id": f"seg{i}", "char_span": [a, b]}
                    for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:], strict=True))
                ],
                continuation=document.text[end:candidate_end],
                text_char_length=len(text),
                estimated_tokens=len(text) / 4,
                estimated_tokens_rule=TOKEN_ESTIMATION_RULE,
            )


def write_readme(output_dir, metadata):
    created = datetime.fromisoformat(metadata["created_at"])
    stamp = created.strftime("%Y%m%d %H:%M:%S UTC+08:00")
    config = DataPreparationConfig.from_mapping(metadata["config"])
    rows = []
    for split, stats in metadata["statistics"].items():
        interval = (
            f"{stats['text_char_length_min']:,}–{stats['text_char_length_max']:,}"
            if stats["trajectories"]
            else "—"
        )
        rows.append(
            f"| {split} | {stats['trajectories']:,} | {stats['documents']:,} | {interval} |"
        )
    total = sum(s["trajectories"] for s in metadata["statistics"].values())
    text = f"""# {created:%Y%m%d}_FineWeb 多段文本数据集

创建时间：{stamp}

最后修订时间：{stamp}

`{output_dir.name}` 基于 [FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 英文网页构建，用于多段写入、正文重构与续文预测。每篇文档可生成多条互不重叠的轨迹，保存正文、固定字符分段和紧邻正文的续文候选，无需原始 Parquet 即可读取。

## 规模与规则

共 {total:,} 条轨迹，引用 {metadata["referenced_documents"]:,} 篇文档；分 {metadata["source_statistics"]["source_batches"]:,} 批扫描 {metadata["source_statistics"]["scanned_documents"]:,} 篇来源文档。

| 划分 | 轨迹数 | 来源文档数 | 正文字符数范围 |
|---|---:|---:|---:|
{chr(10).join(rows)}

| 参数 | 取值 |
|---|---|
| 构建基准 K | {config.window.capacity} |
| 每段倍率 | {config.window.min_segment_ratio:g}–{config.window.max_segment_ratio:g} × K |
| 每段字符数（含余量） | {config.window.min_segment_chars}–{config.window.max_segment_chars} |
| 每条段数 | {config.window.min_segments}–{config.window.max_segments} |
| 续文目标 Q | {config.window.continuation_tokens} tokens |
| 各段与续文余量系数 α | {config.window.content_reserve_ratio:g} |
| 每批原始读取量 | {config.source_batch_size:,} 篇 |

K 与倍率指定余量前的名义长度；保存的正文包含逐段余量，`estimated_tokens` 按实际正文字符数÷4计算，不等同于名义长度，也不约束 tokenizer 的实际 token 数。目录名由构造参数和 `run_id={metadata["run_id"]}` 组成，train 规模是目标训练轨迹数的简写，精确配额见 `preparation.json.config`。

## 构造流程

1. 按固定随机流无放回分批读取，质量及长度过滤后加入累计候选池。按文档 ID、规范化 URL、正文及近重复关系聚类，排除已有数据使用的来源及其匹配簇；每簇至多选一篇，按簇划分 train/dev/test，同篇全部轨迹归于同一划分。配额按轨迹计数，不足继续读取，来源耗尽则报错。
2. 从原文起点顺序构造窗口，根据剩余字符预算抽取段数及每段名义长度 `lᵢ`。各段独立扩充为 `ceil(4 × lᵢ × α)` 个字符，逐段取整后求和得到正文长度；采样时为剩余段保留最低字符预算。
3. 正文后保留独立的 `ceil(4 × Q × α)` 字符续文窗口，下一条从该续文末尾开始，保证正文及续文完整窗口无重叠。余文不足最小窗口时停止；配额最后一篇仅保留所需轨迹。正文保存为 `text`，尾部保存为 `continuation`，按最终字符长度记录分段。读取时先切段再分别分词，实际续文不足 Q tokens 时过滤。

## 文件与字段

- `train.jsonl`、`dev.jsonl`、`test.jsonl`：每行一条轨迹，共有字段与 FactQA 一致。
- `preparation.json`：配置、创建时间、来源规则、排除目录、实际统计及 `used_sources_file`。本次显式排除 {len(metadata["previous_datasets"])} 份数据集；后续构造可通过 `--previous-datasets` 传入本目录。
- `used-sources.jsonl`：按文档 ID 去重排序的实际已用原文，仅登记本次生成轨迹的文档。

| 字段 | 含义 |
|---|---|
| `trajectory_id` / `document_id` / `dedup_cluster` / `split` | 轨迹、来源文档、去重簇及数据划分 |
| `source` | 原文 `file`、`row_group`、`row_index` |
| `window_char_span` | 正文在原文中的左闭右开字符区间 |
| `text` / `segments` | 正文及分段；每段含 `segment_id` 和相对正文的 `char_span` |
| `text_char_length` / `estimated_tokens` / `estimated_tokens_rule` | 正文字符数、估算 token 数及规则 `len(text) / 4` |
| `continuation` | 紧接正文的连续尾部，仅含续文目标及其自身余量 |

字符下标按 Python 字符串索引计数，分段无间隙地覆盖正文；不保存 token 切点。上表报告构建轨迹数，实际可用数量受 tokenizer 及输入预算影响。原始语料与许可说明见 [FineWeb 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb)。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def prepare_dataset(config, output_root, run_id=None, previous_datasets=()):
    created_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    if run_id is None:
        run_id = created_at.strftime("%Y%m%d")
    output_dir = Path(output_root) / config.dataset_name(run_id)
    if output_dir.exists():
        raise FileExistsError(f"use a new dataset directory: {output_dir}")
    paths = source_files(config.source_dir)
    previous_datasets = [str(path) for path in previous_datasets]
    previous = load_previous_sources(previous_datasets)
    recipe = PreparationConfig()
    sources, source_statistics = collect_documents(paths, config, previous, recipe)
    print(
        f"Source quotas filled; saving {source_statistics['selected_trajectories']:,} "
        f"trajectories from {len(sources):,} documents to {output_dir}",
        flush=True,
    )
    documents = []
    for row in sources:
        documents.append(
            Document(
                row["record"]["id"],
                row["record"]["text"],
                row["split"],
                row["cluster"],
                {
                    "file": row["location"]["source_file"],
                    "row_group": row["location"]["row_group"],
                    "row_index": row["location"]["row_index"],
                },
                row["windows"],
            )
        )
    output_dir.mkdir(parents=True)
    used, statistics = {}, {}
    for split in ("train", "dev", "test"):
        doc_ids, segment_counts = set(), Counter()
        segment_min, segment_max, total_min, total_max = None, None, None, None
        characters = continuation_characters = trajectories = 0
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for sample in build_samples(documents, config, split):
                stream.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                parts = [b - a for a, b in (s["char_span"] for s in sample.segments)]
                segment_counts[len(parts)] += 1
                segment_min = min(parts) if segment_min is None else min(segment_min, *parts)
                segment_max = max(parts) if segment_max is None else max(segment_max, *parts)
                total_min = (
                    len(sample.text) if total_min is None else min(total_min, len(sample.text))
                )
                total_max = (
                    len(sample.text) if total_max is None else max(total_max, len(sample.text))
                )
                characters += len(sample.text)
                continuation_characters += len(sample.continuation)
                trajectories += 1
                doc_ids.add(sample.document_id)
                used[sample.document_id] = {
                    "document_id": sample.document_id,
                    "dedup_cluster": sample.dedup_cluster,
                    "source": sample.source,
                }
        statistics[split] = {
            "trajectories": trajectories,
            "documents": len(doc_ids),
            "segment_counts": dict(sorted(segment_counts.items())),
            "segment_char_length_min": segment_min,
            "segment_char_length_max": segment_max,
            "text_char_length_min": total_min,
            "text_char_length_max": total_max,
            "text_characters": characters,
            "continuation_characters": continuation_characters,
        }
    metadata = {
        "preparation_id": str(uuid.uuid4()),
        "created_at": created_at.isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset": output_dir.name,
        "source_dataset": "HuggingFaceFW/fineweb",
        "source_subset": "sample-10BT",
        "config": config.to_dict(),
        "length_estimation": "Segment characters = ceil(4 * sampled nominal tokens * content_reserve_ratio); continuation characters = ceil(4 * continuation_tokens * content_reserve_ratio); sum the independently rounded windows; estimated_tokens = len(text) / 4 includes segment reserves",
        "source_recipe": {
            name: getattr(recipe, name)
            for name in (
                "min_document_chars",
                "near_duplicate_threshold",
                "near_duplicate_min_words",
            )
        },
        "source_files": [str(p) for p in paths],
        "source_statistics": dict(source_statistics),
        "previous_datasets": previous_datasets,
        "excluded_source_count": len(previous),
        "used_sources_file": USED_SOURCES_FILE,
        "referenced_documents": len(used),
        "statistics": statistics,
    }
    write_used_sources(output_dir, used.values())
    write_readme(output_dir, metadata)
    (output_dir / "preparation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {sum(s['trajectories'] for s in statistics.values())} text trajectories to {output_dir}",
        flush=True,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="构造自包含 FineWeb 多段文本、固定字符分段与续文候选"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="构造配置 JSON 路径；K64/K512 配置位于 configs/data_preparation/fineweb-multisegment/",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--run-id", help="本次产物标识；默认使用 Asia/Shanghai 当前日期 YYYYMMDD")
    parser.add_argument(
        "--previous-datasets",
        type=Path,
        nargs="+",
        default=[],
        metavar="DIR",
        help="本次需排除的已有数据集目录；读取各自 used-sources.jsonl（兼容旧 preparation.json.used_sources）",
    )
    args = parser.parse_args()
    config = DataPreparationConfig.from_mapping(json.loads(args.config.read_text(encoding="utf-8")))
    prepare_dataset(config, args.output_root, args.run_id, args.previous_datasets)
