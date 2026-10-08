"""从 FineWeb 构造完整候选窗口、来源记录与多段 token 计划。"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from glob import glob
from itertools import accumulate
import json
from pathlib import Path
import random
import uuid
from zoneinfo import ZoneInfo

from latent_working_memory.data_preparation.fineweb_multisegment.config import DataPreparationConfig
from latent_working_memory.data_preparation.fineweb_multisegment.records import MultisegmentSample
from latent_working_memory.data_preparation.fineweb_multisegment.sources import collect_documents
from latent_working_memory.data_preparation.pretrain.config import PreparationConfig


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    split: str
    dedup_cluster: str
    source: dict


def segment_lengths(count, minimum, maximum, available, rng):
    """先采样各段长度；available 只约束可用预算，不是待分配的目标总长。"""
    remaining, parts = available, []
    for slots in range(count, 0, -1):
        high = min(maximum, remaining - minimum * (slots - 1))
        part = rng.randint(minimum, high)
        parts.append(part)
        remaining -= part
    rng.shuffle(parts)
    return parts


def build_samples(documents, config, split):
    """最终选中来源逐篇构造一次；同一文档的分段随机流不受读取批次影响。"""
    for i, document in enumerate(d for d in documents if d.split == split):
        rng = random.Random(f"{config.seed}:multisegment:{document.document_id}")
        available = config.available_content_tokens(len(document.text))
        max_count = min(config.max_segments, available // config.min_segment_tokens)
        count = rng.randint(config.min_segments, max_count)
        parts = segment_lengths(
            count, config.min_segment_tokens, config.max_segment_tokens, available, rng
        )
        chars = config.candidate_chars(sum(parts))
        start = rng.randint(0, len(document.text) - chars)
        yield MultisegmentSample(
            f"{split}/{i:06d}",
            document.document_id,
            document.dedup_cluster,
            document.text[start : start + chars],
            tuple(accumulate(parts)),
            dict(document.source, char_span=[start, start + chars]),
        )


def load_previous_sources(datasets):
    """仅合并显式指定数据集实际用过的文档，不递归读取它们的历史依赖。"""
    sources = {}
    for directory in datasets:
        metadata = json.loads((Path(directory) / "preparation.json").read_text(encoding="utf-8"))
        for item in metadata["used_sources"]:
            identity = item["document_id"]
            if identity in sources and sources[identity]["source"] != item["source"]:
                raise ValueError(f"previous document has inconsistent source locations: {identity}")
            sources[identity] = item
    return list(sources.values())


def write_readme(output_dir, metadata):
    created = datetime.fromisoformat(metadata["created_at"])
    stamp = created.strftime("%Y%m%d %H:%M:%S UTC+08:00")
    config = DataPreparationConfig(**metadata["config"])
    rows = []
    for split, stats in metadata["statistics"].items():
        interval = (
            f"{stats['content_tokens_min']:,}–{stats['content_tokens_max']:,}"
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

`{output_dir.name}` 基于 [FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 英文网页文本构建，用于单次压缩、多段写入、历史重构及续文预测。每行保存一条完整候选窗口与累计 token 写入切点，训练无需原始 Parquet。

本次运行标识为 `{metadata["run_id"]}`；目录名由构造参数与该标识组成，创建时间记录实际执行时间。本次显式排除 {len(metadata["previous_datasets"])} 份已有数据集，目录清单见 `preparation.json.previous_datasets`。

## 规模与规则

共 {total:,} 条候选轨迹，引用 {metadata["referenced_documents"]:,} 篇来源文档；分 {metadata["source_statistics"]["source_batches"]:,} 批扫描 {metadata["source_statistics"]["scanned_documents"]:,} 篇原始文档。

| 划分 | 候选轨迹 | 来源文档 | 计划正文 tokens 范围 |
|---|---:|---:|---:|
{chr(10).join(rows)}

| 参数 | 取值 |
|---|---|
| 构建基准 K | {config.capacity} |
| 每段倍率 | {config.min_segment_ratio:g}–{config.max_segment_ratio:g} × K |
| 每段计划长度 | {config.min_segment_tokens}–{config.max_segment_tokens} tokens |
| 每条段数 | {config.min_segments}–{config.max_segments} |
| 正文自然范围 | {config.min_segments * config.min_segment_tokens}–{config.max_segments * config.max_segment_tokens} tokens，不含续文与字符缓冲 |
| 续文目标 | {config.continuation_tokens} tokens |
| 整体字符余量系数 | {config.content_reserve_ratio:g} |
| 每批原始读取量 | {config.source_batch_size:,} 篇；不足配额则继续读取 |

名称中的 train 规模为目标训练轨迹数的简写，精确配额与实际产量见元数据；K 用于构建段长，不强制下游模型采用相同记忆容量。每篇最终选中的文档只生成一条轨迹，累计合格来源池中每个去重簇至多选一篇，三份划分按来源簇隔离。

## 构建与使用

1. 从同一 Parquet 随机流无放回地分批读取，基础及长度过滤后加入累计候选池；每轮重新聚类、排除旧数据集已用来源及其匹配簇，再按来源簇划分并填充配额。配额不足继续读，原始来源耗尽则报错。
2. 对最终选中的每篇文档，先在原文可容纳的范围内抽取段数和各段长度，再计算计划正文长度 `L = sum(各段长度)`；没有独立固定总长或总长上限。
3. 令 Q 为续文目标 {config.continuation_tokens} tokens，按 `ceil(4 × (L + Q) × {config.content_reserve_ratio:g})` 估算完整窗口字符数，再随机选择窗口起点；构建时不加载 tokenizer。
4. 使用时对全文一次分词，按 `write_token_ends` 切分；单次压缩取完整前缀，首次写入取首段。按需要右裁剪后，续文从实际终点紧邻读取；余量不参与正文监督。

## 文件与字段

- `train.jsonl`、`dev.jsonl`、`test.jsonl`：每行含 `sample_id`、`document_id`、`dedup_cluster`、`text`、`write_token_ends` 和 `source`。
- `source` 保存原文文件、row group、组内行号及左闭右开的字符区间，仅用于追溯；正文已包含在 `text`。
- [preparation.json](preparation.json)：精确构建参数、运行标识、来源规则、排除记录、统计及 `used_sources`。后续构建可通过 `--previous-datasets` 显式传入本目录，排除这些已用文档。

写入切点是目标 token 位置，字符缓冲可能比分词后的实际需求更长。可用样本数依 tokenizer、输入裁剪与续文要求而变；本表报告构建候选数。原始语料与许可说明见 [FineWeb 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb)。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def prepare_dataset(config, output_root, run_id=None, previous_datasets=()):
    created_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    if run_id is None:
        run_id = created_at.strftime("%Y%m%d")
    output_dir = Path(output_root) / config.dataset_name(run_id)
    if output_dir.exists():
        raise FileExistsError(f"use a new dataset directory: {output_dir}")
    paths = [Path(p) for p in sorted(glob(config.source_glob, recursive=True))]
    if not paths:
        raise ValueError(f"no FineWeb Parquet files match {config.source_glob}")
    previous_datasets = [str(path) for path in previous_datasets]
    previous = load_previous_sources(previous_datasets)
    print("Reading source batches until all split quotas are filled", flush=True)
    recipe = PreparationConfig()
    sources, source_statistics = collect_documents(paths, config, previous, recipe)
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
            )
        )
    output_dir.mkdir(parents=True)
    used, statistics = {}, {}
    for split in ("train", "dev", "test"):
        doc_ids, segment_counts = set(), Counter()
        segment_min, segment_max, total_min, total_max = None, None, None, None
        planned_tokens = characters = trajectories = 0
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for sample in build_samples(documents, config, split):
                stream.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
                ends = sample.write_token_ends
                parts = [b - a for a, b in zip((0,) + ends[:-1], ends, strict=True)]
                segment_counts[len(parts)] += 1
                segment_min = min(parts) if segment_min is None else min(segment_min, *parts)
                segment_max = max(parts) if segment_max is None else max(segment_max, *parts)
                total_min = ends[-1] if total_min is None else min(total_min, ends[-1])
                total_max = ends[-1] if total_max is None else max(total_max, ends[-1])
                planned_tokens += ends[-1]
                characters += len(sample.text)
                trajectories += 1
                doc_ids.add(sample.document_id)
                used[sample.document_id] = {
                    "document_id": sample.document_id,
                    "dedup_cluster": sample.dedup_cluster,
                    "source": {k: v for k, v in sample.source.items() if k != "char_span"},
                }
        statistics[split] = {
            "trajectories": trajectories,
            "documents": len(doc_ids),
            "segment_counts": dict(sorted(segment_counts.items())),
            "segment_tokens_min": segment_min,
            "segment_tokens_max": segment_max,
            "content_tokens_min": total_min,
            "content_tokens_max": total_max,
            "planned_content_tokens": planned_tokens,
            "candidate_text_characters": characters,
        }
    metadata = {
        "preparation_id": str(uuid.uuid4()),
        "created_at": created_at.isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset": output_dir.name,
        "source_dataset": "HuggingFaceFW/fineweb",
        "source_subset": "sample-10BT",
        "config": asdict(config),
        "length_estimation": "L = sum(sampled segment lengths); candidate characters = ceil(4 * (L + continuation_tokens) * content_reserve_ratio); write_token_ends are target token positions",
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
        "excluded_sources": previous,
        "used_sources": sorted(used.values(), key=lambda item: item["document_id"]),
        "referenced_documents": len(used),
        "statistics": statistics,
    }
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
    parser = argparse.ArgumentParser(description="构造自包含 FineWeb 多段文本与 token 写入计划")
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
        help="本次需排除的已有数据集目录；读取各自 preparation.json.used_sources",
    )
    args = parser.parse_args()
    config = DataPreparationConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    prepare_dataset(config, args.output_root, args.run_id, args.previous_datasets)
