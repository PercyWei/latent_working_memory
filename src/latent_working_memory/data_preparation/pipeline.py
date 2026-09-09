from __future__ import annotations

import hashlib
import itertools
import json
import uuid
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping

from transformers import PreTrainedTokenizerBase

from latent_working_memory.data_preparation.audit import audit_preparation
from latent_working_memory.data_preparation.config import PreparationConfig
from latent_working_memory.data_preparation.dedup import cluster_documents, source_key
from latent_working_memory.data_preparation.fineweb import (
    data_contract,
    document_episodes,
    document_split,
)
from latent_working_memory.data_preparation.quality import QUALITY_FILTER, document_rejection_reason
from latent_working_memory.data_preparation.scoring import DocumentScorer
from latent_working_memory.v1.config import ExperimentConfig


def prepare_fineweb(
    records: Iterable[Mapping[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    output_dir: Path,
    preparation: PreparationConfig,
    topic_annotations: dict[str, dict[str, Any]] | None = None,
    scorer: DocumentScorer | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"use a new preparation directory: {output_dir}")
    if (
        preparation.review_model_name_or_path or preparation.fluency_model_name_or_path
    ) and scorer is None:
        raise ValueError("the preparation recipe requires a local DocumentScorer")
    if config.granularity_weights[-1] > 0 and topic_annotations is None:
        raise ValueError("positive topic_group weight requires topic annotations")
    candidates = list(itertools.islice(records, preparation.max_documents))
    # Validate external document fields before clustering, and retain the full candidate pool.
    coarse = [document_rejection_reason(record) for record in candidates]
    clusters = cluster_documents(candidates, preparation)
    output_dir.mkdir(parents=True)
    limits = (
        dict(zip(("train", "dev", "test"), preparation.split_document_limits, strict=True))
        if preparation.split_document_limits
        else None
    )
    counters = Counter(documents_read=len(candidates))
    quality_counts = Counter()
    seen_clusters = set()
    seen_fragments: dict[bytes, str] = {}
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context((output_dir / f"{name}.jsonl").open("w", encoding="utf-8"))
            for name in ("train", "dev", "test", "documents", "candidates", "document-decisions")
        }
        for position, (record, cluster, rejection) in enumerate(
            zip(candidates, clusters, coarse, strict=True)
        ):
            handles["candidates"].write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            split = document_split(cluster, config)
            decision = {
                "document_id": record["id"],
                "cluster": cluster,
                "split": split,
                "status": "accepted",
                "reason": None,
                "scores": {},
            }
            if rejection:
                quality_counts[f"documents_rejected_{rejection}"] += 1
                decision.update(status="quality_rejected", reason=rejection)
            elif cluster in seen_clusters:
                counters["duplicates_removed"] += 1
                decision.update(status="duplicate", reason="same_source_or_near_duplicate_cluster")
            elif limits and counters[f"{split}_documents"] == limits[split]:
                decision.update(status="quota_skipped", reason="split_document_limit")
            else:
                scores = scorer.score(dict(record)) if scorer is not None else {}
                decision["scores"] = scores
                reviews = scores.get("review", [])
                excluded_spans = tuple(
                    tuple(span) for review in reviews for span in review["excluded_spans"]
                )
                annotation = (
                    topic_annotations.get(record["id"]) if topic_annotations is not None else None
                )
                episodes = document_episodes(
                    record, tokenizer, config, annotation, quality_counts, excluded_spans
                )
                unique = []
                source_id = source_key(record["url"])
                for episode in episodes:
                    if len(episode.input_ids) >= QUALITY_FILTER["dedup_input_min_tokens"]:
                        key = hashlib.blake2b(
                            " ".join(episode.reads[0].references[0].text.split()).encode()
                        ).digest()
                        if key in seen_fragments and seen_fragments[key] != source_id:
                            counters["duplicate_fragments_removed"] += 1
                            continue
                        seen_fragments[key] = source_id
                    episode.sources[0].provenance["dedup_cluster"] = cluster
                    unique.append(episode)
                if not unique:
                    counters["documents_without_legal_views"] += 1
                    decision.update(status="quality_rejected", reason="no_usable_original_span")
                else:
                    seen_clusters.add(cluster)
                    counters[f"{split}_documents"] += 1
                    handles["documents"].write(
                        json.dumps(
                            {
                                "split": split,
                                "cluster": cluster,
                                "excluded_spans": excluded_spans,
                                "record": dict(record),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    for episode in unique:
                        handles[split].write(
                            json.dumps(episode.to_record(), ensure_ascii=False) + "\n"
                        )
                        counters[f"{split}_episodes"] += 1
            counters[f"document_status/{decision['status']}"] += 1
            handles["document-decisions"].write(json.dumps(decision, ensure_ascii=False) + "\n")
            if (position + 1) % 500 == 0:
                print(
                    json.dumps({"prepared_candidates": position + 1, "statistics": dict(counters)}),
                    flush=True,
                )
    if limits and any(counters[f"{key}_documents"] != count for key, count in limits.items()):
        raise ValueError("document budget exhausted before requested split counts were reached")
    if not any(counters[f"{split}_episodes"] for split in ("train", "dev", "test")):
        raise ValueError("the candidate pool contains no usable pretraining views")
    audit = audit_preparation(output_dir, tokenizer, config, preparation)
    metadata = {
        "preparation_id": str(uuid.uuid4()),
        "contract": data_contract(config),
        "recipe": preparation.to_dict(),
        "sentence_splitter": "pysbd==0.3.4",
        "statistics": dict(counters),
        "quality_statistics": dict(quality_counts),
        "split_document_limits": limits,
        "lengths": audit["lengths"],
        "scoring_protocols": scorer.protocols if scorer is not None else {},
        "audit": audit,
    }
    # This is the completion record consumed by training. Failed builds never publish it.
    (output_dir / "preparation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    return metadata
