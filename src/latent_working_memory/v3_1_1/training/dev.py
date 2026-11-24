"""训练中的固定主倍率／边界策略验证，保留逐正文、逐 QA 结果。"""

import hashlib

import torch

from latent_working_memory.data_preparation.personamem.audit import scores
from ..compression import compress_documents_batch
from ..rollout import execute_batch


def _questions(document, qa_objective):
    if qa_objective:
        return [(qa.prompt_ids, qa.target_ids, qa.question, qa.answer) for qa in document.qas]
    return [(document.prompt_ids, document.target_ids, document.task, None)]


def _add_record(totals, record):
    prefix = record["condition"]
    for name, value in record["metrics"].items():
        # 分段倍率按压缩块平均，其余正文指标按文档平均。
        weight = record["metrics"]["memory_blocks"] if name == "segment_compression_ratio" else 1
        pair = totals.setdefault(f"{prefix}/{name}", [0.0, 0.0])
        pair[0] += value * weight
        pair[1] += weight
    for answer in record["qas"]:
        for name in ("em", "f1"):
            if name in answer:
                pair = totals.setdefault(f"{prefix}/{name}", [0.0, 0.0])
                pair[0] += answer[name]
                pair[1] += 1


def _compress(model, documents, config, condition, ratio, collect_boundary):
    compression = config.training.compression
    contexts = [document.input_ids for document in documents]
    if condition == "policy":
        return execute_batch(
            model, contexts, config.model.structure.pending_stride,
            pending_size=config.model.structure.pending_size, recent_size=config.recent_size(),
            collect_boundary=collect_boundary,
        )
    if condition == "sampled":
        # 每篇独立固定种子，样本分片、排序和 batch 大小不改变验证边界。
        results = []
        for document in documents:
            digest = hashlib.blake2b(
                f"{config.seed}:dev:{document.id}".encode(), digest_size=8,
            ).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest, "big"))
            results.extend(compress_documents_batch(
                model, [document.input_ids], compression, generator=generator,
                recent_size=config.recent_size(),
            ))
        return results
    return compress_documents_batch(
        model, contexts, compression, fixed_ratio=ratio,
        recent_size=config.recent_size(), pending_size=config.model.structure.pending_size,
    )


def _evaluate_batch(model, documents, config, condition, ratio, qa_objective, boundary_records):
    with model.forward_context():
        paths = _compress(model, documents, config, condition, ratio, boundary_records is not None)
    questions = [_questions(document, qa_objective) for document in documents]
    reader_prefix_ids = documents[0].reader_prefix_ids if qa_objective else ()
    answers = [[] for _ in documents]
    for question_index in range(max(map(len, questions))):
        active = [index for index, values in enumerate(questions) if question_index < len(values)]
        selected = [questions[index][question_index] for index in active]
        prompts = [paths[index].retained_ids + question[0]
                   for index, question in zip(active, selected, strict=True)]
        memories = [paths[index].blocks for index in active]
        with model.forward_context():
            losses = model.answer_nll_batch(
                memories, prompts, [question[1] for question in selected],
                reader_prefix_ids=reader_prefix_ids,
            )
            generated = (model.generate_batch(
                memories, prompts, [config.training.dev.max_new_tokens] * len(active),
                reader_prefix_ids=reader_prefix_ids,
            ) if qa_objective else None)
        for row, index in enumerate(active):
            prompt, target, question, reference = selected[row]
            answer = {
                "qa_index": question_index, "question": question,
                "nll": float(losses[row]), "target_tokens": len(target),
            }
            if generated is not None:
                result = generated[row]
                prediction = model.tokenizer.decode(result["token_ids"], skip_special_tokens=True).strip()
                answer.update(
                    reference=reference, prediction=prediction,
                    generated_tokens=len(result["token_ids"]), stop_reason=result["stop_reason"],
                    **scores(prediction, reference),
                )
            answers[index].append(answer)

    records = []
    for document, path, rows in zip(documents, paths, answers, strict=True):
        nll = sum(answer["nll"] for answer in rows) / len(rows)
        slots = len(path.blocks) * model.memory_slots
        recent = len(path.recent_ids)
        pending = len(path.pending_ids)
        retention = (slots + pending + recent) / len(document.input_ids)
        prefix_length = len(document.input_ids) - recent
        prefix_retention = (slots + pending) / prefix_length if prefix_length else 0.0
        nll_name = "qa_nll" if qa_objective else f"{document.task}_nll"
        metrics = {
            nll_name: nll, "memory_blocks": len(path.blocks), "memory_slots": slots,
            "recent_tokens": recent, "pending_tokens": pending, "retention_ratio": retention,
            "compression_retention_ratio": prefix_retention,
        }
        if slots:
            metrics["segment_compression_ratio"] = sum(path.block_lengths) / slots
        if config.uses_boundary_policy:
            objective = config.training.objective
            cost = prefix_retention ** objective.memory_cost_power
            metrics.update(memory_cost=cost, objective=nll + objective.memory_cost_weight * cost)
        record = {
            "id": document.id, "document_id": document.document_id,
            "condition": condition, "context_tokens": len(document.input_ids),
            "block_lengths": list(path.block_lengths),
            "block_ratios": [length / model.memory_slots for length in path.block_lengths],
            "recent_tokens": recent, "pending_tokens": pending, "metrics": metrics, "qas": rows,
        }
        if condition == "policy":
            if boundary_records is not None:
                boundary_records.append({"id": document.id, **path.boundary_trace})
            mask = torch.tensor(path.decision_mask, device=path.entropy.device, dtype=torch.bool)
            decisions = int(mask.sum())
            metrics.update(
                policy_entropy=float(path.entropy[mask].mean()) if decisions else 0.0,
                commit_rate=sum(path.actions) / decisions if decisions else 0.0,
            )
            record.update(actions=list(path.actions), decision_mask=list(path.decision_mask))
        records.append(record)
    return records


@torch.no_grad()
def evaluate_dev_conditions(model, documents, config, baseline_cache=None, boundary_records=None):
    """返回可跨 rank 汇总的 (sum,count) 指标及本地逐样本记录。"""
    model.eval()
    qa_objective = config.training.objective.type == "qa"
    conditions = [(f"ratio_{ratio}", ratio) for ratio in config.training.dev.fixed_ratios]
    if config.uses_boundary_policy:
        conditions.insert(0, ("policy", None))
    elif not conditions:
        conditions.append(("single" if config.training.compression.mode == "single" else "sampled", None))
    totals, records = {}, []
    batch_size = config.training.loop.micro_batch_size
    for condition, ratio in conditions:
        cacheable = (baseline_cache is not None and config.uses_boundary_policy
                     and config.training.update_mode == "boundary_only" and ratio is not None)
        pending = []
        for document in documents:
            key = (condition, document.id)
            if cacheable and key in baseline_cache:
                record = baseline_cache[key]
                records.append(record)
                _add_record(totals, record)
            else:
                pending.append(document)
        for start in range(0, len(pending), batch_size):
            batch_records = _evaluate_batch(
                model, pending[start:start + batch_size], config, condition, ratio, qa_objective,
                boundary_records,
            )
            for record in batch_records:
                records.append(record)
                _add_record(totals, record)
                if cacheable:
                    baseline_cache[(condition, record["id"])] = record
    return totals, records
