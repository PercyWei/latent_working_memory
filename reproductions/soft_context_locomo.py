"""用同一段对话记忆评估 LoCoMo 的各个问题，逐题保留可续跑的结果。"""

import fcntl
from itertools import groupby
import json
from pathlib import Path
from statistics import mean

from benchmarks.data import locomo_history, locomo_question_prompt
from benchmarks.protocol import data_identity
from benchmarks.scoring import score_sample, summarize_scores
from latent_working_memory.prompts import QA_SYSTEM_PROMPT
from reproductions.evaluation_runtime import source_record, write_json


def checkpoint_identity(model_dir):
    """绑定本地 Hugging Face 模型与 tokenizer，续评时识别同路径替换。"""
    files = {}
    for path in sorted(model_dir.iterdir()):
        if path.is_file() and path.suffix in {".json", ".model", ".bin", ".safetensors"}:
            files[path.name] = path.stat().st_mtime_ns
    return files


def add_evaluation_arguments(parser):
    parser.add_argument("--benchmark-data", type=Path, default=Path("data/benchmarks/locomo"),
                        help="包含 locomo10.json 的数据目录")
    parser.add_argument("--resume", action="store_true", help="继续同一配置，只回答未完成的问题")


def _read_progress(stream, samples):
    """记录按数据原序追加。只有断写的最后一行可以舍弃。"""
    records = []
    stream.seek(0)
    for line in stream:
        if not line.endswith(b"\n"):
            stream.truncate(stream.tell() - len(line))
            break
        record = json.loads(line)
        if len(records) >= len(samples) or record["id"] != samples[len(records)].id:
            raise ValueError(f"LoCoMo progress is out of order at {record['id']}")
        records.append(record)
    return records


def _conversation_memory(records):
    conversations = {}
    for row in records:
        cid = row["metadata"]["conversation_id"]
        stats = row["memory"]
        if conversations.setdefault(cid, stats) != stats:
            raise ValueError(f"inconsistent memory budget for conversation {cid}")
    return {
        "conversations": conversations,
        "mean_retention_ratio": mean(
            (item["memory_tokens"] + item["raw_history_tokens"]) / item["context_tokens"]
            for item in conversations.values()
        ),
    }


def run_evaluation(args, samples, method, settings, load_model):
    """适配层负责模型及原生读写接口；本层只向写入端传入对话正文。"""
    plan = {
        "benchmark": "locomo", "method": method, "settings": settings,
        "data": data_identity(samples, args.benchmark_data, "locomo"),
        "protocol": {
            "categories": [1, 2, 3, 4], "decoding": "greedy", "max_new_tokens": 50,
            "system_prompt": QA_SYSTEM_PROMPT, "scoring": "official LoCoMo QA F1, mean over questions",
            "history": "complete chronological dialogue; one query-independent memory per conversation",
        },
    }
    output = args.output_dir.expanduser().resolve()
    config_path = output / "evaluation_config.json"
    if args.resume and not config_path.is_file():
        raise FileNotFoundError(f"--resume requires an existing evaluation: {config_path}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    with (output / ".eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if saved["evaluation"] != plan:
                raise ValueError(f"LoCoMo settings or data differ from {config_path}")
        else:
            if any(p.name != ".eval.lock" for p in output.iterdir()):
                raise FileExistsError(f"{output} is not empty; use --resume for the same evaluation")
            write_json(config_path, {"evaluation": plan, "source": source_record()})

        with (output / "details.jsonl").open("a+b") as stream:
            records = _read_progress(stream, samples)
            if len(records) < len(samples):
                if summary_path.exists():
                    raise ValueError(f"incomplete details for a completed evaluation in {output}")
                print(f"{method} | LoCoMo {len(records)}/{len(samples)}", flush=True)
                adapter = load_model()
                pending = samples[len(records):]
                for cid, group in groupby(pending, key=lambda s: s.metadata["conversation_id"]):
                    questions = list(group)
                    memory, stats = adapter.encode_history(locomo_history(questions[0]))
                    if records and records[-1]["metadata"]["conversation_id"] == cid and records[-1]["memory"] != stats:
                        raise ValueError(f"resumed memory budget differs for conversation {cid}")
                    for question in questions:
                        meta = question.metadata
                        prompt = locomo_question_prompt(meta["question"], meta["category"])
                        prediction = adapter.answer(memory, prompt, question.max_new_tokens)
                        record = {
                            "id": question.id, "question": meta["question"], "metadata": meta,
                            "references": list(question.references), "prediction": prediction,
                            "score": score_sample("locomo", question, prediction), "memory": stats,
                        }
                        stream.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode())
                        stream.flush()
                        records.append(record)
                    del memory
                    print(f"{method} | {cid} | LoCoMo {len(records)}/{len(samples)}", flush=True)
        summary = {
            **plan, **summarize_scores("locomo", records), "memory": _conversation_memory(records),
        }
        write_json(summary_path, summary)
        print(f"{method} | LoCoMo F1 {100 * summary['score']:.2f} | {summary_path}", flush=True)
    return summary
