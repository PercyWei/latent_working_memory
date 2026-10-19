"""读取 benchmark 原生文件；prompt 中不包含参考答案或标注 evidence。"""

from dataclasses import dataclass
import json
from pathlib import Path
import re


RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
DEFAULT_RULER_LENGTHS = (8192, 16384, 32768)
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
RULER_BUDGETS = {
    **{task: 128 for task in RULER_TASKS if task.startswith("niah_")},
    "vt": 30,
    "cwe": 120,
    "fwe": 50,
    "qa_1": 32,
    "qa_2": 32,
}
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    prompt: str
    references: tuple[str, ...]
    max_new_tokens: int
    metadata: dict


def _nonempty_string(value, location):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: expected a nonempty string")
    return value


def _load_ruler(data_dir, lengths):
    if not lengths or any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("RULER lengths must be positive integers")
    if len(set(lengths)) != len(lengths):
        raise ValueError("RULER lengths must be unique")
    samples = []
    for length in lengths:
        for task in RULER_TASKS:
            path = data_dir / str(length) / task / "test.jsonl"
            sample_start = len(samples)
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    location = f"{path}:{line_number}"
                    row = json.loads(line)
                    if not isinstance(row, dict) or not {"index", "input", "outputs"} <= row.keys():
                        raise ValueError(f"{location}: expected index, input and outputs")
                    index = row["index"]
                    if type(index) is not int or index < 0:
                        raise ValueError(f"{location}: index must be a nonnegative integer")
                    prompt = _nonempty_string(row["input"], f"{location}.input")
                    references = row["outputs"]
                    if not isinstance(references, list) or not references:
                        raise ValueError(f"{location}.outputs: expected a nonempty string list")
                    references = tuple(
                        _nonempty_string(value, f"{location}.outputs") for value in references
                    )
                    metadata = {"task": task, "length": length, "index": index}
                    if "length" in row:
                        metadata["source_length"] = row["length"]
                    samples.append(
                        BenchmarkSample(
                            # 官方 NIAH 的 index 是答案字符位置，可能重复。
                            id=f"ruler/{length}/{task}/{line_number - 1}",
                            prompt=prompt,
                            references=references,
                            max_new_tokens=RULER_BUDGETS[task],
                            metadata=metadata,
                        )
                    )
            if len(samples) == sample_start:
                raise ValueError(f"{path}: no RULER samples")
    return tuple(samples)


def _locomo_history(conversation, location):
    if not isinstance(conversation, dict):
        raise ValueError(f"{location}: expected a conversation object")
    speaker_a = _nonempty_string(conversation["speaker_a"], f"{location}.speaker_a")
    speaker_b = _nonempty_string(conversation["speaker_b"], f"{location}.speaker_b")
    session_keys = sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.split("_")[1]),
    )
    if not session_keys:
        raise ValueError(f"{location}: no sessions")
    parts = [
        f"Below is a conversation between two people: {speaker_a} and {speaker_b}. "
        "The conversation takes place over multiple days and the date of each conversation "
        "is written at the beginning of the conversation.\n"
    ]
    turn_count = 0
    for key in session_keys:
        date = _nonempty_string(conversation[f"{key}_date_time"], f"{location}.{key}_date_time")
        turns = conversation[key]
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{location}.{key}: expected a nonempty turn list")
        parts.append(f"\nDATE: {date}\nCONVERSATION:\n")
        for turn_index, turn in enumerate(turns):
            turn_location = f"{location}.{key}[{turn_index}]"
            if not isinstance(turn, dict):
                raise ValueError(f"{turn_location}: expected a turn object")
            speaker = _nonempty_string(turn["speaker"], f"{turn_location}.speaker")
            text = turn["text"]
            if not isinstance(text, str):
                raise ValueError(f"{turn_location}.text: expected a string")
            parts.append(f'{speaker} said, "{text}"')
            if "blip_caption" in turn:
                caption = turn["blip_caption"]
                if not isinstance(caption, str):
                    raise ValueError(f"{turn_location}.blip_caption: expected a string")
                parts.append(f" and shared {caption}.")
            parts.append("\n\n")
            turn_count += 1
    return "".join(parts), len(session_keys), turn_count


def _load_locomo(data_dir):
    path = data_dir / "locomo10.json"
    with path.open(encoding="utf-8") as stream:
        conversations = json.load(stream)
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{path}: expected a nonempty conversation list")
    samples, conversation_ids = [], set()
    for conversation_index, row in enumerate(conversations):
        location = f"{path}[{conversation_index}]"
        if not isinstance(row, dict) or not {"sample_id", "conversation", "qa"} <= row.keys():
            raise ValueError(f"{location}: expected sample_id, conversation and qa")
        conversation_id = _nonempty_string(row["sample_id"], f"{location}.sample_id")
        if conversation_id in conversation_ids:
            raise ValueError(f"{location}: duplicate sample_id {conversation_id}")
        conversation_ids.add(conversation_id)
        history, sessions, turns = _locomo_history(row["conversation"], f"{location}.conversation")
        if not isinstance(row["qa"], list) or not row["qa"]:
            raise ValueError(f"{location}.qa: expected a nonempty QA list")
        for question_index, qa in enumerate(row["qa"]):
            qa_location = f"{location}.qa[{question_index}]"
            if not isinstance(qa, dict) or "category" not in qa:
                raise ValueError(f"{qa_location}: expected a QA object with category")
            category = qa["category"]
            if type(category) is not int or category not in {1, 2, 3, 4, 5}:
                raise ValueError(f"{qa_location}.category: expected an integer from 1 to 5")
            if category == 5:
                continue
            question = _nonempty_string(qa["question"], f"{qa_location}.question")
            answer = qa["answer"]
            if type(answer) not in {str, int, float}:
                raise ValueError(f"{qa_location}.answer: expected a text or numeric answer")
            answer = _nonempty_string(str(answer), f"{qa_location}.answer")
            instruction = question
            if category == 2:
                instruction += " Use DATE of CONVERSATION to answer with an approximate date."
            prompt = (
                history
                + "\nBased on the above conversations, write a short answer for the following "
                "question in a few words. Do not write complete and lengthy sentences. "
                "Answer with exact words from the conversations whenever possible.\n\n"
                + f"Question: {instruction}\n"
            )
            samples.append(
                BenchmarkSample(
                    id=f"locomo/{conversation_id}/{question_index}",
                    prompt=prompt,
                    references=(answer,),
                    max_new_tokens=50,
                    metadata={
                        "conversation_id": conversation_id,
                        "question_index": question_index,
                        "category": category,
                        "category_name": LOCOMO_CATEGORIES[category],
                        "question": question,
                        "sessions": sessions,
                        "turns": turns,
                    },
                )
            )
    if not samples:
        raise ValueError(f"{path}: no QA samples in categories 1–4")
    return tuple(samples)


def load_samples(benchmark: str, data_dir: Path, lengths: tuple[int, ...] = DEFAULT_RULER_LENGTHS):
    """data_dir 为对应 benchmark 的原生数据根目录。"""
    data_dir = Path(data_dir).expanduser().resolve()
    if benchmark == "ruler":
        return _load_ruler(data_dir, lengths)
    if benchmark == "locomo":
        return _load_locomo(data_dir)
    raise ValueError("benchmark must be ruler or locomo")
