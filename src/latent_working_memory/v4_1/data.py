"""正文、问题与答案使用明确的 token 拼接边界；同文档 QA 共享正文。"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


QA_SYSTEM_PROMPT = (
    "Answer questions using the provided document. Use exact details whenever possible. "
    "Answer with a short span, yes, no, or a concise phrase. "
    "If the document does not contain the answer, reply exactly: No information available."
)


@dataclass(frozen=True)
class Episode:
    id: str
    document_id: str
    context_ids: tuple[int, ...]
    question_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


class EpisodeEncoder:
    """分别分词三个片段后拼接，不等同于整条聊天字符串的一次分词。

    context = chat 前缀 + Document + 正文；question = Question + 问题 + assistant 前缀；
    target = 答案 + assistant 结束符。teacher 提示复用 question 的编码方式。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        marker = "__LWM_DOCUMENT_PLACEHOLDER__"
        answer_marker = "__LWM_ANSWER_PLACEHOLDER__"
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user", "content": marker},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        self.prefix, self.suffix = rendered.split(marker)
        complete = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": answer_marker}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        answer_prefix, self.answer_suffix = complete.split(answer_marker)
        if answer_prefix != rendered:
            raise ValueError("chat template must expose the same assistant generation prefix")

    def _encode(self, text):
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def context(self, text):
        return self._encode(f"{self.prefix}Document:\n{text}")

    def question(self, question):
        return self._encode(f"\n\nQuestion:\n{question}{self.suffix}")

    def target(self, answer):
        return self._encode(f"{answer}{self.answer_suffix}")

    def identity(self):
        return {
            "serialization": "separately-tokenized-context-question-target",
            "system_prompt": QA_SYSTEM_PROMPT,
            "context_prefix": self.prefix + "Document:\n",
            "question_prefix": "\n\nQuestion:\n",
            "question_suffix": self.suffix,
            "answer_suffix": self.answer_suffix,
        }


def document_id(text):
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def encode_episode(text, question, answer, tokenizer, document_id=None, qa_index=0):
    """调用者必须给出明确的正文与问题；不从完整 prompt 猜测分界。"""
    encoder = EpisodeEncoder(tokenizer)
    doc_id = document_id or hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    return Episode(
        f"{doc_id}:{qa_index}", doc_id, encoder.context(text),
        encoder.question(question), encoder.target(answer),
    )


def compression_boundaries(context_length, pending_size, recent_size):
    """返回 (write_index, processed_end, compressed_end)，write_index 从 1 开始。"""
    return tuple(
        (index, index * pending_size + recent_size, index * pending_size)
        for index in range(1, max(0, context_length - recent_size) // pending_size + 1)
    )


def dataset_identity(episodes):
    """绑定选样次序与实际 token 内容，用于 checkpoint 的实验身份检查。"""
    digest = hashlib.blake2b(digest_size=20)
    for episode in episodes:
        digest.update(json.dumps([
            episode.id, episode.document_id, episode.context_ids,
            episode.question_ids, episode.target_ids,
        ], separators=(",", ":")).encode())
    return {"episodes": len(episodes), "tokenized_content": digest.hexdigest()}


def load_episodes(
    path, tokenizer, split="train", min_text_tokens=1, max_text_tokens=None,
    model_window=None, max_samples=None, seed=42,
):
    """读取 {split}.jsonl 或一个 JSONL；长度筛选不截断正文，限量按 QA 选择。"""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / f"{split}.jsonl"
    encoder = EpisodeEncoder(tokenizer)
    episodes, references, documents = [], {}, set()
    filtered_text, filtered_window, raw_qas = 0, 0, 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            location = f"{path}:{line_number}"
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"text", "qas"}:
                raise ValueError(f"{location}: expected exactly text and qas")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"{location}: text must be a nonempty string")
            if not isinstance(row["qas"], list) or not row["qas"]:
                raise ValueError(f"{location}: qas must be a nonempty list")
            doc_id = document_id(row["text"])
            if doc_id in documents:
                raise ValueError(f"{location}: duplicate document; group all its QA in one row")
            documents.add(doc_id)
            text_length = len(tokenizer.encode(row["text"], add_special_tokens=False))
            keep_text = text_length >= min_text_tokens and (
                max_text_tokens is None or text_length <= max_text_tokens
            )
            context_ids = encoder.context(row["text"])
            for qa_index, qa in enumerate(row["qas"]):
                if not isinstance(qa, dict) or set(qa) != {"question", "answer"}:
                    raise ValueError(f"{location}: QA must contain exactly question and answer")
                if any(not isinstance(value, str) or not value.strip() for value in qa.values()):
                    raise ValueError(f"{location}: question and answer must be nonempty strings")
                raw_qas += 1
                if not keep_text:
                    filtered_text += 1
                    continue
                episode = Episode(
                    f"{doc_id}:{qa_index}", doc_id, context_ids,
                    encoder.question(qa["question"]), encoder.target(qa["answer"]),
                )
                length = sum(map(len, (context_ids, episode.question_ids, episode.target_ids)))
                if model_window is not None and length > model_window:
                    filtered_window += 1
                    continue
                episodes.append(episode)
                references[episode.id] = qa
    if max_samples is not None and max_samples < len(episodes):
        episodes.sort(key=lambda item: hashlib.blake2b(
            f"{seed}:{split}:{item.id}".encode(), digest_size=16
        ).digest())
        episodes = episodes[:max_samples]
    if not episodes:
        raise ValueError(f"{path}: no episodes remain after filtering")
    selected_ids = {episode.id for episode in episodes}
    return tuple(episodes), {key: references[key] for key in selected_ids}, {
        "file": str(path), "split": split, "raw_documents": len(documents),
        "raw_document_ids": sorted(documents),
        "raw_qas": raw_qas, "episodes": len(episodes),
        "documents": len({episode.document_id for episode in episodes}),
        "filtered_text": filtered_text, "filtered_model_window": filtered_window,
        "selection_seed": seed, "max_samples": max_samples,
    }
