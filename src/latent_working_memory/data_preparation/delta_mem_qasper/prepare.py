"""将 δ-Mem QASPER episodes 聚合为 text/qas 文档，并按论文固定划分。"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import tempfile
from zoneinfo import ZoneInfo


SPLITS = ("train", "dev", "test")
SOURCE_SYSTEM = (
    "You answer questions using earlier notes, passages, or dialogue history. "
    "Use exact details from them whenever possible. "
    "If they do not contain the answer, reply exactly: No information available."
)
QUESTION_PREFIX = "Question:\n"
QUESTION_SUFFIX = (
    "\n\nAnswer with a short span, yes, no, or a concise phrase. "
    "If the paper does not contain the answer, reply exactly: No information available."
)


@dataclass(frozen=True)
class PreparationConfig:
    input_file: Path
    output_dir: Path
    source_repo: str
    upstream_revision: str
    seed: int
    dev_fraction: float
    test_fraction: float


def load_config(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("preparation config must be a JSON object")
    for key in ("input_file", "output_dir", "source_repo", "upstream_revision"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if type(data["seed"]) is not int:
        raise ValueError("seed must be an integer")
    for key in ("dev_fraction", "test_fraction"):
        if type(data[key]) not in (int, float) or not 0 < data[key] < 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if data["dev_fraction"] + data["test_fraction"] >= 1:
        raise ValueError("dev_fraction + test_fraction must be less than 1")
    for key in ("input_file", "output_dir"):
        data[key] = (path.parent / Path(data[key]).expanduser()).resolve()
    return PreparationConfig(**data)


def read_documents(path):
    documents = {}
    question_ids = set()
    source_splits = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: episode must be a JSON object")
            for key in ("paper_id", "question_id", "source_split"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"line {line_number}: {key} must be a nonempty string")
            if row.get("source_dataset") != "qasper":
                raise ValueError(f"line {line_number}: source_dataset must be qasper")
            if row["question_id"] in question_ids:
                raise ValueError(f"line {line_number}: duplicate question_id")
            question_ids.add(row["question_id"])
            source_splits.add(row["source_split"])
            text, qa = extract_text_and_qa(row, line_number)
            paper_id = row["paper_id"]
            if paper_id not in documents:
                documents[paper_id] = {
                    "paper_id": paper_id,
                    "text": text,
                    "qas": [],
                    "question_ids": [],
                }
            document = documents[paper_id]
            if document["text"] != text:
                raise ValueError(f"line {line_number}: inconsistent text for paper_id {paper_id}")
            document["qas"].append(qa)
            document["question_ids"].append(row["question_id"])
    if not documents:
        raise ValueError("input episodes must not be empty")
    return list(documents.values()), sorted(source_splits)


def extract_text_and_qa(row, line_number):
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 4:
        raise ValueError(f"line {line_number}: expected system, document chunks, question, answer")
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"line {line_number}: message {index} must contain role and content")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError(f"line {line_number}: message {index} content must be nonempty text")
        expected_role = (
            "system" if index == 0 else "assistant" if index == len(messages) - 1 else "user"
        )
        if message["role"] != expected_role:
            raise ValueError(f"line {line_number}: unexpected role for message {index}")
    if messages[0]["content"] != SOURCE_SYSTEM:
        raise ValueError(f"line {line_number}: unexpected source system prompt")
    chunks = messages[1:-2]
    if type(row.get("num_context_chunks")) is not int or row["num_context_chunks"] != len(chunks):
        raise ValueError(f"line {line_number}: num_context_chunks disagrees with messages")
    text_parts = []
    for index, chunk in enumerate(chunks, 1):
        prefix = f"History turn {index} (User):\nPaper chunk {index}:\n"
        content = chunk["content"]
        if not content.startswith(prefix) or not content[len(prefix) :].strip():
            raise ValueError(f"line {line_number}: unexpected document chunk {index}")
        text_parts.append(content[len(prefix) :])
    question_content = messages[-2]["content"]
    if not question_content.startswith(QUESTION_PREFIX) or not question_content.endswith(
        QUESTION_SUFFIX
    ):
        raise ValueError(f"line {line_number}: unexpected question wrapper")
    question = question_content[len(QUESTION_PREFIX) : -len(QUESTION_SUFFIX)]
    if not question.strip():
        raise ValueError(f"line {line_number}: question must be nonempty text")
    return "\n\n".join(text_parts), {"question": question, "answer": messages[-1]["content"]}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_readme(output_dir, metadata):
    created = datetime.fromisoformat(metadata["created_at"])
    stamp = created.strftime("%Y%m%d %H:%M:%S Asia/Shanghai（UTC+08:00）")
    recipe = metadata["recipe"]
    rows = "\n".join(
        f"| {split} | {stats['papers']} | {stats['qas']} |"
        for split, stats in metadata["statistics"].items()
    )
    source = metadata["source"]
    text = f"""# {created:%Y%m%d}_δ-Mem QASPER 数据划分

创建时间：{stamp}

最后修订时间：{stamp}

- 来源：[δ-Mem 构造的 QASPER episodes](https://huggingface.co/datasets/{source["repo"]})，
  revision 为 `{source["revision"]}`。
- 三个划分均来自上游 `source_split=train`；dev/test 是本实验按论文留出的集合，
  **不是 QASPER 官方 dev/test**。

## 数据格式

- 每行是一篇文档，仅含 `text` 和 `qas`：

```json
{{"text": "完整正文", "qas": [{{"question": "问题", "answer": "参考答案"}}]}}
```

- `text`：按上游顺序移除各正文块的固定消息前缀，再用两个换行符连接；保留正文内容。
- `qas`：聚合该论文的全部 QA；问题移除固定提问包装，答案原样保存。
- 文档按论文在上游首次出现的顺序排列，文档内 QA 保持上游顺序。

## 划分与规模

| 划分 | 论文数 | QA 数 |
|---|---:|---:|
{rows}

- 对全部原始 `paper_id` 排序，再用 `random.Random({recipe["seed"]}).shuffle` 打乱。
- 前 `round(N × {recipe["dev_fraction"]})` 篇分给 dev，
  随后 `round(N × {recipe["test_fraction"]})` 篇分给 test，其余分给 train；同篇全部 QA 属于同一划分。
- 不分词、不做长度或答案过滤、不截断文本；使用端读取固定划分后再执行各自的长度过滤。

## 文件与字段

- `train.jsonl`、`dev.jsonl`、`test.jsonl`：采用上述 `text` / `qas` 格式。
- `preparation.json`：来源及 revision、构造配置、转换与划分规则、各划分统计。
  `statistics.<split>.document_sources` 按 JSONL 行顺序记录 `paper_id` 和 `question_ids`，
  其中 `question_ids` 与该行 `qas` 顺序一致。
- `README.md`：本说明。
"""
    (Path(output_dir) / "README.md").write_text(text, encoding="utf-8")


def prepare_dataset(config):
    output_dir = config.output_dir
    if output_dir.exists():
        raise FileExistsError(f"dataset directory already exists: {output_dir}")
    documents, source_splits = read_documents(config.input_file)
    paper_ids = sorted(document["paper_id"] for document in documents)
    random.Random(config.seed).shuffle(paper_ids)
    dev_count = round(len(paper_ids) * config.dev_fraction)
    test_count = round(len(paper_ids) * config.test_fraction)
    if not dev_count or not test_count or dev_count + test_count >= len(paper_ids):
        raise ValueError("rounded paper counts must leave nonempty train/dev/test")
    split_papers = {
        "train": paper_ids[dev_count + test_count :],
        "dev": paper_ids[:dev_count],
        "test": paper_ids[dev_count : dev_count + test_count],
    }
    paper_splits = {paper: split for split, papers in split_papers.items() for paper in papers}
    split_documents = {
        split: [document for document in documents if paper_splits[document["paper_id"]] == split]
        for split in SPLITS
    }
    config_data = asdict(config)
    config_data["input_file"] = str(config.input_file)
    config_data["output_dir"] = str(output_dir)
    metadata = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source": {
            "repo": config.source_repo,
            "revision": config.upstream_revision,
            "input_file": str(config.input_file),
            "source_dataset": "qasper",
            "source_splits": source_splits,
            "system_prompt": SOURCE_SYSTEM,
        },
        "config": config_data,
        "recipe": {
            "seed": config.seed,
            "unit": "paper_id",
            "paper_order": "sort all paper_id values, then random.Random(seed).shuffle",
            "allocation": "dev first, test next, train remainder",
            "rounding": "round(total_papers * fraction)",
            "dev_fraction": config.dev_fraction,
            "test_fraction": config.test_fraction,
            "text": {
                "chunk_prefix": "History turn {i} (User):\nPaper chunk {i}:\n",
                "operation": "remove exact chunk prefix; join all chunk contents in source order",
                "separator": "\n\n",
            },
            "qas": {
                "question_prefix": QUESTION_PREFIX,
                "question_suffix": QUESTION_SUFFIX,
                "question": "remove exact prefix and suffix",
                "answer": "preserve final assistant content exactly",
            },
            "aggregation": "one document per paper_id; require identical text across episodes",
            "document_order": "first occurrence of paper_id in source episodes",
            "qa_order": "source episode order within each paper",
            "tokenization": False,
            "filtering": False,
        },
        "statistics": {
            split: {
                "papers": len(split_papers[split]),
                "qas": sum(len(document["qas"]) for document in split_documents[split]),
                "paper_ids": sorted(split_papers[split]),
                "document_sources": [
                    {"paper_id": document["paper_id"], "question_ids": document["question_ids"]}
                    for document in split_documents[split]
                ],
            }
            for split in SPLITS
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for split in SPLITS:
            with (staging_dir / f"{split}.jsonl").open("w", encoding="utf-8") as target:
                for document in split_documents[split]:
                    row = {"text": document["text"], "qas": document["qas"]}
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(staging_dir / "preparation.json", metadata)
        write_readme(staging_dir, metadata)
        staging_dir.rename(output_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    metadata = prepare_dataset(config)
    print(f"Saved unfiltered QASPER splits to {config.output_dir}", flush=True)
    for split, statistics in metadata["statistics"].items():
        print(f"{split}: {statistics['qas']} QA / {statistics['papers']} papers", flush=True)
