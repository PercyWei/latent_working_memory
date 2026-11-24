"""跨方法展示的数据范围，不参与运行身份或缓存校验。"""

from collections import defaultdict
import hashlib
import json


def comparison_data(samples, benchmark):
    """按实际输入、参考答案和预算标识数据；忽略目录、ID 和执行顺序。"""
    grouped = defaultdict(list)
    for sample in samples:
        length = sample.metadata["length"] if benchmark == "ruler" else None
        record = {
            "prompt": sample.prompt,
            "references": sorted(sample.references),
            "max_new_tokens": sample.max_new_tokens,
            "task": sample.metadata["task"] if benchmark == "ruler" else sample.metadata["category"],
        }
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        grouped[length].append(hashlib.blake2b(encoded, digest_size=32).digest())
    result = []
    for length, records in sorted(grouped.items()):
        fingerprint = hashlib.blake2b(digest_size=32)
        for record in sorted(records):
            fingerprint.update(record)
        result.append({"length": length, "samples": len(records),
                       "fingerprint": fingerprint.hexdigest()})
    return result
