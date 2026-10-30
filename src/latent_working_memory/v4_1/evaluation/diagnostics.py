"""评估输出的字符展示与行为统计，不修改原始答案或评分。"""

from collections import Counter
from itertools import groupby
import re
import unicodedata


_CHARACTERS = {
    " ": "空格", "\n": "换行", "\r": "回车", "\t": "制表符",
    "\u00a0": "不换行空格", "\u200b": "零宽空格", "\u202f": "窄不换行空格",
    "\u3000": "全角空格", "\ufffd": "替换字符",
}


def visible_text(value):
    """显示原文中的换行、特殊空格、控制字符及连续空格；不影响评分。"""
    result = []
    whitespace_only = bool(value) and not value.strip()
    for character, group in groupby(value):
        count = sum(1 for _ in group)
        category = unicodedata.category(character)
        marked = character.isspace() or category.startswith("C") or character == "\ufffd"
        if character == " " and count == 1 and not whitespace_only:
            marked = False
        if marked:
            label = _CHARACTERS.get(character, unicodedata.name(character, "异常字符"))
            repeat = f" ×{count}" if count > 1 else ""
            result.append(f"⟦{label} U+{ord(character):04X}{repeat}⟧")
        else:
            result.append(character * count)
    return "".join(result) if value else "⟦空字符串⟧"


def output_flags(record):
    prediction = record["prediction"]
    return {
        "blank": not prediction.strip(),
        "whitespace_repeat": re.search(r"\s{12,}", prediction) is not None,
        "abnormal_character": any(
            character == "\ufffd"
            or (unicodedata.category(character).startswith("C") and not character.isspace())
            for character in prediction
        ),
        "budget_exhausted": record["stop_reason"] == "max_new_tokens",
    }


def summarize_outputs(records):
    counts = Counter()
    for record in records:
        counts.update({key: int(value) for key, value in output_flags(record).items()})
    return dict(counts)
