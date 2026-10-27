"""从保存的评估结果生成可离线打开的 HTML，明确显示不可见输出。"""

import argparse
from collections import Counter
from html import escape
from itertools import groupby
import json
from pathlib import Path
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


def build_report(results_dir, output=None):
    results_dir = Path(results_dir).expanduser().resolve()
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (results_dir / "details.jsonl").read_text().splitlines()]
    output = Path(output).expanduser().resolve() if output else results_dir / "report" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    counters = summarize_outputs(records)
    metrics = summary["metrics"]
    cards = "".join(
        f"<div><strong>{escape(key)}</strong><span>{value:.4f}</span></div>"
        for key, value in metrics.items() if isinstance(value, (int, float))
    )
    labels = {
        "blank": "空白答案", "whitespace_repeat": "连续空白 ≥12 字符",
        "abnormal_character": "异常字符输出", "budget_exhausted": "生成触顶",
    }
    diagnostics = " · ".join(f"{labels[key]}：{counters[key]}" for key in labels)
    examples = []
    for record in records:
        flags = output_flags(record)
        badges = " · ".join(labels[key] for key, value in flags.items() if value)
        quality = (
            f"EM {record['em']:.3f} · F1 {record['f1']:.3f}"
            if "em" in record else f"{metrics['metric']} {record['score']:.3f}"
        )
        question = record.get("question", "")
        references = record["references"]
        examples.append(
            "<article>"
            f"<h3>{escape(record['id'])}</h3>"
            f"<p class='meta'>{quality} · {record['prompt_tokens']} prompt tokens · "
            f"{record['generated_tokens']}/{record['max_new_tokens']} generated tokens · "
            f"{escape(record['stop_reason'])}</p>"
            + (f"<p>{escape(question)}</p>" if question else "")
            + f"<p><b>参考答案</b> {escape(' / '.join(references))}</p>"
            + (f"<p class='warning'>{escape(badges)}</p>" if badges else "")
            + f"<pre>{escape(visible_text(record['prediction']))}</pre>"
            + "<details><summary>原始解码与 token IDs</summary>"
            + f"<pre>{escape(record['raw_prediction'])}</pre>"
            + f"<code>{escape(str(record['generated_token_ids']))}</code></details></article>"
        )
    content = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>v4.1 评估报告</title><style>"
        "body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;"
        "color:#172337;background:#f6f8fb}h1,h3{line-height:1.3}.cards{display:flex;flex-wrap:wrap;gap:12px}"
        ".cards div,article{background:white;border:1px solid #dbe3ee;border-radius:10px;padding:18px}"
        ".cards span{display:block;font-size:25px}article{margin:16px 0}.meta{font-size:13px;color:#52657b}"
        "pre,code{white-space:pre-wrap;overflow-wrap:anywhere}pre{background:#eff3f9;padding:14px;"
        "border-radius:6px}.warning{color:#9b3c13}summary{cursor:pointer}</style>"
        f"<h1>v4.1 · {escape(summary['task'])} 评估</h1>"
        f"<p class='meta'>{escape(str(summary['checkpoint']))}</p><div class='cards'>{cards}</div>"
        f"<p>{escape(diagnostics)}</p>"
        "<p class='meta'>标记只改变展示；评分与保存的答案保留原文。异常字符统计包含替换字符、"
        "非空白控制／格式字符及未分配字符。连续空白按至少 12 个字符统计。</p>"
        + "".join(examples) + "</html>"
    )
    output.write_text(content, encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="含 summary.json 和 details.jsonl 的目录")
    parser.add_argument("--output", help="HTML 文件；默认 results/report/index.html")
    args = parser.parse_args(argv)
    print(build_report(args.results, args.output))


if __name__ == "__main__":
    main()
