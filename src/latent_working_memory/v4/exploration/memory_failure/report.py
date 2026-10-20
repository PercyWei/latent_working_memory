"""从固定子集的逐题记录重建离线 HTML、CSV 和诊断汇总。"""

import argparse
import csv
import fcntl
from html import escape
import json
from pathlib import Path
import re

from latent_working_memory.benchmarks.data import LOCOMO_CATEGORIES
from latent_working_memory.benchmarks.scoring import summarize_scores
from latent_working_memory.v4.exploration.memory_failure.constants import (
    COLORS,
    CONDITIONS,
    DESCRIPTIONS,
    LABELS,
)
from latent_working_memory.v4.io import write_json


BEHAVIORS = {
    "blank": "空白",
    "immediate_eos": "立即 EOS",
    "budget_exhausted": "生成触顶",
    "no_information": "无信息回答片段",
    "prompt_copy": "指令片段复制",
    "long_zero_repeat": "连续零复读",
}
PAIRS = (
    ("full-context", "no-read"),
    ("ttt", "no-read"),
    ("ttt", "fixed-initial"),
    ("fixed-initial", "no-read"),
)
BEHAVIOR_NOTE = (
    "空白：去除首尾空白后为空；立即 EOS：唯一生成 token 为 EOS；"
    "生成触顶：finish_reason 为 max_new_tokens。"
    "无信息回答片段：输出包含 “no information available”，忽略大小写；"
    "指令片段复制：输出包含 “Based on the above conversations”，忽略大小写；"
    "连续零复读：出现至少 12 个严格相邻的 0。"
    "各标记可重叠，只用于定位输出行为，不据此剔除样本或修改 F1。"
)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def behavior_flags(record, eos_token_id):
    """仅使用保存的原始输出和 token，不依赖模型或模糊语义判断。"""
    prediction = record["prediction"]
    return {
        "blank": not prediction.strip(),
        "immediate_eos": record["generated_token_ids"] == [eos_token_id],
        "budget_exhausted": record["finish_reason"] == "max_new_tokens",
        "no_information": "no information available" in prediction.casefold(),
        "prompt_copy": "based on the above conversations" in prediction.casefold(),
        "long_zero_repeat": re.search(r"0{12}", prediction) is not None,
    }


def _load_results(root, plan):
    records, sources = {}, {}
    sample_ids = plan["selection"]["sample_ids"]
    reference_rows = None
    for condition in CONDITIONS:
        directory = root / condition
        if not (directory / "summary.json").exists():
            continue
        rows = [
            json.loads(line)
            for line in (directory / "details.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if [row["id"] for row in rows] != sample_ids:
            raise ValueError(f"{condition}: completed records do not match the selected sample IDs")
        if reference_rows is not None and any(
            row["references"] != reference["references"]
            or row["prompt_tokens"] != reference["prompt_tokens"]
            or row["metadata"] != reference["metadata"]
            for row, reference in zip(rows, reference_rows)
        ):
            raise ValueError(f"{condition}: paired references, metadata or prompt lengths differ")
        reference_rows = rows
        records[condition] = rows
        sources[condition] = _read_json(directory / "summary.json")["source"]
    return records, sources


def _aggregate(plan, records, sources):
    conditions = {}
    for condition, rows in records.items():
        flags = [behavior_flags(row, plan["eos_token_id"]) for row in rows]
        conditions[condition] = {
            "samples": len(rows),
            "metrics": summarize_scores("locomo", rows),
            "behaviors": {
                name: {
                    "count": sum(row[name] for row in flags),
                    "rate": sum(row[name] for row in flags) / len(rows),
                }
                for name in BEHAVIORS
            },
            "mean_generated_tokens": sum(row["generated_tokens"] for row in rows) / len(rows),
            "source": sources[condition],
        }
    paired = []
    for left, right in PAIRS:
        if left not in records or right not in records:
            continue
        differences = [
            first["score"] - second["score"]
            for first, second in zip(records[left], records[right])
        ]
        paired.append({
            "left": left,
            "right": right,
            "samples": len(differences),
            "mean_f1_difference": sum(differences) / len(differences),
            "wins": sum(value > 1e-12 for value in differences),
            "ties": sum(abs(value) <= 1e-12 for value in differences),
            "losses": sum(value < -1e-12 for value in differences),
        })
    return {
        "checkpoint": plan["checkpoint"],
        "selected_samples": len(plan["selection"]["sample_ids"]),
        "completed_conditions": list(records),
        "missing_conditions": [condition for condition in CONDITIONS if condition not in records],
        "conditions": conditions,
        "paired": paired,
        "behavior_definitions": BEHAVIOR_NOTE,
    }


def _comparison_csv(path, summary):
    columns = [
        "condition", "label", "status", "samples", "f1", "mean_generated_tokens",
        *[f"category_{category}_f1" for category in LOCOMO_CATEGORIES],
        *[f"category_{category}_samples" for category in LOCOMO_CATEGORIES],
        *[f"{behavior}_{field}" for behavior in BEHAVIORS for field in ("count", "rate")],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for condition in CONDITIONS:
            row = {"condition": condition, "label": LABELS[condition], "status": "missing"}
            if condition in summary["conditions"]:
                result = summary["conditions"][condition]
                row.update(
                    status="complete", samples=result["samples"], f1=result["metrics"]["score"],
                    mean_generated_tokens=result["mean_generated_tokens"],
                )
                for category in result["metrics"]["groups"]:
                    row[f"category_{category['category']}_f1"] = category["score"]
                    row[f"category_{category['category']}_samples"] = category["samples"]
                for name, values in result["behaviors"].items():
                    row.update({f"{name}_{key}": value for key, value in values.items()})
            writer.writerow(row)


def _details_csv(path, plan, records):
    result_fields = (
        "prediction", "score", "generated_tokens", "generation_budget", "finish_reason",
        *BEHAVIORS,
    )
    columns = [
        "id", "conversation_id", "category", "question", "references", "prompt_tokens",
        *[f"{condition}/{name}" for condition in CONDITIONS for name in result_fields],
        *[f"{left}_minus_{right}/f1" for left, right in PAIRS],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, sample_id in enumerate(plan["selection"]["sample_ids"]):
            row = {"id": sample_id}
            if records:
                reference = next(iter(records.values()))[index]
                row.update(
                    conversation_id=reference["metadata"]["conversation_id"],
                    category=reference["metadata"]["category"],
                    question=reference["metadata"]["question"],
                    references=json.dumps(reference["references"], ensure_ascii=False),
                    prompt_tokens=reference["prompt_tokens"],
                )
                for condition, values in records.items():
                    result = values[index]
                    fields = {
                        **result, **behavior_flags(result, plan["eos_token_id"]),
                    }
                    row.update({f"{condition}/{name}": fields[name] for name in result_fields})
                for left, right in PAIRS:
                    if left in records and right in records:
                        row[f"{left}_minus_{right}/f1"] = (
                            records[left][index]["score"] - records[right][index]["score"]
                        )
            writer.writerow(row)


def _table(headers, rows):
    head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _label(condition):
    return (
        f"<span class='dot' style='background:{COLORS[condition]}'></span>"
        f"{escape(LABELS[condition])}"
    )


def _summary_tables(plan, summary):
    rows = []
    for condition in CONDITIONS:
        result = summary["conditions"].get(condition)
        rows.append([
            _label(condition), escape(DESCRIPTIONS[condition]),
            str(result["samples"]) if result else "待完成",
            f"{100 * result['metrics']['score']:.2f}%" if result else "—",
            f"{result['mean_generated_tokens']:.2f}" if result else "—",
        ])
    content = "<h2>同一固定子集上的总体结果</h2>" + _table(
        ("条件", "设置", "题数", "F1", "平均生成 tokens"), rows,
    )
    paired_rows = [
        [
            f"{_label(row['left'])} − {_label(row['right'])}",
            f"{100 * row['mean_f1_difference']:+.2f}",
            str(row["wins"]), str(row["ties"]), str(row["losses"]),
        ]
        for row in summary["paired"]
    ]
    content += "<h2>逐题配对差值</h2><p>差值为左侧减右侧，单位为百分点；胜／平／负按逐题 F1 比较。</p>"
    content += _table(("比较", "平均 F1 差", "胜", "平", "负"), paired_rows)
    category_rows = []
    for category, name in LOCOMO_CATEGORIES.items():
        cells = [escape(name)]
        for condition in CONDITIONS:
            result = summary["conditions"].get(condition)
            group = next(
                (row for row in result["metrics"]["groups"] if row["category"] == category), None,
            ) if result else None
            cells.append(
                f"{100 * group['score']:.2f}% <small>（{group['samples']} 题）</small>"
                if group else "—"
            )
        category_rows.append(cells)
    content += "<h2>分类 F1</h2>" + _table(
        ("类别", *[LABELS[name] for name in CONDITIONS]), category_rows,
    )
    behavior_rows = []
    for name, label in BEHAVIORS.items():
        cells = [escape(label)]
        for condition in CONDITIONS:
            result = summary["conditions"].get(condition)
            value = result["behaviors"][name] if result else None
            cells.append(f"{value['count']}（{100 * value['rate']:.1f}%）" if value else "—")
        behavior_rows.append(cells)
    content += "<h2>生成行为</h2>" + _table(
        ("标记", *[LABELS[name] for name in CONDITIONS]), behavior_rows,
    )
    content += f"<p>{escape(BEHAVIOR_NOTE)}</p>"
    content += "<h2>来源与抽样</h2>"
    source_rows = []
    for condition in CONDITIONS:
        result = summary["conditions"].get(condition)
        source_rows.append([
            _label(condition),
            escape(result["source"]["kind"]) if result else "待完成",
            escape(result["source"]["directory"] or "本次计算") if result else "—",
        ])
    content += _table(("条件", "来源类型", "来源目录"), source_rows)
    selection_rows = [
        [
            escape(row["conversation_id"]), str(row["available"]), str(row["selected"]),
            escape("；".join(
                f"{LOCOMO_CATEGORIES[item['category']]} {item['selected']}/{item['available']}"
                for item in row["categories"]
            )),
        ]
        for row in plan["selection"]["conversations"]
    ]
    content += _table(("对话", "可用题数", "入选题数", "分类：入选／可用"), selection_rows)
    content += (
        f"<p>抽样 seed：{plan['selection']['seed']}；"
        f"生成 seed：{plan['seed']}。F1 保留官方评分口径，按入选题平均；"
        "缺失条件显示为待完成，不补零。此子集用于机制诊断，不代替全量 LoCoMo 分数。</p>"
    )
    return content


def _f1_svg(summary):
    rows = []
    for index, condition in enumerate(CONDITIONS):
        y = 28 + index * 44
        result = summary["conditions"].get(condition)
        value = 100 * result["metrics"]["score"] if result else None
        rows.append(f"<text x='0' y='{y + 17}'>{escape(LABELS[condition])}</text>")
        rows.append(f"<rect x='180' y='{y}' width='500' height='24' fill='#eef0f3' rx='4'/>")
        if value is not None:
            rows.append(
                f"<rect x='180' y='{y}' width='{5 * value:.3f}' height='24' "
                f"fill='{COLORS[condition]}' rx='4'/>"
            )
        label = f"{value:.2f}%" if value is not None else "待完成"
        rows.append(f"<text x='695' y='{y + 17}'>{label}</text>")
    return (
        "<svg viewBox='0 0 790 220' role='img' aria-label='四条件总体 F1，横轴 0 到 100%'>"
        "<text x='180' y='16'>0%</text><text x='640' y='16'>100%</text>"
        + "".join(rows) + "</svg>"
    )


def _example_content(plan, records):
    if not records:
        return "<h2>逐题答案</h2><p>尚无完成条件。</p>"
    examples = []
    reference_rows = next(iter(records.values()))
    for index, reference in enumerate(reference_rows):
        metadata = reference["metadata"]
        cells, flags = [], set()
        for condition in CONDITIONS:
            if condition not in records:
                cells.append(f"<div class='answer missing'><h3>{_label(condition)}</h3>待完成</div>")
                continue
            row = records[condition][index]
            marked = [key for key, value in behavior_flags(row, plan["eos_token_id"]).items() if value]
            flags.update(marked)
            badges = "".join(f"<span class='badge'>{BEHAVIORS[key]}</span>" for key in marked)
            cells.append(
                f"<div class='answer' style='border-top-color:{COLORS[condition]}'>"
                f"<h3>{_label(condition)}</h3>"
                f"<p>F1 {100 * row['score']:.2f}% · "
                f"{row['generated_tokens']}/{row['generation_budget']} tokens · "
                f"{escape(row['finish_reason'])}</p>{badges}"
                f"<pre>{escape(row['prediction']) if row['prediction'] else '（空白输出）'}</pre></div>"
            )
        examples.append(
            f"<article class='example' data-category='{metadata['category']}' "
            f"data-flags='{escape(' '.join(sorted(flags)), quote=True)}'>"
            f"<p class='muted'>{escape(reference['id'])} · "
            f"{escape(metadata['category_name'])} · {reference['prompt_tokens']} prompt tokens</p>"
            f"<h3>{escape(metadata['question'])}</h3>"
            f"<p><strong>参考答案：</strong>{escape(' | '.join(reference['references']))}</p>"
            f"<div class='answers'>{''.join(cells)}</div></article>"
        )
    options = "".join(
        f"<option value='{category}'>{escape(name)}</option>"
        for category, name in LOCOMO_CATEGORIES.items()
    )
    behaviors = "".join(
        f"<option value='{key}'>{escape(label)}</option>" for key, label in BEHAVIORS.items()
    )
    return (
        "<h2>逐题并列答案</h2><div class='filters'>"
        "<label>搜索<input id='search' type='search' placeholder='问题、答案或样本 ID'></label>"
        f"<label>类别<select id='category'><option value=''>全部类别</option>{options}</select></label>"
        f"<label>行为<select id='behavior'><option value=''>全部输出</option>{behaviors}</select></label>"
        "<span id='shown'></span></div>"
        + "".join(examples)
        + """<script>
const search = document.getElementById('search');
const category = document.getElementById('category');
const behavior = document.getElementById('behavior');
const entries = Array.from(document.querySelectorAll('.example'));
const searchable = entries.map(entry => entry.textContent.toLowerCase());
function filterExamples() {
  const query = search.value.trim().toLowerCase();
  let count = 0;
  entries.forEach((entry, index) => {
    const visible = (!query || searchable[index].includes(query))
      && (!category.value || entry.dataset.category === category.value)
      && (!behavior.value || entry.dataset.flags.split(' ').includes(behavior.value));
    entry.hidden = !visible;
    if (visible) count += 1;
  });
  document.getElementById('shown').textContent = count + ' / ' + entries.length + ' 题';
}
[search, category, behavior].forEach(control => control.addEventListener('input', filterExamples));
filterExamples();
</script>"""
    )


def _document(title, content):
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>"
        "body{font:15px/1.55 system-ui,-apple-system,sans-serif;color:#243041;background:#f6f7f9;"
        "margin:0;padding:28px}main{max-width:1450px;margin:auto}h1{font-size:25px}"
        "h2{font-size:20px;margin:30px 0 12px}h3{font-size:15px;margin:8px 0}"
        "p{margin:9px 0}small,.muted{color:#697386}.scroll{overflow-x:auto}"
        "table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}"
        "th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e2e6eb}"
        "th{background:#eaf0f5;white-space:nowrap}td{vertical-align:top}"
        ".dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px}"
        ".answer{background:white;padding:12px;border:1px solid #e0e5eb;border-top:4px solid;"
        "border-radius:5px;min-width:0}.answer p{font-size:12px;color:#64748b}"
        ".answers{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}"
        ".example{margin:20px 0;padding:16px;background:#edf1f5;border-radius:8px}"
        "pre{font:14px/1.5 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}"
        ".badge{display:inline-block;padding:2px 6px;background:#fce5db;font-size:12px;"
        "border-radius:4px;margin:2px 4px 2px 0}.missing{color:#6b7280;border-top-color:#b6beca}"
        ".filters{display:flex;align-items:end;flex-wrap:wrap;gap:14px}"
        "label{display:flex;flex-direction:column;font-size:13px}"
        "input,select{padding:8px;border:1px solid #cbd5e1;border-radius:5px;font:inherit}"
        "input{min-width:260px}svg{width:100%;max-width:900px;font-size:14px}"
        "code{overflow-wrap:anywhere}a{color:#235aa5}details{margin:12px 0}"
        "@media(max-width:1000px){.answers{grid-template-columns:repeat(2,minmax(0,1fr))}}"
        "@media(max-width:560px){body{padding:14px}.answers{grid-template-columns:1fr}}"
        "</style></head><body><main>"
        f"<h1>{escape(title)}</h1>{content}</main></body></html>"
    )


def build_report(output_dir):
    """允许部分条件完成；只从 plan 和完成条件的原始记录重建报告。"""
    root = Path(output_dir)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    with (report / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = _read_json(root / "plan.json")
        records, sources = _load_results(root, plan)
        summary = _aggregate(plan, records, sources)
        write_json(report / "summary.json", summary)
        _comparison_csv(report / "comparison.csv", summary)
        _details_csv(report / "details.csv", plan, records)
        content = (
            f"<p>Checkpoint step {plan['checkpoint']['step']} · "
            f"{len(plan['selection']['sample_ids'])} 道固定题 · "
            f"{len(records)}/{len(CONDITIONS)} 个条件完成</p>"
            f"<p><code>{escape(plan['checkpoint']['path'])}</code></p>"
            "<p><a href='comparison.csv'>下载汇总 CSV</a> · "
            "<a href='details.csv'>下载逐题 CSV</a> · "
            "<a href='summary.json'>完整统计 JSON</a></p>"
            + _f1_svg(summary) + _summary_tables(plan, summary) + _example_content(plan, records)
        )
        (report / "index.html").write_text(
            _document("LoCoMo 记忆贡献诊断", content), encoding="utf-8",
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = build_report(args.output_dir)
    print(
        f"report: {args.output_dir / 'report' / 'index.html'} "
        f"({len(summary['completed_conditions'])}/{len(CONDITIONS)} conditions)"
    )


if __name__ == "__main__":
    main()
