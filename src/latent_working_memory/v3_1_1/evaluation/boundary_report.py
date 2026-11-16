"""同一 dev 的边界干预配对统计与本地结果看板。"""

import argparse
import csv
from html import escape
import json
from pathlib import Path

import numpy as np

from ..io import write_json
from .diagnostics import summarize_outputs
from .records import read_details


CONDITIONS = ("policy", "fixed-1", "fixed-2", "fixed-4", "fixed-8")
PAIRS = (
    ("policy", "fixed-1"),
    ("fixed-1", "fixed-2"),
    ("fixed-1", "fixed-4"),
    ("fixed-1", "fixed-8"),
    ("fixed-2", "fixed-4"),
    ("fixed-4", "fixed-8"),
)


def _document_values(records):
    return np.asarray([
        [
            np.mean([answer["nll"] for answer in row["qas"]]),
            sum(answer["em"] for answer in row["qas"]),
            sum(answer["f1"] for answer in row["qas"]),
            len(row["qas"]), row["blocks"], row["memory_slots"],
        ]
        for row in records
    ], dtype=np.float64)


def _paired_records(directory, plan):
    records = {}
    expected_ids = None
    expected_questions = None
    for condition in CONDITIONS:
        root = directory / condition
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        loaded = read_details(root / "details.jsonl")
        indexed = {row["id"]: row for row in loaded}
        if len(indexed) != len(loaded) or len(loaded) != plan["documents"]:
            raise ValueError(f"{condition}: incomplete or duplicate documents")
        if summary["metrics"]["documents"] != len(loaded):
            raise ValueError(f"{condition}: summary and detail document counts differ")
        if expected_ids is None:
            expected_ids = list(indexed)
        if set(indexed) != set(expected_ids):
            raise ValueError(f"{condition}: paired document identities differ")
        ordered = [indexed[identifier] for identifier in expected_ids]
        identities = [
            (row["id"], row["document_id"], row["context_tokens"], tuple(
                (answer["qa_index"], answer["question"], answer["reference"])
                for answer in row["qas"]
            )) for row in ordered
        ]
        if expected_questions is None:
            expected_questions = identities
        elif identities != expected_questions:
            raise ValueError(f"{condition}: paired questions or references differ")
        questions = sum(len(row["qas"]) for row in ordered)
        if questions != plan["questions"]:
            raise ValueError(f"{condition}: incomplete questions")
        records[condition] = ordered
    return records


def _condition_summary(condition, records, values, slot_cost):
    means = values.mean(axis=0)
    questions = int(values[:, 3].sum())
    return {
        "condition": condition,
        "documents": len(records), "questions": questions,
        "qa_nll": float(means[0]),
        "em": float(values[:, 1].sum() / questions),
        "f1": float(values[:, 2].sum() / questions),
        "blocks": float(means[4]), "memory_slots": float(means[5]),
        "memory_cost": float(slot_cost * means[5]),
        "reward": float(-means[0] - slot_cost * means[5]),
        "outputs": summarize_outputs(records),
    }


def _estimate(value, samples):
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"mean": float(value), "ci95": [float(lower), float(upper)]}


def _compare(baseline, candidate, arrays, indices, slot_cost):
    base, other = arrays[baseline], arrays[candidate]
    nll_gain = base[:, 0] - other[:, 0]
    em_gain, f1_gain = other[:, 1] - base[:, 1], other[:, 2] - base[:, 2]
    extra_blocks, extra_slots = other[:, 4] - base[:, 4], other[:, 5] - base[:, 5]
    reward_gain = nll_gain - slot_cost * extra_slots
    questions = base[:, 3]
    bootstrap_questions = questions[indices].sum(axis=1)
    boot_nll = nll_gain[indices].mean(axis=1)
    boot_blocks = extra_blocks[indices].mean(axis=1)
    boot_slots = extra_slots[indices].mean(axis=1)
    result = {
        "baseline": baseline, "candidate": candidate,
        "nll_gain": _estimate(nll_gain.mean(), boot_nll),
        "em_gain": _estimate(em_gain.sum() / questions.sum(),
                             em_gain[indices].sum(axis=1) / bootstrap_questions),
        "f1_gain": _estimate(f1_gain.sum() / questions.sum(),
                             f1_gain[indices].sum(axis=1) / bootstrap_questions),
        "extra_blocks": _estimate(extra_blocks.mean(), boot_blocks),
        "extra_slots": _estimate(extra_slots.mean(), boot_slots),
        "reward_gain": _estimate(reward_gain.mean(), reward_gain[indices].mean(axis=1)),
        "nll_gain_per_extra_block": None,
        "lambda_break_even": None,
        "documents_nll_better": int((nll_gain > 0).sum()),
        "documents_nll_worse": int((nll_gain < 0).sum()),
        "documents_nll_equal": int((nll_gain == 0).sum()),
        "documents_reward_better": int((reward_gain > 0).sum()),
        "documents_reward_worse": int((reward_gain < 0).sum()),
        "documents_reward_equal": int((reward_gain == 0).sum()),
        "document_reward_win_rate": float((reward_gain > 0).mean()),
    }
    # 同成本对照（例如 policy 与 fixed-1）不定义单位新增槽位的收益。
    if extra_blocks.mean() > 0 and np.all(boot_blocks > 0):
        result["nll_gain_per_extra_block"] = _estimate(
            nll_gain.mean() / extra_blocks.mean(), boot_nll / boot_blocks,
        )
    if extra_slots.mean() > 0 and np.all(boot_slots > 0):
        result["lambda_break_even"] = _estimate(
            nll_gain.mean() / extra_slots.mean(), boot_nll / boot_slots,
        )
    return result


def _length_strata(records, arrays, plan):
    ordered = sorted(range(plan["documents"]),
                     key=lambda index: (records["policy"][index]["context_tokens"],
                                        records["policy"][index]["id"]))
    strata = []
    for number, selected in enumerate(np.array_split(ordered, 3), 1):
        grouped = {condition: values[selected] for condition, values in arrays.items()}
        local = {condition: [rows[index] for index in selected]
                 for condition, rows in records.items()}
        comparisons = []
        # 分层仅给出点估计，不用很少的文档重复构造区间。
        local_indices = np.arange(len(selected))[None, :]
        for base, other in PAIRS:
            comparison = _compare(base, other, grouped, local_indices, plan["slot_cost"])
            comparisons.append({key: value["mean"] if isinstance(value, dict) else value
                                for key, value in comparison.items()})
        lengths = [row["context_tokens"] for row in local["policy"]]
        strata.append({
            "stratum": number, "documents": len(selected),
            "context_tokens_min": min(lengths), "context_tokens_max": max(lengths),
            "conditions": [_condition_summary(condition, local[condition], grouped[condition],
                                               plan["slot_cost"]) for condition in CONDITIONS],
            "comparisons": comparisons,
        })
    return strata


def _policy_single_agreement(records):
    policy, single = records["policy"], records["fixed-1"]
    questions = [(left, right) for base, other in zip(policy, single, strict=True)
                 for left, right in zip(base["qas"], other["qas"], strict=True)]
    return {
        "identical_action_documents": sum(base["actions"] == other["actions"]
                                          for base, other in zip(policy, single, strict=True)),
        "identical_block_length_documents": sum(base["block_lengths"] == other["block_lengths"]
                                                for base, other in zip(policy, single, strict=True)),
        "identical_prediction_questions": sum(left["prediction"] == right["prediction"]
                                              for left, right in questions),
        "max_absolute_question_nll_difference": max(abs(left["nll"] - right["nll"])
                                                    for left, right in questions),
    }


def _csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _comparison_csv(comparisons):
    records = []
    for row in comparisons:
        record = {"baseline": row["baseline"], "candidate": row["candidate"]}
        for key, value in row.items():
            if key in record:
                continue
            if isinstance(value, dict):
                record.update({key: value["mean"], key + "_ci95_low": value["ci95"][0],
                               key + "_ci95_high": value["ci95"][1]})
            elif value is None:
                record.update({key: None, key + "_ci95_low": None, key + "_ci95_high": None})
            else:
                record[key] = value
        records.append(record)
    return records


def _tradeoff_svg(summaries, metric, title, scale=1):
    colors = dict(zip(CONDITIONS, ("#7c3aed", "#2563eb", "#0d9488", "#d97706", "#dc2626"), strict=True))
    width, height = 420, 245
    left, right, top, bottom = 55, 18, 25, 45
    xs = [row["memory_slots"] for row in summaries]
    ys = [row[metric] * scale for row in summaries]
    xmin, xmax = min(xs), max(xs)
    span = max(ys) - min(ys)
    margin = max(span * 0.15, 0.005 if scale == 1 else 0.5)
    ymin, ymax = min(ys) - margin, max(ys) + margin

    def point(x, y):
        return (left + (x - xmin) / max(1, xmax - xmin) * (width - left - right),
                top + (ymax - y) / (ymax - ymin) * (height - top - bottom))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    for value in np.linspace(ymin, ymax, 5):
        _, y = point(xmin, value)
        parts.append(f'<path d="M{left} {y:.2f}H{width-right}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end">{value:.3f}</text>')
    for value in sorted(set(xs)):
        x, _ = point(value, ymin)
        parts.append(f'<text x="{x:.2f}" y="{height-bottom+20}" text-anchor="middle">{value:.0f}</text>')
    fixed = sorted((row for row in summaries if row["condition"] != "policy"), key=lambda row: row["memory_slots"])
    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in
                    (point(row["memory_slots"], row[metric] * scale) for row in fixed))
    parts.append(f'<polyline points="{line}" fill="none" stroke="#94a3b8" stroke-width="2"/>')
    for row in summaries:
        x, y = point(row["memory_slots"], row[metric] * scale)
        condition = row["condition"]
        fill = "white" if condition == "policy" else colors[condition]
        radius = 7 if condition == "policy" else 4
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{fill}" stroke="{colors[condition]}" stroke-width="2"><title>{escape(condition)}: {row[metric] * scale:.5f}</title></circle>')
    parts.append(f'<text x="{width/2}" y="{height-6}" text-anchor="middle">平均 memory slots</text></svg>')
    return "".join(parts)


def _format_estimate(estimate, multiplier=1, digits=4):
    if estimate is None:
        return "—"
    point = estimate["mean"] * multiplier
    low, high = (value * multiplier for value in estimate["ci95"])
    return f"{point:.{digits}f}<small>[{low:.{digits}f}, {high:.{digits}f}]</small>"


def _dashboard(path, plan, summary, records):
    conditions = summary["conditions"]
    header = "<tr><th>条件</th><th>QA NLL ↓</th><th>EM ↑</th><th>F1 ↑</th><th>块数</th><th>槽位数</th><th>记忆成本 ↓</th><th>reward ↑</th></tr>"
    condition_rows = []
    for row in conditions:
        condition_rows.append(
            f'<tr><th>{escape(row["condition"])}</th><td>{row["qa_nll"]:.4f}</td>'
            f'<td>{row["em"]:.2%}</td><td>{row["f1"]:.2%}</td><td>{row["blocks"]:.3f}</td>'
            f'<td>{row["memory_slots"]:.1f}</td><td>{row["memory_cost"]:.4f}</td><td>{row["reward"]:.4f}</td></tr>'
        )
    comparison_rows = []
    for row in summary["comparisons"]:
        cells = [_format_estimate(row[name], multiplier, digits) for name, multiplier, digits in (
            ("nll_gain", 1, 4), ("em_gain", 100, 2), ("f1_gain", 100, 2),
            ("nll_gain_per_extra_block", 1, 4), ("lambda_break_even", 1, 6), ("reward_gain", 1, 4),
        )]
        comparison_rows.append(
            f'<tr><th>{escape(row["baseline"])} → {escape(row["candidate"])}</th>'
            + "".join(f"<td>{cell}</td>" for cell in cells)
            + f'<td>{row["documents_nll_better"]}/{row["documents_nll_worse"]}/{row["documents_nll_equal"]}</td>'
            + f'<td>{row["document_reward_win_rate"]:.1%}</td></tr>'
        )
    strata_rows = []
    for stratum in summary["length_strata"]:
        for row in stratum["conditions"]:
            pair = next((pair for pair in stratum["comparisons"]
                         if pair["baseline"] == "fixed-1" and pair["candidate"] == row["condition"]), None)
            threshold = "—" if pair is None else f'{pair["lambda_break_even"]:.6f}'
            strata_rows.append(
                f'<tr><th>第 {stratum["stratum"]} 组 · {stratum["context_tokens_min"]}–{stratum["context_tokens_max"]} tokens</th>'
                f'<td>{stratum["documents"]}</td><td>{row["condition"]}</td><td>{row["qa_nll"]:.4f}</td>'
                f'<td>{row["em"]:.2%}</td><td>{row["f1"]:.2%}</td><td>{threshold}</td></tr>'
            )
    examples = []
    for index, row in enumerate(records["policy"]):
        examples.append({
            "id": row["id"], "context_tokens": row["context_tokens"],
            "conditions": {condition: {
                "blocks": records[condition][index]["blocks"],
                "block_lengths": records[condition][index]["block_lengths"],
                "qas": [{key: answer[key] for key in (
                    "qa_index", "question", "reference", "prediction", "nll", "em", "f1", "stop_reason",
                )} for answer in records[condition][index]["qas"]],
            } for condition in CONDITIONS},
        })
    data = json.dumps(examples, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    charts = "".join(
        f'<section class="chart"><h3>{title}</h3>{_tradeoff_svg(conditions, metric, title, scale)}</section>'
        for metric, title, scale in (("qa_nll", "QA NLL 与记忆用量", 1),
                                    ("em", "生成 EM（%）与记忆用量", 100),
                                    ("reward", "当前成本下的 reward", 1))
    )
    page = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>边界分块与记忆成本对照</title>
<style>
:root{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;color:#172033;background:#f5f7fb}body{max-width:1440px;margin:auto;padding:28px}h1{font-size:27px}h2{margin-top:32px;font-size:21px}p{line-height:1.7}code{background:#e8edf5;padding:2px 5px;border-radius:4px}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid #e2e8f0;padding:11px;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}thead{background:#edf1f7}small{display:block;color:#64748b;margin-top:5px;white-space:nowrap}.scroll{overflow:auto;border:1px solid #e2e8f0;border-radius:10px}.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.chart{background:white;padding:12px;border:1px solid #e2e8f0;border-radius:10px}.chart h3{font-size:16px;margin:8px}.chart svg{width:100%;font-size:11px;fill:#475569}.note{background:#e9f0fc;border-left:4px solid #2563eb;padding:12px 16px}.legend{display:flex;gap:20px;flex-wrap:wrap}.legend span:before{content:'●';color:var(--color);margin-right:6px}button,select,input{font:inherit;border:1px solid #cbd5e1;background:white;border-radius:6px;padding:8px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:15px 0}.qa{background:white;border:1px solid #e2e8f0;border-radius:8px;padding:15px;margin:16px 0}.qa h3{font-size:16px}.qa td:last-child{text-align:left;white-space:pre-wrap;overflow-wrap:anywhere;min-width:220px}.qa td{font-size:14px}.meta{color:#64748b;font-size:14px;overflow-wrap:anywhere}@media(max-width:900px){body{padding:14px}.charts{grid-template-columns:1fr}th,td{padding:8px}}
</style>
<h1>最后一个 checkpoint 的边界分块与记忆成本对照</h1>
<p class="meta">__IDENTITY__</p>
<p>所有条件使用同一批 dev、同一个 checkpoint、相同问题与生成设置。固定 1／2／4／8 块在原始边界决策网格上均匀划分；policy 使用当前确定性边界。QA NLL 先对每题的目标 token 平均，再对题目与文档平均。EM 为规范化文本完全匹配，F1 为 token overlap；它们不等同于人工语义正确率。</p>
<div class="note">当前 slot_cost = __COST__。每增加一块（__SLOTS__ 个槽位），需要平均 QA NLL 降低超过 __BLOCK_COST__ 才能提高训练奖励。配对表中的收益均以正数表示候选条件更好；EM／F1 的单位为百分点。</div>
<h2>条件汇总</h2><div class="scroll"><table><thead>__HEADER__</thead><tbody>__CONDITIONS__</tbody></table></div>
<p class="legend"><span style="--color:#7c3aed">policy</span><span style="--color:#2563eb">fixed-1</span><span style="--color:#0d9488">fixed-2</span><span style="--color:#d97706">fixed-4</span><span style="--color:#dc2626">fixed-8</span></p>
<div class="charts">__CHARTS__</div>
<h2>配对收益与成本临界值</h2>
<p>方括号为按文档整体重采样的 95% bootstrap 区间（__BOOTSTRAP__ 次，seed=__SEED__），同篇内各题始终一起抽取。NLL 收益 = 基线 − 候选；reward 收益 = NLL 收益 − 当前 slot_cost × 新增槽位。临界 slot_cost = NLL 收益／新增槽位；高于此值时，平均奖励不支持该额外记忆预算。同成本条件不定义临界值。</p>
<div class="scroll"><table><thead><tr><th>基线 → 候选</th><th>NLL 收益</th><th>EM 收益（pp）</th><th>F1 收益（pp）</th><th>每新增块 NLL 收益</th><th>临界 slot_cost</th><th>reward 收益</th><th>NLL 更好/更差/相同文档</th><th>reward 更好文档</th></tr></thead><tbody>__COMPARISONS__</tbody></table></div>
<p class="meta">区间描述固定模型在这批 dev 上的文档抽样不确定性，不包含训练 seed 或模型选择的不确定性；多组比较没有多重检验校正。此实验冻结最终 checkpoint，仅在推理时干预分段：已适应单块的模型可能低估重新训练多块的潜力，不能直接推断调整 slot_cost 后的训练收益。若临界 slot_cost ≤ 0，表示额外块的平均 NLL 没有改善，不能据此建议更低的正成本；只有 NLL 收益为正但小于记忆成本时，才是当前成本抵消了这部分收益。</p>
<p class="meta">policy 与 fixed-1：__AGREEMENT__</p>
<h2>按输入长度分层</h2><p>按含提示前缀的压缩输入 token 数排序后，将同一批文档分成三个近等大组；同长度按文档 ID 排序。这些组不是训练课程档位。下表仅给出点估计，临界 slot_cost 以同组 fixed-1 为基线。</p>
<div class="scroll"><table><thead><tr><th>长度分组</th><th>文档数</th><th>条件</th><th>QA NLL</th><th>EM</th><th>F1</th><th>临界 slot_cost</th></tr></thead><tbody>__STRATA__</tbody></table></div>
<h2>同一输入的生成结果</h2>
<div class="controls"><button id="previous">上一篇</button><select id="document"></select><button id="next">下一篇</button><input id="search" placeholder="按问题、答案或文档 ID 搜索"><label><input id="differences" type="checkbox">只看条件间输出不同的题</label></div>
<p id="document-meta" class="meta"></p><div id="examples"></div>
<script id="results" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('results').textContent);
const names=['policy','fixed-1','fixed-2','fixed-4','fixed-8'];
const selector=document.getElementById('document'),search=document.getElementById('search'),differences=document.getElementById('differences');
function node(tag,text){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;return el;}
data.forEach((row,index)=>{const option=node('option',`${index+1}. ${row.id.slice(0,18)}…`);option.value=index;selector.append(option);});
function render(){const row=data[Number(selector.value)],query=search.value.trim().toLowerCase();
document.getElementById('document-meta').textContent=`${Number(selector.value)+1}/${data.length} · ${row.id} · 输入 ${row.context_tokens} tokens · `+names.map(name=>`${name}: ${row.conditions[name].blocks} 块 [${row.conditions[name].block_lengths.join(', ')}]`).join('；');
const root=document.getElementById('examples');root.replaceChildren();
row.conditions.policy.qas.forEach((qa,index)=>{const answers=names.map(name=>row.conditions[name].qas[index]);
if(differences.checked&&new Set(answers.map(answer=>answer.prediction)).size===1)return;
if(query&&!JSON.stringify([row.id,qa.question,qa.reference,...answers.map(a=>a.prediction)]).toLowerCase().includes(query))return;
const card=node('section');card.className='qa';card.append(node('h3',`题 ${index+1}：${qa.question}`),node('p',`参考答案：${qa.reference}`));
const wrap=node('div');wrap.className='scroll';const table=node('table'),head=node('tr');['条件','NLL','EM','F1','结束原因','生成答案'].forEach(text=>head.append(node('th',text)));table.append(head);
answers.forEach((answer,i)=>{const tr=node('tr');[names[i],answer.nll.toFixed(4),(100*answer.em).toFixed(1)+'%',(100*answer.f1).toFixed(1)+'%',answer.stop_reason,answer.prediction||'⟦空字符串⟧'].forEach(text=>tr.append(node('td',text)));table.append(tr);});wrap.append(table);card.append(wrap);root.append(card);});
if(!root.children.length)root.append(node('p','当前筛选条件下没有题目。'));}
selector.onchange=render;search.oninput=render;differences.onchange=render;
document.getElementById('previous').onclick=()=>{selector.value=Math.max(0,Number(selector.value)-1);render();};
document.getElementById('next').onclick=()=>{selector.value=Math.min(data.length-1,Number(selector.value)+1);render();};render();
</script></html>"""
    substitutions = {
        "__IDENTITY__": escape(f"checkpoint step {plan['checkpoint_step']} · {plan['documents']} 篇 / {plan['questions']} 题 · {plan['checkpoint']}"),
        "__COST__": str(plan["slot_cost"]), "__SLOTS__": str(plan["memory_slots"]),
        "__BLOCK_COST__": f"{plan['slot_cost'] * plan['memory_slots']:.4f}",
        "__HEADER__": header, "__CONDITIONS__": "".join(condition_rows), "__CHARTS__": charts,
        "__BOOTSTRAP__": str(summary["bootstrap"]["resamples"]),
        "__SEED__": str(summary["bootstrap"]["seed"]),
        "__COMPARISONS__": "".join(comparison_rows),
        "__STRATA__": "".join(strata_rows),
        "__AGREEMENT__": escape(
            f"动作完全相同 {summary['policy_single_agreement']['identical_action_documents']}/{plan['documents']} 篇；"
            f"生成文本完全相同 {summary['policy_single_agreement']['identical_prediction_questions']}/{plan['questions']} 题；"
            f"逐题 NLL 最大绝对差 {summary['policy_single_agreement']['max_absolute_question_nll_difference']:.8g}。"
        ),
    }
    for key, value in substitutions.items():
        page = page.replace(key, value)
    path.write_text(page.replace("__DATA__", data), encoding="utf-8")


def build_report(output_dir, bootstrap_samples=5000, seed=42):
    directory = Path(output_dir).expanduser().resolve()
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if tuple(plan["conditions"]) != CONDITIONS:
        raise ValueError("boundary report requires policy and fixed 1/2/4/8-block conditions")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    records = _paired_records(directory, plan)
    arrays = {condition: _document_values(rows) for condition, rows in records.items()}
    indices = np.random.default_rng(seed).integers(0, plan["documents"],
                                                size=(bootstrap_samples, plan["documents"]))
    conditions = [_condition_summary(condition, records[condition], arrays[condition], plan["slot_cost"])
                  for condition in CONDITIONS]
    comparisons = [_compare(base, other, arrays, indices, plan["slot_cost"]) for base, other in PAIRS]
    summary = {
        "checkpoint": plan["checkpoint"], "checkpoint_step": plan["checkpoint_step"],
        "documents": plan["documents"], "questions": plan["questions"],
        "slot_cost": plan["slot_cost"], "memory_slots_per_block": plan["memory_slots"],
        "bootstrap": {"unit": "document cluster; all QA retained together", "paired": True,
                      "resamples": bootstrap_samples, "seed": seed, "interval": "percentile 95%",
                      "multiple_comparison_correction": False},
        "definitions": {
            "qa_nll": "token mean within QA, QA mean within document, then document mean",
            "em_f1": "question mean; bootstrap resamples documents then pools their questions",
            "nll_gain": "baseline NLL minus candidate NLL; positive is better",
            "em_f1_gain": "candidate minus baseline; stored as fraction, displayed in percentage points",
            "reward_gain": "nll_gain minus slot_cost times extra_slots; positive is better",
            "lambda_break_even": "nll_gain divided by extra_slots; null when no strictly positive added budget",
        },
        "conditions": conditions, "comparisons": comparisons,
        "policy_single_agreement": _policy_single_agreement(records),
        "length_strata_definition": "Three near-equal document groups sorted by compressed input tokens including prefix, then document ID; not training curriculum bins; point estimates only.",
        "length_strata": _length_strata(records, arrays, plan),
    }
    destination = directory / "comparison"
    destination.mkdir(exist_ok=True)
    write_json(destination / "summary.json", summary)
    write_json(destination / "comparisons.json", comparisons)
    _csv(destination / "conditions.csv", [{key: value for key, value in row.items() if key != "outputs"}
                                          for row in conditions])
    _csv(destination / "comparisons.csv", _comparison_csv(comparisons))
    _dashboard(destination / "dashboard.html", plan, summary, records)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="包含 plan.json 与全部条件结果的目录")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    summary = build_report(args.output_dir, args.bootstrap_samples, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
