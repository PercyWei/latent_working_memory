"""Summarize document records and write a compact, denominator-aware report."""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean
from zoneinfo import ZoneInfo

from .metrics import summarize


DOCUMENT_REPORT_METRICS = (
    ("doc_tokens", "文档 tokens"),
    ("unit_count", "自然单元数"),
    ("unit_tokens_p50", "篇内单元长度 P50"),
    ("entity_density_mean", "实体提及密度均值"),
    ("entity_density_cv", "实体提及密度 CV"),
    ("number_density_mean", "数字密度均值"),
    ("number_density_cv", "数字密度 CV"),
)
QA_REPORT_METRICS = (
    ("evidence_span_count", "证据区间数"),
    ("evidence_tokens", "证据 tokens"),
    ("evidence_extent_tokens", "证据首尾跨度 tokens"),
    ("entity_evidence_background_difference", "证据－背景实体提及密度"),
    ("number_evidence_background_difference", "证据－背景数字密度"),
)


def _ledger(statuses):
    counts = Counter(statuses)
    retained = counts.pop("retained", 0)
    return {
        "scanned": retained + sum(counts.values()),
        "retained": retained,
        "excluded": sum(counts.values()),
        "exclusion_reasons": dict(sorted(counts.items())),
    }


def _qa_metrics(by_document):
    metrics = {}
    names = next(iter(by_document.values()))[0] if by_document else {}
    for name in names:
        means = []
        qa_count = qa_missing = 0
        for rows in by_document.values():
            values = [row[name] for row in rows if row[name] is not None]
            qa_count += len(values)
            qa_missing += len(rows) - len(values)
            means.append(fmean(values) if values else None)
        metrics[name] = {**summarize(means), "qa_count": qa_count, "qa_missing": qa_missing}
    return metrics


def build_summary(records: list[dict], merge_splits: bool = False) -> dict:
    """Pool scanned documents, preserving their source identities for QA weighting."""
    document_groups = defaultdict(list)
    qa_groups = defaultdict(list)
    selection_groups = defaultdict(list)
    segmentation_groups = defaultdict(list)
    recent_groups = defaultdict(lambda: defaultdict(list))
    for record in records:
        dataset = record["dataset"]
        split = "all" if merge_splits else record["split"]
        document_key = (record["split"], record["doc_id"])
        document_groups[(dataset, split, record["unit_kind"])].append(record)
        retained_qas = defaultdict(list)
        for qa in record["qas"]:
            granularity = qa["evidence_granularity"]
            qa_groups[(dataset, split, granularity)].append((document_key, qa))
            if record["status"] == "retained" and qa["status"] == "retained":
                retained_qas[granularity].append(qa)
                for recent in qa["recent"]:
                    key = (dataset, split, granularity, recent["recent_tokens"])
                    recent_groups[key][document_key].append(
                        {"tail_overlap": recent["tail_overlap"]}
                    )
        if record["status"] == "retained":
            for setting in record["segmentations"]:
                for granularity, qas in retained_qas.items():
                    key = (
                        dataset,
                        split,
                        record["unit_kind"],
                        granularity,
                        setting["blocks"],
                        setting["min_block_tokens"],
                        setting["max_block_tokens"],
                    )
                    segmentation_groups[key].append((document_key, setting, qas))
        source_ref = record["source_ref"]
        if "source_qa_count" in source_ref and "selected_qa_count" in source_ref:
            selection_groups[(dataset, split, source_ref["qa_selection_rule"])].append(source_ref)

    documents = []
    for (dataset, split, unit_kind), group in sorted(document_groups.items()):
        retained = [record for record in group if record["status"] == "retained"]
        entry = {"dataset": dataset, "split": split, "unit_kind": unit_kind}
        entry["ledger"] = _ledger(record["status"] for record in group)
        entry["metrics"] = {
            name: summarize([record["metrics"][name] for record in retained])
            for name in (retained[0]["metrics"] if retained else {})
        }
        documents.append(entry)

    questions = []
    for (dataset, split, granularity), group in sorted(qa_groups.items()):
        by_document = defaultdict(list)
        for doc_id, qa in group:
            if qa["status"] == "retained":
                by_document[doc_id].append(qa["metrics"])
        entry = {"dataset": dataset, "split": split, "evidence_granularity": granularity}
        entry["ledger"] = _ledger(qa["status"] for _, qa in group)
        entry["document_count"] = len(by_document)
        entry["metrics"] = _qa_metrics(by_document)
        questions.append(entry)

    segmentations = []
    for key, group in sorted(
        segmentation_groups.items(), key=lambda item: (*item[0][:-1], item[0][-1] or 0)
    ):
        dataset, split, unit_kind, granularity, blocks, minimum, maximum = key
        by_document = defaultdict(list)
        for doc_id, setting, qas in group:
            for qa in qas:
                geometry = next(row for row in qa["blocks"] if row["blocks"] == blocks)
                by_document[doc_id].append(
                    {
                        name: geometry[name] if setting["uniform_feasible"] else None
                        for name in ("evidence_cut", "evidence_block_count")
                    }
                )
        segmentations.append(
            {
                "dataset": dataset,
                "split": split,
                "unit_kind": unit_kind,
                "evidence_granularity": granularity,
                "blocks": blocks,
                "min_block_tokens": minimum,
                "max_block_tokens": maximum,
                "doc_count": len(group),
                "uniform_feasible_count": sum(
                    setting["uniform_feasible"] for _, setting, _ in group
                ),
                "natural_feasible_count": sum(
                    setting["natural_feasible"] for _, setting, _ in group
                ),
                **_qa_metrics(by_document),
            }
        )
    recent = []
    for (dataset, split, granularity, tokens), by_document in sorted(recent_groups.items()):
        recent.append(
            {
                "dataset": dataset,
                "split": split,
                "evidence_granularity": granularity,
                "recent_tokens": tokens,
                "doc_count": len(by_document),
                **_qa_metrics(by_document),
            }
        )
    qa_selection = []
    for (dataset, split, rule), group in sorted(selection_groups.items()):
        source = sum(ref["source_qa_count"] for ref in group)
        selected = sum(ref["selected_qa_count"] for ref in group)
        entry = {"dataset": dataset, "split": split, "document_count": len(group)}
        entry["rule"] = rule
        entry["source_qas"], entry["selected_qas"] = source, selected
        entry["not_selected_qas"] = source - selected
        qa_selection.append(entry)
    return {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
            "%Y%m%d %H:%M:%S Asia/Shanghai"
        ),
        "document_ledger": _ledger(record["status"] for record in records),
        "qa_ledger": _ledger(qa["status"] for record in records for qa in record["qas"]),
        "qa_selection": qa_selection,
        "document_groups": documents,
        "qa_groups": questions,
        "segmentation_groups": segmentations,
        "recent_groups": recent,
    }


def _number(value):
    return "—" if value is None else f"{value:.4g}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _source_label(group):
    if group["split"] is None:
        return group["dataset"]
    return f"{group['dataset']} / {group['split']}"


def _denominator(metric):
    result = f"{metric['count']}/{metric['count'] + metric['missing']}"
    if "qa_count" in metric:
        result += f"；{metric['qa_count']}/{metric['qa_count'] + metric['qa_missing']}"
    return result


def _metric_rows(groups, fields, kind):
    rows = []
    for group in groups:
        label = _cell(f"{_source_label(group)} / {group[kind]}")
        if not group["metrics"]:
            rows.append(f"| {label} | 无有效记录 | — | — | — | — |")
            continue
        for name, title in fields:
            metric = group["metrics"][name]
            values = " | ".join(_number(metric[key]) for key in ("mean", "p50", "p90"))
            rows.append(f"| {label} | {title} | {values} | {_denominator(metric)} |")
    return rows


def write_report(output_dir: Path, summary: dict) -> None:
    """Write report.md without copying source text, references, or configuration."""
    generated_at = summary["generated_at"]
    documents, questions = summary["document_ledger"], summary["qa_ledger"]
    lines = [
        f"""# {generated_at[:8]}_边界相关文本属性统计

创建时间：{generated_at}

最后修订时间：{generated_at}

- 本次扫描 {documents["scanned"]} 篇文档，保留 {documents["retained"]} 篇，排除 {documents["excluded"]} 篇；仅覆盖实际扫描范围，`--max-samples` 截取结果不代表输入全集。
- 按数据集范围选中 {questions["scanned"]} 道 QA，保留 {questions["retained"]} 道，排除 {questions["excluded"]} 道；未选中的题目不计为指标检查失败。
- 分段参数与末尾窗口各自分组报告，不挑选最优组合；组合数不增加独立样本量（例如 36 组不等于 36 倍样本），末尾统计不随分段组合重复累计。
- QA 指标先取每篇文档的有效题目均值，再按文档等权汇总；缺失不填 0，分母逐指标列出。
- 实体指标仅为 capitalized mention proxy（大写形式的提及代理），数字与提及密度以每百词计；这些文本属性未经模型验证，不能据此认定边界敏感或作因果判断。

## 过滤收支

文档和 QA 各自使用互斥主原因；`document_excluded` 表示所属文档被排除，具体原因在文档账本。未扫描数量不在本报告中推算。

| 层级 | 数据源 | 单元或证据粒度 | 扫描 | 保留 | 排除 | 互斥排除原因（数量 / 扫描数） |
|---|---|---|---:|---:|---:|---|"""
    ]
    for level, groups, kind in (
        ("文档", summary["document_groups"], "unit_kind"),
        ("QA", summary["qa_groups"], "evidence_granularity"),
    ):
        for group in groups:
            source = _cell(_source_label(group))
            granularity = _cell(group[kind])
            ledger = group["ledger"]
            reasons = (
                "；".join(
                    f"{_cell(reason)}: {count}/{ledger['scanned']}"
                    for reason, count in ledger["exclusion_reasons"].items()
                )
                or "—"
            )
            lines.append(
                f"| {level} | {source} | {granularity} | {ledger['scanned']} | "
                f"{ledger['retained']} | {ledger['excluded']} | {reasons} |"
            )
    if summary["qa_selection"]:
        lines.append("""
QA 选择账本仅覆盖提供该信息的文档；角色、类别等范围选择与后续证据检查分开计数。

| 数据源 | 选择规则 | 覆盖文档 | 源 QA | 选中 | 未选中 |
|---|---|---:|---:|---:|---:|""")
        for group in summary["qa_selection"]:
            lines.append(
                f"| {_cell(_source_label(group))} | {_cell(group['rule'])} | "
                f"{group['document_count']} | {group['source_qas']} | "
                f"{group['selected_qas']} | {group['not_selected_qas']} |"
            )
    lines.append("""
## 核心指标

以下仅汇总保留记录。N 为有效文档数 / 保留文档数，QA 表另列有效 QA 数 / 保留 QA 数。分母之差为缺失数，`—` 表示无有效值。QA 分位数针对文档均值，不是全部题目的混合分布。

| 数据源 / 自然单元 | 文档指标 | 均值 | P50 | P90 | N（文档） |
|---|---|---:|---:|---:|---:|""")
    lines += _metric_rows(summary["document_groups"], DOCUMENT_REPORT_METRICS, "unit_kind")
    lines.append("""
| 数据源 / 证据粒度 | QA 指标 | 均值 | P50 | P90 | N（文档；QA） |
|---|---|---:|---:|---:|---:|""")
    lines += _metric_rows(summary["qa_groups"], QA_REPORT_METRICS, "evidence_granularity")
    lines.append("""
## 参数扫描

每行固定数据源、自然单元和证据粒度；可行数的分母为该组文档数。等长分段不可行时，仅该组合的 QA 劈裂率和占用块数记为缺失，不改变原始几何或基础统计。两项共用有效分母，N 列出有效文档 / 全部文档、有效 QA / 全部保留 QA。

| 数据源 / 自然单元 / 证据粒度 | B | 最短块 | 最长块 | 等长可行 | 自然边界可行 | 劈裂率均值 | 占用块数均值 | N（文档；QA） |
|---|---:|---:|---:|---:|---:|---:|---:|---|""")
    for group in summary["segmentation_groups"]:
        label = _cell(
            f"{_source_label(group)} / {group['unit_kind']} / {group['evidence_granularity']}"
        )
        cut, occupied = group["evidence_cut"], group["evidence_block_count"]
        maximum = group["max_block_tokens"] if group["max_block_tokens"] is not None else "不限"
        lines.append(
            f"| {label} | {group['blocks']} | {group['min_block_tokens']} | {maximum} | "
            f"{group['uniform_feasible_count']}/{group['doc_count']} | "
            f"{group['natural_feasible_count']}/{group['doc_count']} | {_number(cut['mean'])} | "
            f"{_number(occupied['mean'])} | {_denominator(cut)} |"
        )
    lines.append("""
末尾交集率按窗口长度独立统计，分母不乘以分段组合数；`R=0` 表示没有原文末尾窗口。

| 数据源 / 证据粒度 | R（tokens） | 末尾交集率均值 | N（文档；QA） |
|---|---:|---:|---|""")
    for group in summary["recent_groups"]:
        label = _cell(f"{_source_label(group)} / {group['evidence_granularity']}")
        metric = group["tail_overlap"]
        lines.append(
            f"| {label} | {group['recent_tokens']} | {_number(metric['mean'])} | {_denominator(metric)} |"
        )
    lines += ["", "完整统计见独立汇总 JSON；运行参数和源引用见配置及逐文档 JSON 产物。", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
