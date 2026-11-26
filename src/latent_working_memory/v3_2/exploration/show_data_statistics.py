"""Compare saved data statistics with Streamlit; no metric computation or data loading."""

import argparse
import json
from pathlib import Path

import streamlit as st

from latent_working_memory.v3_2.exploration.data_statistics.report import (
    DOCUMENT_REPORT_METRICS,
    QA_REPORT_METRICS,
)

METRIC_LABELS = dict(DOCUMENT_REPORT_METRICS + QA_REPORT_METRICS) | {
    "evidence_cut": "证据劈裂率",
    "evidence_block_count": "证据占用块数",
    "uniform_feasible": "等长分段可行率",
    "natural_feasible": "自然边界分段可行率",
    "tail_overlap": "末尾证据覆盖率",
}
STATISTICS = {"mean": "均值", "p50": "P50", "p90": "P90", "p10": "P10", "std": "标准差"}


def _source(name, group):
    parts = [name, group["dataset"]]
    if group["split"] is not None:
        parts.append(group["split"])
    parts.extend(
        dict.fromkeys(group[key] for key in ("unit_kind", "evidence_granularity") if key in group)
    )
    return " / ".join(parts)


def _metric_row(name, group, metric):
    row = {"来源": _source(name, group)}
    row.update({label: metric[key] for key, label in STATISTICS.items()})
    row["有效文档"] = f"{metric['count']}/{metric['count'] + metric['missing']}"
    if "qa_count" in metric:
        row["有效 QA"] = f"{metric['qa_count']}/{metric['qa_count'] + metric['qa_missing']}"
    return row


def _show(rows, value, axis=None):
    if not rows:
        st.info("所选运行没有这项统计。")
        return
    points = [row for row in rows if row[value] is not None]
    if points:
        if axis is None:
            st.bar_chart(
                [{"来源": row["来源"], value: row[value]} for row in points],
                x="来源",
                y=value,
            )
        else:
            st.scatter_chart(
                [{"来源": row["来源"], axis: row[axis], value: row[value]} for row in points],
                x=axis,
                y=value,
                color="来源",
            )
    else:
        st.info("当前条件下没有有效值，分母见下表。")
    st.dataframe(rows, hide_index=True, width="stretch")


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--run", nargs="+", action="append", required=True, metavar="NAME DIR")
    args = parser.parse_args()
    if any(len(values) % 2 for values in args.run):
        parser.error("--run requires pairs of NAME DIR")
    run_specs = [
        (values[index], values[index + 1])
        for values in args.run
        for index in range(0, len(values), 2)
    ]
    names = [name for name, _ in run_specs]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        parser.error("run names must be nonempty and unique")

    st.set_page_config(page_title="数据统计对比", layout="wide")
    st.title("数据统计对比")
    selected = st.sidebar.multiselect("统计结果", names, default=names)
    if not selected:
        st.info("请选择至少一个统计结果。")
        return
    runs = []
    overview = []
    for name, directory in run_specs:
        if name not in selected:
            continue
        root = Path(directory).expanduser().resolve()
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        runs.append((name, summary))
        documents, qas = summary["document_ledger"], summary["qa_ledger"]
        overview.append(
            {
                "运行": name,
                "数据集": config["dataset"],
                "split": config["split"],
                "tokenizer": config["tokenizer"],
                "tokenizer 类型": config["tokenizer_class"],
                "密度窗口": config["metrics"]["window_tokens"],
                "最短文档": config["min_tokens"],
                "最长文档": config["max_tokens"],
                "样本上限": config["max_samples"],
                "保留文档": f"{documents['retained']}/{documents['scanned']}",
                "保留 QA": f"{qas['retained']}/{qas['scanned']}",
                "目录": str(root),
            }
        )
    with st.expander("运行条件与样本量", expanded=True):
        st.dataframe(overview, hide_index=True, width="stretch")
        st.caption("空白表示未设置；比较时核对 tokenizer、密度窗口和样本筛选条件。")
    st.caption(
        "来源标签保留结构／证据粒度；各运行独立展示，不合并样本。"
        "QA 统计按文档等权，缺失不补 0；标准差不是置信区间。"
    )

    view = st.sidebar.radio("视图", ("基础指标", "分段扫描", "末尾覆盖"))
    if view == "基础指标":
        level = st.radio("统计层级", ("文档", "QA"), horizontal=True)
        key = "document_groups" if level == "文档" else "qa_groups"
        groups = [(name, group) for name, summary in runs for group in summary[key]]
        metrics = sorted({metric for _, group in groups for metric in group["metrics"]})
        if not metrics:
            st.info("所选运行没有保留样本的基础统计。")
            return
        metric = st.selectbox(
            "指标",
            metrics,
            index=metrics.index("doc_tokens" if level == "文档" else "evidence_tokens"),
            format_func=lambda key: METRIC_LABELS.get(key, key),
        )
        statistic = st.selectbox("统计量", list(STATISTICS), format_func=STATISTICS.__getitem__)
        rows = [
            _metric_row(name, group, group["metrics"][metric])
            for name, group in groups
            if metric in group["metrics"]
        ]
        _show(rows, STATISTICS[statistic])
    elif view == "分段扫描":
        groups = [
            (name, group) for name, summary in runs for group in summary["segmentation_groups"]
        ]
        constraints = sorted(
            {(group["min_block_tokens"], group["max_block_tokens"]) for _, group in groups},
            key=lambda pair: (pair[0], float("inf") if pair[1] is None else pair[1]),
        )
        if not constraints:
            st.info("所选运行没有分段扫描结果。")
            return
        minimum, maximum = st.selectbox(
            "块长约束",
            constraints,
            format_func=lambda pair: f"最短 {pair[0]} / 最长 {pair[1] if pair[1] is not None else '不限'}",
        )
        metric = st.selectbox(
            "指标",
            ("evidence_cut", "evidence_block_count", "uniform_feasible", "natural_feasible"),
            format_func=METRIC_LABELS.__getitem__,
        )
        feasibility = metric in ("uniform_feasible", "natural_feasible")
        value = (
            "可行率"
            if feasibility
            else STATISTICS[
                st.selectbox("统计量", list(STATISTICS), format_func=STATISTICS.__getitem__)
            ]
        )
        rows = []
        for name, group in groups:
            if (group["min_block_tokens"], group["max_block_tokens"]) != (minimum, maximum):
                continue
            if feasibility:
                count, total = group[f"{metric}_count"], group["doc_count"]
                row = {"来源": _source(name, group), "可行率": count / total if total else None}
            else:
                row = _metric_row(name, group, group[metric])
            row.update(
                {
                    "块数": group["blocks"],
                    "等长可行": f"{group['uniform_feasible_count']}/{group['doc_count']}",
                    "自然边界可行": f"{group['natural_feasible_count']}/{group['doc_count']}",
                }
            )
            rows.append(row)
        st.caption("固定块长约束比较不同块数；只画已运行且有有效值的点，缺失组合不补零。")
        _show(rows, value, "块数")
    else:
        statistic = st.selectbox("统计量", list(STATISTICS), format_func=STATISTICS.__getitem__)
        rows = []
        for name, summary in runs:
            for group in summary["recent_groups"]:
                row = _metric_row(name, group, group["tail_overlap"])
                row["末尾 tokens"] = group["recent_tokens"]
                rows.append(row)
        st.caption("末尾范围独立比较，样本量不随分段组合数增加。")
        _show(rows, STATISTICS[statistic], "末尾 tokens")


if __name__ == "__main__":
    main()
