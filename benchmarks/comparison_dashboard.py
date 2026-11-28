"""公共 benchmark 看板的运行选择、核心指标图与逐题对照布局。"""

import argparse
from pathlib import Path

import streamlit as st

from benchmarks.comparison_runs import prepare_runs
from benchmarks.dashboard_data import judge_config, judge_groups


def select_runs(available, scope):
    """按方法、checkpoint 和实际评估设置勾选，单一节点省略多余层级。"""
    def run_key(run):
        return f"comparison-run:{scope}:{run['directory']}"

    def toggle_group(parent_key, child_keys):
        for key in child_keys:
            st.session_state[key] = st.session_state[parent_key]

    def group_checkbox(label, key, runs):
        child_keys = [run_key(run) for run in runs]
        count = sum(st.session_state[child] for child in child_keys)
        st.session_state[key] = count == len(child_keys)
        with st.container(horizontal=True, vertical_alignment="center"):
            st.checkbox(label, key=key, on_change=toggle_group, args=(key, child_keys))
            st.caption(f"已选 {count}/{len(child_keys)}", width="content")

    def checkpoint_checkboxes(label, runs):
        if len(runs) == 1 and not runs[0]["setting"]:
            st.checkbox(label, key=run_key(runs[0]))
            return
        first = runs[0]
        key = f"comparison-checkpoint:{scope}:{first['method']}:{first['checkpoint_id']}"
        group_checkbox(label, key, runs)
        counts = {}
        for run in runs:
            counts[run["setting"]] = counts.get(run["setting"], 0) + 1
        with st.columns([1, 30], gap="small")[1]:
            for run in runs:
                setting = run["setting"]
                if not setting:
                    setting = run["alias"]
                elif counts[setting] > 1:
                    setting = f"{setting} · {run['alias']}"
                st.checkbox(setting, key=run_key(run))

    by_method = {}
    for run in available:
        st.session_state.setdefault(run_key(run), True)
        by_method.setdefault(run["method"], {}).setdefault(run["checkpoint_id"], []).append(run)
    for method, checkpoints in by_method.items():
        if len(checkpoints) == 1:
            runs = next(iter(checkpoints.values()))
            checkpoint_checkboxes(f"{method} · {runs[0]['checkpoint']}", runs)
            continue
        method_runs = [run for runs in checkpoints.values() for run in runs]
        group_checkbox(method, f"comparison-method:{scope}:{method}", method_runs)
        with st.columns([1, 30], gap="small")[1]:
            for runs in checkpoints.values():
                checkpoint_checkboxes(runs[0]["checkpoint"], runs)
    return [run for run in available if st.session_state[run_key(run)]]


def show_score_chart(runs, scores_by_label, colors, color_labels=None):
    """同一量纲的指标合并为分组柱图，沿用已有配色并只展示所选运行的图例。"""
    labels = [run["label"] for run in runs]
    color_labels = labels if color_labels is None else color_labels
    color_scale = {"domain": color_labels, "scheme": "tableau20"} if colors is None else {
        "domain": color_labels, "range": [colors[label] for label in color_labels],
    }
    metrics = list(dict.fromkeys(metric for scores in scores_by_label.values() for metric in scores))
    chart = [
        {"运行": label, "指标": metric, "得分": score}
        for label in labels for metric, score in scores_by_label[label].items() if score is not None
    ]
    st.vega_lite_chart(chart, {
        "encoding": {
            "x": {"field": "指标", "type": "nominal", "sort": metrics,
                  "axis": {"title": None, "labelAngle": 0}},
            "xOffset": {"field": "运行", "type": "nominal", "sort": labels},
            "y": {"field": "得分", "type": "quantitative", "scale": {"domain": [0, 1]},
                  "axis": {"format": ".0%"}},
            "tooltip": [{"field": "运行"}, {"field": "指标"},
                        {"field": "得分", "type": "quantitative", "format": ".2%"}],
        },
        "layer": [
            {"mark": "bar", "encoding": {
                "color": {"field": "运行", "type": "nominal",
                          "scale": color_scale,
                          "legend": {"orient": "top", "direction": "vertical", "title": None,
                                     "values": labels, "labelLimit": 0}},
            }},
            {"mark": {"type": "text", "dy": -7, "fontSize": 11},
             "encoding": {"text": {"field": "得分", "type": "quantitative", "format": ".1%"}}},
        ],
        "height": 360,
    }, width="stretch")


def show_score_panels(runs, score_panels, colors):
    """按指标面板绘图，并在各图中沿用同一运行顺序与配色。"""
    panels = {run["label"]: score_panels(run["summary"]["metrics"]) for run in runs}
    for panel in next(iter(panels.values())):
        st.subheader(panel)
        show_score_chart(runs, {label: scores[panel] for label, scores in panels.items()}, colors)


def show_judge_accuracy_chart(runs, colors=None, color_labels=None):
    scored = [run for run in runs if run.get("judge")]
    if not scored:
        return
    st.subheader("Judge Accuracy")
    config = judge_config(scored[0]["judge"])
    if any(judge_config(run["judge"]) != config for run in scored[1:]):
        st.warning("所选运行的裁判配置不同。各运行的 Judge Accuracy 可在详细统计中查看。")
        return
    st.caption(f"裁判模型 {config['model']} · 所选运行中 {len(scored)}/{len(runs)} 份已完成裁判评分")
    if config["benchmark"] == "qasper":
        st.caption("答案类型沿用各运行的 F1 归类。")
    scores_by_label = {
        run["label"]: {scope: group["score"] for scope, group in judge_groups(run["judge"])}
        for run in scored
    }
    show_score_chart(scored, scores_by_label, colors, color_labels)


def parse_run_paths(title, argv=None):
    """合并两种入口的目录，统一解析路径、检查重复并自动生成标签。"""
    parser = argparse.ArgumentParser(description=f"{title} 跨方法、checkpoint 和评估设置比较。")
    parser.add_argument("--run", nargs="+", action="extend", default=[],
                        metavar="DIRECTORY",
                        help="一个或多个直接包含 summary.json、details.jsonl 的结果目录；相对路径以项目根目录为基准")
    parser.add_argument("--run-file", type=Path, metavar="FILE",
                        help="UTF-8 目录清单，每行一个结果目录；相对路径以项目根目录为基准，忽略空行和 # 注释行")
    args = parser.parse_args(argv)
    if not args.run and args.run_file is None:
        parser.error("至少提供 --run 或 --run-file。")
    entries = [(f"--run 第 {number} 项", directory) for number, directory in enumerate(args.run, 1)]

    if args.run_file is not None:
        run_file = args.run_file.expanduser().resolve()
        try:
            lines = run_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            parser.error(f"无法读取运行目录清单 {run_file}：{error}")
        file_entries = [(f"{run_file}:{number}", line.strip()) for number, line in enumerate(lines, 1)
                        if line.strip() and not line.lstrip().startswith("#")]
        if not file_entries:
            parser.error(f"运行目录清单为空：{run_file}")
        entries.extend(file_entries)

    project_root = Path(__file__).resolve().parents[1]
    paths, seen = {}, set()
    for source, directory in entries:
        if not directory.strip():
            parser.error(f"{source} 的结果目录不能为空。")
        path = (project_root / Path(directory).expanduser()).resolve()
        if path in seen:
            parser.error(f"{source} 重复指定结果目录：{path}")
        seen.add(path)
        for depth in range(1, len(path.parts) + 1):
            label = str(Path(*path.parts[-depth:]))
            if label not in paths:
                paths[label] = path
                break
    return paths


def run_comparison(title, load_runs, scope_labels, score_panels, show_tables, show_answers, argv=None):
    """保留 benchmark 的读取和评分逻辑，统一多方法比较的交互。"""
    paths = parse_run_paths(title, argv)

    st.set_page_config(page_title=f"{title} · 运行比较", layout="wide")
    st.title(f"{title} · 运行比较")
    try:
        available = prepare_runs(load_runs(paths.items()))
        for run in available:
            if run["summary"]["benchmark"] not in scope_labels:
                raise ValueError(f"{run['alias']} 的 benchmark 与当前看板不一致。")
    except (OSError, KeyError, ValueError) as error:
        st.error(f"无法比较评估结果。{error}")
        return

    st.caption("按方法、checkpoint / 模型及评估设置勾选；图表、统计表和逐题回答同步更新。")
    selected = []
    for scope, scope_label in scope_labels.items():
        scope_runs = [run for run in available if run["summary"]["benchmark"] == scope]
        if not scope_runs:
            continue
        with st.container(border=True):
            st.header(scope_label)
            with st.expander("参与比较的运行", expanded=True):
                runs = select_runs(scope_runs, scope)
            if not runs:
                st.info("请勾选需要比较的运行。")
                continue
            selected.extend(runs)
            st.caption(f"已选 {len(runs)} / {len(scope_runs)} 份结果。")
            colors = {run["label"]: run["color"] for run in runs}
            show_score_panels(runs, score_panels, colors)
            show_judge_accuracy_chart(runs, colors)
            with st.expander("详细统计"):
                st.dataframe([
                    {"运行": run["label"], "方法": run["method"], "checkpoint / 模型": run["checkpoint"],
                     "评估设置": run["setting"] or "—", "运行标签": run["alias"]}
                    for run in runs
                ], hide_index=True)
                show_tables(runs)
            with st.expander("运行配置"):
                for run in runs:
                    st.markdown(f"**{run['label']}**")
                    config = {"目录": str(run["directory"]), **run["summary"]}
                    if run.get("judge"):
                        config["裁判配置"] = judge_config(run["judge"])
                    st.json(config, expanded=False)

    if selected:
        selected_directories = {run["directory"] for run in selected}
        selected = [run for run in available if run["directory"] in selected_directories]
        with st.container(border=True):
            st.header("逐题结果")
            if len({run["summary"]["benchmark"] for run in selected}) > 1:
                st.caption("对照上方各数据设置中已勾选运行的同一道题。")
            show_answers(selected)
