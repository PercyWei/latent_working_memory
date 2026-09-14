"""跨阶段共享的表格、配色、图表和原生曲线面板。"""

from __future__ import annotations
import colorsys
import secrets
from typing import Any
import swanlab


def _table(rows: list[dict[str, Any]]) -> Any:
    headers = list(dict.fromkeys(key for row in rows for key in row))
    return swanlab.echarts.Table().add(
        headers,
        [
            [round(row[k], 4) if isinstance(row.get(k), float) else row.get(k) for k in headers]
            for row in rows
        ],
    )


CONDITION_COLORS = {
    "memory": "#2459A6",
    "wrong_memory": "#B45B18",
    "no_memory": "#626B73",
    "full_context": "#28764A",
    "base_full_context": "#7951A0",
}


def _shade(color: str, source_index: int, source_count: int) -> str:
    color = color.lstrip("#")
    rgb = [int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    h, light, saturation = colorsys.rgb_to_hls(*rgb)
    light += 0.25 * source_index / max(source_count - 1, 1)
    return "#" + "".join(f"{round(c * 255):02x}" for c in colorsys.hls_to_rgb(h, light, saturation))


def _bar(
    labels: list[str],
    series: dict[str, list[Any]],
    hue_groups: list[str],
    hue_colors: dict[str, str],
    hue_dimension: str,
) -> Any:
    chart = swanlab.echarts.Bar().add_xaxis(
        [label.replace("_", "\n").replace("/r", "\nr") for label in labels]
    )
    for group in dict.fromkeys(hue_groups):
        for source_index, (source, values) in enumerate(series.items()):
            points = [round(value, 4) if isinstance(value, float) else value for value in values]
            points = [
                value if hue == group else None
                for value, hue in zip(points, hue_groups, strict=True)
            ]
            chart.add_yaxis(
                f"{hue_dimension}={group} / test={source}",
                points,
                stack=source,
                label_opts={"show": False},
                itemstyle_opts={"color": _shade(hue_colors[group], source_index, len(series))},
            )
    chart.set_global_opts(
        tooltip_opts={"trigger": "axis"},
        legend_opts={"type": "scroll", "top": 0},
        xaxis_opts={"axisLabel": {"interval": 0, "rotate": 0, "fontSize": 10, "lineHeight": 11}},
        yaxis_opts={"minInterval": 0.001, "splitNumber": 3},
    )
    chart.options["grid"] = {
        "left": "12%",
        "right": "4%",
        "top": "20%",
        "height": "55%",
        "containLabel": False,
    }
    return chart


def configure_line_panels(run, panels, mode, style):
    """Register dev columns directly into shared native panels before logging data."""
    if mode != "online":
        return
    api = swanlab.Api()
    project_path = run.url.split("/@", 1)[1].split("/runs/", 1)[0]
    remote = api.run(f"{project_path}/{run.id}")
    base = f"/experiment/{remote.run_id}"

    def checked(response):
        if not response.ok:
            raise RuntimeError(response.errmsg)
        return response.data

    sections = checked(api._get(f"{base}/sections", params={"size": 100}))
    section = next((s for s in sections if s["name"] == "dev"), None)
    charts = (
        []
        if section is None
        else [checked(api._get(f"{base}/chart/{index}/info")) for index in section["chartIndex"]]
    )
    panels_by_index = {}
    columns = []
    for title, original in panels.items():
        panel = {**original, "custom": style(original, remote.run_id)}
        existing = next((c for c in charts if c["title"] == title and c["type"] == "LINE"), None)
        if existing is not None:
            panels_by_index[existing["index"]] = panel
            continue
        index = secrets.token_hex(4)
        panels_by_index[index] = panel
        for axis in panel["config"]["yAxis"]:
            columns.append(
                {
                    "key": axis["key"],
                    "type": "FLOAT",
                    "class": "CUSTOM",
                    "sectionName": "dev",
                    "chartName": title,
                    "chartIndex": index,
                    "metricName": "/".join(axis["key"].split("/")[-2:]),
                }
            )
    if columns:
        # This project's v0 endpoint binds each new column directly to its shared chart.
        # Register before SDK log() so it never creates an individual fallback panel.
        checked(api._post(f"{base}/columns", data=columns))
    for index, panel in panels_by_index.items():
        checked(api._put(f"{base}/chart/{index}/info/line", data=panel))
