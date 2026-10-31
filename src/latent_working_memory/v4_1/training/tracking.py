"""将同一次 inner 写入的前后标量合并为原生增量曲线。"""

import secrets


def configure_inner_am_panels(api, remote):
    """配置默认视图，保留已有标量数据和其他运行的曲线样式。"""
    def checked(response):
        if not response.ok:
            raise RuntimeError(response.errmsg)
        return response.data

    project_path = remote.url.split("/@", 1)[1].split("/runs/", 1)[0]
    project = checked(api._get(f"/project/{project_path}"))
    view = project["viewIndex"][0]
    sections_path = f"/sections/{project_path}/{view}"
    sections = checked(api._get(sections_path, params={"size": 100}))
    charts_path = f"/charts/{project_path}/{view}"
    for split in ("train", "dev"):
        title = f"{split}/inner_am"
        series = (
            (f"{split}/inner_before", "更新前", "#74a9cf"),
            (f"{split}/inner_after", "更新后", "#2b8cbe"),
        )
        names = {title, *(metric for metric, _, _ in series)}
        metrics = {metric for metric, _, _ in series}
        section = next((row for row in sections if row["name"] == split), None)
        if section is None:
            section = checked(api._post(
                sections_path,
                data={"name": split, "index": secrets.token_hex(3), "position": "above"},
            ))
            section["chartIndex"] = []
        charts = [
            checked(api._get(f"{charts_path}/xxxxxx/{index}"))
            for index in section["chartIndex"]
        ]
        existing = next(
            (chart for chart in charts if chart["title"] == title and chart["type"] == "LINE"),
            None,
        )
        custom = {}
        for chart in charts:
            if chart["type"] == "LINE" and chart["title"] in names:
                custom.update(chart.get("custom") or {})
        custom.update((existing or {}).get("custom") or {})
        for metric, label, color in series:
            custom[f"{remote.run_id}-{metric}"] = {
                "name": f"{remote.name}/{label}", "colors": [color, color],
            }
        body = {
            "type": "LINE",
            "title": title,
            "custom": custom,
            "config": {
                **((existing or {}).get("config") or {}),
                "xAxis": {"key": "step", "type": "SYSTEM", "class": "SCALAR"},
                "yAxis": [
                    {"key": metric, "type": "FLOAT", "class": "SCALAR"}
                    for metric, _, _ in series
                ],
            },
        }
        if existing is None:
            existing = checked(api._post(f"{charts_path}/{section['index']}", data=body))
        else:
            checked(api._put(f"{charts_path}/xxxxxx/{existing['index']}", data=body))
        for chart in charts:
            if chart["index"] == existing["index"] or chart["type"] != "LINE":
                continue
            keys = {axis["key"] for axis in chart["config"]["yAxis"]}
            if chart["title"] in names and keys and keys <= metrics:
                checked(api._delete(f"{charts_path}/{section['index']}/{chart['index']}"))
