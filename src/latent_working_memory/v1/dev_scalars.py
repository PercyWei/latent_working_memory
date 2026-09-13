"""Incremental dev scalars and native SwanLab panels sharing optimizer-step axes."""

import secrets

import swanlab

from latent_working_memory.v1.reporting import CONDITION_COLORS, OVERVIEW_METRICS, _shade


def development_scalars(reports):
    return {
        f"dev/overview/{task}/{metric}/{source}/{condition}": group[metric]
        for source, report in reports.items()
        for task, metric in OVERVIEW_METRICS
        for condition in CONDITION_COLORS
        if metric in (group := report["groups"].get(f"all/{task}/{condition}", {}))
    }


def development_panels(sources):
    panels = {}
    for task, metric in OVERVIEW_METRICS:
        conditions = [c for c in CONDITION_COLORS if task != "ae" or c != "no_memory"]
        # The cloud native LINE API permits at most eight Y metrics per panel.
        groups = [("", sources)] if len(sources) * len(conditions) <= 8 else [
            (f"/{source}", [source]) for source in sources
        ]
        for suffix, selected in groups:
            title = f"dev/overview/{task}/{metric}{suffix}"
            panels[title] = {
                "title": title,
                "config": {
                    "xAxis": {"key": "step", "name": "step", "type": "FLOAT", "class": "SYSTEM"},
                    "yAxis": [{"key": key, "name": key, "type": "FLOAT", "class": "CUSTOM"}
                              for condition in conditions for source in selected
                              for key in [f"dev/overview/{task}/{metric}/{source}/{condition}"]],
                    "xName": "optimizer step", "yName": metric,
                },
            }
    return panels


def development_panel_style(panel, sources, run_id):
    custom = {}
    for axis in panel["config"]["yAxis"]:
        source, condition = axis["key"].split("/")[-2:]
        color = _shade(CONDITION_COLORS[condition], sources.index(source), len(sources))
        custom[f"{run_id}-{axis['key']}"] = {
            "name": f"{source}/{condition}", "colors": [color, color],
        }
    return custom


def configure_development_panels(run, sources, mode):
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
    charts = [] if section is None else [
        checked(api._get(f"{base}/chart/{index}/info")) for index in section["chartIndex"]
    ]
    panels_by_index = {}
    columns = []
    for title, panel in development_panels(list(sources)).items():
        panel["custom"] = development_panel_style(panel, list(sources), remote.run_id)
        existing = next((c for c in charts if c["title"] == title and c["type"] == "LINE"), None)
        if existing is not None:
            panels_by_index[existing["index"]] = panel
            continue
        index = secrets.token_hex(4)
        panels_by_index[index] = panel
        for axis in panel["config"]["yAxis"]:
            columns.append({
                "key": axis["key"], "type": "FLOAT", "class": "CUSTOM",
                "sectionName": "dev", "chartName": title, "chartIndex": index,
                "metricName": "/".join(axis["key"].split("/")[-2:]),
            })
    if columns:
        # This project's v0 endpoint binds each new column directly to its shared chart.
        # Register before SDK log() so it never creates an individual fallback panel.
        checked(api._post(f"{base}/columns", data=columns))
    for index, panel in panels_by_index.items():
        checked(api._put(f"{base}/chart/{index}/info/line", data=panel))


def remove_individual_dev_panels(run, sources):
    """Delete automatic single-series panels, retaining their data and grouped curves."""
    api = swanlab.Api()
    project = run.url.split("/@", 1)[1].split("/runs/", 1)[0]
    remote = api.run(f"{project}/{run.id}")
    base = f"/experiment/{remote.run_id}"
    keys = {axis["key"] for panel in development_panels(list(sources)).values()
            for axis in panel["config"]["yAxis"]}

    def checked(response):
        if not response.ok:
            raise RuntimeError(response.errmsg)
        return response.data

    public = checked(api._get(f"{base}/sections", params={"size": 100}))
    hidden = checked(api._get(f"{base}/sections/protected", params={"types": "HIDDEN"}))
    deleted = []
    for section in public + hidden:
        if not (section["name"].startswith("dev") or section["type"] == "HIDDEN"):
            continue
        for index in section["chartIndex"]:
            chart = checked(api._get(f"{base}/chart/{index}/info"))
            if (chart["type"] == "LINE" and chart["title"] in keys
                    and len(chart["config"]["yAxis"]) == 1):
                checked(api._delete(f"{base}/chart/{index}/hard"))
                deleted.append(index)
    # Empty auto-created sections are also removed; the main dev section stays.
    for section in checked(api._get(f"{base}/sections", params={"size": 100})):
        if section["name"].startswith("dev/overview/") and not section["chartIndex"]:
            checked(api._delete(f"{base}/section/{section['index']}"))
    return deleted
