"""Incremental dev scalars and native SwanLab panels sharing optimizer-step axes."""

import swanlab

from latent_working_memory.v1.reporting import CONDITION_COLORS, OVERVIEW_METRICS


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
                              for source in selected for condition in conditions
                              for key in [f"dev/overview/{task}/{metric}/{source}/{condition}"]],
                    "xName": "optimizer step", "yName": metric,
                },
            }
    return panels


def configure_development_panels(run, sources, mode):
    for task, metric in OVERVIEW_METRICS:
        run.define_metric(f"dev/overview/{task}/{metric}/*", hidden=True)
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
    if section is None:
        section = checked(api._post(f"{base}/section", data={"name": "dev", "position": "below"}))
    charts = [checked(api._get(f"{base}/chart/{index}/info")) for index in section["chartIndex"]]
    for title, panel in development_panels(list(sources)).items():
        existing = next((c for c in charts if c["title"] == title and c["type"] == "LINE"), None)
        if existing is None:
            checked(api._post(f"{base}/section/{section['index']}/chart/line", data=panel))
