"""Incremental dev scalars and native SwanLab panels sharing optimizer-step axes."""

import swanlab

from latent_working_memory.v1.reporting import configure_line_panels

from latent_working_memory.v1.reporting import CONDITION_COLORS, _shade
from latent_working_memory.v1.pretrain.reporting import OVERVIEW_METRICS


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
        groups = (
            [("", sources)]
            if len(sources) * len(conditions) <= 8
            else [(f"/{source}", [source]) for source in sources]
        )
        for suffix, selected in groups:
            title = f"dev/overview/{task}/{metric}{suffix}"
            panels[title] = {
                "title": title,
                "config": {
                    "xAxis": {"key": "step", "name": "step", "type": "FLOAT", "class": "SYSTEM"},
                    "yAxis": [
                        {"key": key, "name": key, "type": "FLOAT", "class": "CUSTOM"}
                        for condition in conditions
                        for source in selected
                        for key in [f"dev/overview/{task}/{metric}/{source}/{condition}"]
                    ],
                    "xName": "optimizer step",
                    "yName": metric,
                },
            }
    return panels


def development_panel_style(panel, sources, run_id):
    custom = {}
    for axis in panel["config"]["yAxis"]:
        source, condition = axis["key"].split("/")[-2:]
        color = _shade(CONDITION_COLORS[condition], sources.index(source), len(sources))
        custom[f"{run_id}-{axis['key']}"] = {
            "name": f"{source}/{condition}",
            "colors": [color, color],
        }
    return custom


def configure_development_panels(run, sources, mode):
    configure_line_panels(
        run,
        development_panels(list(sources)),
        mode,
        lambda panel, run_id: development_panel_style(panel, list(sources), run_id),
    )


def remove_individual_dev_panels(run, sources):
    """Delete automatic single-series panels, retaining their data and grouped curves."""
    api = swanlab.Api()
    project = run.url.split("/@", 1)[1].split("/runs/", 1)[0]
    remote = api.run(f"{project}/{run.id}")
    base = f"/experiment/{remote.run_id}"
    keys = {
        axis["key"]
        for panel in development_panels(list(sources)).values()
        for axis in panel["config"]["yAxis"]
    }

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
            if (
                chart["type"] == "LINE"
                and chart["title"] in keys
                and len(chart["config"]["yAxis"]) == 1
            ):
                checked(api._delete(f"{base}/chart/{index}/hard"))
                deleted.append(index)
    # Empty auto-created sections are also removed; the main dev section stays.
    for section in checked(api._get(f"{base}/sections", params={"size": 100})):
        if section["name"].startswith("dev/overview/") and not section["chartIndex"]:
            checked(api._delete(f"{base}/section/{section['index']}"))
    return deleted
