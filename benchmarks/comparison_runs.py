"""从已评分运行的配置组织方法、checkpoint 和评估参数。"""

from collections import defaultdict
from colorsys import hls_to_rgb
from pathlib import Path
import re


def natural_key(value):
    parts = re.split(r"([0-9]+)", value)
    return tuple(int(part) if index % 2 else part.casefold()
                 for index, part in enumerate(parts)), value


def run_identity(summary):
    """只读取各方法当前写入的配置，不从目录或运行标签推断参数。"""
    method, settings = summary["method"], summary["settings"]
    if method == "full-context":
        checkpoint = settings["model"]["model_name_or_path"]
        values, setting = (), ""
    elif method == "icae":
        checkpoint = settings["checkpoint"]
        values = (settings["fixed_mem_size"], settings["mean_compression_rate"])
        setting = f"mem{values[0]}_cr{values[1]}"
    elif method == "autocompressors":
        checkpoint = settings["checkpoint"]
        values = (settings["segment_length"],)
        setting = f"seg{values[0]}"
    elif method == "sdcc":
        checkpoint = settings["checkpoint"]
        values = (settings["compress_ratio_scale"],)
        setting = f"scale{values[0]:g}"
    elif method in ("lact", "e2-ttt"):
        checkpoint = settings["checkpoint"]
        values, setting = (), ""
    elif method == "in-place-ttt":
        checkpoint = settings["checkpoint"]
        values = (settings["mlp_chunk_size"],)
        setting = f"mlp{values[0]}"
        if summary["benchmark"] in ("longmemeval_s_cleaned", "longmemeval_oracle"):
            values += (settings["max_seq_len"],)
            setting += f"_len{values[1]}"
    else:
        raise ValueError(f"尚未定义方法 {method!r} 的 checkpoint 与评估参数，请先补充 run_identity")
    name = Path(checkpoint).stem if method == "icae" else Path(checkpoint).name
    return checkpoint, name, values, setting


def scope_name(summary):
    benchmark = summary["benchmark"]
    if benchmark == "longmemeval_s_cleaned":
        return "S"
    if benchmark == "longmemeval_oracle":
        return "Oracle"
    if benchmark in ("qasper", "musique"):
        return summary["split"]
    return benchmark


def prepare_runs(runs):
    """在筛选前确定排序、完整标签和颜色，返回副本并保留原始 alias。"""
    prepared = []
    checkpoints = defaultdict(dict)
    for run in runs:
        summary = run["summary"]
        checkpoint_id, checkpoint, values, setting = run_identity(summary)
        item = {**run, "alias": run["label"], "method": summary["method"],
                "checkpoint_id": checkpoint_id, "checkpoint": checkpoint,
                "setting_sort": values, "setting": setting}
        prepared.append(item)
        checkpoints[(item["method"], checkpoint)][checkpoint_id] = item

    # 同名 checkpoint 使用能区分它们的最短路径后缀，内部仍按完整路径分组。
    checkpoint_names = {}
    for (method, name), identities in checkpoints.items():
        if len(identities) == 1:
            identity = next(iter(identities))
            checkpoint_names[(method, identity)] = name
            continue
        parts = {identity: Path(identity).parts for identity in identities}
        depth = 1
        while len({tuple(value[-depth:]) for value in parts.values()}) != len(parts):
            depth += 1
        for identity, value in parts.items():
            path = Path(*value[-depth:])
            checkpoint_names[(method, identity)] = str(path.with_suffix("") if method == "icae" else path)
    for run in prepared:
        run["checkpoint"] = checkpoint_names[(run["method"], run["checkpoint_id"])]
    prepared.sort(key=lambda run: (
        natural_key(run["method"]), natural_key(run["checkpoint"]), run["checkpoint_id"],
        run["setting_sort"], natural_key(run["alias"]), str(run["directory"]),
    ))

    by_checkpoint, by_label = defaultdict(list), defaultdict(list)
    for run in prepared:
        base_label = " · ".join(part for part in (
            run["method"], run["checkpoint"], run["setting"],
        ) if part)
        by_label[base_label].append(run)
        by_checkpoint[(run["method"], run["checkpoint_id"])].append(run)

    multiple_scopes = len({scope_name(run["summary"]) for run in prepared}) > 1
    for base_label, group in by_label.items():
        scopes = defaultdict(list)
        for run in group:
            scopes[scope_name(run["summary"])].append(run)
        for scope, scoped_runs in scopes.items():
            for run in scoped_runs:
                label = base_label
                if multiple_scopes:
                    label += f" · {scope}"
                if len(scoped_runs) > 1:
                    label += f" · {run['alias']}"
                run["display_label"] = label
                run["label"] = label

    color_keys, method_keys = {}, defaultdict(set)
    for group in by_checkpoint.values():
        repeats = defaultdict(list)
        for run in group:
            repeats[(run["setting_sort"], scope_name(run["summary"]))].append(run)
        for (setting, _scope), scoped_runs in repeats.items():
            for repeat_index, run in enumerate(scoped_runs):
                color_keys[id(run)] = (setting, repeat_index)
                if setting:
                    method_keys[run["method"]].add((setting, repeat_index))

    for index, ((method, _checkpoint), group) in enumerate(by_checkpoint.items()):
        keys = sorted(method_keys[method] if group[0]["setting_sort"] else
                      {color_keys[id(run)] for run in group})
        lightness = {key: 0.50 if len(keys) == 1 else 0.28 + 0.48 * i / (len(keys) - 1)
                     for i, key in enumerate(keys)}
        hue = (0.60 + index * 0.618) % 1
        for run in group:
            rgb = hls_to_rgb(hue, lightness[color_keys[id(run)]], 0.65)
            run["color"] = "#" + "".join(f"{round(value * 255):02x}" for value in rgb)
    return prepared
