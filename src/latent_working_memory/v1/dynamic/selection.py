"""动态训练来源与独立 dev/test 来源的配置契约。"""

import json
from pathlib import Path
import re

from latent_working_memory.v1.dynamic.squad import SquadDataset
from latent_working_memory.v1.dynamic.personamem import PersonaMemDataset


DATASETS = {"squad": SquadDataset, "personamem": PersonaMemDataset}


def load_selection(path, prepared=False):
    spec = json.loads(path.read_text())
    if set(spec) != {"sources", "training", "evaluation"} or not spec["sources"]:
        raise ValueError("selection requires sources, training and evaluation")
    fields = {"dataset", "dataset_dir"} | ({"evaluation_plan"} if prepared else set())
    for name, source in spec["sources"].items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise ValueError("source names must be lowercase path-safe identifiers")
        if set(source) != fields or source["dataset"] not in DATASETS:
            raise ValueError("source must specify a supported dataset and its paths")
        if any(not isinstance(source[k], str) or not source[k] for k in fields):
            raise ValueError("source paths must be nonempty strings")
    if spec["training"] not in spec["sources"] or set(spec["evaluation"]) != {"dev", "test"}:
        raise ValueError("training and dev/test must reference defined sources")
    used = {spec["training"]}
    for names in spec["evaluation"].values():
        if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
                or len(names) != len(set(names)) or not set(names) <= spec["sources"].keys()):
            raise ValueError("evaluation sources must be unique defined source names")
        used.update(names)
    if used != set(spec["sources"]):
        raise ValueError("selection contains unused sources")
    return spec


def load_source(source, tokenizer):
    return DATASETS[source["dataset"]](Path(source["dataset_dir"]), tokenizer)
