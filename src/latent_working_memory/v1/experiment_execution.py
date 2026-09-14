"""实验脚本共用的命令记录、GPU 分配与执行状态。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d %H:%M:%S UTC+08:00")


class ExperimentExecution:
    def __init__(self, plan: Path, gpus: list[int]):
        self.plan, self.gpus = plan, gpus
        self.commands = []
        self.status = {
            "started": now(), "pid": os.getpid(), "status": "running", "jobs": {},
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "gpus": gpus,
        }
        self.save_status()

    def save_status(self):
        self.status["updated"] = now()
        temporary = self.plan / "status.tmp"
        temporary.write_text(json.dumps(self.status, indent=2) + "\n")
        temporary.replace(self.plan / "status.json")

    def execute(self, name, argv, devices):
        env = os.environ | {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, devices)),
            "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
        }
        with (self.plan / f"{name}.log").open("a") as handle:
            handle.write(f"\n{now()}\n")
            handle.flush()
            return subprocess.run(argv, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode

    def stage(self, jobs):
        allocated = set()
        for name, argv, devices in jobs:
            if not set(devices) <= set(self.gpus) or allocated.intersection(devices):
                raise ValueError("parallel jobs must use disjoint allocated GPUs")
            allocated.update(devices)
            self.status["jobs"][name] = {"status": "running", "started": now()}
            self.commands.append({"name": name, "argv": argv, "CUDA_VISIBLE_DEVICES": devices,
                                  "LWM_ALLOWED_PHYSICAL_GPUS": os.environ.get("LWM_ALLOWED_PHYSICAL_GPUS", "4,5,6,7")})
        (self.plan / "commands.json").write_text(json.dumps(self.commands, indent=2) + "\n")
        self.save_status()
        failures = []
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [(name, executor.submit(self.execute, name, argv, devices))
                       for name, argv, devices in jobs]
            for name, future in futures:
                code = future.result()
                self.status["jobs"][name].update(status="failed" if code else "complete",
                                                exit_code=code, finished=now())
                self.save_status()
                if code:
                    failures.append(name)
        if failures:
            raise RuntimeError(f"failed jobs: {failures}; see plan logs")

    def finish(self, error=None):
        self.status.update(status="failed" if error else "complete", finished=now())
        if error:
            self.status["error"] = str(error)
        self.save_status()


def load_experiment(path, extra_fields):
    """Load one experiment; child config paths are relative to its manifest."""
    spec = json.loads(path.read_text())
    expected = {"name", "model", "selection", "gpus", "swanlab_project"} | set(extra_fields)
    if set(spec) != expected:
        raise ValueError(f"experiment fields must be {sorted(expected)}")
    if not spec["name"] or Path(spec["name"]).name != spec["name"] or spec["name"] in {".", ".."}:
        raise ValueError("experiment name must be a simple directory name")
    gpus = spec["gpus"]
    if (not isinstance(gpus, list) or not gpus or len(gpus) != len(set(gpus))
            or any(type(g) is not int or g < 0 for g in gpus)):
        raise ValueError("gpus must contain distinct physical GPU indices")
    allowed = {int(g) for g in os.environ.get("LWM_ALLOWED_PHYSICAL_GPUS", "4,5,6,7").split(",")}
    if not set(gpus) <= allowed:
        raise ValueError(f"experiment GPUs exceed allowed physical GPUs {sorted(allowed)}")
    if not isinstance(spec["swanlab_project"], str) or not spec["swanlab_project"]:
        raise ValueError("swanlab_project is required")
    for key in ("model", "selection"):
        spec[key] = str((path.parent / spec[key]).resolve())
        if not Path(spec[key]).is_file():
            raise FileNotFoundError(spec[key])
    return spec


def experiment_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New artifact series directory")
    parser.add_argument("--swanlab-group")
    parser.add_argument("--swanlab-tag", action="append", default=[])
    parser.add_argument("--swanlab-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--plan-only", action="store_true", help="Validate configs and record commands without execution")
    return parser


def run_experiments(args, load, build):
    """Snapshot selected experiments and execute stage-owned commands in order."""
    specs = [load(path) for path in args.experiments]
    names = [s["name"] for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("selected experiments need distinct names")
    if len(specs) > 1 and not args.swanlab_group:
        raise ValueError("multiple experiments require an explicit swanlab-group")
    if args.output_dir.exists():
        raise FileExistsError("use a new series directory; resume an existing run with its train entry")
    output = args.output_dir.resolve()
    # Build and validate all commands before making output directories.
    stages, snapshots, reports = [], [], []
    for path, original in zip(args.experiments, specs, strict=True):
        spec = dict(original)
        destination = output / "plan" / spec["name"]
        for key in ("model", "selection"):
            source = Path(spec[key])
            target = destination / f"{key}.json"
            snapshots.append((target, source.read_text()))
            spec[key] = str(target)
        snapshots.append((destination / "experiment.json", json.dumps(
            spec | {"model": "model.json", "selection": "selection.json"}, indent=2) + "\n"))
        jobs, entries = build(spec, original, output, args)
        stages.extend(jobs)
        reports.extend(entries)
    plan = output / "plan"
    plan.mkdir(parents=True)
    for path, content in snapshots:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (plan / "run.json").write_text(json.dumps({
        "experiments": [str(p.resolve()) for p in args.experiments],
        "group": args.swanlab_group, "tags": args.swanlab_tag, "mode": args.swanlab_mode,
    }, indent=2) + "\n")
    (plan / "reports.json").write_text(json.dumps(reports, indent=2) + "\n")
    execution = ExperimentExecution(plan, sorted({g for s in specs for g in s["gpus"]}))
    try:
        if args.plan_only:
            commands = [{"name": name, "argv": argv, "CUDA_VISIBLE_DEVICES": devices}
                        for jobs in stages for name, argv, devices in jobs]
            (plan / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            execution.status.update(status="planned", finished=now())
            execution.save_status()
        else:
            for jobs in stages:
                execution.stage(jobs)
            execution.finish()
    except BaseException as error:
        execution.finish(error)
        raise

    return specs
