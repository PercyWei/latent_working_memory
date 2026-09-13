"""实验脚本共用的命令记录、GPU 分配与执行状态。"""
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
            "LWM_ALLOWED_PHYSICAL_GPUS": ",".join(map(str, self.gpus)),
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
                                  "LWM_ALLOWED_PHYSICAL_GPUS": self.gpus})
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
