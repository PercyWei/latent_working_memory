"""独立评估 worker：一题一行提交，重启后只处理新的分工。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .. import method
from ..execution.runtime import seed_all
from ..execution.state import read_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args(argv)
    request = json.loads((args.output_dir / "eval.json").read_text(encoding="utf-8"))
    work = json.loads((args.output_dir / "work.json").read_text(encoding="utf-8"))
    assigned = work["workers"][args.rank]
    state = read_checkpoint(args.checkpoint)
    config = state["config"]
    samples, identity, _ = method.load_samples(
        config,
        "test",
        data_dir=request["data"],
        max_samples=request["max_samples"],
    )
    if identity != request["data_identity"]:
        raise ValueError("evaluation data changed after work was assigned")
    selected = {sample.id: sample for sample in samples}
    if args.device == "cuda":
        torch.cuda.set_device(args.rank)
        device = torch.device("cuda", args.rank)
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")
    seed_all(config["seed"], device)
    model = method.build_model(config, device)
    method.load_trainable_state_dict(model, state["model"])
    del state
    model.eval()
    progress = args.output_dir / "progress"
    progress.mkdir(exist_ok=True)
    with (progress / f"worker_{args.rank}.jsonl").open("a", encoding="utf-8") as stream:
        for sample_id in assigned:
            # 样本随机性与 worker 编号、分片方式及重跑次数无关。
            digest = hashlib.sha256(f"{config['seed']}:{sample_id}".encode()).digest()
            seed_all(int.from_bytes(digest[:4], "big"), device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            record = method.predict(model, selected[sample_id], request)
            if record["id"] != sample_id:
                raise ValueError("method prediction returned another sample id")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            entry = {
                "record": record,
                "resources": {"sample_seconds": time.perf_counter() - started},
            }
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"evaluation worker={args.rank} completed={sample_id}", flush=True)


if __name__ == "__main__":
    main()
