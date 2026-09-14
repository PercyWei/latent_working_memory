"""CPU-only synthetic text benchmark using the supplied real tokenizer, without token caches."""

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import time

from transformers import AutoTokenizer

from latent_working_memory.data_preparation.pretrain.text_samples import TextSample, input_text_key
from latent_working_memory.v1.config import ExperimentConfig
from latent_working_memory.v1.pretrain.prepared_data import eligible_input_length
from latent_working_memory.v1.pretrain.sampling import read_tokens
from latent_working_memory.v1.pretrain.tokenization import TokenizationPool, text_blocks


def scalar_inspection(tokenizer, config, blocks):
    prompts = {
        t: len(tokenizer.encode(p, add_special_tokens=False))
        for t, p in (("ae", config.ae_prompt), ("continuation", config.lm_prompt))
    }
    output = []
    for block in blocks:
        for offset, line in block:
            sample = TextSample(**json.loads(line))
            size = eligible_input_length(sample, tokenizer, config, False, prompts)
            target = sample.text if sample.task == "ae" else sample.continuation
            content = (
                hashlib.blake2b(
                    json.dumps(
                        (sample.task, input_text_key(sample.text), " ".join(target.split()))
                    ).encode()
                ).hexdigest()
                if size is not None
                else None
            )
            output.append(
                (
                    offset,
                    sample.sample_id,
                    sample.document_id,
                    sample.source_id,
                    sample.dedup_cluster,
                    sample.task,
                    sample.boundary_method,
                    size,
                    content,
                )
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    config = replace(
        ExperimentConfig(),
        model_name_or_path=str(args.tokenizer),
        max_input_tokens=2048,
        max_continuation_tokens=2048,
    )
    paragraphs = [
        "The laboratory recorded the temperature of every container before the morning shift. ",
        "Each observation includes a timestamp, an instrument number, and a short written explanation. ",
        "研究人员记录了实验条件，并且核对每次测量的数值。 ",
        "Names such as café, naïve and 東京 require consistent Unicode handling.\n",
        "The next section discusses how the experiment changed after adjusting the sampling interval. ",
    ]
    results = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "tokenizer": str(args.tokenizer),
        "samples": args.samples,
        "workload": "synthetic mixed English/Chinese/Unicode AE and LM; no real training data",
        "index": [],
        "batch": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="lwm-tokenization-") as directory:
        path = Path(directory) / "train.jsonl"
        offsets = []
        with path.open("wb") as f:
            for i in range(args.samples):
                x = "".join(
                    paragraphs[(i + j) % len(paragraphs)] for j in range(12 + i % 48)
                ) + str(i)
                y = (
                    "".join(paragraphs[(i + j + 2) % len(paragraphs)] for j in range(8 + i % 32))
                    if i % 2
                    else None
                )
                sample = TextSample(
                    str(i),
                    str(i),
                    str(i),
                    str(i),
                    "continuation" if y else "ae",
                    x,
                    y,
                    [0, len(x)],
                    [len(x), len(x) + len(y)] if y else None,
                    "pysbd_conservative",
                    1,
                    1,
                )
                offsets.append(f.tell())
                f.write((json.dumps(sample.to_record(), ensure_ascii=False) + "\n").encode())
        start = time.perf_counter()
        expected = scalar_inspection(tokenizer, config, text_blocks(path, 256))
        seconds = time.perf_counter() - start
        results["index"].append(
            {
                "mode": "original_scalar",
                "seconds": seconds,
                "samples_per_second": args.samples / seconds,
            }
        )
        batches = [
            [(path, offset, "semantic") for offset in offsets[i : i + 8]]
            for i in range(0, min(args.samples, 2048), 8)
        ]
        # Original on-demand path: encode X, target and prompt for each sample.
        start = time.perf_counter()
        expected_batches = []
        for batch in batches:
            encoded = []
            for _, offset, variant in batch:
                with path.open("rb") as f:
                    f.seek(offset)
                    sample = TextSample(**json.loads(f.readline()))
                episode = sample.to_episode(tokenizer, config, variant)
                ae, lm = read_tokens(episode, tokenizer)
                encoded.append((episode, (ae or lm).prompt_ids, (ae or lm).target_ids))
            expected_batches.append(encoded)
        results["batch"].append(
            {
                "mode": "original_scalar",
                "seconds": time.perf_counter() - start,
                "batches": len(batches),
            }
        )
        for workers in (0, 2, 4, 8):
            start = time.perf_counter()
            with TokenizationPool(tokenizer, config, workers) as pool:
                # Force all configured workers to start; report initialization separately.
                warm = [pool.submit_batch(batches[0]) for _ in range(max(1, 2 * workers))]
                for f in warm:
                    f.result()
                startup = time.perf_counter() - start
                durations = []
                for _ in range(2):
                    start = time.perf_counter()
                    actual = [
                        row for rows in pool.inspect_blocks(text_blocks(path, 256)) for row in rows
                    ]
                    durations.append(time.perf_counter() - start)
                    assert actual == expected
                results["index"].append(
                    {
                        "mode": f"batch256_workers{workers}",
                        "startup_seconds": startup,
                        "seconds": min(durations),
                        "trials_seconds": durations,
                        "samples_per_second": args.samples / min(durations),
                    }
                )
                start = time.perf_counter()
                pending = deque()
                source = iter(batches)
                for _ in range(2):
                    pending.append(pool.submit_batch(next(source)))
                for expected_batch in expected_batches:
                    batch = pending.popleft().result()
                    assert [
                        (s.episode, s.prompt_ids, s.target_ids) for s in batch
                    ] == expected_batch
                    item = next(source, None)
                    if item is not None:
                        pending.append(pool.submit_batch(item))
                results["batch"].append(
                    {
                        "mode": f"batch8_workers{workers}_prefetch2",
                        "seconds": time.perf_counter() - start,
                        "batches": len(batches),
                    }
                )
            (args.output_dir / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results["index"][-1]), flush=True)
            print(json.dumps(results["batch"][-1]), flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
