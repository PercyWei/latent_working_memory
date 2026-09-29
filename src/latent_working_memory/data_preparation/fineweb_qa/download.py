"""Download the fixed FineWeb sample-10BT Parquet files without a proxy."""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DATASET = "HuggingFaceFW/fineweb"
SOURCE_DIR = "sample/10BT"
REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
SHARD_NAMES = tuple(f"{index:03d}_00000.parquet" for index in range(15))
MAX_WORKERS = 3
MAX_ATTEMPTS = 8


def download_shard(name: str, directory: Path, endpoint: str, revision: str) -> bool:
    """Resume one .part file and rename it only after curl succeeds."""
    final = directory / name
    if final.exists():
        print(f"{name}: already present", flush=True)
        return False
    partial = directory / f"{name}.part"
    url = f"{endpoint.rstrip('/')}/datasets/{DATASET}/resolve/{revision}/{SOURCE_DIR}/{name}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if final.exists():
            print(f"{name}: already present", flush=True)
            return False
        print(f"{name}: downloading (attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--noproxy",
                "*",
                "--proxy",
                "",
                "--connect-timeout",
                "30",
                "--speed-limit",
                "1024",
                "--speed-time",
                "60",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            if final.exists():
                raise RuntimeError(
                    f"{name}: final file appeared during transfer; keeping {partial}"
                )
            partial.rename(final)
            print(f"{name}: ready", flush=True)
            return True
        detail = result.stderr.strip() or f"curl exited with code {result.returncode}"
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(
                f"{name}: transfer failed after {MAX_ATTEMPTS} attempts; "
                f"partial file kept at {partial}: {detail}"
            )
        print(f"{name}: retrying after {detail}", flush=True)
        time.sleep(min(5 * 2 ** (attempt - 1), 60))
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args(argv)
    if args.revision != REVISION:
        parser.error(f"--revision must be the fixed FineWeb commit {REVISION}")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    if not args.endpoint.startswith("https://"):
        parser.error("--endpoint must use HTTPS")

    try:
        args.directory.mkdir(parents=True, exist_ok=True)
        print(f"FineWeb {SOURCE_DIR}: {len(SHARD_NAMES)} fixed Parquet files", flush=True)
        downloaded = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(SHARD_NAMES), args.workers):
                futures = [
                    pool.submit(download_shard, name, args.directory, args.endpoint, args.revision)
                    for name in SHARD_NAMES[start : start + args.workers]
                ]
                for future in as_completed(futures):
                    downloaded += future.result()
        print(
            f"All {len(SHARD_NAMES)} FineWeb files ready; {downloaded} downloaded in this run",
            flush=True,
        )
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"FineWeb download failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
