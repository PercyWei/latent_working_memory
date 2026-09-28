"""Download the FineWeb 10BT source files directly, without an HTTP proxy."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def download_file(directory: Path, endpoint: str, revision: str, index: int) -> None:
    name = f"{index:03d}_00000.parquet"
    destination = directory / name
    if destination.exists():
        print(f"Already downloaded: {name}", flush=True)
        return
    partial = directory / f"{name}.part"
    url = f"{endpoint}/datasets/HuggingFaceFW/fineweb/resolve/{revision}/sample/10BT/{name}"
    print(f"Downloading: {name}", flush=True)
    subprocess.run(
        [
            "curl",
            "--noproxy",
            "*",
            "--proxy",
            "",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--retry",
            "3",
            "--retry-all-errors",
            "--retry-delay",
            "60",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(destination)
    print(f"Downloaded: {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(
            pool.map(
                lambda index: download_file(args.directory, args.endpoint, args.revision, index),
                range(15),
            )
        )


if __name__ == "__main__":
    main()
