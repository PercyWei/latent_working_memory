"""按完整正文的实际 token 数，将所选 split 离线分档。"""

import argparse
from bisect import bisect_left
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
from zoneinfo import ZoneInfo

from transformers import AutoTokenizer

from .cli import positive_int


def _scan(path, tokenizer):
    lengths, fields = [], None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) not in ({"text"}, {"text", "qas"}):
                    raise ValueError("expected exactly text, or text and qas")
                if fields is None:
                    fields = set(row)
                elif set(row) != fields:
                    raise ValueError("all samples must use the same pretrain or post format")
                if not isinstance(row["text"], str) or not row["text"].strip():
                    raise ValueError("text must be a nonempty string")
                lengths.append(len(tokenizer.encode(row["text"], add_special_tokens=False)))
            except (ValueError, TypeError, KeyError) as error:
                error.add_note(f"{path}:{line_number}")
                raise
            if line_number % 1000 == 0:
                print(f"[prepare-length] scanned={line_number}", flush=True)
    if not lengths:
        raise ValueError("the selected split must contain at least one sample")
    return lengths


def _bins(lengths, requested, split):
    minimum, maximum = min(lengths), max(lengths)
    width = maximum - minimum + 1
    if requested > width:
        raise ValueError(
            f"length_bins={requested} exceeds the {width} integer token lengths "
            f"in [{minimum}, {maximum}]"
        )
    bounds = [(minimum + index * width // requested,
               minimum + (index + 1) * width // requested - 1)
              for index in range(requested)]
    upper = [right for _, right in bounds]
    assignments = [bisect_left(upper, length) for length in lengths]
    groups = [[] for _ in bounds]
    for length, index in zip(lengths, assignments, strict=True):
        groups[index].append(length)
    bins = [{
        "file": f"{split}/bin_{index:02d}.jsonl", "min_tokens": lower, "max_tokens": upper,
        "samples": len(values), "tokens_min": min(values, default=None),
        "tokens_max": max(values, default=None),
        "tokens_mean": sum(values) / len(values) if values else None,
    } for index, ((lower, upper), values) in enumerate(zip(bounds, groups, strict=True))]
    return bins, assignments


def prepare_data(source_dir, split, model_path, output_dir, length_bins=4):
    if type(length_bins) is not int or length_bins < 1:
        raise ValueError("length_bins must be a positive integer")
    directory = Path(source_dir).expanduser().resolve()
    source = directory / f"{split}.jsonl"
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    model_path = str(Path(model_path).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"[prepare-length] scanning complete {split} texts: {source}", flush=True)
    lengths = _scan(source, tokenizer)
    bins, assignments = _bins(lengths, length_bins, split)
    metadata = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source_file": str(source), "split": split, "tokenizer": model_path,
        "requested_length_bins": length_bins, "samples": len(lengths),
        "min_tokens": min(lengths), "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths), "bins": bins,
    }
    print(f"[prepare-length] samples={len(lengths)} "
          f"tokens=[{metadata['min_tokens']}, {metadata['max_tokens']}] "
          f"mean={metadata['mean_tokens']:.2f} bins={len(bins)}/{length_bins}", flush=True)
    for entry in bins:
        print(f"[prepare-length] {entry['file']} "
              f"range=[{entry['min_tokens']}, {entry['max_tokens']}] "
              f"samples={entry['samples']} "
              f"actual=[{entry['tokens_min']}, {entry['tokens_max']}] "
              f"mean={entry['tokens_mean']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.preparing-", dir=output.parent))
    try:
        (staging / split).mkdir()
        with ExitStack() as stack:
            destinations = [stack.enter_context((staging / entry["file"]).open("w", encoding="utf-8"))
                            for entry in bins]
            with source.open(encoding="utf-8") as stream:
                for line, index in zip(stream, assignments, strict=True):
                    destinations[index].write(line if line.endswith("\n") else line + "\n")
        for other in sorted(directory.glob("*.jsonl")):
            if other != source:
                shutil.copyfile(other, staging / other.name)
        (staging / f"preparation-{split}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"preparation complete: {output}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", required=True, help="directory containing prepared split JSONL files")
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True,
                        help="split to bin; other split JSONL files are copied unchanged")
    parser.add_argument("--model-path", required=True, help="tokenizer path used by training")
    parser.add_argument("--output-dir", required=True, help="new dataset directory")
    parser.add_argument("--length-bins", type=positive_int, default=4,
                        help="number of equal-width token-length bins, default 4")
    args = parser.parse_args(argv)
    return prepare_data(args.prepare_data, args.split, args.model_path, args.output_dir,
                        args.length_bins)


if __name__ == "__main__":
    main()
