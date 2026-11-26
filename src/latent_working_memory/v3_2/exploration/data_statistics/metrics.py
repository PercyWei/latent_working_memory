"""Text statistics only: no dataset fields, model calls, or filesystem access."""

from bisect import bisect_left, bisect_right
import re
from statistics import fmean, pstdev

from .schema import Document, MetricConfig, Span

_WORDS = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,:/-]\d+)*%?")
_ENTITIES = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:[ \t]+(?:[A-Z][a-z]+|[A-Z]{2,}))*\b")
_NUMBERS = re.compile(r"(?<!\w)[+-]?\d+(?:[.,:/-]\d+)*(?:%|\b)")
_SENTENCE_STARTS = re.compile(r"""(?:^|[.!?]["'”’)\]}]*\s+|[\r\n]+)[\s"'“‘(\[{«]*""")


def summarize(values):
    """Linear-interpolated percentiles; missing values never become zero."""
    observed = sorted(float(value) for value in values if value is not None)
    result = {"count": len(observed), "missing": len(values) - len(observed)}
    for name in ("mean", "std", "p10", "p50", "p90"):
        result[name] = None
    if observed:
        result.update(mean=fmean(observed), std=pstdev(observed))
        for name, fraction in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9)):
            position = (len(observed) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(observed) - 1)
            result[name] = observed[lower] + (observed[upper] - observed[lower]) * (
                position - lower
            )
    return result


def _merge_overlapping(spans):
    merged = []
    for start, end in sorted(spans):
        # Adjacent annotated evidence pieces remain distinct for cut detection.
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _natural_feasible(boundaries, length, blocks, min_block_tokens, max_block_tokens):
    """Exact B-block reachability under min/max lengths, using sorted predecessors."""
    upper = max_block_tokens if max_block_tokens is not None else length
    if not blocks * min_block_tokens <= length <= blocks * upper:
        return False
    reachable = [0]
    for block in range(1, blocks + 1):
        endpoints = [length] if block == blocks else boundaries
        next_reachable = []
        for end in endpoints:
            index = bisect_left(reachable, end - upper)
            if index < len(reachable) and reachable[index] <= end - min_block_tokens:
                next_reachable.append(end)
        reachable = next_reachable
        if not reachable:
            return False
    return True


def _density_windows(text, starts, ends, size):
    length = len(starts)
    windows = []
    for start in range(0, length, size):
        end = min(start + size, length)
        if end - start >= (size + 1) // 2:
            windows.append({"start": start, "end": end, "words": 0, "entity": 0, "number": 0})
    sentence_starts = {match.end() for match in _SENTENCE_STARTS.finditer(text)}
    for kind, pattern in (("words", _WORDS), ("entity", _ENTITIES), ("number", _NUMBERS)):
        for match in pattern.finditer(text):
            if (
                kind == "entity"
                and match.start() in sentence_starts
                and len(match.group().split()) == 1
                and not match.group().isupper()
            ):
                continue
            # Assign whole mentions by their first overlapping token, never re-match a window.
            token = bisect_right(ends, match.start())
            index = token // size
            if index < len(windows):
                windows[index][kind] += 1
    for window in windows:
        for kind in ("entity", "number"):
            window[f"{kind}_density"] = (
                100 * window[kind] / window["words"] if window["words"] else None
            )
    return windows


def compute_metrics(document: Document, offsets: list[Span], config: MetricConfig):
    """Measure one final document/slice using offsets from a fast tokenizer.

    Tokenize the exact text with add_special_tokens=False. Unit spans must be
    ordered and nonoverlapping; evidence may overlap. Invalid evidence is a QA
    exclusion, while broken unit/tokenizer contracts raise normally. Density and
    evidence mapping are shared by all configurations. QA block geometry has no
    min/max constraints; consumers apply the corresponding feasibility result.
    """
    starts = [span[0] for span in offsets]
    ends = [span[1] for span in offsets]
    length = len(offsets)
    if not length or starts != sorted(starts) or ends != sorted(ends):
        raise ValueError("expected nonempty, ordered tokenizer offsets")
    if any(not 0 <= start < end <= len(document.text) for start, end in offsets):
        raise ValueError("offsets must reference text without special tokens")
    previous_end = 0
    unit_lengths = []
    for start, end in document.unit_spans:
        if not previous_end <= start < end <= len(document.text):
            raise ValueError(f"{document.doc_id}: invalid or overlapping natural units")
        unit_lengths.append(bisect_left(starts, end) - bisect_right(ends, start))
        previous_end = end
    boundaries = sorted(
        {bisect_left(starts, start) for start, _ in document.unit_spans[1:]} - {0, length}
    )
    cuts_by_blocks = {
        blocks: [index * length // blocks for index in range(1, blocks)]
        if length >= blocks
        else None
        for blocks in config.blocks
    }
    length_ranges = {
        blocks: (length // blocks, (length + blocks - 1) // blocks) for blocks in config.blocks
    }
    segmentations = []
    for blocks, minimum, maximum in config.segmentations:
        # Equal partitions have only floor(D/B) and ceil(D/B) block lengths.
        shortest, longest = length_ranges[blocks]
        segmentations.append(
            {
                "blocks": blocks,
                "min_block_tokens": minimum,
                "max_block_tokens": maximum,
                "uniform_feasible": shortest >= minimum and (maximum is None or longest <= maximum),
                "natural_feasible": _natural_feasible(boundaries, length, blocks, minimum, maximum),
            }
        )
    windows = _density_windows(document.text, starts, ends, config.window_tokens)
    units = summarize(unit_lengths)
    metrics = {
        "doc_tokens": length,
        "unit_count": len(document.unit_spans),
        "unit_tokens_p50": units["p50"],
        "unit_tokens_p90": units["p90"],
        "window_count": len(windows),
        "density_window_count": sum(window["words"] > 0 for window in windows),
        "density_covered_tokens": sum(window["end"] - window["start"] for window in windows),
    }
    for kind in ("entity", "number"):
        summary = summarize([window[f"{kind}_density"] for window in windows])
        for name in ("mean", "p10", "p90"):
            metrics[f"{kind}_density_{name}"] = summary[name]
        metrics[f"{kind}_density_cv"] = (
            summary["std"] / summary["mean"] if summary["mean"] else None
        )

    questions = []
    for qa in document.qas:
        row = {
            "qa_id": qa.qa_id,
            "evidence_granularity": qa.evidence_granularity,
            "status": qa.exclusion_reason or "retained",
            "metrics": {},
        }
        questions.append(row)
        if qa.exclusion_reason:
            continue
        if not qa.evidence_spans:
            row["status"] = "missing_evidence_annotation"
            continue
        if any(not 0 <= start < end <= len(document.text) for start, end in qa.evidence_spans):
            row["status"] = "evidence_locator_failed"
            continue
        spans = [
            (bisect_right(ends, start), bisect_left(starts, end))
            for start, end in qa.evidence_spans
        ]
        if any(start >= end for start, end in spans):
            row["status"] = "evidence_locator_failed"
            continue
        spans = _merge_overlapping(spans)
        result = {
            "evidence_span_count": len(qa.evidence_spans),
            "evidence_tokens": sum(end - start for start, end in spans),
            "evidence_extent_tokens": spans[-1][1] - spans[0][0],
        }
        row["blocks"] = []
        for blocks, cuts in cuts_by_blocks.items():
            if cuts is None:
                cut, occupied_count = None, None
            else:
                cut = any(start < point < end for start, end in spans for point in cuts)
                occupied = {
                    block
                    for start, end in spans
                    for block in range(bisect_right(cuts, start), bisect_left(cuts, end) + 1)
                }
                occupied_count = len(occupied)
            row["blocks"].append(
                {
                    "blocks": blocks,
                    "evidence_cut": cut,
                    "evidence_block_count": occupied_count,
                }
            )
        row["recent"] = [
            {
                "recent_tokens": recent,
                "tail_overlap": spans[-1][1] > max(0, length - recent),
            }
            for recent in config.recent_tokens
        ]
        groups = {True: [], False: []}
        for window in windows:
            overlaps = any(start < window["end"] and window["start"] < end for start, end in spans)
            groups[overlaps].append(window)
        for label, overlaps in (("evidence", True), ("background", False)):
            selected = groups[overlaps]
            result[f"{label}_window_count"] = len(selected)
            result[f"{label}_density_window_count"] = sum(w["words"] > 0 for w in selected)
            for kind in ("entity", "number"):
                values = [
                    w[f"{kind}_density"] for w in selected if w[f"{kind}_density"] is not None
                ]
                result[f"{kind}_{label}_density"] = fmean(values) if values else None
        for kind in ("entity", "number"):
            ev, bg = result[f"{kind}_evidence_density"], result[f"{kind}_background_density"]
            result[f"{kind}_evidence_background_difference"] = (
                ev - bg if ev is not None and bg is not None else None
            )
        row["metrics"] = result
    metrics["qa_count"] = len(questions)
    metrics["valid_qa_count"] = sum(row["status"] == "retained" for row in questions)
    return {"metrics": metrics, "qas": questions, "segmentations": segmentations}
