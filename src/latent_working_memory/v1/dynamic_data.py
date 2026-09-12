"""Deterministic SQuAD text construction and micro-epoch scheduling."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import random


@dataclass(frozen=True)
class DynamicConfig:
    capacities: tuple[int, ...] = (64, 128, 256, 512, 1024)
    ratios: tuple[int, ...] = (2, 4, 8)
    micro_epochs_per_capacity: int = 1
    samples_per_micro_epoch: int = 100
    stage_ends: tuple[float, ...] = (0.3, 0.7, 1.0)
    ratio_weights: tuple[tuple[float, ...], ...] = (
        (0.6, 0.3, 0.1),
        (0.3, 0.4, 0.3),
        (0.1, 0.3, 0.6),
    )
    new_count: int = 1
    history_count: int = 1
    max_visits: int = 2
    bptt_unit: str = "tokens"
    bptt_span: int = 0
    batch_size: int = 2
    gradient_checkpointing: bool = True
    learning_rate: float = 0.00003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    generation_tokens: int = 64
    seed: int = 42

    def __post_init__(self):
        # JSON arrays have one canonical in-memory representation.
        for name in ("capacities", "ratios", "stage_ends"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "ratio_weights", tuple(tuple(w) for w in self.ratio_weights))
        for name in ("capacities", "ratios"):
            values = getattr(self, name)
            if (
                not values
                or len(set(values)) != len(values)
                or any(type(v) is not int or v <= 0 for v in values)
            ):
                raise ValueError(f"{name} must contain unique positive integers")
        for name in (
            "micro_epochs_per_capacity",
            "samples_per_micro_epoch",
            "batch_size",
            "max_visits",
            "generation_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("new_count", "history_count", "bptt_span", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.samples_per_micro_epoch < self.batch_size:
            raise ValueError("samples_per_micro_epoch must contain a complete batch")
        if self.new_count + self.history_count == 0:
            raise ValueError("empty reading policy")
        if self.bptt_unit not in {"tokens", "updates"}:
            raise ValueError("bptt_unit must be tokens or updates")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if (
            not self.stage_ends
            or self.stage_ends[-1] != 1
            or any(not math.isfinite(v) or v <= 0 for v in self.stage_ends)
            or tuple(sorted(set(self.stage_ends))) != self.stage_ends
            or len(self.ratio_weights) != len(self.stage_ends)
            or any(
                len(w) != len(self.ratios)
                or any(not math.isfinite(v) or v < 0 for v in w)
                or not math.isclose(sum(w), 1.0)
                for w in self.ratio_weights
            )
        ):
            raise ValueError("invalid compression curriculum stages or weights")
        if not (
            0 < self.learning_rate < float("inf")
            and 0 < self.gradient_clip < float("inf")
            and 0 <= self.weight_decay < float("inf")
        ):
            raise ValueError("invalid optimizer parameters")

    @property
    def micro_epochs_per_epoch(self):
        return len(self.capacities) * self.micro_epochs_per_capacity

    @property
    def steps_per_micro_epoch(self):
        return self.samples_per_micro_epoch // self.batch_size

    def capacity_order(self, epoch):
        order = list(self.capacities) * self.micro_epochs_per_capacity
        random.Random(f"{self.seed}:capacity:{epoch}").shuffle(order)
        return order

    def weights(self, epoch, epochs):
        return next(
            weights
            for end, weights in zip(self.stage_ends, self.ratio_weights, strict=True)
            if epoch / max(epochs - 1, 1) <= end
        )


def allocate_counts(total, weights):
    """Largest-remainder allocation; ties follow the configured ratio order."""
    raw = [total * w for w in weights]
    counts = [math.floor(n) for n in raw]
    order = sorted(range(len(weights)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return counts


def text_bounds(capacity, ratio):
    return (9 * ratio * capacity + 9) // 10, 3 * ratio * capacity // 2


def write_boundaries(episode, capacity):
    """One initial compression strictly above 1.5K, followed by paragraph updates."""
    initial = next((i for i, end in enumerate(episode.write_ends) if 2 * end > 3 * capacity), None)
    if initial is None or initial == len(episode.write_ends) - 1:
        raise ValueError("text needs initialization above 1.5K and a subsequent update")
    return episode.write_ends[initial:]


@dataclass(frozen=True)
class TrainingText:
    document_id: str
    paragraph_start: int
    paragraph_end: int
    input_tokens: int
    initial_tokens: int
    updates: int
    ratio: int

    def episode(self, data):
        return data.episode(
            self.document_id,
            paragraph_count=self.paragraph_end - self.paragraph_start,
            paragraph_start=self.paragraph_start,
        )


class DynamicTextSampler:
    def __init__(self, data, recipe, write_context_tokens):
        self.data, self.recipe = data, recipe
        self.write_context_tokens = write_context_tokens
        self.pools = {}
        self.candidate_counts = {}

    def pool(self, split, capacity, ratio):
        key = split, capacity, ratio
        if key in self.pools:
            return self.pools[key]
        lower, upper = text_bounds(capacity, ratio)
        candidates = [
            doc
            for doc, record in self.data.records.items()
            if record["split"] == split and record["input_tokens"] > upper
        ]
        texts = []
        for doc in candidates:
            lengths = self.data.records[doc]["paragraph_tokens"]
            start, length = 0, 0
            for i, size in enumerate(lengths):
                if length and length + size > upper:
                    self._append(texts, doc, start, i, capacity, ratio, lower)
                    start, length = i, 0
                if size > upper:
                    start, length = i + 1, 0
                else:
                    length += size
            if length:
                self._append(texts, doc, start, len(lengths), capacity, ratio, lower)
        self.pools[key] = texts
        self.candidate_counts[key] = len(candidates)
        return texts

    def _append(self, texts, doc, start, end, capacity, ratio, lower):
        lengths = self.data.records[doc]["paragraph_tokens"][start:end]
        total = sum(lengths)
        if total < lower:
            return
        cumulative, initial_index = 0, None
        for i, length in enumerate(lengths):
            cumulative += length
            if 2 * cumulative > 3 * capacity:
                initial_index = i
                break
        if initial_index is None or initial_index == len(lengths) - 1:
            return
        if max([cumulative, *lengths[initial_index + 1 :]]) + 1 > self.write_context_tokens:
            return
        paragraphs = self.data.articles[doc]["paragraphs"][start:end]
        initial_count = initial_index + 1
        has_new = self.recipe.new_count and any(p["qas"] for p in paragraphs[initial_count:])
        has_history = self.recipe.history_count and any(p["qas"] for p in paragraphs[:-1])
        if not (has_new or has_history):
            return
        texts.append(
            TrainingText(doc, start, end, total, cumulative, len(lengths) - initial_count, ratio)
        )

    def select(self, split, capacity, counts, seed, independent_documents=False):
        """Select exact quotas without overlapping source spans across ratios."""
        rng = random.Random(seed)
        used = defaultdict(list)
        selected = []
        pools = [self.pool(split, capacity, r) for r in self.recipe.ratios]
        # Allocate scarce ratios first; shuffle articles and retain their paragraph order.
        order = sorted(
            (i for i, n in enumerate(counts) if n), key=lambda i: len(pools[i]) / counts[i]
        )
        for i in order:
            by_document = defaultdict(list)
            for text in pools[i]:
                by_document[text.document_id].append(text)
            documents = list(by_document)
            rng.shuffle(documents)
            candidates = [t for doc in documents for t in by_document[doc]]
            if independent_documents:
                # Prefer one text per source before using another text from that source.
                candidates.sort(key=lambda t: by_document[t.document_id].index(t))
            chosen = 0
            for text in candidates:
                spans = used[text.document_id]
                if any(text.paragraph_start < b and a < text.paragraph_end for a, b in spans):
                    continue
                selected.append(text)
                spans.append((text.paragraph_start, text.paragraph_end))
                chosen += 1
                if chosen == counts[i]:
                    break
            if chosen != counts[i]:
                raise ValueError(
                    f"insufficient non-overlapping {split} texts: K={capacity}, "
                    f"r={self.recipe.ratios[i]}, requested={counts[i]}, available={chosen}"
                )
        rng.shuffle(selected)
        return selected

    def micro_epoch(self, epoch, micro_epoch, epochs):
        capacity = self.recipe.capacity_order(epoch)[micro_epoch]
        counts = allocate_counts(
            self.recipe.samples_per_micro_epoch, self.recipe.weights(epoch, epochs)
        )
        texts = self.select(
            "train", capacity, counts, f"{self.recipe.seed}:texts:{epoch}:{micro_epoch}"
        )
        used = texts[: self.recipe.steps_per_micro_epoch * self.recipe.batch_size]
        report = {
            "epoch": epoch,
            "micro_epoch": micro_epoch,
            "capacity": capacity,
            "requested_counts": dict(zip(self.recipe.ratios, counts, strict=True)),
            "used_counts": dict(Counter(t.ratio for t in used)),
            "selected_samples": len(texts),
            "used_samples": len(used),
            "dropped_samples": len(texts) - len(used),
            "ratios": {
                str(r): {
                    "candidate_articles": self.candidate_counts["train", capacity, r],
                    "valid_texts": len(self.pool("train", capacity, r)),
                }
                for r in self.recipe.ratios
            },
            "texts": [asdict(t) for t in texts],
        }
        return capacity, used, report

    def evaluation_texts(self, split, count_per_ratio):
        panel = {}
        for capacity in self.recipe.capacities:
            texts = self.select(
                split,
                capacity,
                [count_per_ratio] * len(self.recipe.ratios),
                f"{self.recipe.seed}:panel:{split}:{capacity}",
                independent_documents=True,
            )
            if len({t.document_id for t in texts}) < 2:
                raise ValueError(f"evaluation K={capacity} requires two independent documents")
            panel[capacity] = texts
        return panel
