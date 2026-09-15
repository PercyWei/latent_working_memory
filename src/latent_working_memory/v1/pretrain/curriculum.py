"""Epoch distributions and exact maximal, without-replacement quotas."""

from fractions import Fraction
import math


def normalized(weights):
    values = {k: Fraction(str(v)) for k, v in weights.items()}
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def validate_schedule(schedule, keys, tasks=False):
    if schedule is None:
        return
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("a schedule must be null or a nonempty list starting at epoch 1")
    previous = 0
    for point in schedule:
        if not isinstance(point, dict):
            raise ValueError("schedule nodes must be mappings")
        fields = set(point)
        allowed_fields = {"epoch", "weights", "tasks"} if tasks else {"epoch", "weights"}
        if not {"epoch", "weights"} <= fields or not fields <= allowed_fields:
            raise ValueError("invalid schedule node fields")
        epoch = point["epoch"]
        if type(epoch) is not int or epoch <= previous or (previous == 0 and epoch != 1):
            raise ValueError("schedule epochs must start at 1 and strictly increase")
        selected = point.get("tasks", list(keys))
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(k, str) or k not in keys for k in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("tasks must be a nonempty set of allowed task names")
        weights = point["weights"]
        if weights is not None and (
            not isinstance(weights, dict)
            or set(weights) != set(selected)
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in weights.values()
            )
            or not sum(weights.values())
        ):
            raise ValueError("schedule weights must cover the configured cells and be nonnegative")
        previous = epoch


def schedule_point(schedule, epoch):
    if schedule is None:
        return None
    point = schedule[0]
    for candidate in schedule[1:]:
        if candidate["epoch"] > epoch:
            break
        point = candidate
    return point


def distribution_at(schedule, epoch, interpolate=False):
    if schedule is None:
        return None
    left = schedule[0]
    for right in schedule[1:]:
        if epoch < right["epoch"]:
            if not interpolate or left["weights"] is None or right["weights"] is None:
                break
            progress = Fraction(epoch - left["epoch"], right["epoch"] - left["epoch"])
            a, b = normalized(left["weights"]), normalized(right["weights"])
            return {k: a[k] * (1 - progress) + b[k] * progress for k in a}
        left = right
    return None if left["weights"] is None else normalized(left["weights"])


def validate_curriculum(training, sources, bounds, max_input_tokens=None):
    required = {"source_schedule", "task_schedule", "length_schedule"}
    if not isinstance(training, dict) or not required <= set(training) <= required | {"input_tokens"}:
        raise ValueError("training requires source, task and length schedules")
    if "input_tokens" in training:
        limits = training["input_tokens"]
        if (
            not isinstance(limits, dict)
            or set(limits) != {"min", "max"}
            or any(type(v) is not int or v <= 0 for v in limits.values())
            or limits["min"] > limits["max"]
            or (max_input_tokens is not None and limits["max"] > max_input_tokens)
        ):
            raise ValueError("input_tokens requires 0 < min <= max <= model max_input_tokens")
    validate_schedule(training["source_schedule"], sources)
    validate_schedule(training["task_schedule"], ("ae", "continuation"), tasks=True)
    validate_schedule(training["length_schedule"], [str(b) for b in bounds])


def tasks_at(training, epoch):
    point = schedule_point(training["task_schedule"], epoch)
    return ("ae", "continuation") if point is None else point.get("tasks", ("ae", "continuation"))


def epoch_distribution(training, epoch):
    # None in a cell key pools that entire dimension without imposing its proportions.
    source = distribution_at(training["source_schedule"], epoch)
    task = distribution_at(training["task_schedule"], epoch)
    length = distribution_at(training["length_schedule"], epoch, interpolate=True)
    return {
        (s, t, int(b) if b is not None else None): ps * pt * pl
        for s, ps in (source if source is not None else {None: Fraction(1)}).items()
        for t, pt in (task if task is not None else {None: Fraction(1)}).items()
        for b, pl in (length if length is not None else {None: Fraction(1)}).items()
        if ps * pt * pl > 0
    }


def maximal_quotas(available, probabilities, batch_size, max_samples_per_epoch=None):
    if max_samples_per_epoch is not None and (
        type(max_samples_per_epoch) is not int or max_samples_per_epoch <= 0
    ):
        raise ValueError("max_samples_per_epoch must be a positive integer or null")
    unit = math.lcm(batch_size, *(p.denominator for p in probabilities.values()))
    bounds = {k: Fraction(available.get(k, 0)) / p for k, p in probabilities.items()}
    maximum = min(bounds.values())
    if max_samples_per_epoch is not None:
        maximum = min(maximum, max_samples_per_epoch)
    total = int(maximum) // unit * unit
    if total == 0:
        raise ValueError(
            f"no complete epoch satisfies exact proportions and batch size; available={available}"
        )
    quotas = {k: int(total * p) for k, p in probabilities.items()}
    bottlenecks = [k for k, n in bounds.items() if n == maximum]
    return quotas, bottlenecks, unit
