"""Epoch distributions and exact maximal, without-replacement quotas."""

from fractions import Fraction
import math


def normalized(weights):
    values = {k: Fraction(str(v)) for k, v in weights.items()}
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def validate_schedule(schedule, keys):
    if not isinstance(schedule, list) or not schedule or schedule[0]["epoch"] != 1:
        raise ValueError("a schedule must start at epoch 1")
    previous = 0
    for point in schedule:
        if (
            set(point) != {"epoch", "weights"}
            or type(point["epoch"]) is not int
            or point["epoch"] <= previous
        ):
            raise ValueError("schedule epochs must be strictly increasing integers")
        weights = point["weights"]
        if (
            set(weights) != set(keys)
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in weights.values()
            )
            or not sum(weights.values())
        ):
            raise ValueError("schedule weights must cover the configured cells and be nonnegative")
        previous = point["epoch"]


def distribution_at(schedule, epoch, interpolate=False):
    left = schedule[0]
    for right in schedule[1:]:
        if epoch < right["epoch"]:
            if not interpolate:
                break
            progress = Fraction(epoch - left["epoch"], right["epoch"] - left["epoch"])
            a, b = normalized(left["weights"]), normalized(right["weights"])
            return {k: a[k] * (1 - progress) + b[k] * progress for k in a}
        left = right
    return normalized(left["weights"])


def validate_curriculum(training, sources, bounds):
    if set(training) != {"source_schedule", "task_schedule", "length_schedule"}:
        raise ValueError("training requires source, task and length schedules")
    validate_schedule(training["source_schedule"], sources)
    validate_schedule(training["task_schedule"], ("ae", "continuation"))
    validate_schedule(training["length_schedule"], [str(b) for b in bounds])


def epoch_distribution(training, epoch):
    source = distribution_at(training["source_schedule"], epoch)
    task = distribution_at(training["task_schedule"], epoch)
    length = distribution_at(training["length_schedule"], epoch, interpolate=True)
    return {
        (s, t, int(b)): ps * pt * pl
        for s, ps in source.items()
        for t, pt in task.items()
        for b, pl in length.items()
        if ps * pt * pl > 0
    }


def maximal_quotas(available, probabilities, batch_size):
    unit = math.lcm(batch_size, *(p.denominator for p in probabilities.values()))
    bounds = {k: Fraction(available.get(k, 0)) / p for k, p in probabilities.items()}
    maximum = min(bounds.values())
    total = int(maximum) // unit * unit
    if total == 0:
        raise ValueError(
            f"no complete epoch satisfies exact proportions and batch size; available={available}"
        )
    quotas = {k: int(total * p) for k, p in probabilities.items()}
    bottlenecks = [k for k, n in bounds.items() if n == maximum]
    return quotas, bottlenecks, unit
