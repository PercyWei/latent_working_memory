from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def score_generation_records(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    rows = list(records)
    if not rows:
        raise ValueError("generation metrics require at least one record")
    predictions: list[list[str]] = []
    references: list[list[str]] = []
    weighted_loss_sum = 0.0
    loss_tokens = 0
    for row in rows:
        prediction = row.get("prediction")
        reference = row.get("reference")
        loss = row.get("loss")
        tokens = row.get("loss_tokens")
        if not isinstance(prediction, str) or not isinstance(reference, str):
            raise TypeError("prediction and reference must be strings")
        if not isinstance(loss, (int, float)) or not isinstance(tokens, int) or tokens < 1:
            raise TypeError("loss must be numeric and loss_tokens must be a positive integer")
        predictions.append(_tokenize(prediction))
        references.append(_tokenize(reference))
        weighted_loss_sum += float(loss) * tokens
        loss_tokens += tokens

    nll = weighted_loss_sum / loss_tokens
    rouge = _corpus_rouge(predictions, references)
    return {
        "samples": len(rows),
        "loss_tokens": loss_tokens,
        "token_weighted_loss": nll,
        "ppl": math.exp(nll),
        "bleu": _corpus_bleu(predictions, references),
        **rouge,
    }


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def _corpus_bleu(predictions: list[list[str]], references: list[list[str]]) -> float:
    clipped = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    prediction_length = 0
    reference_length = 0
    for prediction, reference in zip(predictions, references, strict=True):
        prediction_length += len(prediction)
        reference_length += len(reference)
        for order in range(1, 5):
            predicted = Counter(_ngrams(prediction, order))
            reference_counts = Counter(_ngrams(reference, order))
            clipped[order - 1] += sum(
                min(count, reference_counts[ngram]) for ngram, count in predicted.items()
            )
            totals[order - 1] += sum(predicted.values())
    if prediction_length == 0 or any(total == 0 for total in totals):
        return 0.0
    precisions = [match / total for match, total in zip(clipped, totals, strict=True)]
    if any(precision == 0.0 for precision in precisions):
        return 0.0
    brevity_penalty = (
        1.0
        if prediction_length >= reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / 4.0)


def _corpus_rouge(
    predictions: list[list[str]],
    references: list[list[str]],
) -> dict[str, float]:
    scores = {name: [] for name in ("rouge_1", "rouge_2", "rouge_l")}
    recalls = {name: [] for name in ("rouge_1_recall", "rouge_2_recall", "rouge_l_recall")}
    for prediction, reference in zip(predictions, references, strict=True):
        for order, name in ((1, "rouge_1"), (2, "rouge_2")):
            overlap = sum(
                (Counter(_ngrams(prediction, order)) & Counter(_ngrams(reference, order))).values()
            )
            predicted_total = max(0, len(prediction) - order + 1)
            reference_total = max(0, len(reference) - order + 1)
            precision = overlap / predicted_total if predicted_total else 0.0
            recall = overlap / reference_total if reference_total else 0.0
            scores[name].append(_f1(precision, recall))
            recalls[f"{name}_recall"].append(recall)
        lcs = _lcs_length(prediction, reference)
        precision = lcs / len(prediction) if prediction else 0.0
        recall = lcs / len(reference) if reference else 0.0
        scores["rouge_l"].append(_f1(precision, recall))
        recalls["rouge_l_recall"].append(recall)
    return {
        name: sum(values) / len(values)
        for name, values in {**scores, **recalls}.items()
    }


def _ngrams(tokens: list[str], order: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)]


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
