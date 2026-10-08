"""共享分段的整数预算、字符位置与持久化布局契约。"""

import copy
import random

import pytest

from latent_working_memory.data_preparation.segmentation import (
    SegmentationConfig,
    sample_windows,
    segment_lengths,
    validate_text_layout,
)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"capacity": True}, "capacity"),
        ({"continuation_tokens": -1}, "continuation_tokens"),
        ({"continuation_tokens": True}, "continuation_tokens"),
        ({"min_segment_ratio": 3, "max_segment_ratio": 2}, "ratios"),
        ({"capacity": 1, "min_segment_ratio": 1.1, "max_segment_ratio": 1.9}, "integer"),
        ({"min_segments": 1}, "segment counts"),
        ({"content_reserve_ratio": 0.5}, "content_reserve_ratio"),
    ],
)
def test_invalid_segmentation_parameters(changes, match):
    with pytest.raises(ValueError, match=match):
        SegmentationConfig(**changes)


def test_fractional_budget_rounding_and_minimum_window_fit():
    config = SegmentationConfig(
        capacity=7,
        min_segment_ratio=0.5,
        max_segment_ratio=1.25,
        min_segments=2,
        max_segments=4,
        continuation_tokens=3,
        content_reserve_ratio=1.3,
    )
    assert (config.min_segment_tokens, config.max_segment_tokens) == (4, 8)
    assert (config.min_segment_chars, config.max_segment_chars) == (21, 42)
    assert config.continuation_chars == 16
    assert config.minimum_window_chars == 58
    assert config.candidate_chars([4, 4]) == 58
    assert list(sample_windows(58, "doc", 17, config)) == [(0, [4, 4], 58)]
    assert list(sample_windows(57, "doc", 17, config)) == []
    for seed in range(30):
        windows = list(sample_windows(201, "doc", seed, config))
        assert windows == list(sample_windows(201, "doc", seed, config))
        previous_end = 0
        for start, parts, candidate_end in windows:
            assert start == previous_end
            assert 2 <= len(parts) <= 4
            assert all(4 <= part <= 8 for part in parts)
            body_chars = sum(config.reserved_chars(part) for part in parts)
            assert 0 <= start < start + body_chars < candidate_end <= 201
            assert candidate_end - start == config.candidate_chars(parts)
            assert candidate_end - start - body_chars == config.continuation_chars
            previous_end = candidate_end
        assert 201 - previous_end < config.minimum_window_chars


@pytest.mark.parametrize("continuation_tokens,minimum", [(0, 10), (1, 15)])
def test_each_segment_and_continuation_round_up_independently(continuation_tokens, minimum):
    config = SegmentationConfig(
        capacity=1,
        min_segment_ratio=1,
        max_segment_ratio=1,
        min_segments=2,
        max_segments=2,
        continuation_tokens=continuation_tokens,
        content_reserve_ratio=1.1,
    )
    assert config.reserved_chars(1) == 5
    assert config.reserved_chars(15) == 66  # Avoid floating-point 66.00000000000001.
    assert config.minimum_window_chars == config.candidate_chars([1, 1]) == minimum
    # Rounding only the total loses a character in both the Q=0 and Q=1 cases.
    assert config.reserved_chars(2 + continuation_tokens) == minimum - 1
    assert list(sample_windows(minimum, "doc", 17, config)) == [(0, [1, 1], minimum)]
    assert list(sample_windows(minimum - 1, "doc", 17, config)) == []


@pytest.mark.parametrize("continuation_tokens", [0, 2])
def test_multiple_windows_cover_full_budget_without_overlap_and_discard_short_tail(
    continuation_tokens,
):
    config = SegmentationConfig(
        capacity=2,
        min_segment_ratio=1,
        max_segment_ratio=1,
        min_segments=2,
        max_segments=2,
        continuation_tokens=continuation_tokens,
        content_reserve_ratio=1.3,
    )
    window_chars = config.minimum_window_chars
    text_length = window_chars * 4 + window_chars - 1
    windows = list(sample_windows(text_length, "doc", 17, config))
    assert windows == [
        (index * window_chars, [2, 2], (index + 1) * window_chars) for index in range(4)
    ]
    for start, parts, candidate_end in windows:
        body_end = start + sum(config.reserved_chars(part) for part in parts)
        assert candidate_end - body_end == config.continuation_chars
    assert text_length - windows[-1][2] == window_chars - 1


def test_sampling_uses_repeatable_document_rng_and_changes_with_seed_or_document():
    config = SegmentationConfig(
        capacity=8,
        min_segments=2,
        max_segments=4,
        continuation_tokens=3,
        content_reserve_ratio=1.3,
    )
    windows = list(sample_windows(4000, "doc", 17, config))
    assert len(windows) > 1
    assert windows == list(sample_windows(4000, "doc", 17, config))
    assert windows != list(sample_windows(4000, "doc", 18, config))
    assert windows != list(sample_windows(4000, "other", 17, config))
    assert list(sample_windows(0, "doc", 17, config)) == []


def test_sampling_reserves_each_remaining_segments_rounded_minimum():
    config = SegmentationConfig(
        capacity=1,
        min_segment_ratio=1,
        max_segment_ratio=3,
        min_segments=2,
        max_segments=3,
        content_reserve_ratio=1.1,
    )
    # Two minimum segments use ten characters; thirteen cannot fit one 2-token
    # segment (nine characters) plus one minimum segment (five characters).
    for seed in range(30):
        assert segment_lengths(2, 13, config, random.Random(seed)) == [1, 1]
        [(start, parts, candidate_end)] = sample_windows(14, "doc", seed, config)
        assert len(parts) == 2
        assert 0 <= start < candidate_end <= 14
        assert sum(config.reserved_chars(part) for part in parts) == candidate_end - start


def test_segment_length_validation_uses_the_reserved_character_grid():
    config = SegmentationConfig(
        capacity=7,
        min_segment_ratio=0.5,
        max_segment_ratio=1.25,
        content_reserve_ratio=1.3,
    )
    expected = {21, 26, 32, 37, 42}
    assert {
        characters for characters in range(50) if config.is_valid_segment_length(characters)
    } == expected
    assert all(
        config.is_valid_segment_length(config.reserved_chars(tokens)) for tokens in range(4, 9)
    )
    assert not config.is_valid_segment_length(24)  # A multiple of four need not be valid.
    assert config.is_valid_segment_length(21)  # A valid length need not be a multiple of four.


def document():
    return {
        "trajectory_id": "doc:11:27",
        "document_id": "doc",
        "dedup_cluster": "cluster",
        "split": "train",
        "source": {"file": "raw/source.parquet", "row_group": 0, "row_index": 3},
        "window_char_span": [11, 27],
        "text": "abcdefghABCDEFGH",
        "segments": [
            {"segment_id": "seg0", "char_span": [0, 8]},
            {"segment_id": "seg1", "char_span": [8, 16]},
        ],
    }


def test_layout_returns_frozen_character_boundaries_without_mutation():
    record = document()
    original = copy.deepcopy(record)
    assert validate_text_layout(record) == ("doc", {"seg0": (0, 8), "seg1": (8, 16)})
    assert record == original


@pytest.mark.parametrize("invalid", ["source", "window", "gap", "last", "segment_id", "split"])
def test_layout_rejects_invalid_provenance_and_incomplete_coverage(invalid):
    record = document()
    if invalid == "source":
        record["source"]["char_span"] = [11, 27]
    elif invalid == "window":
        record["window_char_span"][1] += 1
    elif invalid == "gap":
        record["segments"][1]["char_span"][0] += 1
    elif invalid == "last":
        record["segments"][1]["char_span"][1] -= 1
    elif invalid == "segment_id":
        record["segments"][1]["segment_id"] = "seg0"
    else:
        record["split"] = "validation"
    with pytest.raises(ValueError):
        validate_text_layout(record)
