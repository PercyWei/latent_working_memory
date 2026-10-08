from collections import Counter
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.fineweb_factqa import analyze_lengths
from latent_working_memory.data_preparation.fineweb_source import write_used_sources


def test_weighted_distribution_matches_expanded_lengths_and_threshold_boundaries():
    values = [64, 1024, 1024, 4096, 18431, 18432, 24576, 40960, 70000]
    result = analyze_lengths.summarize_lengths(Counter(values))
    assert result["documents"] == len(values)
    assert result["mean_chars"] == np.mean(values)
    for name, value in result["quantiles_chars"].items():
        assert np.isclose(value, np.quantile(values, float(name[1:]) / 100))
    assert sum(row["documents"] for row in result["bins"]) == len(values)
    counts = {row["minimum_chars"]: row["documents"] for row in result["thresholds"]}
    assert counts[18432] == 4
    assert counts[24576] == 3


def test_census_excludes_used_identity_url_and_normalized_text_without_deduping_remainder(tmp_path):
    original = {"id": "old", "url": "https://EXAMPLE.org/old/", "text": "Old content " * 10}
    records = [
        original,
        {"id": "same-url", "url": "http://example.org/old#fragment", "text": "a" * 100},
        {"id": "same-text", "url": "https://example.org/another", "text": "Old\ncontent\t" * 10},
        {"id": "short", "url": "https://example.org/short", "text": "tiny"},
        {"id": "new", "url": "https://example.org/new", "text": "x" * 18432},
        {"id": "new-copy", "url": "https://example.org/new-copy", "text": "x" * 18432},
    ]
    path = tmp_path / "000_00000.parquet"
    pq.write_table(pa.Table.from_pylist(records), path)
    previous = tmp_path / "previous"
    write_used_sources(
        previous,
        [
            {
                "document_id": "old",
                "dedup_cluster": "example.org/old",
                "source": {"file": str(path), "row_group": 0, "row_index": 0},
            }
        ],
    )
    config = {
        "source_dir": str(tmp_path),
        "source_seed": 17,
        "previous_datasets": [str(previous)],
    }
    result = analyze_lengths.analyze(config, tmp_path / "analysis")
    assert result["counts"] == {
        "scanned": 6,
        "excluded_id": 1,
        "excluded_url": 1,
        "excluded_text": 1,
        "remaining_before_basic": 3,
        "basic_too_short": 1,
        "remaining": 2,
    }
    assert result["distribution"]["mean_chars"] == 18432
    replay = analyze_lengths.analyze(config, tmp_path / "analysis")
    assert replay["counts"] == result["counts"]
    assert replay["distribution"] == result["distribution"]
    saved = json.loads((tmp_path / "analysis/config.json").read_text())
    assert len(saved["excluded_sources"]) == 1
    write_used_sources(previous, [])
    with pytest.raises(ValueError, match="analysis inputs changed"):
        analyze_lengths.analyze(config, tmp_path / "analysis")


def test_census_without_previous_datasets_keeps_first_source(tmp_path):
    records = [{"id": "first", "url": "https://example.org/first", "text": "x" * 100}]
    pq.write_table(pa.Table.from_pylist(records), tmp_path / "arbitrary-shard.parquet")
    config = {"source_dir": str(tmp_path), "source_seed": 17, "previous_datasets": []}
    result = analyze_lengths.analyze(config, tmp_path / "analysis")
    assert result["counts"] == {"scanned": 1, "remaining_before_basic": 1, "remaining": 1}
    assert result["distribution"]["documents"] == 1


def test_empty_distribution_has_no_invented_mean_or_quantiles():
    result = analyze_lengths.summarize_lengths(Counter())
    assert result["documents"] == 0
    assert result["mean_chars"] is None and result["quantiles_chars"] == {}
    assert all(row["documents"] == row["fraction"] == 0 for row in result["thresholds"])
