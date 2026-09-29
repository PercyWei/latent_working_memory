from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from latent_working_memory.data_preparation.fineweb_qa import analyze_lengths


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


def test_census_excludes_old_identity_url_and_normalized_text_without_deduping_remainder(
    tmp_path, monkeypatch
):
    original = {"id": "old", "url": "https://EXAMPLE.org/old/", "text": "Old content " * 10}
    records = [
        original,
        {"id": "same-url", "url": "http://example.org/old#fragment", "text": "a" * 100},
        {"id": "same-text", "url": "https://example.org/another", "text": "Old\ncontent\t" * 10},
        {"id": "short", "url": "https://example.org/short", "text": "tiny"},
        {"id": "new", "url": "https://example.org/new", "text": "x" * 18432},
        {"id": "new-copy", "url": "https://example.org/new-copy", "text": "x" * 18432},
    ]
    pq.write_table(pa.Table.from_pylist(records), tmp_path / "000_00000.parquet")
    monkeypatch.setattr(
        analyze_lengths, "parquet_records", lambda *_: ((original, {}) for _ in range(1))
    )
    config = {
        "source": {
            "raw_dir": str(tmp_path),
            "file_count": 1,
            "data_seed": 17,
            "old_pool_documents": 1,
        }
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
