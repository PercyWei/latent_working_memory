import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.pretrain.config import PreparationConfig
from latent_working_memory.data_preparation.pretrain.dedup import (
    cluster_documents,
    matching_clusters,
    source_key,
)
from latent_working_memory.data_preparation.pretrain.sources import referenced_records


def test_matching_excludes_current_cluster_after_representative_changes():
    words = [f"word{i:03d}" for i in range(120)]
    near = words.copy()
    near[20] = "changed"
    transitive = near.copy()
    transitive[70] = "altered"
    previous = {"id": "previous", "url": "https://example.org/z", "text": " ".join(words)}
    candidates = [
        {"id": "near", "url": "https://example.org/b", "text": " ".join(near)},
        {"id": "transitive", "url": "https://example.org/a", "text": " ".join(transitive)},
        {
            "id": "unrelated",
            "url": "https://example.org/other",
            "text": " ".join(f"other{i}" for i in range(120)),
        },
    ]
    recipe = PreparationConfig(near_duplicate_threshold=0.9, near_duplicate_min_words=64)
    old_cluster = cluster_documents([previous], recipe)[0]
    clusters = cluster_documents(candidates, recipe)
    assert clusters[0] == clusters[1] != old_cluster
    assert matching_clusters([previous], [candidates[1]], [clusters[1]], recipe) == set()
    excluded = matching_clusters(iter([previous]), candidates, clusters, recipe)
    assert excluded == {clusters[0]}
    assert [
        record["id"] for record, cluster in zip(candidates, clusters) if cluster not in excluded
    ] == ["unrelated"]


@pytest.mark.parametrize("identity", ["id", "url", "text"])
def test_matching_uses_raw_identity_url_and_normalized_text(identity):
    previous = {
        "id": "previous",
        "url": "https://Example.org/used/#section",
        "text": "first  second\nthird",
    }
    candidate = {"id": "different", "url": "https://elsewhere.org/new", "text": "unrelated"}
    candidate[identity] = {
        "id": previous["id"],
        "url": "http://example.org/used",
        "text": "first\nsecond  third",
    }[identity]
    recipe = PreparationConfig(near_duplicate_min_words=1000)
    cluster = source_key(candidate["url"])
    assert matching_clusters([previous], [candidate], [cluster], recipe) == {cluster}


def test_referenced_records_group_reads_and_verify_registered_identity(tmp_path, monkeypatch):
    path = tmp_path / "source.parquet"
    records = [
        {"id": str(i), "url": f"https://example.org/{i}", "text": f"document {i}"} for i in range(5)
    ]
    pq.write_table(pa.Table.from_pylist(records), path, row_group_size=2)
    references = [
        {
            "document_id": str(i),
            "source": {"file": str(path), "row_group": i // 2, "row_index": i % 2},
        }
        for i in (0, 1, 4)
    ]
    reads = []
    original = pq.ParquetFile.read_row_group

    def read_group(parquet, group, *args, **kwargs):
        reads.append(group)
        return original(parquet, group, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", read_group)
    assert list(referenced_records(references)) == [records[i] for i in (0, 1, 4)]
    assert reads == [0, 2]
    with pytest.raises(ValueError, match="differs from its recorded location"):
        list(referenced_records([dict(references[0], document_id="other-document")]))
