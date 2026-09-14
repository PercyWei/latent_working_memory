from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from latent_working_memory.v1.dynamic.empty_memory import EMPTY_CONDITIONS, ORIGINAL_CONDITIONS
from latent_working_memory.v1.dynamic.evaluation import aggregate_qa
from latent_working_memory.v1.dynamic import supplement_publication as publication


@pytest.fixture
def package(tmp_path):
    source = tmp_path / "train"
    source.mkdir()
    initial = tmp_path / "initial.pt"
    recipe = {"generation_tokens": 2, "seed": 42}
    (source / "provenance.json").write_text(
        json.dumps({"target_steps": 2, "config": recipe, "initial_checkpoint": str(initial)})
    )
    original = [
        dict(
            capacity=4,
            episode_id="doc",
            read_id="q",
            prefix_end=12,
            document_id="source",
            condition=c,
            question="What?",
            references=["First"],
            target_tokens=2,
            nll_sum=4.0,
            em=0.0,
            f1=0.0,
            prediction="Other",
            hit_limit=False,
            kind="delayed",
            delay_tokens=6,
        )
        for c in sorted(ORIGINAL_CONDITIONS)
    ]
    origin = tmp_path / "original.jsonl"
    origin.write_text("".join(json.dumps(r) + "\n" for r in original))
    added = [
        dict(next(r for r in original if r["condition"] == "no_memory"), condition=c)
        for c in EMPTY_CONDITIONS
    ]
    report = tmp_path / "test-step-000002.json"
    rows = original + added
    report.write_text(json.dumps(aggregate_qa(rows)))
    report.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {
                "dataset": "squad",
                "split": "test",
                "checkpoint_step": 2,
                "checkpoint": str(source / "checkpoints/dynamic-step-000002.pt"),
                "config": recipe,
                "supplemental_baselines": {
                    "source_report": str(origin.with_suffix(".json")),
                    "pretrain_checkpoint": str(initial),
                },
            }
        )
    )
    manifest = tmp_path / "reports.json"
    manifest.write_text(json.dumps({"squad": report.name}))
    display = tmp_path / "display"
    display.mkdir()
    identity = {"id": "same-run", "url": "https://swanlab.cn/@owner/project/runs/same-run"}
    (display / "swanlab.json").write_text(json.dumps(identity))
    (display / "republication.json").write_text(
        json.dumps({"new_run": identity, "source": {"training_run": str(source)}})
    )
    return display, manifest, source, report


def test_reject_modified_old_scores_and_mismatched_training_run(package):
    _, manifest, source, report = package
    reports, _, step = publication.load_reports(manifest, source)
    assert step == 2 and len(reports["squad"][1]) == 7
    rows = [json.loads(x) for x in report.with_suffix(".jsonl").read_text().splitlines()]
    rows[0]["f1"] = 0.9
    report.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="original evaluation records changed"):
        publication.load_reports(manifest, source)


def test_reject_mismatched_checkpoint(package):
    _, manifest, source, report = package
    path = report.parent / "evaluation.json"
    info = json.loads(path.read_text())
    info["checkpoint"] = str(source / "checkpoints/dynamic-step-000001.pt")
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="different training run"):
        publication.load_reports(manifest, source)


def test_render_does_not_touch_cloud(package, tmp_path, monkeypatch):
    display, manifest, _, _ = package

    def forbidden():
        pytest.fail("render-only must not access cloud")

    monkeypatch.setattr(publication.swanlab, "Api", forbidden)
    output = tmp_path / "rendered"
    publication.publish_supplement(display, manifest, output, "disabled")
    assert json.loads((output / "publication.json").read_text())["status"] == "prepared"
    chart = json.loads((output / "media.json").read_text())[publication.PREFIX + "/f1"][0]
    assert len(chart["series"]) == 7


def test_failed_upload_verification_never_removes_old_panels(package, tmp_path, monkeypatch):
    display, manifest, _, _ = package
    reports, _, _ = publication.load_reports(manifest, package[2])
    keys = list(publication.qa_media(reports, prefix=publication.PREFIX))
    old_keys = [k.replace(publication.PREFIX, publication.OLD_PREFIX) for k in keys]
    old_charts = [{"index": str(i), "title": key} for i, key in enumerate(old_keys)]
    deleted = []
    remote = SimpleNamespace(
        state="FINISHED",
        run_id="internal",
        series=lambda **kwargs: SimpleNamespace(json=lambda: {"keys": old_keys}),
    )

    class Api:
        def run(self, path):
            return remote

        def _get(self, path, **kwargs):
            data = (
                [{"name": "evaluation", "chartIndex": [c["index"] for c in old_charts]}]
                if path.endswith("sections")
                else old_charts[int(path.split("/")[-2])]
            )
            return SimpleNamespace(ok=True, data=data)

        def _delete(self, path):
            deleted.append(path)
            return SimpleNamespace(ok=True, data={})

    @contextmanager
    def session(path):
        assert path == display
        yield SimpleNamespace(log=lambda values, step: None)

    monkeypatch.setattr(publication.swanlab, "Api", Api)
    monkeypatch.setattr(publication, "snapshot_run", lambda remote: {"unchanged": True})
    monkeypatch.setattr(publication, "media_contents", lambda remote, keys, step: {})
    monkeypatch.setattr(publication, "swanlab_training_run", session)
    with pytest.raises(ValueError, match="differ from the prepared"):
        publication.publish_supplement(display, manifest, tmp_path / "failed", "online")
    assert deleted == []
