from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from latent_working_memory.data_preparation.__main__ import main
from latent_working_memory.data_preparation.inspection import judge_inspection, sample_inspection
from latent_working_memory.data_preparation.scoring import SampleScorer, parse_review
from latent_working_memory.v1.config import write_resolved_config


@pytest.fixture
def quality_service():
    requests = []
    behavior = {"finish_reason": "stop", "error": False, "invalid": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            assert self.path == "/v1/chat/completions"
            if behavior["error"]:
                self.send_error(503)
                return
            sample = json.loads(payload["messages"][1]["content"])
            review = {
                "decision": "reject" if sample["X"] == "bad" else "keep",
                "reason": "Test service judgment",
            }
            invalid = (
                behavior["invalid"] is True
                or behavior["invalid"] == sample["X"]
                or behavior.pop("invalid_once", False)
            )
            content = "invalid JSON" if invalid else json.dumps(review)
            encoded = json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": behavior["finish_reason"],
                            "message": {"content": content},
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests, behavior
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_prompt_contract_cache_and_independent_inspection(
    tmp_path,
    preparation_recipe,
    quality_service,
    monkeypatch,
):
    url, requests, _ = quality_service
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    recipe = replace(preparation_recipe, review_base_url=url)
    cache = tmp_path / "cache.jsonl"
    samples = [
        {
            "boundary_variant": variant,
            "task": "continuation",
            "X": "A sentence.",
            "Y": " Next sentence.",
        }
        for variant in ("semantic", "random")
    ]
    scorer = SampleScorer(recipe, cache)
    assert all(r["decision"] == "keep" for r in scorer.score_batch(samples))
    assert len(requests) == 2
    assert {json.loads(r["messages"][1]["content"])["boundary_variant"] for r in requests} == {
        "semantic",
        "random",
    }
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert requests[0]["response_format"]["json_schema"]["strict"]
    reason_schema = requests[0]["response_format"]["json_schema"]["schema"]["properties"]["reason"]
    assert reason_schema == {"type": "string", "minLength": 1, "maxLength": 240}
    again = SampleScorer(recipe, cache)
    assert again.score_batch(samples) == scorer.score_batch(samples)
    assert len(requests) == 2
    inspector = SampleScorer(recipe, cache, "inspection")
    inspector.score_batch(samples)
    assert len(requests) == 4
    assert all("Independently audit" in row["messages"][0]["content"] for row in requests[2:])
    changed = [samples[0] | {"X": "bad"}]
    assert scorer.score_batch(changed)[0]["decision"] == "reject"
    SampleScorer(replace(recipe, review_model="another-model"), cache).score_batch(samples[:1])
    assert len(requests) == 6


@pytest.mark.parametrize("failure", ["error", "invalid", "finish_reason"])
def test_model_failure_never_becomes_a_keep_decision(
    tmp_path, preparation_recipe, quality_service, failure
):
    url, _, behavior = quality_service
    behavior[failure] = "length" if failure == "finish_reason" else True
    scorer = SampleScorer(
        replace(preparation_recipe, review_base_url=url), tmp_path / "cache.jsonl"
    )
    samples = [{"boundary_variant": "semantic", "task": "ae", "X": "A sentence.", "Y": None}]
    if failure == "error":
        with pytest.raises(HTTPError):
            scorer.score_batch(samples)
        assert not scorer.cache
    else:
        assert scorer.score_batch(samples)[0]["decision"] == "error"
        assert len(next(iter(scorer.cache.values()))["failures"]) == 2


def test_invalid_sample_does_not_discard_valid_batch_results(
    tmp_path, preparation_recipe, quality_service
):
    url, requests, behavior = quality_service
    behavior["invalid"] = "broken"
    recipe = replace(preparation_recipe, review_base_url=url)
    path = tmp_path / "cache.jsonl"
    scorer = SampleScorer(recipe, path)
    samples = [
        {"boundary_variant": "semantic", "task": "ae", "X": x, "Y": None}
        for x in ("broken", "A sentence.", "bad")
    ]
    result = scorer.score_batch(samples)
    assert [r["decision"] for r in result] == ["error", "keep", "reject"]
    assert len(requests) == 4
    assert SampleScorer(recipe, path).score_batch(samples) == result
    assert len(requests) == 4
    error_row = scorer.cache[result[0]["cache_key"]]
    assert all(
        r["response"]["choices"][0]["message"]["content"] == "invalid JSON"
        for r in error_row["failures"]
    )


def test_format_retry_can_recover_a_valid_judgment(tmp_path, preparation_recipe, quality_service):
    url, requests, behavior = quality_service
    behavior["invalid_once"] = True
    scorer = SampleScorer(
        replace(preparation_recipe, review_base_url=url), tmp_path / "cache.jsonl"
    )
    review = scorer.score_batch(
        [{"boundary_variant": "semantic", "task": "ae", "X": "A sentence.", "Y": None}]
    )[0]
    assert review["decision"] == "keep"
    assert len(requests) == 2
    assert len(scorer.cache[review["cache_key"]]["failures"]) == 1


@pytest.mark.parametrize(
    "raw",
    [
        "{}",
        '{"decision":"maybe","reason":"ok"}',
        '{"decision":"keep","reason":""}',
        '{"decision":"keep","reason":"ok","extra":1}',
        json.dumps({"decision": "keep", "reason": "x" * 241}),
        '{"decision":"keep","reason":"invalid\tcontrol"}',
    ],
)
def test_quality_response_requires_exact_contract(raw):
    with pytest.raises(ValueError):
        parse_review(raw)


def test_cli_stages_and_posthoc_model_inspection(
    tmp_path,
    tiny_config,
    tokenizer,
    preparation_records,
    preparation_recipe,
    quality_service,
):
    url, requests, behavior = quality_service
    model = tmp_path / "tokenizer"
    tokenizer.save_pretrained(model)
    config = replace(tiny_config, model_name_or_path=str(model))
    config_path, recipe_path = tmp_path / "config.json", tmp_path / "recipe.json"
    write_resolved_config(config, config_path)
    recipe = replace(preparation_recipe, review_base_url=url)
    recipe_path.write_text(json.dumps(recipe.to_dict()))
    source = tmp_path / "HuggingFaceFW-fineweb/sample-10BT"
    source.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(preparation_records), source / "input.parquet")
    root, cache = tmp_path / "data", tmp_path / "scores.jsonl"
    args = ["--config", str(config_path), "--recipe", str(recipe_path), "--output-dir", str(root)]
    main([*args, "--stage", "sources", "--dataset-dir", str(source.parent)])
    assert not requests
    for stage in ("semantic", "random"):
        main([*args, "--stage", stage, "--score-cache", str(cache)])
        assert (root / stage / "preparation.json").is_file()
    assert (root / "comparison.json").is_file()
    assert not list(root.rglob("*.pt"))
    for variant in ("semantic", "random"):
        inspection = tmp_path / f"inspection-{variant}"
        sample_inspection(root / variant, inspection, examples=4)
        before = len(requests)
        scorer = SampleScorer(recipe, tmp_path / f"inspection-{variant}-cache.jsonl", "inspection")
        summary = judge_inspection(inspection, scorer)
        assert len(requests) > before
        assert summary["panels"]["random-views"]["all"]["complete"]
        before = len(requests)
        judge_inspection(inspection, scorer)
        assert len(requests) == before
    behavior["invalid"] = True
    inspection = tmp_path / "inspection-format-errors"
    sample_inspection(root / "semantic", inspection, examples=2)
    scorer = SampleScorer(recipe, tmp_path / "inspection-errors-cache.jsonl", "inspection")
    report = judge_inspection(inspection, scorer)["panels"]["random-views"]["all"]
    assert report["unreviewed"] == report["samples"]
    assert not report["complete"]
    assert report["failure_rate"] is None
