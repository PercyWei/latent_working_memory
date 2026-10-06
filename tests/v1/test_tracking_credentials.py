"""共享 tracking 的凭据传递测试；所有 SDK 入口均使用替身，不读取保存的凭据。"""

from copy import deepcopy
import json
import socket
from types import SimpleNamespace

import pytest

from latent_working_memory.v1 import tracking


API_KEY = "unit-test-api-key-not-a-secret"


@pytest.fixture
def sdk(monkeypatch):
    calls = SimpleNamespace(settings=[], init=[], api=[], remote_paths=[], restored=[])

    class Run:
        id = "training-run"
        url = "https://swanlab.example/@my-workspace/my-project/runs/training-run"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    remote = SimpleNamespace(
        state="FINISHED",
        name="original-training-name",
        profile={
            "config": {
                "method": {"value": "memory_change", "desc": "", "sort": 1},
                "seed": {"value": 42, "desc": "", "sort": 0},
            }
        },
    )

    def settings(**kwargs):
        calls.settings.append(kwargs)
        return SimpleNamespace(**kwargs)

    def initialize(**kwargs):
        calls.init.append(kwargs)
        return Run()

    def query_run(path):
        calls.remote_paths.append(path)
        return remote

    def api(**kwargs):
        calls.api.append(kwargs)
        return SimpleNamespace(run=query_run)

    def forbidden(*args, **kwargs):
        raise AssertionError("credential tests must not log in or access the network")

    monkeypatch.setattr(tracking.swanlab, "Settings", settings)
    monkeypatch.setattr(tracking.swanlab, "init", initialize)
    monkeypatch.setattr(tracking.swanlab, "Api", api)
    monkeypatch.setattr(tracking.swanlab, "login", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(tracking, "capture_rng_state", lambda: "saved-rng")
    monkeypatch.setattr(tracking, "restore_rng_state", calls.restored.append)
    return calls


@pytest.mark.parametrize("credential", ["explicit", "none", "omitted"])
def test_training_passes_key_only_to_sdk_settings(tmp_path, sdk, credential, capsys):
    config = {"method": "memory_change", "training": {"seed": 42, "learning_rate": 0.0001}}
    original = deepcopy(config)
    options = (
        {}
        if credential == "omitted"
        else {"api_key": API_KEY if credential == "explicit" else None}
    )

    with tracking.swanlab_run(
        tmp_path,
        config,
        mode="online",
        project="my-project",
        group="credential-unit-test",
        **options,
    ) as run:
        assert run.id == "training-run"

    expected_key = API_KEY if credential == "explicit" else None
    assert len(sdk.init) == len(sdk.settings) == 1
    assert sdk.settings[0].get("api_key") == expected_key
    if expected_key is None:
        assert "api_key" not in sdk.settings[0]
    assert sdk.init[0]["settings"].__dict__.get("api_key") == expected_key
    assert sdk.settings[0]["interactive"] is False
    assert sdk.init[0]["config"] == config == original
    assert sdk.init[0]["project"] == "my-project"
    assert sdk.init[0]["resume"] == "never"
    assert sdk.restored == ["saved-rng"]
    assert sdk.api == []
    identity = (tmp_path / "swanlab.json").read_text()
    assert json.loads(identity)["id"] == "training-run"
    assert API_KEY not in identity
    assert API_KEY not in json.dumps(sdk.init[0]["config"])
    assert "api_key" not in json.loads(identity)
    captured = capsys.readouterr()
    assert API_KEY not in captured.out + captured.err
    assert {path.name for path in tmp_path.iterdir()} == {"swanlab.json"}


@pytest.mark.parametrize("credential", ["explicit", "none", "omitted"])
def test_evaluation_passes_same_key_to_api_and_resume_init_without_persisting_it(
    tmp_path, sdk, credential, capsys
):
    identity = {
        "id": "training-run",
        "project": "my-project",
        "group": "original-group",
        "tags": ["scope:main"],
        "job_type": "train",
        "mode": "online",
        "url": "https://swanlab.example/@my-workspace/my-project/runs/training-run",
    }
    path = tmp_path / "swanlab.json"
    path.write_text(json.dumps(identity))
    original_identity = path.read_bytes()
    options = (
        {}
        if credential == "omitted"
        else {"api_key": API_KEY if credential == "explicit" else None}
    )

    with tracking.swanlab_training_run(tmp_path, **options) as run:
        assert run.id == "training-run"

    expected_key = API_KEY if credential == "explicit" else None
    assert len(sdk.api) == len(sdk.init) == len(sdk.settings) == 1
    assert sdk.api[0].get("api_key") == expected_key
    assert sdk.settings[0].get("api_key") == expected_key
    if expected_key is None:
        assert "api_key" not in sdk.settings[0]
    assert sdk.init[0]["settings"].__dict__.get("api_key") == expected_key
    assert sdk.remote_paths == ["my-workspace/my-project/training-run"]
    assert sdk.init[0]["workspace"] == "my-workspace"
    assert sdk.init[0]["project"] == "my-project"
    assert sdk.init[0]["name"] == "original-training-name"
    assert sdk.init[0]["id"] == "training-run"
    assert sdk.init[0]["resume"] == "must"
    assert sdk.init[0]["config"] == {"seed": 42, "method": "memory_change"}
    assert list(sdk.init[0]["config"]) == ["seed", "method"]
    assert path.read_bytes() == original_identity
    assert API_KEY not in json.dumps(sdk.init[0]["config"])
    assert API_KEY not in path.read_text()
    captured = capsys.readouterr()
    assert API_KEY not in captured.out + captured.err
    assert sdk.restored == ["saved-rng"]
    assert {item.name for item in tmp_path.iterdir()} == {"swanlab.json"}


def test_disabled_tracking_does_not_touch_sdk_or_persist_key(tmp_path, sdk):
    with tracking.swanlab_run(tmp_path, {}, mode="disabled", api_key=API_KEY) as run:
        assert run is None
    assert sdk.settings == sdk.init == sdk.api == sdk.remote_paths == sdk.restored == []
    assert list(tmp_path.iterdir()) == []
