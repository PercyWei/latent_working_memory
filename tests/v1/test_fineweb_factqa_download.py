import subprocess

import pytest

from latent_working_memory.data_preparation.fineweb_factqa import download


ENDPOINT = "https://hf-mirror.com"
NAME = "000_00000.parquet"


def test_existing_final_file_is_trusted_and_skipped(monkeypatch, tmp_path):
    final = tmp_path / NAME
    final.write_bytes(b"previous successful transfer")
    partial = tmp_path / f"{NAME}.part"
    partial.write_bytes(b"untouched")

    def unexpected_curl(*args, **kwargs):
        raise AssertionError("curl must not run when the final file exists")

    monkeypatch.setattr(download.subprocess, "run", unexpected_curl)
    assert not download.download_shard(NAME, tmp_path, ENDPOINT, download.REVISION)
    assert final.read_bytes() == b"previous successful transfer"
    assert partial.read_bytes() == b"untouched"


def test_partial_file_is_resumed_and_renamed_after_curl_success(monkeypatch, tmp_path):
    partial = tmp_path / f"{NAME}.part"
    partial.write_bytes(b"first half")
    final = tmp_path / NAME
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert command[command.index("--noproxy") + 1] == "*"
        assert command[command.index("--proxy") + 1] == ""
        assert command[command.index("--continue-at") + 1] == "-"
        assert command[command.index("--output") + 1] == str(partial)
        assert command[-1] == (
            f"{ENDPOINT}/datasets/HuggingFaceFW/fineweb/resolve/"
            f"{download.REVISION}/sample/10BT/{NAME}"
        )
        assert kwargs["capture_output"] and kwargs["text"]
        with partial.open("ab") as output:
            output.write(b" second half")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    assert download.download_shard(NAME, tmp_path, ENDPOINT, download.REVISION)
    assert final.read_bytes() == b"first half second half"
    assert not partial.exists()
    assert len(calls) == 1


def test_failed_curl_keeps_partial_file_for_another_run(monkeypatch, tmp_path):
    partial = tmp_path / f"{NAME}.part"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        with partial.open("ab") as output:
            output.write(b"partial")
        return subprocess.CompletedProcess(command, 28, "", "curl: (28) timeout")

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(download, "MAX_ATTEMPTS", 2)
    with pytest.raises(RuntimeError, match="partial file kept"):
        download.download_shard(NAME, tmp_path, ENDPOINT, download.REVISION)
    assert len(calls) == 2
    assert partial.read_bytes() == b"partialpartial"
    assert not (tmp_path / NAME).exists()


def test_main_uses_fixed_shards_and_rejects_other_revision(monkeypatch, tmp_path):
    requested = []

    def fake_download(name, directory, endpoint, revision):
        requested.append((name, directory, endpoint, revision))
        return False

    monkeypatch.setattr(download, "download_shard", fake_download)
    args = [
        "--directory",
        str(tmp_path),
        "--endpoint",
        ENDPOINT,
        "--revision",
        download.REVISION,
        "--workers",
        "1",
    ]
    assert download.main(args) == 0
    assert [item[0] for item in requested] == list(download.SHARD_NAMES)
    assert all(item[1:] == (tmp_path, ENDPOINT, download.REVISION) for item in requested)
    with pytest.raises(SystemExit) as exc:
        download.main(args[:5] + ["0" * 40] + args[6:])
    assert exc.value.code == 2
