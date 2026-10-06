"""在临时项目中验证凭据优先级，不读取本机登录文件。"""

import pytest

from latent_working_memory.v3.tracking_credentials import swanlab_api_key


@pytest.mark.parametrize(
    "content,terminal,expected",
    [
        (None, "terminal-key", "terminal-key"),
        ("SWANLAB_API_KEY=project-key\n", "terminal-key", "project-key"),
        ('SWANLAB_API_KEY="literal-${OTHER_TOKEN}"\n', "terminal-key", "literal-${OTHER_TOKEN}"),
        ("SWANLAB_API_KEY=project-key\n", None, "project-key"),
    ],
)
def test_explicit_credentials(tmp_path, monkeypatch, content, terminal, expected):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OTHER_TOKEN", "must-not-substitute")
    if terminal is None:
        monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    else:
        monkeypatch.setenv("SWANLAB_API_KEY", terminal)
    if content is not None:
        (tmp_path / ".env").write_text(content)
    assert swanlab_api_key() == expected


@pytest.mark.parametrize("content", ["", "UNRELATED=value\n", "SWANLAB_API_KEY=\n", 'SWANLAB_API_KEY="  "\n'])
def test_existing_file_without_key_does_not_fall_back(tmp_path, monkeypatch, content):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SWANLAB_API_KEY", "must-not-use-terminal")
    (tmp_path / ".env").write_text(content)
    with pytest.raises(ValueError, match=r"project \.env"):
        swanlab_api_key()


def test_does_not_search_parent_or_saved_login(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SWANLAB_API_KEY=parent-key\n")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="terminal environment"):
        swanlab_api_key()
