import os
from pathlib import Path

import pytest

from meetrag.config import load_environment


def test_env_file_exports_provider_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=from-file\nGOOGLE_API_KEY=also-from-file\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "already-set")

    assert load_environment(env_file)
    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"
    assert os.environ["GOOGLE_API_KEY"] == "already-set"  # the real environment wins
    monkeypatch.delenv("ANTHROPIC_API_KEY")


def test_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_environment(tmp_path / "absent.env") is False
