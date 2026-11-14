import json
import subprocess
from pathlib import Path

import pytest

from toxic_mlops.cli import main
from toxic_mlops.train import code_version


def test_cli_dispatch(settings, monkeypatch, capsys):
    monkeypatch.setattr("toxic_mlops.cli.load_settings", lambda path: settings)
    monkeypatch.setattr("toxic_mlops.cli.download_data", lambda config: Path("data.csv"))
    main(["download"])
    assert "data.csv" in capsys.readouterr().out
    monkeypatch.setattr("toxic_mlops.cli.prepare_data", lambda config: {"rows": 123})
    main(["prepare"])
    assert json.loads(capsys.readouterr().out) == {"rows": 123}
    monkeypatch.setattr("toxic_mlops.train.train", lambda *a: {"selected": {"name": "winner"}})
    main(["train"])
    assert "winner" in capsys.readouterr().out
    monkeypatch.setattr("uvicorn.run", lambda *a, **kw: None)
    main(["serve"])
    monkeypatch.setattr("toxic_mlops.release.bundle", lambda *a: Path("model.zip"))
    main(["bundle"])
    assert "model.zip" in capsys.readouterr().out
    monkeypatch.setattr("toxic_mlops.release.fetch_model", lambda *a: {"model_version": "v-test"})
    main(["fetch-model"])
    assert "v-test" in capsys.readouterr().out
    monkeypatch.setattr("toxic_mlops.release.restore_tracking", lambda *a: [{"run": "restored"}])
    main(["restore-tracking"])
    assert "restored" in capsys.readouterr().out


def test_code_version(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-b", "main")
    git("config", "user.email", "test@example.test")
    git("config", "user.name", "Test")
    path = tmp_path / "config.toml"
    path.write_text("seed = 42\n")
    git("add", "config.toml")
    git("commit", "-m", "test fixture")
    result = code_version(tmp_path)
    assert len(result["commit_sha"]) == 40
    assert len(result["source_sha256"]) == 64
    path.write_text("seed = 43\n")
    with pytest.raises(ValueError, match="Commit"):
        code_version(tmp_path)
