import io
import json

import pytest

from toxic_mlops.release import bundle, fetch_model, restore_tracking
from toxic_mlops.train import train


def test_training_tracking_and_release(raw_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "toxic_mlops.train.code_version",
        lambda root: {
            "commit_sha": "a" * 40,
            "source_sha256": "b" * 64,
        },
    )
    result = train(raw_dataset, tmp_path)
    assert len(result["experiments"]) == 6
    assert len({e["run_id"] for e in result["experiments"]}) == 6
    assert result["selected"]["test"]["f1"] >= 0.8
    assert result["selected"]["validation"]["f1"] == max(
        r["validation"]["f1"] for r in result["experiments"]
    )
    archive = bundle(raw_dataset, tmp_path)
    checksums = json.loads((tmp_path / "dist/bundle-checksums.json").read_text())
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({"url": "https://example.test/model.zip", **checksums}))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(archive.read_bytes()))
    downloaded = fetch_model(manifest, tmp_path / "downloaded")
    assert downloaded["model_version"] == result["selected"]["model_version"]
    mapping = restore_tracking(raw_dataset, raw_dataset.train.report_dir / "experiments.json")
    assert len(mapping) == 6
    checksums["archive_sha256"] = "0" * 64
    manifest.write_text(json.dumps({"url": "https://example.test/model.zip", **checksums}))
    with pytest.raises(ValueError, match="checksum"):
        fetch_model(manifest, tmp_path / "bad")
