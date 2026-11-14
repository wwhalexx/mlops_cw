"""Portable release assets: checksum-verified download and tracking restoration."""

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from toxic_mlops.config import Settings
from toxic_mlops.data import sha256_file, write_json
from toxic_mlops.model import load_model


def bundle(settings: Settings, root: Path) -> Path:
    _, metadata = load_model(settings.train.output_dir)
    target = root / "dist"
    target.mkdir(exist_ok=True)
    archive_path = target / "model-bundle.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("model.joblib", "metadata.json"):
            archive.write(settings.train.output_dir / name, f"model/{name}")
        archive.write(settings.train.report_dir / "experiments.json", "experiments.json")
    write_json(
        target / "bundle-checksums.json",
        {
            "archive_sha256": sha256_file(archive_path),
            "model_version": metadata["model_version"],
            "training_commit": metadata["commit_sha"],
        },
    )
    return archive_path


def fetch_model(manifest_path: Path, destination: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(manifest["url"], timeout=120) as response:
        content = response.read(50 * 1024 * 1024 + 1)
    if hashlib.sha256(content).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Release archive checksum mismatch")
    with tempfile.TemporaryDirectory() as temporary:
        archive_path = Path(temporary) / "model.zip"
        archive_path.write_bytes(content)
        with zipfile.ZipFile(archive_path) as archive:
            expected = {"model/model.joblib", "model/metadata.json", "experiments.json"}
            if set(archive.namelist()) != expected:
                raise ValueError("Unexpected release archive members")
            for name in expected:
                if archive.getinfo(name).file_size > 100 * 1024 * 1024:
                    raise ValueError("Release archive member too large")
                path = Path(temporary) / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(archive.read(name))
        _, metadata = load_model(Path(temporary) / "model")
        if metadata["model_version"] != manifest["model_version"]:
            raise ValueError("Release model version mismatch")
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("model.joblib", "metadata.json"):
            shutil.copy2(Path(temporary) / "model" / name, destination / name)
    return metadata


def restore_tracking(settings: Settings, report: Path) -> list[dict]:
    """Import portable run records; new MLflow run IDs retain original IDs as tags."""
    import mlflow
    from mlflow.tracking import MlflowClient

    records = json.loads(report.read_text(encoding="utf-8"))
    settings.train.tracking_artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.train.tracking_uri)
    client = MlflowClient()
    name = settings.train.experiment_name + "-restored"
    experiment = client.get_experiment_by_name(name)
    identifier = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            name, artifact_location=settings.train.tracking_artifacts.as_uri()
        )
    )
    mapping = []
    for row in records["experiments"]:
        with mlflow.start_run(experiment_id=identifier, run_name=row["name"]) as run:
            mlflow.set_tags(
                {
                    "original_run_id": row["run_id"],
                    "restored": "true",
                    "mlflow.source.git.commit": row["commit_sha"],
                    "source_sha256": row["source_sha256"],
                }
            )
            mlflow.log_params(row["parameters"])
            mlflow.log_metrics({f"validation_{k}": v for k, v in row["validation"].items()})
            mlflow.log_dict(records["data"], "data_manifest.json")
            if row["run_id"] == records["selected"]["run_id"]:
                _, model_metadata = load_model(settings.inference.model_dir)
                if model_metadata["model_version"] != records["selected"]["model_version"]:
                    raise ValueError("Local model is not the selected model in this report")
                mlflow.set_tag("selected", "true")
                mlflow.log_metrics({f"test_{k}": v for k, v in records["selected"]["test"].items()})
                mlflow.log_artifacts(str(settings.inference.model_dir), artifact_path="model")
            mapping.append({"original_run_id": row["run_id"], "local_run_id": run.info.run_id})
    return mapping
