"""Six comparable tracked experiments; select on validation, evaluate test once."""

import hashlib
import json
import subprocess
import time
import warnings
from importlib import metadata as package_metadata
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from toxic_mlops.config import Settings
from toxic_mlops.data import prepare_data, sha256_file, write_json
from toxic_mlops.model import make_pipeline, save_model


def evaluate(labels, scores) -> dict:
    prediction = np.asarray(scores) >= 0.5
    return {
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def code_version(root: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    sha = git("rev-parse", "HEAD")
    # Generated reports may change without affecting executable source.
    paths = ["src", "config.toml", "pyproject.toml", "uv.lock"]
    dirty = git("status", "--porcelain", "--", *paths)
    if dirty:
        raise ValueError("Commit source/config/lock changes before recording experiments")
    digest = hashlib.sha256()
    for name in sorted(git("ls-files", "--", *paths).splitlines()):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return {"commit_sha": sha, "source_sha256": digest.hexdigest()}


def train(settings: Settings, root: Path) -> dict:
    import mlflow
    import pandas as pd
    from mlflow.tracking import MlflowClient

    version = code_version(root)
    manifest = prepare_data(settings)
    frame = pd.read_csv(settings.data.prepared_dir / "v1.csv", keep_default_na=False)
    parts = {split: frame.loc[frame.split.eq(split)] for split in ("train", "validation", "test")}
    output = settings.train.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    settings.train.tracking_artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.train.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(settings.train.experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            settings.train.experiment_name,
            artifact_location=settings.train.tracking_artifacts.as_uri(),
        )
    )
    results = []
    best_pipeline = None
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for spec in settings.train.experiments:
            pipeline = make_pipeline(spec, settings.train)
            start = time.perf_counter()
            pipeline.fit(parts["train"].text.tolist(), parts["train"].label)
            seconds = time.perf_counter() - start
            scores = pipeline.predict_proba(parts["validation"].text.tolist())[:, 1]
            metrics = evaluate(parts["validation"].label, scores)
            with mlflow.start_run(experiment_id=experiment_id, run_name=spec.name) as run:
                mlflow.set_tags({**version, "mlflow.source.git.commit": version["commit_sha"]})
                params = {
                    **spec.model_dump(),
                    "seed": settings.train.seed,
                    "class_weight": "balanced",
                    "solver": "liblinear",
                    "min_df": settings.train.min_df,
                    "max_features": settings.train.max_features,
                    "max_iter": settings.train.max_iter,
                    "sublinear_tf": True,
                    "threshold": 0.5,
                    "raw_sha256": manifest["raw_sha256"],
                    "data_sha256": manifest["versions"][spec.preprocessing]["sha256"],
                    "split_sha256": manifest["split_sha256"],
                }
                mlflow.log_params(params)
                mlflow.log_metrics({f"validation_{k}": v for k, v in metrics.items()})
                mlflow.log_metric("fit_seconds", seconds)
                mlflow.log_dict(manifest, "data_manifest.json")
                # Configuration from the repository contains no environment credentials.
                mlflow.log_artifact(str(root / "config.toml"), artifact_path="configuration")
                row = {
                    "name": spec.name,
                    "run_id": run.info.run_id,
                    "experiment_id": experiment_id,
                    "parameters": params,
                    "validation": metrics,
                    "fit_seconds": seconds,
                    **version,
                }
                results.append(row)
                if best is None or (metrics["f1"], metrics["average_precision"]) > (
                    best["validation"]["f1"],
                    best["validation"]["average_precision"],
                ):
                    best, best_pipeline = row, pipeline
            print(f"{spec.name}: validation F1={metrics['f1']:.6f}", flush=True)
    # Selection has ended; the test set cannot influence the winning configuration.
    test_scores = best_pipeline.predict_proba(parts["test"].text.tolist())[:, 1]
    test_metrics = evaluate(parts["test"].label, test_scores)
    artifact_metadata = {
        "schema_version": 1,
        **version,
        "run_id": best["run_id"],
        "experiment_id": experiment_id,
        "experiment_name": settings.train.experiment_name,
        "preprocessing": best["parameters"]["preprocessing"],
        "parameters": best["parameters"],
        "min_chars": settings.train.min_chars,
        "validation": best["validation"],
        "test": test_metrics,
        "selection_rule": "maximum validation F1, then average precision, then config order",
        "threshold": 0.5,
        "data_manifest": manifest,
        "packages": {
            p: package_metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "joblib", "toxic-mlops")
        },
    }
    artifact_metadata = save_model(best_pipeline, output, artifact_metadata)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected", "true")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifacts(str(output), artifact_path="model")
        mlflow.log_dict(
            {
                "labels": [0, 1],
                "matrix": confusion_matrix(
                    parts["test"].label, test_scores >= 0.5, labels=[0, 1]
                ).tolist(),
            },
            "confusion_matrix.json",
        )
    summary = {
        "code": version,
        "data": manifest,
        "experiments": results,
        "selected": artifact_metadata,
        "test_rows": len(parts["test"]),
        "model_bytes": (output / "model.joblib").stat().st_size,
        "metadata_sha256": sha256_file(output / "metadata.json"),
    }
    write_json(settings.train.report_dir / "experiments.json", summary)
    markdown = [
        "# Фактически выполненные эксперименты",
        "",
        f"Код: `{version['commit_sha']}`. Порог: 0.5; выбор только по validation.",
        "",
        "| Эксперимент | Run ID | Validation F1 | AP |",
        "|---|---|---:|---:|",
    ]
    for row in results:
        markdown.append(
            f"| {row['name']} | `{row['run_id']}` | "
            f"{row['validation']['f1']:.6f} | "
            f"{row['validation']['average_precision']:.6f} |"
        )
    markdown.extend(
        [
            "",
            f"Выбран: **{best['name']}**.",
            "",
            "Test (только итоговая модель):",
            "",
            "```json",
            json.dumps(test_metrics, indent=2),
            "```",
            "",
            f"Версия модели: `{artifact_metadata['model_version']}`.",
            "",
            "Время fit — одиночное измерение, не статистический benchmark.",
        ]
    )
    (settings.train.report_dir / "EXPERIMENTS.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return summary
