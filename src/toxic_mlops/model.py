"""A single serialized sklearn pipeline, with integrity-checked metadata."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from toxic_mlops.config import Experiment, TrainConfig
from toxic_mlops.data import sha256_file, write_json
from toxic_mlops.preprocessing import TextPreprocessor


def make_pipeline(experiment: Experiment, train: TrainConfig) -> Pipeline:
    return Pipeline(
        [
            ("preprocessing", TextPreprocessor(experiment.preprocessing)),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=experiment.analyzer,
                    ngram_range=(experiment.ngram_min, experiment.ngram_max),
                    min_df=train.min_df,
                    max_features=train.max_features,
                    sublinear_tf=True,
                    lowercase=True,
                    norm="l2",
                    dtype=np.float32,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=experiment.c,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=train.max_iter,
                    random_state=train.seed,
                ),
            ),
        ]
    )


def save_model(pipeline: Pipeline, directory: Path, metadata: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    digest = sha256_file(model_path)
    result = {**metadata, "model_sha256": digest, "model_version": f"sha256:{digest}"}
    write_json(directory / "metadata.json", result)
    return result


def load_model(directory: Path) -> tuple[Pipeline, dict]:
    # joblib uses pickle: only load artifacts obtained from the documented trusted source.
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    digest = sha256_file(directory / "model.joblib")
    if digest != metadata["model_sha256"] or metadata["model_version"] != f"sha256:{digest}":
        raise ValueError("Model artifact checksum mismatch")
    model = joblib.load(directory / "model.joblib")
    if not isinstance(model, Pipeline) or list(model.classes_) != [0, 1]:
        raise ValueError("Expected a fitted binary sklearn pipeline with classes [0,1]")
    if model.named_steps["preprocessing"].version != metadata["preprocessing"]:
        raise ValueError("Preprocessing version does not match model metadata")
    return model, metadata
