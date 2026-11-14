"""Verified raw data, label-independent near-duplicate grouping and fixed splits."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from toxic_mlops.config import Settings
from toxic_mlops.preprocessing import clean_text, duplicate_key, preprocess


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download_data(settings: Settings) -> Path:
    target = settings.data.path
    if not target.exists():
        request = urllib.request.Request(
            settings.data.download_url, headers={"User-Agent": "toxic-mlops-coursework/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read(30 * 1024 * 1024 + 1)
        if len(content) > 30 * 1024 * 1024:
            raise ValueError("Dataset download exceeds 30 MiB limit")
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [n for n in archive.namelist() if Path(n).name == "labeled.csv"]
            if len(members) != 1 or archive.getinfo(members[0]).file_size > 30 * 1024 * 1024:
                raise ValueError("Archive must contain exactly one small labeled.csv")
            csv_bytes = archive.read(members[0])
        if hashlib.sha256(csv_bytes).hexdigest() != settings.data.sha256:
            raise ValueError("Downloaded dataset checksum mismatch; refusing new dataset version")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(csv_bytes)
    if sha256_file(target) != settings.data.sha256:
        raise ValueError("Raw dataset checksum mismatch")
    return target


def group_near_duplicates(texts, threshold: float, min_chars: int) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    parent = np.arange(len(texts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(5, 5),
        n_features=2**18,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )
    matrix = vectorizer.transform(texts)
    long = np.array([len(t) >= min_chars for t in texts])
    for start in range(0, len(texts), 256):
        similarity = (matrix[start : start + 256] @ matrix.T).tocoo()
        keep = similarity.data >= threshold
        for local_i, j in zip(similarity.row[keep], similarity.col[keep], strict=True):
            i = start + int(local_i)
            if j > i and long[i] and long[j]:
                a, b = find(i), find(int(j))
                parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(len(texts))])


def prepare_data(settings: Settings) -> dict:
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    path = settings.data.path
    if not path.is_file() or sha256_file(path) != settings.data.sha256:
        raise ValueError("Verified labeled.csv is required; run download or copy the original CSV")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["comment", "toxic"]:
        raise ValueError("Expected columns comment,toxic")
    if frame.toxic.isna().any() or set(frame.toxic.unique()) != {0, 1}:
        raise ValueError("Labels must contain exactly 0 and 1 without missing values")
    frame["row_id"] = np.arange(len(frame))
    frame["label"] = frame.toxic.astype("int64")
    frame["text"] = frame.comment.fillna("").map(clean_text)
    frame["key"] = frame.text.map(duplicate_key)
    raw_rows = len(frame)
    frame = frame.loc[frame.text.str.len().ge(settings.train.min_chars)].copy()
    after_length = len(frame)
    conflicting = frame.groupby("key").label.nunique().loc[lambda x: x > 1].index
    conflict_rows = int(frame.key.isin(conflicting).sum())
    frame = frame.loc[~frame.key.isin(conflicting)].drop_duplicates("key").reset_index(drop=True)
    frame["group"] = group_near_duplicates(
        frame.key.tolist(),
        settings.train.near_duplicate_threshold,
        settings.train.near_duplicate_min_chars,
    )
    fold = np.full(len(frame), -1)
    splitter = StratifiedGroupKFold(
        n_splits=settings.train.split_folds, shuffle=True, random_state=settings.train.seed
    )
    for k, (_, indices) in enumerate(splitter.split(frame.text, frame.label, frame.group)):
        fold[indices] = k
    frame["split"] = np.where(
        fold < settings.train.test_folds,
        "test",
        np.where(
            fold < settings.train.test_folds + settings.train.validation_folds,
            "validation",
            "train",
        ),
    )
    if frame.groupby("group").split.nunique().max() != 1:
        raise RuntimeError("Group leakage detected")
    if frame.groupby("key").split.nunique().max() != 1:
        raise RuntimeError("Duplicate leakage detected")
    output = settings.data.prepared_dir
    output.mkdir(parents=True, exist_ok=True)
    splits_path = output / "splits.csv"
    frame[["row_id", "group", "split"]].to_csv(splits_path, index=False, lineterminator="\n")
    versions = {}
    for version in ("v1", "v2"):
        version_frame = frame[["row_id", "text", "label", "group", "split"]].copy()
        version_frame["text"] = frame.text.map(lambda t, v=version: preprocess(t, v))
        version_path = output / f"{version}.csv"
        version_frame.to_csv(version_path, index=False, lineterminator="\n")
        versions[version] = {"sha256": sha256_file(version_path), "rows": len(version_frame)}
    # Experiments read minimal texts; each persisted pipeline applies its own preprocessing once.
    summary = {
        "source": settings.data.source_url,
        "raw_sha256": sha256_file(path),
        "raw_rows": raw_rows,
        "after_length": after_length,
        "conflict_rows": conflict_rows,
        "clean_rows": len(frame),
        "positive_rows": int(frame.label.sum()),
        "split_sha256": sha256_file(splits_path),
        "versions": versions,
        "splits": frame.groupby("split").label.agg(["size", "sum", "mean"]).to_dict("index"),
        "protocol": {
            "seed": settings.train.seed,
            "min_chars": settings.train.min_chars,
            "split_folds": settings.train.split_folds,
            "test_folds": settings.train.test_folds,
            "validation_folds": settings.train.validation_folds,
            "near_threshold": settings.train.near_duplicate_threshold,
            "near_min_chars": settings.train.near_duplicate_min_chars,
        },
    }
    write_json(output / "manifest.json", summary)
    return summary
