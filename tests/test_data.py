import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from toxic_mlops.data import download_data, group_near_duplicates, prepare_data


def test_prepare_reproducible_and_grouped(raw_dataset):
    config = raw_dataset
    frame = pd.read_csv(config.data.path)
    # Two conflicting rows and a repeated consistent example.
    frame = pd.concat(
        [
            frame,
            frame.iloc[[0]],
            pd.DataFrame(
                [
                    {"comment": "Конфликтующий комментарий один", "toxic": 0},
                    {"comment": "Конфликтующий комментарий один", "toxic": 1},
                    {"comment": "мало", "toxic": 0},
                ]
            ),
        ],
        ignore_index=True,
    )
    frame.to_csv(config.data.path, index=False)
    from toxic_mlops.data import sha256_file

    config.data.sha256 = sha256_file(config.data.path)
    first = prepare_data(config)
    second = prepare_data(config)
    assert first == second
    assert first["clean_rows"] == 80 and first["conflict_rows"] == 2
    assert set(first["versions"]) == {"v1", "v2"}
    v1 = pd.read_csv(config.data.prepared_dir / "v1.csv")
    v2 = pd.read_csv(config.data.prepared_dir / "v2.csv")
    assert v1[["row_id", "split"]].equals(v2[["row_id", "split"]])
    assert v1.groupby("group").split.nunique().max() == 1


def test_near_duplicates():
    text = "Достаточно длинный комментарий для проверки очень близких текстов " * 3
    groups = group_near_duplicates([text, text + "!", "Совсем другой короткий текст"], 0.95, 40)
    assert groups[0] == groups[1] and groups[0] != groups[2]


def test_download_verified(raw_dataset, monkeypatch):
    config = raw_dataset
    content = config.data.path.read_bytes()
    config.data.path.unlink()
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("labeled.csv", content)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(memory.getvalue()))
    assert download_data(config).read_bytes() == content
    assert download_data(config).exists()
    config.data.path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        download_data(config)


def test_invalid_raw(raw_dataset):
    config = raw_dataset
    config.data.sha256 = "0" * 64
    with pytest.raises(ValueError, match="Verified"):
        prepare_data(config)
    from toxic_mlops.data import sha256_file

    for frame in (
        pd.DataFrame({"wrong": [1]}),
        pd.DataFrame({"comment": ["hello", "bye"], "toxic": [0, np.nan]}),
    ):
        frame.to_csv(config.data.path, index=False)
        config.data.sha256 = sha256_file(config.data.path)
        with pytest.raises(ValueError):
            prepare_data(config)
