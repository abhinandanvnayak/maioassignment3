import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge

import src.train as train


def test_v02_trains_ridge_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "VERSION", "v0.2")

    train.main()

    model = joblib.load("model.joblib")
    assert isinstance(model.named_steps["regressor"], Ridge)


def test_v02_saves_correct_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "VERSION", "v0.2")

    train.main()

    with open("metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)

    assert metrics["model_version"] == "v0.2"
    assert metrics["model_type"] == "StandardScaler + Ridge"
    assert metrics["seed"] == 42
    assert metrics["n_train"] == 353
    assert metrics["n_test"] == 89
    assert metrics["rmse"] > 0


def test_v02_model_predicts_finite_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "VERSION", "v0.2")

    train.main()

    model = joblib.load("model.joblib")
    sample = np.zeros((1, 10))

    with pytest.warns(UserWarning):
        prediction = model.predict(sample)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])


def test_v02_improves_on_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "VERSION", "v0.2")

    train.main()

    with open("metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)

    assert metrics["rmse"] < 53.85344583676593
