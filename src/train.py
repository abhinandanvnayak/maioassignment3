import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.2
VERSION = os.getenv("MODEL_VERSION", "v0.1")


def main() -> None:
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    if VERSION == "v0.1":
        regressor = LinearRegression()
        model_type = "StandardScaler + LinearRegression"
    elif VERSION == "v0.2":
        regressor = Ridge(alpha=1.0)
        model_type = "StandardScaler + Ridge"
    else:
        raise ValueError(f"Unsupported model version: {VERSION}")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    metrics = {
        "model_version": VERSION,
        "model_type": model_type,
        "rmse": rmse,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X.columns),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
    }

    joblib.dump(model, "model.joblib")
    Path("metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
