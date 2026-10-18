# Changelog

## v0.1 - Baseline

- StandardScaler + LinearRegression in a single scikit-learn Pipeline.
- Fixed seed (42), 80/20 train/test split, pinned dependencies (Python 3.13).
- FastAPI service with `GET /health` and `POST /predict`; invalid input returns HTTP 422 with a JSON error.
- Docker image with the trained model baked in, listening on port 80.

| Metric | v0.1 |
|---|---|
| RMSE (held-out test set) | 53.85 |