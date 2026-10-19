# Changelog

## v0.2 - Ridge regression

**What changed:** `LinearRegression` replaced by `Ridge(alpha=1.0)` in the same
`StandardScaler` pipeline. `alpha` was chosen by 5-fold cross-validation on the
training set. Data, split, seed and API are unchanged.

| Metric | v0.1 | v0.2 |
|---|---|---|
| Test RMSE | 53.85 | 53.78 |
| CV RMSE (train) | 55.39 | 55.37 |
| Precision (progression >= 200) | 0.85 | 0.85 |
| Recall (progression >= 200) | 0.46 | 0.46 |

**Why it is an improvement:** Ridge reduces variance from the strongly
correlated features and lowers RMSE on both the cross-validation folds and the
held-out test set. The gain is small, because the linear baseline is already
close to the best achievable on this dataset. Model size and latency are
unchanged.

## v0.1 - Baseline

**What changed:** First release. `StandardScaler` + `LinearRegression`, FastAPI
service with `/health` and `/predict`, Docker image with the model baked in.

| Metric | v0.1 |
|---|---|
| Test RMSE | 53.85 |