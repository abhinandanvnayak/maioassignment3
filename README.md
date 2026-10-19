# Virtual Diabetes Clinic Triage

A small ML service for a virtual diabetes clinic. It predicts short-term disease
progression from ten patient features and returns a continuous risk score
(higher means greater expected deterioration). The clinic dashboard uses the
score to sort patients so nurses can prioritise follow-up calls.

The data is scikit-learn's open Diabetes regression dataset, used as a stand-in
for de-identified EHR features. The target acts as a "progression index".

The whole workflow (lint, tests, training, image build, release) runs in GitHub
Actions.

## Released images

Public images are published to GitHub Container Registry. No login is needed.

| Version | Image |
|---|---|
| v0.1 (baseline) | `ghcr.io/abhinandanvnayak/maioassignment3:v0.1` |
| v0.2 (improved) | `ghcr.io/abhinandanvnayak/maioassignment3:v0.2` |

Run a version (both use port 80, so stop one before starting the other):

```bash
docker pull ghcr.io/abhinandanvnayak/maioassignment3:v0.1
docker run -it -p 80:80 ghcr.io/abhinandanvnayak/maioassignment3:v0.1
```

```bash
docker pull ghcr.io/abhinandanvnayak/maioassignment3:v0.2
docker run -it -p 80:80 ghcr.io/abhinandanvnayak/maioassignment3:v0.2
```

To stop a container: press `Ctrl+C`, or run `docker ps` and then
`docker stop <container-name>`.

The images are built for `linux/amd64`. On Apple Silicon Macs Docker runs them
under emulation and prints a platform warning, which is harmless.

## API

The service listens on port 80 inside the container.

### `GET /health`

```bash
curl http://localhost:80/health
```

```json
{"status":"ok","model_version":"v0.1"}
```

`model_version` is `v0.1` or `v0.2` depending on the image.

### `POST /predict`

All ten features are required and must be numbers. They use the (already
standardised) scikit-learn Diabetes feature scale.

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

Successful response:

```json
{"prediction": 235.94963722176266}
```

The number is the predicted progression index. Higher means higher risk, so the
dashboard should sort in descending order.

### Error responses

Invalid input returns **HTTP 422** with a JSON body. The format is FastAPI's
standard validation error:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required",
      "input": {"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02}
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `detail` | List with one entry per problem, so several errors are reported at once |
| `type` | Kind of error, for example `missing` or `float_parsing` |
| `loc` | Where the problem is. `["body", "s6"]` means the field `s6` in the JSON body |
| `msg` | Human-readable description |
| `input` | The value (or whole body, for a missing field) that was received |

What triggers a 422:

- A required feature is missing (any of `age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`).
- A feature is not a number, for example `"bmi": "abc"` or `"bmi": null`.
- The request contains an unknown extra field.
- The body is not valid JSON.

Example, reproducing the response above:

```bash
curl -i -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02}'
```

## Run locally (without Docker)

Requires Python 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Train the model, run the tests and the linter, then start the API:

```bash
python -m src.train
python -m pytest -q
ruff check .
uvicorn app.main:app --port 8000
```

Training writes two files in the repository root: `model.joblib` (the trained
pipeline) and `metrics.json` (RMSE, seed, library versions). The API loads
`model.joblib` at startup, so train before starting it. Both files are
git-ignored because CI and the Docker build regenerate them.

Locally the API is then at `http://localhost:8000`.

### Build the image yourself

```bash
docker build -t diabetes-model:v0.1 .
docker build --build-arg MODEL_VERSION=v0.2 -t diabetes-model:v0.2 .
docker run -it -p 80:80 diabetes-model:v0.1
```

The build installs the pinned dependencies, copies `app/` and `src/`, and trains
the model inside the image, so the image is self-contained and needs no
downloads or volumes at runtime. `MODEL_VERSION` is a build argument that ends up
in `metrics.json` and in the `/health` response.

## Reproducibility

- **Seed:** a fixed `random_state` (42) is used for the train/test split and for
  every model that has randomness.
- **Split:** 80% training / 20% held-out test, identical for every version, so
  metrics are comparable.
- **Environment:** exact versions are pinned in `requirements.txt` (runtime) and
  `requirements-dev.txt` (tests and linting). Python is 3.13 locally, in CI, and
  in the Docker base image (`python:3.13-slim`).
- **Metrics:** every training run writes `metrics.json` with the RMSE, the seed,
  the split sizes, the feature list, and the Python, scikit-learn and NumPy
  versions. CI uploads it as an artifact, and the release attaches it to the
  GitHub Release.

Running `python -m src.train` twice gives identical metrics.

The model is chosen by the `MODEL_VERSION` environment variable (default `v0.1`). Run `MODEL_VERSION=v0.2 python -m src.train` to train the v0.2 model locally.

## Model versions

| Version | Model | Notes |
|---|---|---|
| v0.1 | `StandardScaler` + `LinearRegression` | Baseline, test RMSE 53.85 |
| v0.2 | `StandardScaler` + `Ridge(alpha=1.0)` | Test RMSE 53.78 (v0.1: 53.85), details in [CHANGELOG.md](CHANGELOG.md) |

## CI/CD

Two GitHub Actions workflows live in `.github/workflows/`.

**CI** (`ci.yaml`) runs on every push and on pull requests into `main`:

1. Install pinned dependencies.
2. Lint with `ruff` (the workflow fails on lint errors).
3. Train the model as a smoke test and check the RMSE is below a sanity bound.
4. Run the unit tests with `pytest` (the workflow fails on test failures).
5. Upload `model.joblib` and `metrics.json` as artifacts.

**Release** (`release.yml`) runs when a tag starting with `v` is pushed:

1. Build the Docker image with the tag as `MODEL_VERSION`.
2. Start the container and smoke-test `/health` (including the version),
   `/predict`, and the 422 response for invalid input.
3. Only if the smoke tests pass: push the image to GHCR.
4. Publish a GitHub Release with the metrics and the changelog.

To cut a release, merge to `main` and then:

```bash
git checkout main
git pull
git tag v0.2
git push origin v0.2
```

## Repository layout

```
app/                 FastAPI service (main.py)
src/                 training script (train.py)
tests/               API tests
.github/workflows/   CI and release workflows
Dockerfile           self-contained image, model trained at build time
requirements.txt     pinned runtime dependencies
requirements-dev.txt pinned test and lint dependencies
CHANGELOG.md         what changed between versions, with metrics
```