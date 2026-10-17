# Virtual Diabetes Clinic Triage

## Run locally

Create and activate a virtual environment, then install the development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Train the baseline model and start the API from the repository root:

```bash
python -m src.train
uvicorn app.main:app --host 0.0.0.0 --port 80
```

The training command creates `model.joblib`, which the API loads when it starts.
The service listens on port 80 and provides `/health` and `/predict`.

## API

Check service health:

```bash
curl http://localhost:80/health
```

Example prediction:

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

The response is `{"prediction": <number>}`. Invalid requests return HTTP 422
with FastAPI's JSON validation format, for example:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```
