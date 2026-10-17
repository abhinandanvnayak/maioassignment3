import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
MODEL_PATH = Path(__file__).resolve().parents[1] / "model.joblib"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

model = joblib.load(MODEL_PATH)
app = FastAPI()


class DiabetesFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(features: DiabetesFeatures) -> dict[str, float]:
    values = pd.DataFrame([features.model_dump()], columns=FEATURES)
    prediction = model.predict(values)[0]
    return {"prediction": float(prediction)}