from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deploy_model import get_realistic_sample_input, load_or_train_bundle


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature dictionary using model feature names")


app = FastAPI(
    title="Cirrhosis Best Model API",
    description="FastAPI service that auto-selects the best model by F1 and serves predictions.",
    version="1.0.0",
)

bundle_cache: dict = {}
metadata_cache: dict = {}


def _get_model_assets():
    global bundle_cache, metadata_cache
    if not bundle_cache:
        bundle_cache, metadata_cache = load_or_train_bundle()
    return bundle_cache, metadata_cache


@app.get("/", include_in_schema=False)
def read_home():
    return FileResponse(Path("frontend") / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    bundle, metadata = _get_model_assets()
    return {
        "selected_model": bundle["model_name"],
        "feature_count": len(bundle["feature_names"]),
        "target_labels": list(bundle["target_encoder"].classes_),
        "selection_metric": metadata.get("selection_metric", "test_f1"),
        "trained_metrics": metadata.get("trained_metrics", {}),
    }


@app.get("/model/sample-input")
def sample_input():
    bundle, metadata = _get_model_assets()
    try:
        realistic = get_realistic_sample_input(bundle["feature_names"])
        return {"features": realistic}
    except Exception:
        return {"features": metadata.get("default_input", {})}


@app.post("/predict")
def predict(request: PredictionRequest):
    bundle, _ = _get_model_assets()

    feature_names = bundle["feature_names"]
    received_keys = set(request.features.keys())
    expected_keys = set(feature_names)

    missing = sorted(expected_keys - received_keys)
    extras = sorted(received_keys - expected_keys)

    if missing or extras:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature keys mismatch",
                "missing_features": missing,
                "extra_features": extras,
            },
        )

    input_df = pd.DataFrame([request.features])[feature_names]
    scaled = bundle["scaler"].transform(input_df)

    model = bundle["model"]
    pred_idx = int(model.predict(scaled)[0])
    pred_label = str(bundle["target_encoder"].inverse_transform([pred_idx])[0])

    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
        labels = bundle["target_encoder"].classes_
        probabilities = {str(label): float(value) for label, value in zip(labels, proba)}
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(scaled)[0]
        labels = bundle["target_encoder"].classes_
        if isinstance(scores, np.ndarray):
            exp_scores = np.exp(scores - np.max(scores))
            softmax_scores = exp_scores / exp_scores.sum()
            probabilities = {str(label): float(value) for label, value in zip(labels, softmax_scores)}

    return {
        "model": bundle["model_name"],
        "prediction": pred_label,
        "probabilities": probabilities,
    }


if Path("frontend").exists():
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
