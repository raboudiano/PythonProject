from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deploy_model import (
    get_realistic_sample_input,
    load_or_train_bundle,
    load_or_train_status_ndays_bundle,
    load_or_train_stage_bundle,
)


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(..., description="Feature dictionary using model feature names")


app = FastAPI(
    title="Cirrhosis Best Model API",
    description="FastAPI service that auto-selects the best model by F1 and serves predictions.",
    version="1.0.0",
)

bundle_cache: dict = {}
metadata_cache: dict = {}
status_ndays_bundle_cache: dict = {}
status_ndays_metadata_cache: dict = {}
stage_bundle_cache: dict = {}
stage_metadata_cache: dict = {}


def _get_model_assets():
    global bundle_cache, metadata_cache
    if not bundle_cache:
        bundle_cache, metadata_cache = load_or_train_bundle()
    return bundle_cache, metadata_cache


def _get_status_ndays_assets():
    global status_ndays_bundle_cache, status_ndays_metadata_cache
    if not status_ndays_bundle_cache:
        status_ndays_bundle_cache, status_ndays_metadata_cache = load_or_train_status_ndays_bundle()
    return status_ndays_bundle_cache, status_ndays_metadata_cache


def _get_stage_assets():
    global stage_bundle_cache, stage_metadata_cache
    if not stage_bundle_cache:
        stage_bundle_cache, stage_metadata_cache = load_or_train_stage_bundle()
    return stage_bundle_cache, stage_metadata_cache


def _build_status_ndays_feature_row(raw_features: dict[str, float], feature_base_columns: list[str], feature_names: list[str]) -> pd.DataFrame:
    incoming_df = pd.DataFrame([raw_features])
    base_df = incoming_df.reindex(columns=feature_base_columns, fill_value=0.0)
    encoded_df = pd.get_dummies(base_df, drop_first=False)
    aligned_df = encoded_df.reindex(columns=feature_names, fill_value=0.0)
    return aligned_df


@app.get("/", include_in_schema=False)
def read_home():
    return FileResponse(Path("frontend") / "home.html")


@app.get("/status", include_in_schema=False)
def read_status_page():
    return FileResponse(Path("frontend") / "index.html")


@app.get("/status-ndays", include_in_schema=False)
def read_status_ndays_page():
    return FileResponse(Path("frontend") / "status_ndays.html")


@app.get("/stage", include_in_schema=False)
def read_stage_page():
    return FileResponse(Path("frontend") / "stage.html")


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


@app.get("/model/status-ndays-info")
def status_ndays_info():
    bundle, metadata = _get_status_ndays_assets()
    return {
        "feature_count": len(bundle["feature_names"]),
        "feature_base_columns": bundle["feature_base_columns"],
        "targets": ["Status", "N_Days"],
        "trained_metrics": metadata.get("trained_metrics", {}),
    }


@app.get("/model/stage-info")
def stage_info():
    bundle, metadata = _get_stage_assets()
    stage_labels = [int(label) for label in bundle["stage_encoder"].classes_]
    return {
        "feature_count": len(bundle["feature_names"]),
        "feature_base_columns": bundle["feature_base_columns"],
        "target": "Stage",
        "target_labels": stage_labels,
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


@app.get("/model/status-ndays-sample-input")
def status_ndays_sample_input():
    bundle, _ = _get_status_ndays_assets()
    df = pd.read_csv("Data/cirrhosis_cleaned.csv")
    sample = df[bundle["feature_base_columns"]].median(numeric_only=True).to_dict()
    # Fill non-numeric base columns if they exist.
    for column in bundle["feature_base_columns"]:
        if column not in sample:
            mode_series = df[column].mode(dropna=True)
            sample[column] = mode_series.iloc[0] if not mode_series.empty else 0
    normalized = {key: float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value for key, value in sample.items()}
    return {"features": normalized}


@app.get("/model/stage-sample-input")
def stage_sample_input():
    bundle, _ = _get_stage_assets()
    df = pd.read_csv("Data/cirrhosis_cleaned.csv")
    sample = df[bundle["feature_base_columns"]].median(numeric_only=True).to_dict()
    for column in bundle["feature_base_columns"]:
        if column not in sample:
            mode_series = df[column].mode(dropna=True)
            sample[column] = mode_series.iloc[0] if not mode_series.empty else 0
    normalized = {key: float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value for key, value in sample.items()}
    return {"features": normalized}


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


@app.post("/predict/status-ndays")
def predict_status_ndays(request: PredictionRequest):
    bundle, _ = _get_status_ndays_assets()

    feature_base_columns = bundle["feature_base_columns"]
    feature_names = bundle["feature_names"]

    received_keys = set(request.features.keys())
    expected_keys = set(feature_base_columns)

    missing = sorted(expected_keys - received_keys)
    extras = sorted(received_keys - expected_keys)

    if missing or extras:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature keys mismatch for Status+N_Days endpoint",
                "missing_features": missing,
                "extra_features": extras,
            },
        )

    feature_row = _build_status_ndays_feature_row(
        raw_features=request.features,
        feature_base_columns=feature_base_columns,
        feature_names=feature_names,
    )

    scaled = bundle["scaler"].transform(feature_row)

    status_pred_idx = int(bundle["status_model"].predict(scaled)[0])
    status_pred_label = str(bundle["status_encoder"].inverse_transform([status_pred_idx])[0])

    status_probabilities = {}
    if hasattr(bundle["status_model"], "predict_proba"):
        proba = bundle["status_model"].predict_proba(scaled)[0]
        labels = bundle["status_encoder"].classes_
        status_probabilities = {str(label): float(value) for label, value in zip(labels, proba)}

    ndays_pred = float(bundle["ndays_model"].predict(scaled)[0])

    return {
        "predicted_status": status_pred_label,
        "status_probabilities": status_probabilities,
        "estimated_n_days": max(0.0, ndays_pred),
    }


@app.post("/predict/stage")
def predict_stage(request: PredictionRequest):
    bundle, _ = _get_stage_assets()

    feature_base_columns = bundle["feature_base_columns"]
    feature_names = bundle["feature_names"]

    received_keys = set(request.features.keys())
    expected_keys = set(feature_base_columns)

    missing = sorted(expected_keys - received_keys)
    extras = sorted(received_keys - expected_keys)

    if missing or extras:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature keys mismatch for Stage endpoint",
                "missing_features": missing,
                "extra_features": extras,
            },
        )

    feature_row = _build_status_ndays_feature_row(
        raw_features=request.features,
        feature_base_columns=feature_base_columns,
        feature_names=feature_names,
    )

    scaled = bundle["scaler"].transform(feature_row)

    pred_idx = int(bundle["stage_model"].predict(scaled)[0])
    pred_stage = int(bundle["stage_encoder"].inverse_transform([pred_idx])[0])

    probabilities = {}
    if hasattr(bundle["stage_model"], "predict_proba"):
        proba = bundle["stage_model"].predict_proba(scaled)[0]
        labels = bundle["stage_encoder"].classes_
        probabilities = {str(int(label)): float(value) for label, value in zip(labels, proba)}

    return {
        "predicted_stage": pred_stage,
        "stage_probabilities": probabilities,
    }


if Path("frontend").exists():
    app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
