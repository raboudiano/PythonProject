"""
Deployment utilities for selecting and serving the best trained model.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import MODEL_RANDOM_STATE, PRIMARY_TARGET, TEST_SIZE


MODELS_DIR = Path("Models")
RESULTS_PATH = MODELS_DIR / "training_results.json"
BUNDLE_PATH = MODELS_DIR / "best_model_bundle.joblib"
METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
STATUS_NDAYS_BUNDLE_PATH = MODELS_DIR / "status_ndays_bundle.joblib"
STATUS_NDAYS_METADATA_PATH = MODELS_DIR / "status_ndays_metadata.json"
STAGE_BUNDLE_PATH = MODELS_DIR / "stage_bundle.joblib"
STAGE_METADATA_PATH = MODELS_DIR / "stage_metadata.json"


def _load_results(path: Path = RESULTS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Training results not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def choose_best_model_name(path: Path = RESULTS_PATH) -> str:
    results = _load_results(path)
    if not results:
        raise ValueError("No model results available in training_results.json")

    best_name = max(
        results.keys(),
        key=lambda model_name: results[model_name].get("metrics", {}).get("test_f1", 0.0),
    )
    return best_name


def _build_model(model_name: str):
    builders = {
        "Logistic Regression": lambda: LogisticRegression(random_state=MODEL_RANDOM_STATE, max_iter=1000),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100,
            random_state=MODEL_RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=100,
            random_state=MODEL_RANDOM_STATE,
        ),
        "SVM": lambda: SVC(kernel="rbf", random_state=MODEL_RANDOM_STATE, probability=True),
    }

    if model_name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=100, random_state=MODEL_RANDOM_STATE, verbosity=0)

    if model_name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=100, random_state=MODEL_RANDOM_STATE, verbose=-1)

    if model_name not in builders:
        raise ValueError(f"Unsupported model name in results: {model_name}")

    return builders[model_name]()


def _representative_row_input(features_df: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    """Select a real row closest to the feature medians for a realistic sample payload."""
    medians = features_df[feature_names].median(numeric_only=True)
    distances = ((features_df[feature_names] - medians) ** 2).sum(axis=1)
    representative_idx = distances.idxmin()
    representative_row = features_df.loc[representative_idx, feature_names]
    return {column: float(representative_row[column]) for column in feature_names}


def get_realistic_sample_input(
    feature_names: list[str], cleaned_data_path: str = "Data/cirrhosis_cleaned.csv"
) -> dict[str, float]:
    """Build a realistic sample input from a representative row in cleaned data."""
    df = pd.read_csv(cleaned_data_path)
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in cleaned data: {missing}")

    return _representative_row_input(df, feature_names)


def train_and_save_best_bundle(cleaned_data_path: str = "Data/cirrhosis_cleaned.csv") -> dict:
    df = pd.read_csv(cleaned_data_path)
    if PRIMARY_TARGET not in df.columns:
        raise ValueError(f"Target column '{PRIMARY_TARGET}' is missing in cleaned dataset")

    best_model_name = choose_best_model_name()

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[PRIMARY_TARGET])
    X = df.drop(columns=[PRIMARY_TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = _build_model(best_model_name)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }

    default_input = _representative_row_input(X, list(X.columns))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_name": best_model_name,
        "model": model,
        "scaler": scaler,
        "target_encoder": target_encoder,
        "feature_names": list(X.columns),
        "target_column": PRIMARY_TARGET,
    }

    joblib.dump(bundle, BUNDLE_PATH)

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "test_f1",
        "bundle_path": str(BUNDLE_PATH),
        "trained_metrics": metrics,
        "sample_source": "representative_real_row",
        "default_input": default_input,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    return metadata


def load_or_train_bundle(cleaned_data_path: str = "Data/cirrhosis_cleaned.csv") -> tuple[dict, dict]:
    if not BUNDLE_PATH.exists() or not METADATA_PATH.exists():
        metadata = train_and_save_best_bundle(cleaned_data_path=cleaned_data_path)
    else:
        with METADATA_PATH.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

    bundle = joblib.load(BUNDLE_PATH)
    return bundle, metadata


def train_and_save_status_ndays_bundle(cleaned_data_path: str = "Data/cirrhosis_cleaned.csv") -> dict:
    """Train a two-model bundle for Status classification + N_Days regression."""
    df = pd.read_csv(cleaned_data_path)

    required_cols = {"Status", "N_Days"}
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for Status+N_Days bundle: {missing_required}")

    feature_cols = [col for col in df.columns if col not in ["Status", "N_Days"]]
    if not feature_cols:
        raise ValueError("No features available to train Status+N_Days models")

    X_raw = pd.get_dummies(df[feature_cols], drop_first=False)
    y_status_raw = df["Status"]
    y_ndays_raw = df["N_Days"].astype(float)

    status_encoder = LabelEncoder()
    y_status = status_encoder.fit_transform(y_status_raw)

    X_train, X_test, y_status_train, y_status_test, y_ndays_train, y_ndays_test = train_test_split(
        X_raw,
        y_status,
        y_ndays_raw,
        test_size=TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=y_status,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    status_model = LogisticRegression(random_state=MODEL_RANDOM_STATE, max_iter=2000)
    status_model.fit(X_train_scaled, y_status_train)

    ndays_model = RandomForestRegressor(
        n_estimators=300,
        random_state=MODEL_RANDOM_STATE,
        n_jobs=-1,
    )
    ndays_model.fit(X_train_scaled, y_ndays_train)

    status_pred = status_model.predict(X_test_scaled)
    ndays_pred = ndays_model.predict(X_test_scaled)

    metadata = {
        "selected_status_model": "Logistic Regression",
        "selected_ndays_model": "RandomForestRegressor",
        "bundle_path": str(STATUS_NDAYS_BUNDLE_PATH),
        "feature_names": list(X_raw.columns),
        "feature_base_columns": feature_cols,
        "trained_metrics": {
            "status_test_accuracy": float(accuracy_score(y_status_test, status_pred)),
            "status_test_precision": float(
                precision_score(y_status_test, status_pred, average="weighted", zero_division=0)
            ),
            "status_test_recall": float(recall_score(y_status_test, status_pred, average="weighted", zero_division=0)),
            "status_test_f1": float(f1_score(y_status_test, status_pred, average="weighted", zero_division=0)),
            "ndays_test_mae": float(mean_absolute_error(y_ndays_test, ndays_pred)),
            "ndays_test_r2": float(r2_score(y_ndays_test, ndays_pred)),
        },
    }

    bundle = {
        "status_model": status_model,
        "ndays_model": ndays_model,
        "status_encoder": status_encoder,
        "scaler": scaler,
        "feature_names": list(X_raw.columns),
        "feature_base_columns": feature_cols,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, STATUS_NDAYS_BUNDLE_PATH)

    with STATUS_NDAYS_METADATA_PATH.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    return metadata


def load_or_train_status_ndays_bundle(cleaned_data_path: str = "Data/cirrhosis_cleaned.csv") -> tuple[dict, dict]:
    if not STATUS_NDAYS_BUNDLE_PATH.exists() or not STATUS_NDAYS_METADATA_PATH.exists():
        metadata = train_and_save_status_ndays_bundle(cleaned_data_path=cleaned_data_path)
    else:
        with STATUS_NDAYS_METADATA_PATH.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

    bundle = joblib.load(STATUS_NDAYS_BUNDLE_PATH)
    return bundle, metadata


def train_and_save_stage_bundle(cleaned_data_path: str = "Data/cirrhosis_cleaned.csv") -> dict:
    """Train a dedicated model bundle for Stage prediction."""
    df = pd.read_csv(cleaned_data_path)

    if "Stage" not in df.columns:
        raise ValueError("Missing required target column 'Stage' in cleaned dataset")

    # Avoid leakage from downstream outcomes.
    blocked_features = {"Stage", "Status", "N_Days"}
    feature_cols = [col for col in df.columns if col not in blocked_features]
    if not feature_cols:
        raise ValueError("No valid features available for Stage prediction")

    X_raw = pd.get_dummies(df[feature_cols], drop_first=False)
    y_stage_raw = df["Stage"].astype(int)

    stage_encoder = LabelEncoder()
    y_stage = stage_encoder.fit_transform(y_stage_raw)

    X_train, X_test, y_stage_train, y_stage_test = train_test_split(
        X_raw,
        y_stage,
        test_size=TEST_SIZE,
        random_state=MODEL_RANDOM_STATE,
        stratify=y_stage,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    stage_model = LogisticRegression(random_state=MODEL_RANDOM_STATE, max_iter=2500)
    stage_model.fit(X_train_scaled, y_stage_train)

    y_pred = stage_model.predict(X_test_scaled)

    metadata = {
        "selected_stage_model": "Logistic Regression",
        "bundle_path": str(STAGE_BUNDLE_PATH),
        "feature_names": list(X_raw.columns),
        "feature_base_columns": feature_cols,
        "trained_metrics": {
            "stage_test_accuracy": float(accuracy_score(y_stage_test, y_pred)),
            "stage_test_precision": float(
                precision_score(y_stage_test, y_pred, average="weighted", zero_division=0)
            ),
            "stage_test_recall": float(recall_score(y_stage_test, y_pred, average="weighted", zero_division=0)),
            "stage_test_f1_weighted": float(f1_score(y_stage_test, y_pred, average="weighted", zero_division=0)),
            "stage_test_f1_macro": float(f1_score(y_stage_test, y_pred, average="macro", zero_division=0)),
        },
    }

    bundle = {
        "stage_model": stage_model,
        "stage_encoder": stage_encoder,
        "scaler": scaler,
        "feature_names": list(X_raw.columns),
        "feature_base_columns": feature_cols,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, STAGE_BUNDLE_PATH)

    with STAGE_METADATA_PATH.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    return metadata


def load_or_train_stage_bundle(cleaned_data_path: str = "Data/cirrhosis_cleaned.csv") -> tuple[dict, dict]:
    if not STAGE_BUNDLE_PATH.exists() or not STAGE_METADATA_PATH.exists():
        metadata = train_and_save_stage_bundle(cleaned_data_path=cleaned_data_path)
    else:
        with STAGE_METADATA_PATH.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

    bundle = joblib.load(STAGE_BUNDLE_PATH)
    return bundle, metadata
