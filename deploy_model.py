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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from config import MODEL_RANDOM_STATE, PRIMARY_TARGET, TEST_SIZE


MODELS_DIR = Path("Models")
RESULTS_PATH = MODELS_DIR / "training_results.json"
BUNDLE_PATH = MODELS_DIR / "best_model_bundle.joblib"
METADATA_PATH = MODELS_DIR / "best_model_metadata.json"


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
