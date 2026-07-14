"""Register the committed best model in MLflow Model Registry."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

MODEL_PATH = MODELS_DIR / "churn_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REGISTRY_METADATA_PATH = MODELS_DIR / "registry_metadata.json"

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{ROOT / 'mlflow.db'}",
)

REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "haett_churn_model",
)

MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}

    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def log_numeric_metrics(
    metadata: dict[str, Any],
) -> None:
    metric_sources = [
        metadata,
        metadata.get("metrics", {}),
        metadata.get("best_metrics", {}),
    ]

    logged_names: set[str] = set()

    for metric_source in metric_sources:
        if not isinstance(metric_source, dict):
            continue

        for name, value in metric_source.items():
            if isinstance(value, (int, float)) and name not in logged_names:
                mlflow.log_metric(name, float(value))
                logged_names.add(name)


def register_best_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. " "Run the training pipeline first."
        )

    metadata = load_metadata()
    model = joblib.load(MODEL_PATH)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("haett_churn_registry")

    with mlflow.start_run(run_name="register_best_churn_model") as run:
        selected_model = metadata.get(
            "best_model",
            metadata.get(
                "best_model_name",
                metadata.get(
                    "model_name",
                    "unknown",
                ),
            ),
        )

        mlflow.log_param(
            "selected_model",
            str(selected_model),
        )
        mlflow.log_param(
            "registered_model_name",
            REGISTERED_MODEL_NAME,
        )

        log_numeric_metrics(metadata)

        # cloudpickle supports complex sklearn objects such as
        # CalibratedClassifierCV and its internal calibration objects.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
        )

        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"

    registered_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient(tracking_uri=TRACKING_URI)

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=registered_version.version,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered_version.version,
        key="assessment",
        value="Haett MLOps Internship",
    )

    output = {
        "tracking_uri": TRACKING_URI,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": str(registered_version.version),
        "alias": MODEL_ALIAS,
        "run_id": run_id,
        "model_uri": model_uri,
    }

    REGISTRY_METADATA_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    register_best_model()
