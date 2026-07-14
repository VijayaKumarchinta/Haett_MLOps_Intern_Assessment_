"""Airflow DAG for the non-deployment Haett MLOps workflow."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

PROJECT_ROOT = Path(
    os.getenv(
        "HAETT_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
)
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
FEAST_BINARY = PROJECT_ROOT / ".venv" / "bin" / "feast"


def execute(command: list[str]) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


@dag(
    dag_id="haett_mlops_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["haett", "mlops", "churn"],
    description=(
        "Generate data, preprocess, engineer features, train, "
        "register the model, update Feast, and run tests."
    ),
)
def haett_mlops_pipeline():
    @task
    def generate_data():
        execute(
            [
                str(PROJECT_PYTHON),
                "-c",
                (
                    "from src.data.generate_data "
                    "import generate_all_data; generate_all_data()"
                ),
            ]
        )

    @task
    def preprocess_data():
        execute(
            [
                str(PROJECT_PYTHON),
                "-c",
                (
                    "from src.data.preprocess "
                    "import preprocess_all; preprocess_all()"
                ),
            ]
        )

    @task
    def engineer_features():
        execute(
            [
                str(PROJECT_PYTHON),
                "-c",
                (
                    "from src.data.feature_engineering "
                    "import build_feature_matrix; "
                    "build_feature_matrix()"
                ),
            ]
        )

    @task
    def train_models():
        execute(
            [
                str(PROJECT_PYTHON),
                "-c",
                (
                    "from src.models.train "
                    "import train_all_models; train_all_models()"
                ),
            ]
        )

    @task
    def register_model():
        execute(
            [
                str(PROJECT_PYTHON),
                "scripts/register_best_model.py",
            ]
        )

    @task
    def update_feature_store():
        execute(
            [
                str(PROJECT_PYTHON),
                "scripts/prepare_feast.py",
            ]
        )
        execute([str(FEAST_BINARY), "apply"])
        execute(
            [
                str(FEAST_BINARY),
                "materialize-incremental",
                pendulum.now("UTC").to_iso8601_string(),
            ]
        )

    @task
    def run_tests():
        execute(
            [
                str(PROJECT_PYTHON),
                "-m",
                "pytest",
                "tests",
                "-q",
            ]
        )

    generated = generate_data()
    processed = preprocess_data()
    engineered = engineer_features()
    trained = train_models()

    registered = register_model()
    feature_store = update_feature_store()
    verified = run_tests()

    generated >> processed >> engineered >> trained
    trained >> [registered, feature_store]
    [registered, feature_store] >> verified


haett_mlops_pipeline()
