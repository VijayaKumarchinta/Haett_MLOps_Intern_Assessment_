"""Tests for the local Feast feature-store configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_feature_store_configuration_exists():
    assert (ROOT / "feature_repo" / "feature_store.yaml").exists()


def test_feast_preparation_script_exists():
    assert (ROOT / "scripts" / "prepare_feast.py").exists()


def test_airflow_dag_exists():
    assert (ROOT / "airflow" / "dags" / "haett_mlops_pipeline.py").exists()


def test_prometheus_alert_rules_exist():
    assert (ROOT / "monitoring" / "prometheus_rules.yml").exists()
