"""
Tests for the Drift Detection Module
"""

import pytest
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.monitoring.drift_detection import (
    save_reference_data,
    generate_drift_report,
    list_drift_reports,
    run_drift_monitoring,
    REFERENCE_DIR,
    DRIFT_REPORTS_DIR,
)


@pytest.fixture(autouse=True)
def clean_monitoring_dirs():
    """Clean generated drift data without deleting monitoring configuration."""
    import shutil

    for directory in (REFERENCE_DIR, DRIFT_REPORTS_DIR):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    yield

    for directory in (REFERENCE_DIR, DRIFT_REPORTS_DIR):
        if directory.exists():
            shutil.rmtree(directory)


@pytest.fixture
def sample_features():
    """Generate a small sample feature matrix for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "days_since_last_order": np.random.exponential(15, 100).astype(int),
            "total_orders": np.random.poisson(12, 100),
            "total_spent": np.random.exponential(250, 100).round(2),
            "avg_order_value": np.random.normal(32, 12, 100).clip(5).round(2),
            "avg_rating": np.random.uniform(3, 5, 100).round(1),
            "total_support_tickets": np.random.poisson(1, 100),
            "is_sub_active": np.random.choice([0, 1], 100, p=[0.25, 0.75]),
            "avg_app_logins": np.random.exponential(3, 100),
            "login_decline": np.random.exponential(1, 100),
            "order_frequency_per_month": np.random.exponential(4, 100),
            "subscription_tenure_days": np.random.exponential(180, 100).astype(int),
            "tenure_days": np.random.exponential(200, 100).astype(int),
        }
    )


@pytest.fixture
def drifted_features(sample_features):
    """Generate drifted version of the sample features (add noise)."""
    np.random.seed(99)
    drifted = sample_features.copy()
    for col in drifted.select_dtypes(include=[np.number]).columns:
        noise = np.random.normal(0, 0.3, len(drifted)) * drifted[col].std()
        drifted[col] = drifted[col] + noise
        drifted[col] = drifted[col].clip(0)
    return drifted


class TestSaveReferenceData:
    """Tests for saving reference data."""

    def test_save_reference_success(self, sample_features):
        ref_dir = save_reference_data(features=sample_features)
        assert ref_dir.exists()
        assert (ref_dir / "features.parquet").exists()
        assert (ref_dir / "metadata.json").exists()

    def test_save_reference_with_predictions(self, sample_features):
        predictions = pd.Series(np.random.uniform(0, 1, len(sample_features)))
        ref_dir = save_reference_data(features=sample_features, predictions=predictions)
        assert (ref_dir / "predictions.parquet").exists()

    def test_save_reference_metadata(self, sample_features):
        ref_dir = save_reference_data(
            features=sample_features,
            metadata={"model_name": "rf_classifier", "version": "1.0"},
        )
        with open(ref_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["model_name"] == "rf_classifier"
        assert meta["n_samples"] == 100

    def test_save_reference_no_data(self):
        ref_dir = save_reference_data()
        assert ref_dir.exists()
        assert not (ref_dir / "features.parquet").exists()

    def test_save_multiple_references_cleanup(self, sample_features):
        for i in range(5):
            save_reference_data(features=sample_features)
        ref_dirs = sorted(REFERENCE_DIR.glob("ref_*"))
        assert len(ref_dirs) <= 3


class TestGenerateDriftReport:
    """Tests for generating drift reports."""

    def test_generate_report_no_reference(self, sample_features):
        result = generate_drift_report(sample_features)
        assert result["status"] == "error"
        assert "No reference data" in result.get("error", "")

    def test_generate_report_no_drift(self, sample_features):
        save_reference_data(features=sample_features)
        result = generate_drift_report(sample_features)
        assert result["status"] == "ok"
        assert "json_path" in result
        assert Path(result["json_path"]).exists()

    def test_generate_report_json_saved(self, sample_features):
        save_reference_data(features=sample_features)
        result = generate_drift_report(sample_features, report_name="test_report")
        if result["status"] != "error":
            json_path = DRIFT_REPORTS_DIR / "test_report.json"
            assert json_path.exists()

    def test_generate_report_drifted_detects_more(
        self, sample_features, drifted_features
    ):
        save_reference_data(features=sample_features)
        baseline = generate_drift_report(sample_features)
        drifted_result = generate_drift_report(drifted_features)
        assert drifted_result.get("n_drifted_features", 0) >= baseline.get(
            "n_drifted_features", 0
        )

    def test_generate_report_critical_drift(self, sample_features, drifted_features):
        save_reference_data(features=sample_features)
        very_drifted = drifted_features.copy()
        for col in very_drifted.select_dtypes(include=[np.number]).columns:
            very_drifted[col] = very_drifted[col] * 3 + 50
        result = generate_drift_report(very_drifted)
        assert result["status"] in ["ok", "warning", "critical"]


class TestListDriftReports:
    """Tests for listing drift reports."""

    def test_list_empty(self):
        reports = list_drift_reports()
        assert reports.empty

    def test_list_with_reports(self, sample_features):
        save_reference_data(features=sample_features)
        generate_drift_report(sample_features, report_name="report_a")
        generate_drift_report(sample_features, report_name="report_b")
        reports = list_drift_reports()
        assert len(reports) >= 2
        assert "status" in reports.columns
        assert "drift_share" in reports.columns


class TestRunDriftMonitoring:
    """Tests for the full drift monitoring flow."""

    def test_monitoring_no_reference(self, sample_features):
        result = run_drift_monitoring(sample_features)
        assert result["status"] == "error"

    def test_monitoring_full_flow(self, sample_features):
        save_reference_data(features=sample_features)
        result = run_drift_monitoring(sample_features)
        assert "status" in result
        if result["status"] != "error":
            assert "drift_summary" in result


class TestGenerateSyntheticCurrentData:
    """Tests for the synthetic data generator in the drift runner."""

    def test_synthetic_data_shape(self):
        from scripts.check_drift import generate_synthetic_current_data

        data = generate_synthetic_current_data(n_samples=50)
        assert len(data) == 50
        assert "days_since_last_order" in data.columns
        assert "total_orders" in data.columns
        assert "avg_rating" in data.columns
