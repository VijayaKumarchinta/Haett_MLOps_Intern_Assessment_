"""
Tests for the Metrics Utility Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
    assess_risk_level,
    get_business_recommendation,
)


def test_compute_classification_metrics():
    """Test basic classification metrics computation."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1


def test_compute_metrics_without_proba():
    """Test metrics without probability scores."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])

    metrics = compute_classification_metrics(y_true, y_pred)

    assert "roc_auc" not in metrics
    assert metrics["accuracy"] == 0.75


def test_find_optimal_threshold():
    """Test optimal threshold finding."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 0.95])

    result = find_optimal_threshold(y_true, y_proba)

    assert "optimal_threshold" in result
    assert "max_f1" in result
    assert "precision_at_optimal" in result
    assert "recall_at_optimal" in result
    assert 0 < result["optimal_threshold"] < 1
    assert 0 < result["max_f1"] <= 1


def test_assess_risk_level():
    """Test risk level classification."""
    assert assess_risk_level(0.1) == "Low"
    assert assess_risk_level(0.29) == "Low"
    assert assess_risk_level(0.3) == "Medium"
    assert assess_risk_level(0.5) == "Medium"
    assert assess_risk_level(0.6) == "High"
    assert assess_risk_level(0.9) == "High"


def test_get_business_recommendation_low_risk():
    """Test recommendation for low risk."""
    rec = get_business_recommendation(0.1)
    assert "No action needed" in rec
    assert "low risk" in rec.lower()


def test_get_business_recommendation_medium_risk():
    """Test recommendation for medium risk."""
    rec = get_business_recommendation(0.5)
    assert "Medium" in rec
    assert "15% discount" in rec or "re-engagement" in rec


def test_get_business_recommendation_high_risk():
    """Test recommendation for high risk."""
    rec = get_business_recommendation(0.8)
    assert "HIGH" in rec
    assert "50% off" in rec or "win-back" in rec


def test_get_recommendation_with_features():
    """Test recommendation with feature context."""
    features = {
        "days_since_last_order": 35,
        "avg_rating": 2.0,
        "is_sub_active": False,
        "login_decline": 3,
    }
    rec = get_business_recommendation(0.8, features)
    assert "HIGH" in rec
    assert "no orders" in rec or "ratings" in rec or "subscription" in rec
