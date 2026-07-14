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
    """Test risk level classification with dynamic thresholds.

    With default optimal_threshold=0.5:
        Low:    < 0.25
        Medium: 0.25 - 0.75
        High:   > 0.75
    """
    # Default threshold (0.5): boundaries at 0.25 and 0.75
    assert assess_risk_level(0.1) == "Low"
    assert assess_risk_level(0.2) == "Low"
    assert assess_risk_level(0.3) == "Medium"
    assert assess_risk_level(0.5) == "Medium"
    assert assess_risk_level(0.7) == "Medium"
    assert assess_risk_level(0.8) == "High"
    assert assess_risk_level(0.9) == "High"

    # Custom threshold (e.g., XGBoost typical ~0.18): boundaries at 0.09 and 0.27
    assert assess_risk_level(0.05, optimal_threshold=0.18) == "Low"
    assert assess_risk_level(0.12, optimal_threshold=0.18) == "Medium"
    assert assess_risk_level(0.30, optimal_threshold=0.18) == "High"


def test_get_business_recommendation_low_risk():
    """Test recommendation for low risk."""
    rec = get_business_recommendation(0.1)
    assert "No action needed" in rec
    assert "low risk" in rec.lower()


def test_get_business_recommendation_medium_risk():
    """Test recommendation for medium risk."""
    rec = get_business_recommendation(0.5)
    assert "At-risk" in rec
    assert "Recommended actions" in rec


def test_get_business_recommendation_high_risk():
    """Test recommendation for high risk."""
    rec = get_business_recommendation(0.8)
    assert "HIGH" in rec
    assert "win-back" in rec or "immediate retention" in rec.lower()


def test_get_recommendation_with_shap_explanations():
    """Test recommendation with SHAP-driven feature explanations."""
    shap_explanations = [
        {"feature": "days_since_last_order", "value": 35.0, "impact": 0.15},
        {"feature": "avg_rating", "value": 2.0, "impact": 0.08},
        {"feature": "total_support_tickets", "value": 5.0, "impact": 0.05},
        {"feature": "subscription_tenure_days", "value": 480.0, "impact": -0.10},
    ]
    rec = get_business_recommendation(
        probability=0.8,
        shap_explanations=shap_explanations,
    )
    assert "HIGH" in rec
    assert "ordered" in rec or "disengagement" in rec or "days" in rec
    assert "Risk signals" in rec
    assert "Strengths to leverage" in rec or "reduces risk" in rec
