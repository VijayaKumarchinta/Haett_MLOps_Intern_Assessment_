"""
API Tests for Haett Churn Prediction Service
Tests the FastAPI endpoints using the TestClient with mocked predictor.
"""

import pytest
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app, health_check, predict_churn, predict_churn_batch, root
import src.api.main as api_main
from src.models import predict as predict_module

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Haett Churn Prediction API"
    assert data["version"] == "1.0.0"
    assert "/docs" in data["docs"]
    assert "/health" in data["health"]
    assert "/predict" in data["predict"]


def test_predict_endpoint_with_mock():
    """Test the prediction endpoint with a mocked predictor returning deterministic results."""
    # Create a mock predictor
    mock_predictor = MagicMock()
    mock_predictor.feature_names = ["days_since_last_order", "total_orders", "total_spent", "age"]
    mock_predictor.model = MagicMock()
    mock_predictor.scaler = None
    mock_predictor.predict.return_value = {
        "churn_probability": 0.1234,
        "risk_level": "Low",
        "business_recommendation": "No action needed. User is at low risk of churning.",
    }

    with patch("src.api.main.get_predictor", return_value=mock_predictor):
        request_data = {
            "user_id": 1,
            "days_since_last_order": 5,
            "total_orders": 15,
            "avg_order_value": 33.33,
            "avg_rating": 4.2,
            "total_support_tickets": 1,
            "age": 32,
            "subscription_tenure_days": 180,
            "tenure_days": 200,
            "avg_meals_skipped": 0.5,
            "coupon_usage_rate": 0.1,
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["churn_probability"] == 0.1234
        assert data["risk_level"] == "Low"
        assert data["business_recommendation"] is not None
        assert data["user_id"] == 1


def test_predict_endpoint_with_explain():
    """Test the prediction endpoint with ?explain=true."""
    mock_predictor = MagicMock()
    mock_predictor.feature_names = ["days_since_last_order", "total_orders", "total_spent", "age"]
    mock_predictor.model = MagicMock()
    mock_predictor.scaler = None
    mock_predictor.predict.return_value = {
        "churn_probability": 0.85,
        "risk_level": "High",
        "business_recommendation": "High risk - take action.",
        "explanations": [
            {"feature": "days_since_last_order", "value": 45.0, "impact": 0.35},
            {"feature": "total_spent", "value": 100.0, "impact": -0.12},
        ],
    }

    with patch("src.api.main.get_predictor", return_value=mock_predictor):
        response = client.post("/predict?explain=true", json={"user_id": 1})
        assert response.status_code == 200
        data = response.json()
        assert "explanations" in data
        assert len(data["explanations"]) > 0
        assert "feature" in data["explanations"][0]
        assert "impact" in data["explanations"][0]


def test_batch_predict_endpoint():
    """Test batch prediction endpoint with mocked predictor."""
    mock_predictor = MagicMock()
    mock_predictor.feature_names = ["days_since_last_order", "total_orders", "age"]
    mock_predictor.model = MagicMock()
    mock_predictor.scaler = None
    mock_predictor.predict.return_value = {
        "churn_probability": 0.1234,
        "risk_level": "Low",
        "business_recommendation": "No action needed.",
    }

    with patch("src.api.main.get_predictor", return_value=mock_predictor):
        request_data = {
            "users": [
                {"user_id": 1, "age": 30},
                {"user_id": 2, "age": 40},
            ]
        }
        response = client.post("/predict/batch", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "users" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["users"]) == 2


def test_predict_returns_503_without_model():
    """Test that predict returns 503 if model file doesn't exist."""
    # Since a model may exist on disk, mock a FileNotFoundError
    mock = MagicMock()
    mock.side_effect = FileNotFoundError("Model not found. Run train.py first.")

    with patch("src.api.main.ChurnPredictor", mock):
        with patch("src.api.main.get_predictor", side_effect=FileNotFoundError("Model not found")):
            response = client.post("/predict", json={"user_id": 1})
            assert response.status_code == 503
            assert "detail" in response.json()


def test_predict_validates_input():
    """Test that the API validates required fields."""
    # user_id is required (no default)
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error


def test_predict_handles_out_of_range_values():
    """Test that the API rejects out-of-range values."""
    # age > 100 should be rejected
    response = client.post("/predict", json={"user_id": 1, "age": 150})
    assert response.status_code == 422
