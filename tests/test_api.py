"""
API Tests for Haett Churn Prediction Service
Tests the FastAPI endpoints using the TestClient.
"""

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app, health_check, predict_churn, predict_churn_batch, root
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


@pytest.mark.skip(reason="Requires a trained model. Run train.py first.")
def test_predict_endpoint():
    """Test the prediction endpoint with sample data."""
    request_data = {
        "user_id": 1,
        "days_since_last_order": 5,
        "total_orders": 15,
        "total_spent": 500.0,
        "avg_order_value": 33.33,
        "avg_rating": 4.2,
        "total_support_tickets": 1,
        "age": 32,
        "is_sub_active": True,
        "subscription_tenure_days": 180,
        "tenure_days": 200,
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "business_recommendation" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert 0 <= data["churn_probability"] <= 1


@pytest.mark.skip(reason="Requires a trained model. Run train.py first.")
def test_batch_predict_endpoint():
    """Test batch prediction endpoint."""
    request_data = {
        "users": [
            {
                "user_id": 1,
                "days_since_last_order": 5,
                "total_orders": 15,
                "total_spent": 500.0,
                "avg_order_value": 33.33,
                "avg_rating": 4.2,
                "is_sub_active": True,
                "subscription_tenure_days": 180,
                "tenure_days": 200,
            },
            {
                "user_id": 2,
                "days_since_last_order": 45,
                "total_orders": 3,
                "total_spent": 100.0,
                "avg_order_value": 33.33,
                "avg_rating": 2.5,
                "is_sub_active": False,
                "subscription_tenure_days": 60,
                "tenure_days": 90,
            },
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
    # Save originals for restoration
    original_predictor = predict_module._predictor
    original_predictor_class = predict_module.ChurnPredictor
    predict_module._predictor = None

    try:
        # Monkey-patch the predictor to raise FileNotFoundError
        class MockPredictor:
            def __init__(self):
                raise FileNotFoundError("Model not found. Run train.py first.")

        predict_module.ChurnPredictor = MockPredictor

        # Create a fresh test app
        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        test_app.get("/")(root)
        test_app.get("/health")(health_check)
        test_app.post("/predict")(predict_churn)
        test_app.post("/predict/batch")(predict_churn_batch)

        test_client = TestClient(test_app)
        response = test_client.post("/predict", json={"user_id": 1})

        assert response.status_code == 503
        assert "detail" in response.json()
    finally:
        # Restore originals
        predict_module._predictor = original_predictor
        predict_module.ChurnPredictor = original_predictor_class
