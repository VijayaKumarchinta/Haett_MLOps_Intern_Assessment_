"""
FastAPI Prediction Service for Haett Churn Prediction System
Provides POST /predict endpoint returning churn probability, risk level, and business recommendations.
"""

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from fastapi import Query
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import os

from src.models.predict import ChurnPredictor, get_predictor
from src.utils.config import MODELS_DIR

app = FastAPI(
    title="Haett Churn Prediction API",
    description="ML-powered churn prediction for the Haett meal delivery platform. "
    "Predicts user churn probability within the next 30 days and provides "
    "actionable business recommendations.",
    version="1.0.0",
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup: Auto-train if no model exists ─────────────────────────────────


@app.on_event("startup")
async def auto_train_on_startup():
    """Automatically run the pipeline if no trained model is found.
    This makes the API self-contained: just start the server and it works."""
    model_path = MODELS_DIR / "churn_model.pkl"
    if not model_path.exists():
        print("[startup] No trained model found. Running pipeline automatically...")
        print("[startup] This may take a few minutes.")
        try:
            import subprocess
            import sys

            # Use fast mode (no hyperparameter tuning) for quick startup
            env = {**os.environ, "N_HPARAM_ITER": "5"}
            result = subprocess.run(
                [sys.executable, "src/run_pipeline.py"],
                cwd=MODELS_DIR.parent,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("[startup] Pipeline completed successfully.")
            else:
                print(f"[startup] Pipeline failed:\n{result.stderr}")
        except Exception as e:
            print(f"[startup] Could not auto-train: {e}")
            print("[startup] You can manually run: python run.py")


# ─── Request/Response Schemas ─────────────────────────────────────────────────


class ChurnPredictionRequest(BaseModel):
    """Single user churn prediction request.

    Features align with the assessment criteria:
    - Days since last order, Orders in last 30 days, Average order value
    - Subscription duration, Coupon usage, Meal swap frequency, Order consistency
    """

    user_id: int = Field(..., description="Unique user identifier", ge=1)

    # Recency
    days_since_last_order: float = Field(default=0, ge=0, description="Days since user's last order")
    tenure_days: int = Field(default=0, ge=0, description="Days since first order")

    # Frequency & Order consistency
    total_orders: int = Field(default=0, ge=0, description="Total number of orders placed")
    std_days_between_orders: float = Field(default=0, ge=0, description="Order consistency (std dev of days between orders)")
    orders_last_30_days: int = Field(default=0, ge=0, description="Orders placed in the last 30 days")

    # Monetary & Coupon usage
    avg_order_value: float = Field(default=0, ge=0, description="Average order value ($)")
    avg_rating: float = Field(default=3.5, ge=1, le=5, description="Average order rating (1-5)")
    coupon_usage_count: int = Field(default=0, ge=0, description="Number of orders where coupon was used")
    coupon_usage_rate: float = Field(default=0, ge=0, le=1, description="Fraction of orders with coupon")

    # Subscription
    n_plan_changes: int = Field(default=0, ge=0, description="Number of plan changes")
    monthly_price: float = Field(default=0, ge=0, description="Current monthly subscription price ($)")
    subscription_tenure_days: int = Field(default=0, ge=0, description="Subscription duration in days")

    # Engagement & Meal swap frequency
    avg_app_logins: float = Field(default=0, ge=0, description="Average weekly app logins")
    avg_meals_skipped: float = Field(default=0, ge=0, description="Meal swap frequency (avg meals skipped per week)")
    total_support_tickets: int = Field(default=0, ge=0, description="Total support tickets submitted")

    # Demographic
    age: int = Field(default=30, ge=18, le=100, description="User age")
    age_group_code: int = Field(default=0, ge=0, description="Age group (0=young_adult, 1=adult, 2=middle_age, 3=senior)")


class FeatureExplanation(BaseModel):
    """SHAP explanation for a single feature."""

    feature: str
    value: float
    impact: float = Field(..., description="Positive = increases churn risk, Negative = decreases churn risk")


class ChurnPredictionResponse(BaseModel):
    """Churn prediction response."""

    user_id: int
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., pattern="^(Low|Medium|High)$")
    business_recommendation: str
    explanations: list[FeatureExplanation] | None = Field(
        default=None,
        description="Top 5 feature contributions via SHAP. Only included when ?explain=true. "
        "Positive impact = increases churn risk, Negative = decreases churn risk.",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    users: list[ChurnPredictionRequest] = Field(..., min_length=1, max_length=1000)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# ─── API Endpoints ────────────────────────────────────────────────────────────


def _prepare_features(predictor, request_dict: dict) -> pd.DataFrame:
    """Convert a request dict to a properly ordered feature DataFrame."""
    features = pd.DataFrame([request_dict])

    if predictor.feature_names:
        # Build a feature vector using the model's expected columns
        feature_dict = {}
        for col in predictor.feature_names:
            if col in features.columns:
                feature_dict[col] = features[col].values[0]
            else:
                feature_dict[col] = 0  # default for missing features
        return pd.DataFrame([feature_dict])

    return features


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    try:
        predictor = get_predictor()
        model_loaded = predictor.model is not None
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Prediction"])
async def predict_churn(
    request: ChurnPredictionRequest,
    explain: bool = Query(False, description="Include SHAP feature explanations in the response"),
):
    """
    Predict churn probability for a single user.

    Takes user features and returns:
    - churn_probability: probability of churning within next 30 days
    - risk_level: Low (< 30%), Medium (30-60%), or High (> 60%)
    - business_recommendation: actionable retention recommendation

    Add ?explain=true to also receive the top 5 feature contributions via SHAP.
    """
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features_ordered = _prepare_features(predictor, request.model_dump())

    # Predict with optional SHAP explanations
    result = predictor.predict(features_ordered, explain=explain)

    return ChurnPredictionResponse(
        user_id=request.user_id,
        churn_probability=result["churn_probability"],
        risk_level=result["risk_level"],
        business_recommendation=result["business_recommendation"],
        explanations=result.get("explanations"),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_churn_batch(request: BatchPredictionRequest):
    """
    Predict churn probability for multiple users (up to 1000 at a time).

    Returns a list of predictions with the same structure as the single prediction endpoint.
    """
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    results = []
    for user_request in request.users:
        features_ordered = _prepare_features(predictor, user_request.model_dump())

        result = predictor.predict(features_ordered)

        results.append({
            "user_id": user_request.user_id,
            "churn_probability": result["churn_probability"],
            "risk_level": result["risk_level"],
            "business_recommendation": result["business_recommendation"],
        })

    return {"users": results, "total": len(results)}


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API info and links."""
    return {
        "service": "Haett Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
