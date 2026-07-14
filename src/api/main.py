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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.predict import ChurnPredictor, get_predictor
from src.utils.metrics import get_business_recommendation

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


# ─── Request/Response Schemas ─────────────────────────────────────────────────


class ChurnPredictionRequest(BaseModel):
    """Single user churn prediction request."""

    user_id: int = Field(..., description="Unique user identifier", ge=1)

    # Recency features
    days_since_last_order: float = Field(default=0, ge=0, description="Days since user's last order")
    days_since_high_value: float = Field(default=999, ge=0, description="Days since last order > $50")
    days_since_last_late: float = Field(default=999, ge=0, description="Days since last late delivery")

    # Frequency features
    total_orders: int = Field(default=0, ge=0, description="Total number of orders placed")
    unique_meal_plans: int = Field(default=0, ge=0, description="Number of distinct meal plans tried")
    mean_days_between_orders: float = Field(default=30, ge=0, description="Average days between orders")
    std_days_between_orders: float = Field(default=0, ge=0, description="Std dev of days between orders")
    weekend_order_ratio: float = Field(default=0.28, ge=0, le=1, description="Fraction of orders on weekends")
    order_frequency_per_month: float = Field(default=4, ge=0, description="Orders per month")

    # Monetary features
    total_spent: float = Field(default=0, ge=0, description="Total amount spent ($)")
    avg_order_value: float = Field(default=0, ge=0, description="Average order value ($)")
    max_order_value: float = Field(default=0, ge=0, description="Maximum single order value ($)")
    min_order_value: float = Field(default=0, ge=0, description="Minimum single order value ($)")
    avg_rating: float = Field(default=3.5, ge=1, le=5, description="Average order rating (1-5)")
    late_delivery_count: int = Field(default=0, ge=0, description="Number of late deliveries")
    spending_trend: float = Field(default=0, description="Change in spending (recent vs earlier)")
    value_variability: float = Field(default=0, ge=0, description="Range of order values ($)")

    # Subscription features
    n_plan_changes: int = Field(default=0, ge=0, description="Number of plan changes")
    total_subscription_days: int = Field(default=0, ge=0, description="Total days subscribed")
    is_sub_active: bool = Field(default=True, description="Whether subscription is currently active")
    monthly_price: float = Field(default=0, ge=0, description="Current monthly subscription price ($)")
    days_since_cancellation: float = Field(default=999, ge=0, description="Days since subscription cancelled")
    subscription_tenure_days: int = Field(default=0, ge=0, description="Days since first subscription")
    tenure_days: int = Field(default=0, ge=0, description="Days since first order")

    # Engagement features
    avg_app_logins: float = Field(default=0, ge=0, description="Average weekly app logins")
    avg_recipes_viewed: float = Field(default=0, ge=0, description="Average weekly recipes viewed")
    avg_meals_skipped: float = Field(default=0, ge=0, description="Average weekly meals skipped")
    total_support_tickets: int = Field(default=0, ge=0, description="Total support tickets submitted")
    total_referral_clicks: int = Field(default=0, ge=0, description="Total referral link clicks")
    avg_orders_per_week: float = Field(default=0, ge=0, description="Average orders per week")
    recent_avg_logins: float = Field(default=0, ge=0, description="Average logins in last 4 weeks")
    recent_avg_recipes: float = Field(default=0, ge=0, description="Average recipes viewed in last 4 weeks")
    login_decline: float = Field(default=0, ge=0, description="Decline in login frequency")
    recipe_decline: float = Field(default=0, ge=0, description="Decline in recipe viewing")

    # Demographic features
    age: int = Field(default=30, ge=18, le=100, description="User age")
    age_group: int = Field(default=0, ge=0, description="Age group encoded (0=young_adult, 1=adult, 2=middle_age, 3=senior)")


class ChurnPredictionResponse(BaseModel):
    """Churn prediction response."""

    user_id: int
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., pattern="^(Low|Medium|High)$")
    business_recommendation: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    users: list[ChurnPredictionRequest] = Field(..., min_length=1, max_length=1000)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# ─── API Endpoints ────────────────────────────────────────────────────────────


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
async def predict_churn(request: ChurnPredictionRequest):
    """
    Predict churn probability for a single user.

    Takes user features and returns:
    - churn_probability: probability of churning within next 30 days
    - risk_level: Low (< 30%), Medium (30-60%), or High (> 60%)
    - business_recommendation: actionable retention recommendation
    """
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Convert request to DataFrame
    features = pd.DataFrame([request.model_dump()])

    # We need to map the input features to the model's expected feature names
    # The model expects specific feature names from training
    if predictor.feature_names:
        # Build a feature vector using the model's expected columns
        feature_dict = {}
        for col in predictor.feature_names:
            if col in features.columns:
                feature_dict[col] = features[col].values[0]
            else:
                feature_dict[col] = 0  # default for missing features

        features_ordered = pd.DataFrame([feature_dict])
    else:
        features_ordered = features

    # Predict
    result = predictor.predict(features_ordered)

    return ChurnPredictionResponse(
        user_id=request.user_id,
        churn_probability=result["churn_probability"],
        risk_level=result["risk_level"],
        business_recommendation=result["business_recommendation"],
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
        features = pd.DataFrame([user_request.model_dump()])

        if predictor.feature_names:
            feature_dict = {}
            for col in predictor.feature_names:
                if col in features.columns:
                    feature_dict[col] = features[col].values[0]
                else:
                    feature_dict[col] = 0
            features_ordered = pd.DataFrame([feature_dict])
        else:
            features_ordered = features

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
    """Root endpoint with API documentation links."""
    return {
        "service": "Haett Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
