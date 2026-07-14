"""
FastAPI prediction service for the Haett Churn Prediction System.

Endpoints:
- GET  /
- GET  /health
- POST /predict
- POST /predict/batch
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Self

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.models.predict import ChurnPredictor, get_predictor

logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------


def _get_cors_origins() -> list[str]:
    """
    Read allowed CORS origins from the CORS_ORIGINS environment variable.

    Examples:
        CORS_ORIGINS=*
        CORS_ORIGINS=https://example.com,https://admin.example.com
    """
    raw_origins = os.getenv("CORS_ORIGINS", "*").strip()

    if not raw_origins or raw_origins == "*":
        return ["*"]

    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Load and validate the prediction model when the API starts.

    The API fails immediately when model artifacts are unavailable instead
    of attempting to train a model inside the web service.
    """
    try:
        predictor = get_predictor()

        if predictor.model is None:
            raise RuntimeError("The churn model was not loaded.")

        logger.info(
            "Haett churn model loaded successfully. Threshold: %.4f",
            predictor.optimal_threshold,
        )

    except Exception as exc:
        logger.critical(
            "Application startup failed because model artifacts "
            "could not be loaded.",
            exc_info=True,
        )

        raise RuntimeError(
            "Model startup validation failed. Ensure the required "
            "artifacts exist in the models directory."
        ) from exc

    yield

    logger.info("Haett Churn Prediction API shutting down.")


app = FastAPI(
    title="Haett Churn Prediction API",
    description=(
        "ML-powered churn prediction for the Haett meal delivery platform. "
        "Predicts whether a user may churn within the next 30 days and "
        "provides an actionable retention recommendation."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

cors_origins = _get_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prometheus metrics are exposed at /metrics.
Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
).instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)

# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------


class ChurnPredictionRequest(BaseModel):
    """Input features for one churn prediction."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_id: int = Field(
        ...,
        ge=1,
        description="Unique user identifier.",
    )

    # Recency
    days_since_last_order: float = Field(
        default=0,
        ge=0,
        description="Number of days since the user's latest order.",
    )

    tenure_days: int = Field(
        default=0,
        ge=0,
        description="Number of days since the user's first order.",
    )

    # Frequency and consistency
    total_orders: int = Field(
        default=0,
        ge=0,
        description="Total number of orders placed.",
    )

    std_days_between_orders: float = Field(
        default=0,
        ge=0,
        description=(
            "Standard deviation of days between orders. "
            "Higher values indicate lower ordering consistency."
        ),
    )

    orders_last_30_days: int = Field(
        default=0,
        ge=0,
        description="Orders placed during the previous 30 days.",
    )

    # Monetary and coupon usage
    avg_order_value: float = Field(
        default=0,
        ge=0,
        description="Average monetary value of an order.",
    )

    avg_rating: float = Field(
        default=3.5,
        ge=1,
        le=5,
        description="Average order rating from 1 to 5.",
    )

    coupon_usage_count: int = Field(
        default=0,
        ge=0,
        description="Number of orders where a coupon was used.",
    )

    coupon_usage_rate: float = Field(
        default=0,
        ge=0,
        le=1,
        description="Proportion of orders where a coupon was used.",
    )

    # Subscription
    n_plan_changes: int = Field(
        default=0,
        ge=0,
        description="Number of subscription plan changes.",
    )

    monthly_price: float = Field(
        default=0,
        ge=0,
        description="Current monthly subscription price.",
    )

    subscription_tenure_days: int = Field(
        default=0,
        ge=0,
        description="Current subscription duration in days.",
    )

    # Engagement
    avg_app_logins: float = Field(
        default=0,
        ge=0,
        description="Average number of weekly application logins.",
    )

    avg_meals_skipped: float = Field(
        default=0,
        ge=0,
        description="Average number of meals skipped per week.",
    )

    total_support_tickets: int = Field(
        default=0,
        ge=0,
        description="Total number of support tickets submitted.",
    )

    # Demographic
    age: int = Field(
        default=30,
        ge=18,
        le=100,
        description="User age.",
    )

    age_group_code: int = Field(
        default=0,
        ge=0,
        le=3,
        description=(
            "Age group code: 0=young adult, 1=adult, " "2=middle age, 3=senior."
        ),
    )

    @model_validator(mode="after")
    def validate_business_rules(self) -> Self:
        """Validate relationships between multiple request fields."""

        if self.orders_last_30_days > self.total_orders:
            raise ValueError("orders_last_30_days cannot exceed total_orders.")

        if self.coupon_usage_count > self.total_orders:
            raise ValueError("coupon_usage_count cannot exceed total_orders.")

        if self.subscription_tenure_days > self.tenure_days:
            raise ValueError("subscription_tenure_days cannot exceed tenure_days.")

        if self.tenure_days > 0 and self.days_since_last_order > self.tenure_days:
            raise ValueError("days_since_last_order cannot exceed tenure_days.")

        if self.total_orders == 0:
            if self.orders_last_30_days != 0:
                raise ValueError(
                    "orders_last_30_days must be 0 when total_orders is 0."
                )

            if self.coupon_usage_count != 0:
                raise ValueError("coupon_usage_count must be 0 when total_orders is 0.")

            if self.coupon_usage_rate != 0:
                raise ValueError("coupon_usage_rate must be 0 when total_orders is 0.")

        return self


class FeatureExplanation(BaseModel):
    """SHAP contribution for one model feature."""

    feature: str
    value: float

    impact: float = Field(
        ...,
        description=(
            "Positive values increase churn risk; "
            "negative values decrease churn risk."
        ),
    )


class ChurnPredictionResponse(BaseModel):
    """Response returned for one churn prediction."""

    user_id: int

    churn_probability: float = Field(
        ...,
        ge=0,
        le=1,
    )

    risk_level: str = Field(
        ...,
        pattern="^(Low|Medium|High)$",
    )

    business_recommendation: str

    explanations: list[FeatureExplanation] | None = Field(
        default=None,
        description=("Top model feature contributions. " "Included when explain=true."),
    )


class BatchPredictionRequest(BaseModel):
    """Input for batch churn prediction."""

    model_config = ConfigDict(extra="forbid")

    users: list[ChurnPredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class BatchPredictionItem(BaseModel):
    """One result inside a batch prediction response."""

    user_id: int

    churn_probability: float = Field(
        ...,
        ge=0,
        le=1,
    )

    risk_level: str = Field(
        ...,
        pattern="^(Low|Medium|High)$",
    )

    business_recommendation: str


class BatchPredictionResponse(BaseModel):
    """Response returned for batch prediction."""

    users: list[BatchPredictionItem]

    total: int = Field(
        ...,
        ge=1,
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = Field(
        ...,
        pattern="^(healthy|degraded)$",
    )

    model_loaded: bool
    version: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_ready_predictor() -> ChurnPredictor:
    """Return the loaded predictor or raise an HTTP 503 response."""

    try:
        predictor = get_predictor()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction model artifacts are unavailable.",
        ) from exc

    except Exception as exc:
        logger.exception("Unable to initialize the prediction model.")

        raise HTTPException(
            status_code=503,
            detail="Prediction service is temporarily unavailable.",
        ) from exc

    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    return predictor


def _requests_to_features(
    requests: ChurnPredictionRequest | list[ChurnPredictionRequest],
) -> pd.DataFrame:
    """
    Convert one or more validated requests into a feature DataFrame.

    user_id is excluded because it identifies the API response record and
    should not be used as a machine-learning feature.
    """

    request_list = requests if isinstance(requests, list) else [requests]

    return pd.DataFrame(
        [request.model_dump(exclude={"user_id"}) for request in request_list]
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Report whether the prediction model is available."""

    try:
        predictor = get_predictor()
        model_loaded = predictor.model is not None

    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=API_VERSION,
    )


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["Prediction"],
)
def predict_churn(
    request: ChurnPredictionRequest,
    explain: bool = Query(
        default=False,
        description=("Include the top SHAP feature contributions " "in the response."),
    ),
) -> ChurnPredictionResponse:
    """
    Predict churn for one user.

    Risk levels are calculated using the optimal threshold produced by the
    training pipeline.
    """

    predictor = _get_ready_predictor()
    features = _requests_to_features(request)

    try:
        result = predictor.predict(
            features=features,
            explain=explain,
        )

    except Exception as exc:
        logger.exception(
            "Prediction failed for user_id=%s.",
            request.user_id,
        )

        raise HTTPException(
            status_code=500,
            detail="The prediction could not be completed.",
        ) from exc

    return ChurnPredictionResponse(
        user_id=request.user_id,
        churn_probability=result["churn_probability"],
        risk_level=result["risk_level"],
        business_recommendation=result["business_recommendation"],
        explanations=result.get("explanations"),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
def predict_churn_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """
    Predict churn for between 1 and 1,000 users.

    Probability scoring is vectorized. SHAP explanations are disabled for
    batch predictions to avoid excessive latency.
    """

    predictor = _get_ready_predictor()
    features = _requests_to_features(request.users)

    try:
        prediction_results = predictor.predict_batch(features)

        if not isinstance(prediction_results, list):
            raise TypeError("predict_batch() must return a list.")

        if len(prediction_results) != len(request.users):
            raise ValueError(
                "Prediction count does not match input count. "
                f"Expected {len(request.users)}, "
                f"received {len(prediction_results)}."
            )

    except Exception as exc:
        logger.exception(
            "Batch prediction failed for %d users.",
            len(request.users),
        )

        raise HTTPException(
            status_code=500,
            detail="The batch prediction could not be completed.",
        ) from exc

    response_items = [
        BatchPredictionItem(
            user_id=user_request.user_id,
            churn_probability=result["churn_probability"],
            risk_level=result["risk_level"],
            business_recommendation=result["business_recommendation"],
        )
        for user_request, result in zip(
            request.users,
            prediction_results,
            strict=True,
        )
    ]

    return BatchPredictionResponse(
        users=response_items,
        total=len(response_items),
    )


@app.get(
    "/",
    tags=["System"],
)
def root() -> dict[str, str]:
    """Return basic API information."""

    return {
        "service": "Haett Churn Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "batch_predict": "/predict/batch",
    }
