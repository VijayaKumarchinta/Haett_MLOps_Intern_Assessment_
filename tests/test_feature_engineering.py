"""
Tests for Feature Engineering Module
Covers edge cases: empty data, single-user data, zero orders, etc.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.feature_engineering import (
    compute_recency_features,
    compute_frequency_features,
    compute_monetary_features,
    compute_subscription_features,
    compute_engagement_features,
    compute_demographic_features,
    encode_categorical_features,
    build_feature_matrix,
)
from src.utils.config import SNAPSHOT_DATE


@pytest.fixture
def sample_orders():
    """Fixture with basic order data."""
    return pd.DataFrame({
        "order_id": [101, 102, 103, 201, 202, 301],
        "user_id": [1, 1, 1, 2, 2, 3],
        "order_date": pd.to_datetime([
            SNAPSHOT_DATE - timedelta(days=30),
            SNAPSHOT_DATE - timedelta(days=20),
            SNAPSHOT_DATE - timedelta(days=10),
            SNAPSHOT_DATE - timedelta(days=60),
            SNAPSHOT_DATE - timedelta(days=5),
            SNAPSHOT_DATE - timedelta(days=1),
        ]),
        "order_value": [30.0, 55.0, 25.0, 100.0, 40.0, 15.0],
        "meal_plan": ["weekly_classic", "weekly_vegan", "weekly_classic", "monthly_premium", "weekly_classic", "biweekly_standard"],
        "delivery_hour": ["morning", "afternoon", "morning", "evening", "afternoon", "morning"],
        "rating": [4, 5, 3, 2, 4, 5],
        "on_time_delivery": [True, True, False, True, True, True],
    })


@pytest.fixture
def single_order():
    """Fixture with a user who has only one order."""
    return pd.DataFrame({
        "order_id": [101],
        "user_id": [1],
        "order_date": [SNAPSHOT_DATE - timedelta(days=15)],
        "order_value": [50.0],
        "meal_plan": ["weekly_classic"],
        "delivery_hour": ["afternoon"],
        "rating": [4],
        "on_time_delivery": [True],
    })


@pytest.fixture
def empty_orders():
    """Fixture with no orders."""
    return pd.DataFrame(columns=[
        "user_id", "order_date", "order_value", "meal_plan",
        "delivery_hour", "rating", "on_time_delivery",
    ])


@pytest.fixture
def sample_users():
    """Fixture with basic user data."""
    return pd.DataFrame({
        "user_id": [1, 2, 3],
        "age": [25, 45, 65],
        "signup_date": [
            SNAPSHOT_DATE - timedelta(days=200),
            SNAPSHOT_DATE - timedelta(days=100),
            SNAPSHOT_DATE - timedelta(days=50),
        ],
        "dietary_preference": ["keto", "vegan", "balanced"],
        "referral_source": ["google", "friend", "instagram"],
        "city": ["New York", "Los Angeles", "Chicago"],
    })


@pytest.fixture
def sample_subscriptions():
    """Fixture with basic subscription data."""
    return pd.DataFrame({
        "user_id": [1, 1, 2, 3],
        "plan_type": ["weekly_basic", "monthly_premium", "weekly_basic", "monthly_basic"],
        "start_date": [
            SNAPSHOT_DATE - timedelta(days=180),
            SNAPSHOT_DATE - timedelta(days=90),
            SNAPSHOT_DATE - timedelta(days=120),
            SNAPSHOT_DATE - timedelta(days=60),
        ],
        "end_date": [
            SNAPSHOT_DATE - timedelta(days=100),
            SNAPSHOT_DATE + timedelta(days=30),
            SNAPSHOT_DATE - timedelta(days=10),
            SNAPSHOT_DATE + timedelta(days=60),
        ],
        "monthly_price": [49.0, 79.0, 49.0, 59.0],
        "status": ["cancelled", "active", "cancelled", "active"],
        "cancellation_reason": ["too_expensive", "", "delivery_issues", ""],
    })


@pytest.fixture
def sample_engagement():
    """Fixture with basic engagement data."""
    return pd.DataFrame({
        "user_id": [1, 1, 2],
        "week_date": [
            SNAPSHOT_DATE - timedelta(weeks=3),
            SNAPSHOT_DATE - timedelta(weeks=1),
            SNAPSHOT_DATE - timedelta(weeks=2),
        ],
        "app_logins": [4, 2, 0],
        "recipes_viewed": [10, 5, 1],
        "meals_skipped": [0, 1, 3],
        "support_tickets": [0, 1, 2],
        "referral_clicks": [1, 0, 0],
        "n_orders_this_week": [2, 1, 0],
    })


# ─── Recency Tests ───────────────────────────────────────────────────────────


def test_recency_multiple_users(sample_orders):
    """Test recency features for multiple users."""
    recency = compute_recency_features(sample_orders)
    assert "user_id" in recency.columns
    assert "days_since_last_order" in recency.columns
    assert "tenure_days" in recency.columns
    # User 1 had last order 10 days ago
    assert recency[recency["user_id"] == 1]["days_since_last_order"].values[0] == 10.0
    # User 3 had last order 1 day ago
    assert recency[recency["user_id"] == 3]["days_since_last_order"].values[0] == 1.0


def test_recency_single_order(single_order):
    """Test recency with a user who has only one order."""
    recency = compute_recency_features(single_order)
    assert len(recency) == 1
    assert recency["days_since_last_order"].values[0] == 15.0


def test_recency_high_value_order(sample_orders):
    """Test days_since_high_value feature."""
    recency = compute_recency_features(sample_orders)
    # User 2: high-value order (100.0) was 60 days ago (not 5 days ago - that's 40.0)
    assert recency[recency["user_id"] == 2]["days_since_high_value"].values[0] == 60.0
    # User 3 had no high-value order (15.0), should be 999
    assert recency[recency["user_id"] == 3]["days_since_high_value"].values[0] == 999.0


# ─── Frequency Tests ─────────────────────────────────────────────────────────


def test_frequency_basic(sample_orders):
    """Test basic frequency features."""
    freq = compute_frequency_features(sample_orders)
    assert "total_orders" in freq.columns
    assert "order_frequency_per_month" in freq.columns
    # User 1 has 3 orders
    assert freq[freq["user_id"] == 1]["total_orders"].values[0] == 3


def test_frequency_single_order(single_order):
    """Test frequency with a single order (no inter-order time)."""
    freq = compute_frequency_features(single_order)
    assert freq["total_orders"].values[0] == 1
    # With 1 order, std_days_between_orders should be 0
    assert freq["std_days_between_orders"].values[0] == 0.0


# ─── Monetary Tests ──────────────────────────────────────────────────────────


def test_monetary_basic(sample_orders):
    """Test basic monetary features."""
    mon = compute_monetary_features(sample_orders)
    assert "total_spent" in mon.columns
    assert "avg_order_value" in mon.columns
    # User 1: 30+55+25 = 110
    assert mon[mon["user_id"] == 1]["total_spent"].values[0] == 110.0
    # User 1 late delivery count = 1 (order 3 is late)
    assert mon[mon["user_id"] == 1]["late_delivery_count"].values[0] == 1


# ─── Subscription Tests ──────────────────────────────────────────────────────


def test_subscription_plan_changes(sample_subscriptions):
    """Test that n_plan_changes counts actual changes, not unique plans."""
    sub = compute_subscription_features(sample_subscriptions)
    # User 1: changed from weekly_basic to monthly_premium = 1 change
    assert sub[sub["user_id"] == 1]["n_plan_changes"].values[0] == 1
    # User 2: only 1 plan, no changes
    assert sub[sub["user_id"] == 2]["n_plan_changes"].values[0] == 0


def test_subscription_tenure(sample_subscriptions):
    """Test subscription tenure from first subscription start."""
    sub = compute_subscription_features(sample_subscriptions)
    # User 1: first sub started 180 days ago
    assert sub[sub["user_id"] == 1]["subscription_tenure_days"].values[0] == 180


def test_subscription_no_leakage_fields(sample_subscriptions):
    """Test that target-leaking fields are NOT in the subscription features."""
    sub = compute_subscription_features(sample_subscriptions)
    # These features were removed to prevent target leakage:
    # - is_sub_active: if subscription already cancelled, it can't churn again
    # - total_subscription_days: 0 = never subscribed = can't churn
    # - days_since_cancellation: low value = already cancelled = can't churn
    assert "is_sub_active" not in sub.columns
    assert "total_subscription_days" not in sub.columns
    assert "days_since_cancellation" not in sub.columns
    # The legitimate features should still be present
    assert "subscription_tenure_days" in sub.columns
    assert "n_plan_changes" in sub.columns
    assert "monthly_price" in sub.columns


# ─── Engagement Tests ────────────────────────────────────────────────────────


def test_engagement_decline_signals(sample_engagement):
    """Test engagement decline computation."""
    eng = compute_engagement_features(sample_engagement)
    # User 1: avg logins = 3, recent avg logins = 2, decline = 1
    assert "login_decline" in eng.columns
    assert eng[eng["user_id"] == 1]["login_decline"].values[0] >= 0
    # User 2: only 1 entry, recent = same as avg, decline should be 0
    assert eng[eng["user_id"] == 2]["login_decline"].values[0] == 0.0


# ─── Demographic Tests ───────────────────────────────────────────────────────


def test_demographic_age_groups(sample_users):
    """Test age group encoding."""
    demo = compute_demographic_features(sample_users)
    assert "age_group_code" in demo.columns
    # Age 25 -> young_adult (0)
    assert demo[demo["user_id"] == 1]["age_group_code"].values[0] == 0
    # Age 45 -> adult (1) or middle_age (2)? 25-35=adult, 35-50=middle_age
    # bins=[0, 25, 35, 50, 100], labels=[young_adult, adult, middle_age, senior]
    # Age 45 -> middle_age -> code 2
    assert demo[demo["user_id"] == 2]["age_group_code"].values[0] == 2
    # Age 65 -> senior -> code 3
    assert demo[demo["user_id"] == 3]["age_group_code"].values[0] == 3


def test_demographic_one_hot(sample_users):
    """Test one-hot encoding of dietary preference and referral source."""
    demo = compute_demographic_features(sample_users)
    # Should have diet_ and referral_ columns
    diet_cols = [c for c in demo.columns if c.startswith("diet_")]
    refer_cols = [c for c in demo.columns if c.startswith("referral_")]
    assert len(diet_cols) > 0
    assert len(refer_cols) > 0


# ─── Integration Tests ───────────────────────────────────────────────────────


def test_encode_categorical_handles_no_cats():
    """Test encoding with no categorical columns."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    result = encode_categorical_features(df)
    assert result.shape == df.shape


def test_full_pipeline_smoke():
    """Smoke test: run the full pipeline if feature files exist."""
    feat_path = Path(__file__).resolve().parent.parent / "data" / "features" / "features_encoded.csv"
    if not feat_path.exists():
        pytest.skip("Feature files not found. Run generate + preprocess + feature_engineering first.")
    try:
        X, y = build_feature_matrix()
        assert X.shape[0] > 0
        assert len(y) == X.shape[0]
        assert X.isna().sum().sum() == 0, f"Found {X.isna().sum().sum()} NaN values!"
    except FileNotFoundError:
        pytest.skip("Required data files not found.")
