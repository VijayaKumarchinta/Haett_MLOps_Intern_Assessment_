"""
Feature Engineering Module
Transforms cleaned raw data into predictive features for churn modeling.
Features include recency, frequency, monetary (RFM), subscription, engagement, and demographic features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    PROCESSED_DATA_DIR,
    FEATURES_DIR,
    CHURN_LABEL_DAYS,
    RANDOM_SEED,
)

np.random.seed(RANDOM_SEED)


def compute_recency_features(orders: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Compute recency-based features for each user."""
    reference_date = pd.Timestamp.now()

    # Latest order per user
    latest_orders = orders.groupby("user_id").agg(
        last_order_date=("order_date", "max"),
        last_order_value=("order_value", "last"),
        last_order_rating=("rating", "last"),
    ).reset_index()

    # Days since last order
    latest_orders["days_since_last_order"] = (
        reference_date - latest_orders["last_order_date"]
    ).dt.days

    # Days since last high-value order (> $50)
    high_value_orders = orders[orders["order_value"] > 50]
    if not high_value_orders.empty:
        last_high_val = high_value_orders.groupby("user_id")["order_date"].max().reset_index()
        last_high_val.columns = ["user_id", "last_high_value_date"]
        latest_orders = latest_orders.merge(last_high_val, on="user_id", how="left")
        latest_orders["days_since_high_value"] = (
            reference_date - latest_orders["last_high_value_date"]
        ).dt.days
        latest_orders["days_since_high_value"] = latest_orders["days_since_high_value"].fillna(999)
    else:
        latest_orders["days_since_high_value"] = 999

    # Days since last on-time delivery issue
    late_orders = orders[orders["on_time_delivery"] == False]
    if not late_orders.empty:
        last_late = late_orders.groupby("user_id")["order_date"].max().reset_index()
        last_late.columns = ["user_id", "last_late_date"]
        latest_orders = latest_orders.merge(last_late, on="user_id", how="left")
        latest_orders["days_since_last_late"] = (
            reference_date - latest_orders["last_late_date"]
        ).dt.days
        latest_orders["days_since_last_late"] = latest_orders["days_since_last_late"].fillna(999)
    else:
        latest_orders["days_since_last_late"] = 999

    # First order date (for tenure)
    first_orders = orders.groupby("user_id")["order_date"].min().reset_index()
    first_orders.columns = ["user_id", "first_order_date"]
    latest_orders = latest_orders.merge(first_orders, on="user_id", how="left")
    latest_orders["tenure_days"] = (
        reference_date - latest_orders["first_order_date"]
    ).dt.days

    # Drop intermediate date columns
    date_cols = ["last_order_date", "first_order_date", "last_high_value_date", "last_late_date"]
    latest_orders = latest_orders.drop(columns=[c for c in date_cols if c in latest_orders.columns])

    return latest_orders


def compute_frequency_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Compute frequency-based features."""
    freq = orders.groupby("user_id").agg(
        total_orders=("order_id", "count"),
        unique_meal_plans=("meal_plan", "nunique"),
        preferred_hour=("delivery_hour", lambda x: x.mode().iloc[0] if not x.mode().empty else "afternoon"),
    ).reset_index()

    # Order consistency (std of inter-order time)
    orders_sorted = orders.sort_values(["user_id", "order_date"])
    orders_sorted["prev_order_date"] = orders_sorted.groupby("user_id")["order_date"].shift(1)
    orders_sorted["days_between_orders"] = (
        orders_sorted["order_date"] - orders_sorted["prev_order_date"]
    ).dt.days

    order_consistency = orders_sorted.groupby("user_id")["days_between_orders"].agg(
        mean_days_between_orders="mean",
        std_days_between_orders="std",
    ).reset_index()
    order_consistency["std_days_between_orders"] = order_consistency["std_days_between_orders"].fillna(0)

    freq = freq.merge(order_consistency, on="user_id", how="left")
    freq["order_frequency_per_month"] = freq["total_orders"] / (
        freq["mean_days_between_orders"] * 30 / 30 if freq["mean_days_between_orders"].notna().any() else 1
    )
    # More robust: just use total orders / tenure
    # This will be combined with recency features later

    # Weekend ordering ratio
    orders["day_of_week"] = orders["order_date"].dt.dayofweek
    orders["is_weekend"] = orders["day_of_week"].isin([5, 6])
    weekend_ratio = orders.groupby("user_id")["is_weekend"].mean().reset_index()
    weekend_ratio.columns = ["user_id", "weekend_order_ratio"]
    freq = freq.merge(weekend_ratio, on="user_id", how="left")

    return freq


def compute_monetary_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Compute monetary (value-based) features."""
    monetary = orders.groupby("user_id").agg(
        total_spent=("order_value", "sum"),
        avg_order_value=("order_value", "mean"),
        max_order_value=("order_value", "max"),
        min_order_value=("order_value", "min"),
        avg_rating=("rating", "mean"),
        late_delivery_count=("on_time_delivery", lambda x: (~x.astype(bool)).sum()),
    ).reset_index()

    # Spending trend: compare first half vs second half
    orders_sorted = orders.sort_values(["user_id", "order_date"])
    orders_sorted["order_rank"] = orders_sorted.groupby("user_id").cumcount() + 1
    orders_sorted["total_user_orders"] = orders_sorted.groupby("user_id")["user_id"].transform("count")

    # First half vs second half spending
    first_half = orders_sorted[orders_sorted["order_rank"] <= orders_sorted["total_user_orders"] / 2]
    second_half = orders_sorted[orders_sorted["order_rank"] > orders_sorted["total_user_orders"] / 2]

    first_half_spend = first_half.groupby("user_id")["order_value"].mean().reset_index()
    first_half_spend.columns = ["user_id", "first_half_avg_value"]

    second_half_spend = second_half.groupby("user_id")["order_value"].mean().reset_index()
    second_half_spend.columns = ["user_id", "second_half_avg_value"]

    monetary = monetary.merge(first_half_spend, on="user_id", how="left")
    monetary = monetary.merge(second_half_spend, on="user_id", how="left")

    monetary["spending_trend"] = (
        monetary["second_half_avg_value"].fillna(0) - monetary["first_half_avg_value"].fillna(0)
    )
    monetary["value_variability"] = monetary["max_order_value"] - monetary["min_order_value"]

    return monetary


def compute_subscription_features(subscriptions: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Compute subscription-based features."""
    ref_date = pd.Timestamp.now()

    sub_features = subscriptions.groupby("user_id").agg(
        current_plan=("plan_type", lambda x: x.iloc[-1] if not x.empty else "none"),
        n_plan_changes=("plan_type", "nunique"),
        total_subscription_days=("end_date", lambda x: (pd.to_datetime(x.iloc[-1]) - pd.to_datetime(x.iloc[0])).days if not x.empty else 0),
        is_sub_active=("status", lambda x: (x.iloc[-1] == "active") if not x.empty else False),
        monthly_price=("monthly_price", "last"),
        cancellation_reason=("cancellation_reason", lambda x: x.iloc[-1] if not x.empty else ""),
    ).reset_index()

    # Convert subscriptions dates
    subscriptions["start_date"] = pd.to_datetime(subscriptions["start_date"])
    subscriptions["end_date"] = pd.to_datetime(subscriptions["end_date"])

    # Days since subscription ended (if cancelled)
    cancelled = subscriptions[subscriptions["status"] == "cancelled"].copy()
    if not cancelled.empty:
        last_cancel = cancelled.groupby("user_id")["end_date"].max().reset_index()
        last_cancel.columns = ["user_id", "last_cancel_date"]
        sub_features = sub_features.merge(last_cancel, on="user_id", how="left")
        sub_features["days_since_cancellation"] = (
            ref_date - sub_features["last_cancel_date"]
        ).dt.days
        sub_features["days_since_cancellation"] = sub_features["days_since_cancellation"].fillna(999)
    else:
        sub_features["days_since_cancellation"] = 999

    # Merge with users to get signup date for tenure calculation
    sub_features = sub_features.merge(
        users[["user_id", "signup_date"]], on="user_id", how="left"
    )
    sub_features["subscription_tenure_days"] = (
        ref_date - pd.to_datetime(sub_features["signup_date"])
    ).dt.days

    # Encode cancellation reason
    reason_dummies = pd.get_dummies(sub_features["cancellation_reason"], prefix="cancel_reason")
    sub_features = pd.concat([sub_features, reason_dummies], axis=1)

    # Drop intermediate columns
    drop_cols = ["current_plan", "signup_date", "cancellation_reason", "last_cancel_date"]
    sub_features = sub_features.drop(columns=[c for c in drop_cols if c in sub_features.columns], errors="ignore")

    return sub_features


def compute_engagement_features(engagement: pd.DataFrame) -> pd.DataFrame:
    """Compute engagement-based features from weekly snapshots."""
    engagement["week_date"] = pd.to_datetime(engagement["week_date"])

    # Aggregate weekly data per user
    eng_features = engagement.groupby("user_id").agg(
        avg_app_logins=("app_logins", "mean"),
        avg_recipes_viewed=("recipes_viewed", "mean"),
        avg_meals_skipped=("meals_skipped", "mean"),
        total_support_tickets=("support_tickets", "sum"),
        total_referral_clicks=("referral_clicks", "sum"),
        avg_orders_per_week=("n_orders_this_week", "mean"),
    ).reset_index()

    # Engagement trend (last 4 weeks vs overall)
    max_week = engagement.groupby("user_id")["week_date"].max().reset_index()
    max_week.columns = ["user_id", "max_week_date"]

    recent_engagement = engagement.merge(max_week, on="user_id")
    recent_engagement = recent_engagement[
        recent_engagement["week_date"] >= recent_engagement["max_week_date"] - pd.Timedelta(weeks=4)
    ]

    recent_agg = recent_engagement.groupby("user_id").agg(
        recent_avg_logins=("app_logins", "mean"),
        recent_avg_recipes=("recipes_viewed", "mean"),
    ).reset_index()

    eng_features = eng_features.merge(recent_agg, on="user_id", how="left")

    # Engagement decline signal
    eng_features["login_decline"] = (
        eng_features["avg_app_logins"] - eng_features["recent_avg_logins"]
    ).clip(0)
    eng_features["recipe_decline"] = (
        eng_features["avg_recipes_viewed"] - eng_features["recent_avg_recipes"]
    ).clip(0)

    return eng_features


def compute_demographic_features(users: pd.DataFrame) -> pd.DataFrame:
    """Compute demographic features from user data."""
    demo = users[["user_id", "age", "dietary_preference", "referral_source", "city"]].copy()

    # Age groups
    demo["age_group"] = pd.cut(
        demo["age"],
        bins=[0, 25, 35, 50, 100],
        labels=["young_adult", "adult", "middle_age", "senior"],
    ).astype(str)

    # One-hot encode categoricals
    diet_dummies = pd.get_dummies(demo["dietary_preference"], prefix="diet")
    refer_dummies = pd.get_dummies(demo["referral_source"], prefix="referral")

    demo = pd.concat(
        [demo[["user_id", "age", "age_group"]], diet_dummies, refer_dummies], axis=1
    )

    return demo


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode remaining categorical columns to numeric."""
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
        if col == "user_id":
            continue
        df_encoded[col] = pd.factorize(df_encoded[col])[0]
    return df_encoded


def build_feature_matrix() -> pd.DataFrame:
    """Build the complete feature matrix from all data sources."""
    print("[...] Building feature matrix...")

    # Load cleaned data
    print("  |-- Loading cleaned data...")
    users = pd.read_csv(PROCESSED_DATA_DIR / "users_clean.csv")
    orders = pd.read_csv(PROCESSED_DATA_DIR / "orders_clean.csv")
    subscriptions = pd.read_csv(PROCESSED_DATA_DIR / "subscriptions_clean.csv")
    engagement = pd.read_csv(PROCESSED_DATA_DIR / "engagement_clean.csv")
    churn_labels = pd.read_csv(PROCESSED_DATA_DIR / "churn_labels_clean.csv")

    # Convert date columns
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    users["signup_date"] = pd.to_datetime(users["signup_date"])

    print("  |-- Computing recency features...")
    recency = compute_recency_features(orders, users)

    print("  |-- Computing frequency features...")
    frequency = compute_frequency_features(orders)

    print("  |-- Computing monetary features...")
    monetary = compute_monetary_features(orders)

    print("  |-- Computing subscription features...")
    subscription = compute_subscription_features(subscriptions, users)

    print("  |-- Computing engagement features...")
    engagement_feat = compute_engagement_features(engagement)

    print("  |-- Computing demographic features...")
    demographic = compute_demographic_features(users)

    # Merge all features
    print("  +-- Merging feature sets...")
    feature_matrix = users[["user_id"]].copy()
    feature_matrix = feature_matrix.merge(recency, on="user_id", how="left")
    feature_matrix = feature_matrix.merge(frequency, on="user_id", how="left")
    feature_matrix = feature_matrix.merge(monetary, on="user_id", how="left")
    feature_matrix = feature_matrix.merge(subscription, on="user_id", how="left")
    feature_matrix = feature_matrix.merge(engagement_feat, on="user_id", how="left")
    feature_matrix = feature_matrix.merge(demographic, on="user_id", how="left")

    # Merge churn labels
    feature_matrix = feature_matrix.merge(
        churn_labels[["user_id", "churned"]], on="user_id", how="left"
    )
    feature_matrix["churned"] = feature_matrix["churned"].fillna(0).astype(int)

    # Drop user_id for modeling, keep a copy
    user_ids = feature_matrix["user_id"]
    X = feature_matrix.drop(columns=["user_id", "churned"])
    y = feature_matrix["churned"]

    # Encode categorical features
    X_encoded = encode_categorical_features(X)

    # Add back user_id for reference
    X_encoded["user_id"] = user_ids

    # Save features
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(FEATURES_DIR / "feature_matrix.csv", index=False)
    X_encoded.to_csv(FEATURES_DIR / "features_encoded.csv", index=False)
    pd.DataFrame({"churned": y}).to_csv(FEATURES_DIR / "target.csv", index=False)

    print(f"\n[OK] Feature matrix complete! Shape: {feature_matrix.shape}")
    print(f"   Features: {len(feature_matrix.columns) - 2} (excluding user_id and target)")
    print(f"   Positive class ratio: {y.mean():.1%}")

    return X_encoded, y


if __name__ == "__main__":
    X, y = build_feature_matrix()
