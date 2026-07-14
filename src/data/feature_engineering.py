"""
Feature Engineering Module
Transforms cleaned raw data into predictive features for churn modeling.
Features include recency, frequency, monetary (RFM), subscription, engagement, and demographic features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    PROCESSED_DATA_DIR,
    FEATURES_DIR,
    RANDOM_SEED,
    SNAPSHOT_DATE,
    AGE_GROUP_MAP,
)

np.random.seed(RANDOM_SEED)


def compute_recency_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Compute recency-based features for each user.

    All dates are computed relative to SNAPSHOT_DATE for temporal consistency.
    """
    reference_date = SNAPSHOT_DATE

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
    ).dt.days.clip(lower=0)

    # Drop intermediate date columns
    date_cols = ["last_order_date", "first_order_date", "last_high_value_date", "last_late_date"]
    latest_orders = latest_orders.drop(columns=[c for c in date_cols if c in latest_orders.columns])

    return latest_orders


def compute_frequency_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Compute frequency-based features."""
    orders_sorted = orders.sort_values(["user_id", "order_date"]).copy()

    freq = orders_sorted.groupby("user_id").agg(
        total_orders=("order_id", "count"),
        unique_meal_plans=("meal_plan", "nunique"),
        first_order_date=("order_date", "min"),
        last_order_date=("order_date", "max"),
        preferred_hour=("delivery_hour", lambda x: x.mode().iloc[0] if not x.mode().empty else "afternoon"),
    ).reset_index()

    # Order consistency (std of inter-order time)
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

    # Order frequency per month: total_orders / tenure_days * 30
    freq["tenure_days"] = (SNAPSHOT_DATE - freq["first_order_date"]).dt.days.clip(lower=1)
    freq["order_frequency_per_month"] = freq["total_orders"] / freq["tenure_days"] * 30

    # Weekend ordering ratio
    orders_sorted["is_weekend"] = orders_sorted["order_date"].dt.dayofweek.isin([5, 6])
    weekend_ratio = orders_sorted.groupby("user_id")["is_weekend"].mean().reset_index()
    weekend_ratio.columns = ["user_id", "weekend_order_ratio"]
    freq = freq.merge(weekend_ratio, on="user_id", how="left")

    # Drop intermediate date columns
    freq = freq.drop(columns=["first_order_date", "last_order_date"], errors="ignore")

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


def compute_subscription_features(subscriptions: pd.DataFrame) -> pd.DataFrame:
    """Compute subscription-based features with correct semantics.

    ⚠️ TARGET LEAKAGE FIXED: Removed is_sub_active, days_since_cancellation,
    and total_subscription_days. These leaked future information because:
    - is_sub_active=False means subscription already ended BEFORE snapshot,
      making it impossible to churn (subscription can't end twice)
    - days_since_cancellation directly reveals if the user already cancelled
    - total_subscription_days=0 means never subscribed, can't churn

    Retained: subscription_tenure_days (legitimate tenure metric),
    n_plan_changes (behavioral signal), monthly_price (value signal).
    """
    subs = subscriptions.copy()
    subs["start_date"] = pd.to_datetime(subs["start_date"])
    subs["end_date"] = pd.to_datetime(subs["end_date"])
    subs = subs.sort_values(["user_id", "start_date", "end_date"])

    # Count actual plan changes (consecutive different plan_type, not unique count)
    def count_plan_changes(plans):
        changes = int(plans.ne(plans.shift()).sum() - 1)
        return max(0, changes)

    plan_changes = (
        subs.groupby("user_id")["plan_type"]
        .apply(count_plan_changes)
        .reset_index(name="n_plan_changes")
    )

    # Aggregate subscription info (REMOVED leaking fields: is_sub_active, total_subscription_days, days_since_cancellation)
    agg = subs.groupby("user_id").agg(
        first_subscription_start=("start_date", "min"),
        monthly_price=("monthly_price", "last"),
    ).reset_index()

    # Subscription tenure from first subscription start
    agg["subscription_tenure_days"] = (
        SNAPSHOT_DATE - agg["first_subscription_start"]
    ).dt.days.clip(lower=0)

    # Merge plan changes
    agg = agg.merge(plan_changes, on="user_id", how="left")
    agg["n_plan_changes"] = agg["n_plan_changes"].fillna(0).astype(int)

    # Drop intermediate columns (including cancellation_reason — REMOVED to eliminate residual leakage)
    # cancel_reason_* dummies were removed because a non-empty cancellation reason directly implies
    # the user already cancelled (is_sub_active=False), creating the same leakage signal.
    drop_cols = [
        "first_subscription_start",
    ]
    agg = agg.drop(columns=[c for c in drop_cols if c in agg.columns], errors="ignore")

    return agg


def compute_engagement_features(engagement: pd.DataFrame) -> pd.DataFrame:
    """Compute engagement-based features from weekly snapshots.

    Only uses engagement data up to SNAPSHOT_DATE.
    """
    eng = engagement.copy()
    eng["week_date"] = pd.to_datetime(eng["week_date"])

    # Filter to data up to snapshot date
    eng = eng[eng["week_date"] <= SNAPSHOT_DATE].copy()

    if eng.empty:
        # Return empty feature set if no engagement data
        return pd.DataFrame(columns=["user_id"])

    # Aggregate weekly data per user
    eng_features = eng.groupby("user_id").agg(
        avg_app_logins=("app_logins", "mean"),
        avg_recipes_viewed=("recipes_viewed", "mean"),
        avg_meals_skipped=("meals_skipped", "mean"),
        total_support_tickets=("support_tickets", "sum"),
        total_referral_clicks=("referral_clicks", "sum"),
        avg_orders_per_week=("n_orders_this_week", "mean"),
    ).reset_index()

    # Engagement trend (last 4 weeks vs overall)
    max_week = eng.groupby("user_id")["week_date"].max().reset_index()
    max_week.columns = ["user_id", "max_week_date"]

    recent_engagement = eng.merge(max_week, on="user_id")
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
        eng_features["avg_app_logins"].fillna(0) - eng_features["recent_avg_logins"].fillna(0)
    ).clip(0)
    eng_features["recipe_decline"] = (
        eng_features["avg_recipes_viewed"].fillna(0) - eng_features["recent_avg_recipes"].fillna(0)
    ).clip(0)

    return eng_features


def compute_demographic_features(users: pd.DataFrame) -> pd.DataFrame:
    """Compute demographic features from user data."""
    demo = users[["user_id", "age", "dietary_preference", "referral_source", "city"]].copy()

    # Age groups with deterministic encoding using AGE_GROUP_MAP
    age_group_labels = pd.cut(
        demo["age"],
        bins=[0, 25, 35, 50, 100],
        labels=["young_adult", "adult", "middle_age", "senior"],
    )
    demo["age_group"] = age_group_labels.astype(str)
    demo["age_group_code"] = age_group_labels.map(AGE_GROUP_MAP).fillna(0).astype(int)

    # One-hot encode categoricals
    diet_dummies = pd.get_dummies(demo["dietary_preference"], prefix="diet")
    refer_dummies = pd.get_dummies(demo["referral_source"], prefix="referral")

    demo = pd.concat(
        [demo[["user_id", "age", "age_group_code"]], diet_dummies, refer_dummies], axis=1
    )

    return demo


def compute_rolling_window_features(
    engagement: pd.DataFrame, orders: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 4-week rolling window features from raw engagement and orders data.

    These features capture recent trends that aggregate-only features miss:
    - Recent engagement: meals_skipped, support_tickets, referral_clicks (last 4 weeks)
    - Recent orders: spending, avg_value (last 30 days)
    """
    results = pd.DataFrame()

    # ── Rolling windows from engagement data (last 4 weeks) ──
    eng = engagement.copy()
    eng["week_date"] = pd.to_datetime(eng["week_date"])
    eng = eng[eng["week_date"] <= SNAPSHOT_DATE].copy()

    if not eng.empty:
        # Filter to last 4 weeks from each user's max week
        max_week = eng.groupby("user_id")["week_date"].max().reset_index()
        max_week.columns = ["user_id", "max_week_date"]

        recent_eng = eng.merge(max_week, on="user_id")
        recent_eng = recent_eng[
            recent_eng["week_date"] >= recent_eng["max_week_date"] - pd.Timedelta(weeks=4)
        ]

        rolling = recent_eng.groupby("user_id").agg(
            recent_meals_skipped=("meals_skipped", "mean"),
            recent_support_tickets=("support_tickets", "sum"),
            recent_referral_clicks=("referral_clicks", "sum"),
        ).reset_index()

        results = rolling

    # ── Rolling windows from orders data (last 30 days) ──
    odr = orders.copy()
    odr["order_date"] = pd.to_datetime(odr["order_date"])
    window_start = SNAPSHOT_DATE - pd.Timedelta(days=30)

    recent_orders = odr[odr["order_date"] >= window_start].copy()

    if not recent_orders.empty:
        recent_ord_agg = recent_orders.groupby("user_id").agg(
            recent_spending=("order_value", "sum"),
            recent_order_count=("order_id", "count"),
            recent_avg_order_value=("order_value", "mean"),
        ).reset_index()

        if results.empty:
            results = recent_ord_agg
        else:
            results = results.merge(recent_ord_agg, on="user_id", how="outer")

    if results.empty or "user_id" not in results.columns:
        return pd.DataFrame(columns=["user_id"])

    return results


def compute_interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """Compute interaction and ratio features from the aggregated feature matrix.

    These derived features amplify signal by combining related metrics:
    - Interactions: spending_velocity, satisfaction_score, inactivity_depth
    - Ratios: ticket_rate, value_for_money, late_delivery_rate

    All features are computed safely (no division by zero, no NaN).
    """
    df = features.copy()

    # ── Safe division helper ──
    def safe_div(a, b, default=0):
        return np.where(b > 0, a / b, default)

    # ── Interaction features ──
    # Spending velocity: how much the user spends per unit of frequency
    if "avg_order_value" in df and "order_frequency_per_month" in df:
        df["spending_velocity"] = df["avg_order_value"] * df["order_frequency_per_month"]

    # Satisfaction score: high rating + high engagement = happy user
    if "avg_rating" in df and "avg_app_logins" in df:
        df["satisfaction_score"] = df["avg_rating"] * df["avg_app_logins"]

    # Inactivity depth: how long since last order × how much logins declined
    if "days_since_last_order" in df and "login_decline" in df:
        df["inactivity_depth"] = df["days_since_last_order"] * (df["login_decline"] + 1e-6)

    # Frustration index: more support tickets with lower ratings = frustrated
    if "total_support_tickets" in df and "avg_rating" in df:
        df["frustration_index"] = df["total_support_tickets"] * (5.0 - df["avg_rating"])

    # Order velocity: total orders × avg value per day of tenure
    if "total_orders" in df and "avg_order_value" in df and "tenure_days" in df:
        df["order_velocity"] = safe_div(df["total_orders"] * df["avg_order_value"], df["tenure_days"] + 1)

    # Waste indicator: skipped meals relative to orders
    if "avg_meals_skipped" in df and "total_orders" in df:
        df["waste_indicator"] = df["avg_meals_skipped"] * df["total_orders"]

    # ── Ratio features ──
    # Support ticket rate: tickets per order (normalized effort)
    if "total_support_tickets" in df and "total_orders" in df:
        df["ticket_rate"] = safe_div(df["total_support_tickets"], df["total_orders"])

    # Value for money: avg order value relative to monthly subscription price
    if "avg_order_value" in df and "monthly_price" in df:
        df["value_for_money"] = safe_div(df["avg_order_value"], df["monthly_price"]) * 100

    # Late delivery rate: fraction of orders that were late
    if "late_delivery_count" in df and "total_orders" in df:
        df["late_delivery_rate"] = safe_div(df["late_delivery_count"], df["total_orders"])

    # Meals skipped per active order week
    if "avg_meals_skipped" in df and "avg_orders_per_week" in df:
        df["skip_per_order"] = safe_div(df["avg_meals_skipped"], df["avg_orders_per_week"] + 0.1)

    # Referral rate: clicks per month of tenure
    if "total_referral_clicks" in df and "tenure_days" in df:
        df["referral_rate"] = safe_div(df["total_referral_clicks"] * 30, df["tenure_days"] + 1)

    # Rating consistency: spread between max possible and actual rating
    if "avg_rating" in df:
        df["rating_gap"] = 5.0 - df["avg_rating"]

    # Order concentration: what fraction of total orders are recent (last 4 weeks)
    if "avg_orders_per_week" in df and "order_frequency_per_month" in df:
        df["order_concentration"] = safe_div(df["avg_orders_per_week"] * 4, df["order_frequency_per_month"] / 30 * 28 + 1)

    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode remaining categorical columns to numeric.

    Uses factorize for any remaining object columns that weren't already
    one-hot encoded (e.g. preferred_hour from frequency features).
    """
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
        if col == "user_id":
            continue
        df_encoded[col] = pd.factorize(df_encoded[col])[0]
    return df_encoded


def build_feature_matrix() -> tuple:
    """Build the complete feature matrix from all data sources.

    Returns:
        Tuple of (X_encoded: pd.DataFrame, y: pd.Series)
        X_encoded contains all numeric features including user_id.
        y contains the churn target variable.
    """
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
    recency = compute_recency_features(orders)

    print("  |-- Computing frequency features...")
    frequency = compute_frequency_features(orders)

    print("  |-- Computing monetary features...")
    monetary = compute_monetary_features(orders)

    print("  |-- Computing subscription features...")
    subscription = compute_subscription_features(subscriptions)

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

    # ── Compute rolling window features from raw data ──
    print("  |-- Computing rolling window features...")
    rolling_features = compute_rolling_window_features(engagement, orders)
    feature_matrix = feature_matrix.merge(rolling_features, on="user_id", how="left")

    # ── Compute interaction and ratio features ──
    print("  |-- Computing interaction and ratio features...")
    feature_matrix = compute_interaction_features(feature_matrix)

    # ── Impute NaNs for users with missing data (e.g. zero orders) ──
    # These semantically correct defaults ensure all users get valid features
    default_fill_values = {
        # Recency
        "days_since_last_order": 999,
        "days_since_high_value": 999,
        "days_since_last_late": 999,
        "tenure_days": 0,
        # Frequency
        "total_orders": 0,
        "unique_meal_plans": 0,
        "mean_days_between_orders": 0,
        "std_days_between_orders": 0,
        "weekend_order_ratio": 0,
        "order_frequency_per_month": 0,
        # Monetary
        "total_spent": 0,
        "avg_order_value": 0,
        "max_order_value": 0,
        "min_order_value": 0,
        "avg_rating": 0,
        "late_delivery_count": 0,
        "first_half_avg_value": 0,
        "second_half_avg_value": 0,
        "spending_trend": 0,
        "value_variability": 0,
        # Subscription (REMOVED leaked fields: is_sub_active, total_subscription_days, days_since_cancellation)
        "n_plan_changes": 0,
        "monthly_price": 0,
        "subscription_tenure_days": 0,
        # Engagement
        "avg_app_logins": 0,
        "avg_recipes_viewed": 0,
        "avg_meals_skipped": 0,
        "total_support_tickets": 0,
        "total_referral_clicks": 0,
        "avg_orders_per_week": 0,
        "recent_avg_logins": 0,
        "recent_avg_recipes": 0,
        "login_decline": 0,
        "recipe_decline": 0,
        # Demographic
        "age": 30,
        "age_group_code": 0,
    }
    # Also fill any cancel_reason_* dummies and other unknown columns with 0
    for col in feature_matrix.columns:
        if col not in default_fill_values and col not in ["user_id", "churned"]:
            default_fill_values[col] = 0

    feature_matrix = feature_matrix.fillna(default_fill_values)

    # Drop user_id for modeling, keep a copy
    user_ids = feature_matrix["user_id"]
    X = feature_matrix.drop(columns=["user_id", "churned"])
    y = feature_matrix["churned"]

    # Encode any remaining categorical features
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
    print(f"   NaN count in feature matrix: {feature_matrix.isna().sum().sum()}")

    return X_encoded, y


if __name__ == "__main__":
    X, y = build_feature_matrix()
