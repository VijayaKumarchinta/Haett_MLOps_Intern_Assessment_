"""
Feature Engineering Module
Transforms cleaned raw data into predictive features for churn modeling.
Features include recency, frequency, monetary (RFM), subscription, engagement, and demographic features.
"""

import pandas as pd
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


def compute_recency_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Compute recency-based features.

    Per the assessment criteria: Days since last order
    """
    reference_date = SNAPSHOT_DATE

    # Latest order per user
    latest_orders = orders.groupby("user_id").agg(
        last_order_date=("order_date", "max"),
    ).reset_index()

    # Days since last order — the core recency feature
    latest_orders["days_since_last_order"] = (
        reference_date - latest_orders["last_order_date"]
    ).dt.days

    # First order date (for tenure)
    first_orders = orders.groupby("user_id")["order_date"].min().reset_index()
    first_orders.columns = ["user_id", "first_order_date"]
    latest_orders = latest_orders.merge(first_orders, on="user_id", how="left")
    latest_orders["tenure_days"] = (
        reference_date - latest_orders["first_order_date"]
    ).dt.days.clip(lower=0)

    # Drop intermediate date columns
    latest_orders = latest_orders.drop(columns=["last_order_date", "first_order_date"], errors="ignore")

    return latest_orders


def compute_frequency_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Compute frequency-based features.

    Per the assessment criteria: Order consistency, orders in last 30 days
    """
    orders_sorted = orders.sort_values(["user_id", "order_date"]).copy()

    freq = orders_sorted.groupby("user_id").agg(
        total_orders=("order_id", "count"),
        first_order_date=("order_date", "min"),
    ).reset_index()

    # Order consistency (std of inter-order time) — per assessment criteria
    orders_sorted["prev_order_date"] = orders_sorted.groupby("user_id")["order_date"].shift(1)
    orders_sorted["days_between_orders"] = (
        orders_sorted["order_date"] - orders_sorted["prev_order_date"]
    ).dt.days

    order_consistency = orders_sorted.groupby("user_id")["days_between_orders"].agg(
        std_days_between_orders="std",
    ).reset_index()
    order_consistency["std_days_between_orders"] = order_consistency["std_days_between_orders"].fillna(0)

    freq = freq.merge(order_consistency, on="user_id", how="left")

    # Orders in the last 30 days — per assessment criteria
    window_start = SNAPSHOT_DATE - pd.Timedelta(days=30)
    recent_orders = orders_sorted[orders_sorted["order_date"] >= window_start]
    orders_last_30 = recent_orders.groupby("user_id").size().reset_index(name="orders_last_30_days")
    freq = freq.merge(orders_last_30, on="user_id", how="left")

    freq = freq.drop(columns=["first_order_date"], errors="ignore")

    return freq


def compute_monetary_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Compute monetary and coupon-usage features.

    Per the assessment criteria: Average order value, Coupon usage
    """
    monetary = orders.groupby("user_id").agg(
        avg_order_value=("order_value", "mean"),
        avg_rating=("rating", "mean"),
    ).reset_index()

    # Coupon usage — per assessment criteria
    if "coupon_used" in orders.columns:
        coupon_agg = orders.groupby("user_id").agg(
            coupon_usage_count=("coupon_used", "sum"),
            total_orders_with_coupon_col=("coupon_used", "count"),
        ).reset_index()
        # Ratio of orders where coupon was used
        coupon_agg["coupon_usage_rate"] = (
            coupon_agg["coupon_usage_count"] / coupon_agg["total_orders_with_coupon_col"]
        )
        monetary = monetary.merge(
            coupon_agg[["user_id", "coupon_usage_count", "coupon_usage_rate"]],
            on="user_id", how="left",
        )
    else:
        monetary["coupon_usage_count"] = 0
        monetary["coupon_usage_rate"] = 0.0

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
    """Compute engagement-based features.

    Per the assessment criteria: Meal swap frequency (avg_meals_skipped)
    """
    eng = engagement.copy()
    eng["week_date"] = pd.to_datetime(eng["week_date"])

    # Filter to data up to snapshot date
    eng = eng[eng["week_date"] <= SNAPSHOT_DATE].copy()

    if eng.empty:
        return pd.DataFrame(columns=["user_id"])

    # Core engagement features
    eng_features = eng.groupby("user_id").agg(
        avg_app_logins=("app_logins", "mean"),
        avg_meals_skipped=("meals_skipped", "mean"),
        total_support_tickets=("support_tickets", "sum"),
    ).reset_index()

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

    # Features match the assessment criteria:
    # days_since_last_order, orders_last_30_days, avg_order_value,
    # subscription_tenure_days, coupon_usage_rate, avg_meals_skipped,
    # std_days_between_orders (order consistency)

    # ── Impute NaNs for users with missing data ──
    default_fill_values = {
        "days_since_last_order": 999,
        "tenure_days": 0,
        "total_orders": 0,
        "std_days_between_orders": 0,
        "orders_last_30_days": 0,
        "avg_order_value": 0,
        "avg_rating": 0,
        "coupon_usage_count": 0,
        "coupon_usage_rate": 0.0,
        "n_plan_changes": 0,
        "monthly_price": 0,
        "subscription_tenure_days": 0,
        "avg_app_logins": 0,
        "avg_meals_skipped": 0,
        "total_support_tickets": 0,
        "age": 30,
        "age_group_code": 0,
    }
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
