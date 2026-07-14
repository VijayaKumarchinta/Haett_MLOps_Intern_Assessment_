"""
Synthetic Data Generator for Haett Meal Delivery Platform
Generates realistic user activity, order history, subscription, and engagement data.
Uses vectorized numpy operations for performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    RAW_DATA_DIR,
    RANDOM_SEED,
    N_USERS,
    CHURN_RATE,
)

np.random.seed(RANDOM_SEED)


def generate_users(n_users: int = N_USERS) -> pd.DataFrame:
    """Generate user demographic and signup data."""
    signup_start = datetime(2024, 1, 1)
    signup_end = datetime(2025, 6, 1)

    diet_types = ["balanced", "keto", "vegan", "paleo", "mediterranean", "low_carb"]
    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "San Francisco", "Seattle", "Miami", "Denver", "Austin",
        "Portland", "Boston", "Atlanta", "Dallas", "San Diego",
    ]
    referral_sources = ["google", "facebook", "instagram", "friend", "blog", "tiktok", "direct"]

    # Generate signup dates using vectorized exponential distribution
    signup_days = np.random.exponential(30, n_users).astype(int)
    signup_dates = [signup_start + timedelta(days=int(d)) for d in signup_days]

    users = pd.DataFrame({
        "user_id": range(1, n_users + 1),
        "signup_date": signup_dates,
        "age": np.random.normal(32, 8, n_users).clip(18, 70).astype(int),
        "city": np.random.choice(cities, n_users),
        "dietary_preference": np.random.choice(diet_types, n_users),
        "referral_source": np.random.choice(referral_sources, n_users, p=[0.2, 0.15, 0.15, 0.2, 0.1, 0.1, 0.1]),
        "is_active": np.random.choice([True, False], n_users, p=[0.7, 0.3]),
    })

    # Clamp signup dates
    users["signup_date"] = pd.to_datetime(users["signup_date"]).clip(signup_start, signup_end)
    return users


def generate_orders(users: pd.DataFrame) -> pd.DataFrame:
    """Generate order history using vectorized operations."""
    end_date = datetime(2025, 7, 1)
    meal_plans = ["weekly_classic", "weekly_vegan", "weekly_keto", "monthly_premium", "biweekly_standard"]
    delivery_hours = ["morning", "afternoon", "evening"]

    # Calculate expected number of orders per user based on tenure and activity
    tenure_days = (end_date - users["signup_date"]).dt.days.clip(1)

    # Active users order ~2-6 times/month, inactive ~0.5-3 times/month
    base_freq = np.where(
        users["is_active"],
        np.random.uniform(2, 6, len(users)),
        np.random.uniform(0.5, 3, len(users))
    )
    expected_orders = (base_freq * tenure_days / 30).clip(1)

    # Generate actual order counts from Poisson
    n_orders_per_user = np.random.poisson(expected_orders).astype(int)

    # Build records using numpy repeat
    user_ids = np.repeat(users["user_id"].values, n_orders_per_user)
    signup_dates = np.repeat(users["signup_date"].values, n_orders_per_user)
    tenure_days_repeated = np.repeat(tenure_days.values, n_orders_per_user)

    n_total = len(user_ids)
    if n_total == 0:
        return pd.DataFrame(columns=["user_id", "order_date", "order_value", "meal_plan",
                                      "delivery_hour", "rating", "on_time_delivery"])

    # Generate order dates using exponential inter-arrival times
    inter_arrival = np.random.exponential(
        tenure_days_repeated / np.repeat(n_orders_per_user, n_orders_per_user).clip(1)
    )
    cumulative_days = np.cumsum(inter_arrival).astype(int)

    # Cap at tenure
    cumulative_days = np.minimum(cumulative_days, tenure_days_repeated - 1)
    cumulative_days = np.maximum(cumulative_days, 0)

    # Convert to actual dates
    ref_dates = pd.to_datetime(signup_dates)
    order_dates = ref_dates + pd.to_timedelta(cumulative_days, unit="D")

    # Filter out future dates
    valid_mask = order_dates <= pd.Timestamp(end_date)

    orders = pd.DataFrame({
        "user_id": user_ids[valid_mask],
        "order_date": order_dates[valid_mask],
        "order_value": np.round(np.random.lognormal(mean=3.5, sigma=0.4, size=valid_mask.sum()), 2),
        "meal_plan": np.random.choice(meal_plans, valid_mask.sum()),
        "delivery_hour": np.random.choice(delivery_hours, valid_mask.sum()),
        "rating": np.random.choice([1, 2, 3, 4, 5], valid_mask.sum(), p=[0.02, 0.03, 0.1, 0.35, 0.5]),
        "on_time_delivery": np.random.choice([True, False], valid_mask.sum(), p=[0.92, 0.08]),
    })

    # Add order_id
    orders.insert(0, "order_id", range(1, len(orders) + 1))
    return orders.sort_values(["user_id", "order_date"]).reset_index(drop=True)


def generate_subscriptions(users: pd.DataFrame) -> pd.DataFrame:
    """Generate subscription data with vectorized operations."""
    end_date = datetime(2025, 7, 1)
    plan_types = ["weekly_basic", "weekly_premium", "monthly_basic", "monthly_premium", "biweekly"]
    plan_probs = [0.25, 0.15, 0.3, 0.2, 0.1]
    cancel_reasons = ["too_expensive", "not_enough_variety", "delivery_issues", "diet_change", "traveling"]

    # Duration based on activity
    duration_days = np.where(
        users["is_active"],
        np.random.exponential(180, len(users)).astype(int),
        np.random.exponential(90, len(users)).astype(int)
    )

    start_date = pd.to_datetime(users["signup_date"])
    end_sub_date = start_date + pd.to_timedelta(duration_days, unit="D")
    end_sub_date = end_sub_date.clip(upper=pd.Timestamp(end_date))

    # Status
    is_active_status = (
        users["is_active"].values &
        (end_sub_date >= pd.Timestamp(end_date) - pd.Timedelta(days=30))
    )

    # Cancellation reasons for cancelled subscriptions
    reasons = np.where(
        ~is_active_status,
        np.random.choice(cancel_reasons, len(users)),
        ""
    )

    subscriptions = pd.DataFrame({
        "user_id": users["user_id"],
        "plan_type": np.random.choice(plan_types, len(users), p=plan_probs),
        "start_date": start_date,
        "end_date": end_sub_date,
        "monthly_price": np.round(np.random.uniform(29, 99, len(users)), 2),
        "status": np.where(is_active_status, "active", "cancelled"),
        "cancellation_reason": reasons,
    })

    return subscriptions


def generate_engagement(users: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Generate weekly engagement metrics using vectorized operations."""
    end_date = datetime(2025, 7, 1)

    # Pre-compute weekly order counts per user once (avoid per-user DataFrame filtering)
    if not orders.empty:
        order_weekly = orders.copy()
        order_weekly["week_key"] = order_weekly["order_date"].dt.strftime("%Y-W%V")
        weekly_counts = order_weekly.groupby(["user_id", "week_key"]).size().to_dict()
    else:
        weekly_counts = {}

    # For each user, generate weekly snapshots
    records = []
    for _, user in users.iterrows():
        user_id = user["user_id"]
        signup_date = user["signup_date"]
        tenure_days = (end_date - signup_date).days
        if tenure_days <= 0:
            continue

        n_weeks = max(1, int(tenure_days / 7))
        weeks = np.arange(n_weeks)
        week_dates = [signup_date + timedelta(weeks=int(w)) for w in weeks]

        # Engagement decays over time
        decay = np.exp(-0.02 * weeks)

        for i, wd in enumerate(week_dates):
            wk_key = wd.strftime("%Y-W%V")
            d = decay[i]

            records.append({
                "user_id": user_id,
                "week_date": wd,
                "app_logins": max(0, int(np.random.poisson(d * 4))),
                "recipes_viewed": max(0, int(np.random.poisson(d * 8))),
                "meals_skipped": np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.12, 0.08]) if d < 0.7 else 0,
                "support_tickets": int(np.random.choice([0, 0, 0, 1, 1, 2], p=[0.7, 0.1, 0.05, 0.1, 0.03, 0.02])),
                "referral_clicks": max(0, int(np.random.poisson(d * 0.5))),
                "n_orders_this_week": weekly_counts.get((user_id, wk_key), 0),
            })

    return pd.DataFrame(records)


def generate_churn_labels(users: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
    """Generate churn labels based on subscription and activity patterns."""
    # Merge users with their last subscription
    sub_last = subscriptions.groupby("user_id").last().reset_index()

    merged = users[["user_id", "is_active"]].merge(
        sub_last[["user_id", "status", "end_date"]], on="user_id", how="left"
    )

    # Label churn
    merged["churned"] = (
        (merged["status"] == "cancelled") | (~merged["is_active"])
    ).astype(int)

    merged["churn_date"] = np.where(
        merged["churned"] == 1,
        merged["end_date"],
        None
    )

    return merged[["user_id", "churned", "churn_date"]]


def generate_all_data() -> dict:
    """Generate all synthetic datasets and save to CSV."""
    print("[...] Generating synthetic data for Haett meal delivery platform...")

    print("  |-- Generating users...")
    users = generate_users()

    print("  |-- Generating orders...")
    orders = generate_orders(users)

    print("  |-- Generating subscriptions...")
    subscriptions = generate_subscriptions(users)

    print("  |-- Generating engagement data...")
    engagement = generate_engagement(users, orders)

    print("  +-- Generating churn labels...")
    churn_labels = generate_churn_labels(users, subscriptions)

    # Save raw data
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    users.to_csv(RAW_DATA_DIR / "users.csv", index=False)
    print(f"    - users.csv ({len(users)} records)")
    orders.to_csv(RAW_DATA_DIR / "orders.csv", index=False)
    print(f"    - orders.csv ({len(orders)} records)")
    subscriptions.to_csv(RAW_DATA_DIR / "subscriptions.csv", index=False)
    print(f"    - subscriptions.csv ({len(subscriptions)} records)")
    engagement.to_csv(RAW_DATA_DIR / "engagement.csv", index=False)
    print(f"    - engagement.csv ({len(engagement)} records)")
    churn_labels.to_csv(RAW_DATA_DIR / "churn_labels.csv", index=False)
    print(f"    - churn_labels.csv ({len(churn_labels)} records)")

    print(f"\n[OK] Data generation complete! Files saved to: {RAW_DATA_DIR}")
    print(f"   Churn rate: {churn_labels['churned'].mean():.1%}")

    return {
        "users": users,
        "orders": orders,
        "subscriptions": subscriptions,
        "engagement": engagement,
        "churn_labels": churn_labels,
    }


if __name__ == "__main__":
    generate_all_data()
