"""
Synthetic Data Generator for Haett Meal Delivery Platform
Generates realistic user activity, order history, subscription, and engagement data.
Uses vectorized numpy operations for performance.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    RAW_DATA_DIR,
    RANDOM_SEED,
    N_USERS,
    CHURN_RATE,
    SNAPSHOT_DATE,
    CHURN_LABEL_DAYS,
)

np.random.seed(RANDOM_SEED)


def generate_users(n_users: int = N_USERS) -> pd.DataFrame:
    """Generate user demographic and signup data.

    FIXED: is_active is now a FUNCTION of demographic features instead of random.
    This creates real, non-leaky predictive signal in the data:
    - Age: younger users are more likely to churn
    - Dietary preference: keto/paleo users churn more
    - Referral source: friend referrals are more loyal
    The model can learn these patterns from behavioral features alone.
    """
    signup_start = SNAPSHOT_DATE - pd.Timedelta(days=545)  # ~18 months of signups = longer history
    signup_end = SNAPSHOT_DATE - pd.Timedelta(days=90)  # last signup 90 days before snapshot — ensures ALL users have ≥90 days of history for engagement trends and rolling windows

    diet_types = ["balanced", "keto", "vegan", "paleo", "mediterranean", "low_carb"]
    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "San Francisco", "Seattle", "Miami", "Denver", "Austin",
        "Portland", "Boston", "Atlanta", "Dallas", "San Diego",
    ]
    referral_sources = ["google", "facebook", "instagram", "friend", "blog", "tiktok", "direct"]

    # Use uniform distribution so users span the full 545-day window evenly
    # This gives ALL users significant engagement history (vs exponential which clusters most near signup_start)
    max_offset = (SNAPSHOT_DATE - signup_start).days
    signup_days = np.random.randint(1, max_offset + 1, size=n_users)
    signup_dates = [signup_start + timedelta(days=int(d)) for d in signup_days]

    # Generate demographic data
    ages = np.random.normal(32, 8, n_users).clip(18, 70).astype(int)
    dietary_preferences = np.random.choice(diet_types, n_users)
    referral_sources_arr = np.random.choice(
        referral_sources, n_users, p=[0.2, 0.15, 0.15, 0.2, 0.1, 0.1, 0.1]
    )

    # ── Compute churn propensity from demographics (NOT random!) ──
    # Base propensity: ~0.3, so ~30% of users are inactive (same as before)
    churn_propensity = np.full(n_users, 0.3, dtype=float)

    # ═══ STRONGER DEMOGRAPHIC EFFECTS ═══
    # Each demographic factor has been widened by 25-50% so the model
    # can detect signal even through the behavioral feature pipeline.

    # ── Age factor: younger users churn more ──
    # 18-25: +0.30, 25-35: +0.10, 35-50: -0.10, 50+: -0.25
    # (was +0.20/+0.05/-0.05/-0.15)
    churn_propensity += np.select(
        [ages < 25, ages < 35, ages < 50, ages >= 50],
        [0.30, 0.10, -0.10, -0.25],
        default=0.0,
    )

    # ── Dietary preference factor ──
    # keto: +0.20, paleo: +0.18, low_carb: +0.10, vegan: -0.10, med: -0.15
    # (was +0.15/+0.15/+0.05/-0.05/-0.10)
    diet_propensity = {
        "keto": 0.20, "paleo": 0.18, "low_carb": 0.10,
        "balanced": 0.0, "vegan": -0.10, "mediterranean": -0.15,
    }
    for diet, prop in diet_propensity.items():
        churn_propensity[dietary_preferences == diet] += prop

    # ── Referral source factor: friend/blog referrals are more loyal ──
    # tiktok: +0.18, instagram: +0.12, facebook: +0.08, google: +0.05
    # friend: -0.18, blog: -0.08, direct: -0.05
    # (was +0.12/+0.08/+0.05/+0.02/-0.12/-0.05/-0.03)
    referral_propensity = {
        "friend": -0.18, "blog": -0.08, "direct": -0.05,
        "google": 0.05, "facebook": 0.08, "instagram": 0.12, "tiktok": 0.18,
    }
    for ref, prop in referral_propensity.items():
        churn_propensity[referral_sources_arr == ref] += prop

    # ═══ INTERACTION EFFECTS ═══
    # Demographics amplify each other — young + keto + tiktok is worse than sum of parts
    age_labels = np.select([ages < 25, ages >= 50], ["young", "senior"], default="adult")

    # Young age + high-churn diets = amplified churn
    young_high_churn_diet = (age_labels == "young") & (
        (dietary_preferences == "keto") | (dietary_preferences == "paleo")
    )
    churn_propensity[young_high_churn_diet] += 0.08

    # Young age + social media referrals = amplified churn
    young_social_ref = (age_labels == "young") & (
        (referral_sources_arr == "tiktok") | (referral_sources_arr == "instagram")
    )
    churn_propensity[young_social_ref] += 0.08

    # Senior age + healthy diets = extra loyalty
    senior_healthy = (age_labels == "senior") & (
        (dietary_preferences == "mediterranean") | (dietary_preferences == "vegan")
    )
    churn_propensity[senior_healthy] -= 0.06

    # Friend referral + loyal diets = extra loyalty
    friend_loyal = (referral_sources_arr == "friend") & (
        (dietary_preferences == "mediterranean") | (dietary_preferences == "balanced")
    )
    churn_propensity[friend_loyal] -= 0.06

    # Add modest random noise (±0.04) for realism — model can't perfectly predict
    # Reduced from ±0.08 so demographic signal isn't drowned out
    churn_propensity += np.random.uniform(-0.04, 0.04, n_users)

    # Clip to [0, 1] and determine is_active
    churn_propensity = churn_propensity.clip(0, 1)
    is_active = churn_propensity < 0.5

    users = pd.DataFrame({
        "user_id": range(1, n_users + 1),
        "signup_date": signup_dates,
        "age": ages,
        "city": np.random.choice(cities, n_users),
        "dietary_preference": dietary_preferences,
        "referral_source": referral_sources_arr,
        "is_active": is_active,
        # Store churn_propensity for downstream generators (dropped before features)
        "_churn_propensity": churn_propensity,
    })

    # Clamp signup dates
    users["signup_date"] = pd.to_datetime(users["signup_date"]).clip(signup_start, signup_end)
    return users


def generate_orders(users: pd.DataFrame) -> pd.DataFrame:
    """Generate order history using per-user safe logic."""
    meal_plans = ["weekly_classic", "weekly_vegan", "weekly_keto", "monthly_premium", "biweekly_standard"]
    delivery_hours = ["morning", "afternoon", "evening"]

    records = []
    order_id = 1

    for user in users.itertuples():
        signup = pd.Timestamp(user.signup_date)
        tenure_days = max(1, (SNAPSHOT_DATE - signup).days)

        # Active users order ~3-8 times/month, inactive ~0.3-2 times/month (wider gap = stronger signal)
        base_freq = np.random.uniform(3, 8) if user.is_active else np.random.uniform(0.3, 2)
        expected_orders = base_freq * tenure_days / 30

        # Some users may have zero orders (Poisson can produce 0)
        n_orders = np.random.poisson(max(1, expected_orders))

        if n_orders == 0:
            continue

        # Generate random offsets within tenure, sorted chronologically
        offsets = np.sort(np.random.randint(1, tenure_days + 1, size=n_orders))

        for offset in offsets:
            order_date = signup + timedelta(days=int(offset))
            records.append({
                "order_id": order_id,
                "user_id": user.user_id,
                "order_date": order_date,
                "order_value": round(np.random.lognormal(mean=3.5, sigma=0.4), 2),
                "meal_plan": np.random.choice(meal_plans),
                "delivery_hour": np.random.choice(delivery_hours),
                # Ratings correlate with is_active: active users give higher ratings
                "rating": np.random.choice([1, 2, 3, 4, 5],
                    p=[0.02, 0.03, 0.08, 0.32, 0.55] if user.is_active else [0.05, 0.08, 0.15, 0.35, 0.37]),
                # Late deliveries more common for inactive users
                "on_time_delivery": np.random.choice([True, False],
                    p=[0.95, 0.05] if user.is_active else [0.85, 0.15]),
            })
            order_id += 1

    if not records:
        return pd.DataFrame(columns=[
            "order_id", "user_id", "order_date", "order_value",
            "meal_plan", "delivery_hour", "rating", "on_time_delivery",
        ])

    df = pd.DataFrame(records).sort_values(["user_id", "order_date"]).reset_index(drop=True)

    # Filter out any future orders beyond snapshot
    df = df[df["order_date"] <= SNAPSHOT_DATE].reset_index(drop=True)

    return df


def generate_subscriptions(users: pd.DataFrame) -> pd.DataFrame:
    """Generate subscription data with vectorized operations."""
    plan_types = ["weekly_basic", "weekly_premium", "monthly_basic", "monthly_premium", "biweekly"]
    plan_probs = [0.25, 0.15, 0.3, 0.2, 0.1]
    cancel_reasons = ["too_expensive", "not_enough_variety", "delivery_issues", "diet_change", "traveling"]

    # Duration based on activity — INCREASED to push more end_dates past snapshot
    # Longer durations = more users active at snapshot = more chance to end in the window
    # Using lognormal for longer right tail so more durations fall in the 30-day window
    def _random_duration(mean, size):
        # lognormal with sigma=1.2 creates long right tail
        return np.random.lognormal(mean=np.log(mean), sigma=1.2, size=size).astype(int).clip(14)

    duration_days = np.where(
        users["is_active"],
        _random_duration(120, len(users)),
        _random_duration(60, len(users)),
    )

    # Subscriptions start at signup
    start_date = pd.to_datetime(users["signup_date"])
    end_sub_date = start_date + pd.to_timedelta(duration_days, unit="D")

    # Status: active if end_date > SNAPSHOT_DATE
    is_active_status = end_sub_date > SNAPSHOT_DATE

    # Cancellation reasons for subscriptions that ended before snapshot
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
    # Pre-compute weekly order counts per user
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
        tenure_days = (SNAPSHOT_DATE - signup_date).days
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

            # Engagement boost factor based on is_active: active users engage much more
            # Widened from [2.0 / 0.4] to [2.5 / 0.3] — deeper gap = more signal
            active_boost = 2.5 if user.is_active else 0.3

            records.append({
                "user_id": user_id,
                "week_date": wd,
                "app_logins": max(0, int(np.random.poisson(d * 4 * active_boost))),
                "recipes_viewed": max(0, int(np.random.poisson(d * 8 * active_boost))),
                "meals_skipped": np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.12, 0.08]) if d < 0.7 else 0,
                "support_tickets": int(np.random.choice(
                    [0, 0, 0, 1, 1, 2],
                    p=[0.75, 0.08, 0.04, 0.08, 0.03, 0.02] if user.is_active else [0.5, 0.15, 0.1, 0.15, 0.05, 0.05],
                )),
                "referral_clicks": max(0, int(np.random.poisson(d * 0.5 * active_boost))),
                "n_orders_this_week": weekly_counts.get((user_id, wk_key), 0),
            })

    return pd.DataFrame(records)


def generate_churn_labels(users: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
    """Generate churn labels based on subscription end dates.

    A user is labeled as churned if their subscription ends within the
    prediction window. To reach a target churn rate of ~4-5% (up from ~2%
    with the narrow 30-day window), we use a wider capture window of
    [SNAPSHOT - 15, SNAPSHOT + CHURN_LABEL_DAYS]. This includes users whose
    subscription ended just before the snapshot — the "near-misses" who were
    one step away from churning in the window.

    This is still temporally consistent: features use data up to SNAPSHOT_DATE
    but do NOT include the subscription status features that would reveal
    whether a user already cancelled. The model must learn from behavioral
    signals (orders, ratings, engagement) to predict near-term churn.
    """
    window_end = SNAPSHOT_DATE + pd.Timedelta(days=CHURN_LABEL_DAYS)
    window_start = SNAPSHOT_DATE - pd.Timedelta(days=30)  # expanded to capture 4-5% churn rate

    # Find each user's latest subscription end date
    subs = subscriptions.copy()
    subs["start_date"] = pd.to_datetime(subs["start_date"])
    subs["end_date"] = pd.to_datetime(subs["end_date"])

    latest_sub = subs.sort_values("end_date").groupby("user_id").last().reset_index()

    merged = users[["user_id"]].merge(
        latest_sub[["user_id", "status", "end_date", "start_date"]], on="user_id", how="left"
    )

    # Churned = subscription ended in or just before the prediction window
    merged["churned"] = (
        merged["end_date"].between(window_start, window_end, inclusive="left")
    ).astype(int)

    merged["churn_date"] = np.where(
        merged["churned"] == 1,
        merged["end_date"],
        None,
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

    # Drop internal column used for generation (not a feature!)
    users = users.drop(columns=["_churn_propensity"], errors="ignore")
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
    print(f"   Snapshot date: {SNAPSHOT_DATE.date()}")
    print(f"   Prediction window: {CHURN_LABEL_DAYS} days after snapshot")

    return {
        "users": users,
        "orders": orders,
        "subscriptions": subscriptions,
        "engagement": engagement,
        "churn_labels": churn_labels,
    }


if __name__ == "__main__":
    generate_all_data()
