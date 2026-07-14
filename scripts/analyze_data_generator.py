"""
Analyze the synthetic data generator to understand the root cause
of why subscription features dominate churn predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import RAW_DATA_DIR, FEATURES_DIR, SNAPSHOT_DATE

print("=" * 70)
print("  ROOT CAUSE ANALYSIS: Why Subscription Features Dominate")
print("=" * 70)

# -- Load raw data --
users = pd.read_csv(RAW_DATA_DIR / "users.csv")
subs = pd.read_csv(RAW_DATA_DIR / "subscriptions.csv")
churn = pd.read_csv(RAW_DATA_DIR / "churn_labels.csv")

users["signup_date"] = pd.to_datetime(users["signup_date"])
subs["start_date"] = pd.to_datetime(subs["start_date"])
subs["end_date"] = pd.to_datetime(subs["end_date"])

# -- 1. Causality Chain --
SEP = "-" * 70
HASH = "=" * 70

print(f"\n{HASH}")
print("  [1] THE CAUSALITY CHAIN")
print(SEP)
print("""
  users.is_active  --->  sub.duration  --->  sub.end_date  --->  churn_label
    (random 70/30)     (exponential)      (start + duration)    (end in window?)
""")

# -- 2. Step 1: is_active -> duration --
subs["duration_days"] = (subs["end_date"] - subs["start_date"]).dt.days
merged = users[["user_id", "is_active"]].merge(
    subs[["user_id", "duration_days", "status", "end_date"]], on="user_id"
)
merged["churned"] = churn.set_index("user_id")["churned"]

active_d = merged[merged["is_active"]]["duration_days"]
inactive_d = merged[~merged["is_active"]]["duration_days"]

print(f"\n{HASH}")
print("  [2] STEP 1: users.is_active DRIVES subscription duration")
print(SEP)
print(
    f"  Active   users (is_active=True) : mean={active_d.mean():7.0f}d, median={active_d.median():7.0f}d"
)
print(
    f"  Inactive users (is_active=False): mean={inactive_d.mean():7.0f}d, median={inactive_d.median():7.0f}d"
)
print(f"  Ratio: {active_d.mean() / inactive_d.mean():.1f}x longer for active users")

# -- 3. Step 2: duration -> end_date relative to snapshot --
merged["ends_after_snapshot"] = merged["end_date"] > SNAPSHOT_DATE
ct = pd.crosstab(
    merged["is_active"].map({True: "is_active=True", False: "is_active=False"}),
    merged["ends_after_snapshot"].map(
        {True: "ends_after_snap", False: "ends_before_snap"}
    ),
    margins=True,
)
print(f"\n{HASH}")
print("  [3] STEP 2: Crosstab - is_active x ends_after_snapshot")
print(SEP)
print(ct)

# -- 4. Step 3: end_date vs snapshot -> churn label --
window_end = SNAPSHOT_DATE + pd.Timedelta(days=30)
ends_before = merged["end_date"] < SNAPSHOT_DATE
ends_in_window = (merged["end_date"] >= SNAPSHOT_DATE) & (
    merged["end_date"] <= window_end
)

print(f"\n{HASH}")
print("  [4] STEP 3: Subscription end dates relative to snapshot")
print(SEP)
print(
    f"  Already ended BEFORE snapshot:       {ends_before.sum():5d} users ({ends_before.mean()*100:.1f}%)"
)
print(
    f"  Ends WITHIN 30-day window (CHURN):   {ends_in_window.sum():5d} users ({ends_in_window.mean()*100:.1f}%)"
)
print(
    f"  Ends AFTER 30-day window:            {(~ends_before & ~ends_in_window).sum():5d} users ({(~ends_before & ~ends_in_window).mean()*100:.1f}%)"
)

# -- 5. The leakage proof --
merged["is_sub_active"] = merged["status"] == "active"
cancelled = merged[~merged["is_sub_active"]]
active_sub = merged[merged["is_sub_active"]]

print(f"\n{HASH}")
print("  [5] PROOF OF LEAKAGE: is_sub_active vs churn")
print(SEP)
print(f"  Users with is_sub_active=False (cancelled): {len(cancelled)}")
print(
    f"    Churned: {cancelled['churned'].sum():.0f}, churn rate: {cancelled['churned'].mean()*100:.1f}%"
)
print("    --> ~0% because subscription already ended (can't end twice)")

print(f"\n  Users with is_sub_active=True (active): {len(active_sub)}")
print(
    f"    Churned: {active_sub['churned'].sum():.0f}, churn rate: {active_sub['churned'].mean()*100:.1f}%"
)
print("    --> TRUE at-risk rate for active subscribers")

# -- 5b. Cancel reason cross-check --
print("\n  Cross-check: cancel_reason dummies (residual leakage?)")
for reason in [
    "too_expensive",
    "delivery_issues",
    "diet_change",
    "traveling",
    "not_enough_variety",
]:
    col = f"cancel_reason_{reason}"
    # These features are in the encoded features, not raw data; check just the concept
    has_reason = merged[merged["status"] == "cancelled"]
    if len(has_reason) > 0:
        churn_rate = has_reason["churned"].mean() * 100
        print(
            f"    Users with any cancel_reason: "
            f"{len(has_reason):4d}, churn rate = {churn_rate:.1f}%"
        )
        break

# -- 6. Feature correlations --
features = pd.read_csv(FEATURES_DIR / "features_encoded.csv")
y = pd.read_csv(FEATURES_DIR / "target.csv")["churned"]

correlations = []
for col in features.columns:
    if col != "user_id":
        corr = features[col].corr(y)
        if not np.isnan(corr):
            correlations.append({"feature": col, "correlation": corr})

corr_df = pd.DataFrame(correlations)
corr_df["abs_corr"] = corr_df["correlation"].abs()
corr_df = corr_df.sort_values("abs_corr", ascending=False)

print(f"\n{HASH}")
print("  [6] TOP 15 FEATURES BY CORRELATION WITH CHURN (57 features)")
print(SEP)
print(f"  {'Feature':<35} {'Correlation':>10}  {'Abs':>8}")
print(f"  {'-'*55}")
for _, row in corr_df.head(15).iterrows():
    print(
        f"  {row['feature']:<35} {row['correlation']:>+10.4f}  {row['abs_corr']:>8.4f}"
    )

# -- 7. Feature group comparison --
categories = {
    "Subscription (tenure)": [
        "subscription_tenure_days",
        "monthly_price",
        "n_plan_changes",
    ],
    "Cancel reasons (leakage residue)": [
        c for c in features.columns if c.startswith("cancel_reason_")
    ],
    "Recency": [
        "days_since_last_order",
        "days_since_high_value",
        "days_since_last_late",
    ],
    "Frequency": [
        "total_orders",
        "unique_meal_plans",
        "order_frequency_per_month",
        "weekend_order_ratio",
        "mean_days_between_orders",
    ],
    "Monetary": [
        "total_spent",
        "avg_order_value",
        "max_order_value",
        "min_order_value",
        "avg_rating",
        "spending_trend",
        "value_variability",
    ],
    "Engagement": [
        "avg_app_logins",
        "avg_recipes_viewed",
        "avg_meals_skipped",
        "total_support_tickets",
        "total_referral_clicks",
        "login_decline",
        "recipe_decline",
        "avg_orders_per_week",
        "recent_avg_logins",
        "recent_avg_recipes",
    ],
    "Demographic": [
        "age",
        "age_group_code",
        "diet_balanced",
        "diet_keto",
        "diet_low_carb",
        "diet_mediterranean",
        "diet_paleo",
        "diet_vegan",
    ],
}

print(f"\n{HASH}")
print("  [7] FEATURE GROUP COMPARISON (mean |correlation|)")
print(SEP)
print(f"  {'Group':<35} {'Max|r|':>8}  {'Mean|r|':>8}")
print(f"  {'-'*55}")
for cat_name, cat_features in categories.items():
    valid = [f for f in cat_features if f in corr_df["feature"].values]
    if valid:
        cat_corrs = corr_df[corr_df["feature"].isin(valid)]
        max_c = cat_corrs["abs_corr"].max()
        mean_c = cat_corrs["abs_corr"].mean()
        print(f"  {cat_name:<35} {max_c:>8.4f}  {mean_c:>8.4f}")

# -- 8. Conclusion --
print(f"\n{HASH}")
print("  CONCLUSION")
print(HASH)
print("""
The root cause is a structural issue in the data generator:

  1. users['is_active'] is a RANDOM FLAG (70% True, 30% False) with NO
     relationship to any behavioral features (orders, ratings, etc.)

  2. generate_subscriptions() uses this flag to set subscription duration:
       is_active=True  -> duration ~ Exp(180) -> end_date FAR past snapshot
       is_active=False -> duration ~ Exp(90)  -> end_date SOONER

  3. generate_churn_labels() marks user as churned if end_date falls
     in [snapshot, snapshot+30]. This creates a deterministic link.

  4. Since is_active is random and independent of orders/ratings/engagement,
     the NON-subscription features have NO real signal for churn.
     Only subscription features correlate with churn, because they derive
     from the same is_active -> duration -> end_date -> label chain.

  5. After removing is_sub_active, days_since_cancellation, and
     total_subscription_days, the cancel_reason_* dummies REMAIN as
     residual leakage - a non-empty cancel_reason still means the user
     already cancelled (is_sub_active=False).

  RECOMMENDATION: Make is_active a function of behavioral features:
  - Users with fewer orders (>2 std below mean) -> higher chance of inactive
  - Users with declining ratings or logins -> higher chance of inactive
  - This creates REAL signal in behavioral features for the model to learn
""")

# Save analysis for reference
with open(
    Path(__file__).resolve().parent.parent / "models" / "leakage_analysis.txt", "w"
) as f:
    f.write("Leakage Analysis - Root Cause\n")
    f.write(f"Cancelled users churn rate: {cancelled['churned'].mean()*100:.1f}%\n")
    f.write(f"Active users churn rate: {active_sub['churned'].mean()*100:.1f}%\n")
    f.write("Top 5 feature correlations:\n")
    for _, row in corr_df.head(5).iterrows():
        f.write(f"  {row['feature']:<35} {row['correlation']:+.4f}\n")
