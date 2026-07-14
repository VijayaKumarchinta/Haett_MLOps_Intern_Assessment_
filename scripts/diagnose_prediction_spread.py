"""
Diagnose Prediction Spread
Tests the model on ACTUAL test set users to see if the probability compression
is a real model issue or just caused by manually-crafted curl data not matching
the training distribution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.config import FEATURES_DIR, MODELS_DIR, RANDOM_SEED

print("=" * 70)
print("  DIAGNOSIS: Prediction Probability Spread")
print("=" * 70)

# 1. Load feature matrix
X = pd.read_csv(FEATURES_DIR / "features_encoded.csv")
y = pd.read_csv(FEATURES_DIR / "target.csv")["churned"]

user_ids = X["user_id"]
X_feat = X.drop(columns=["user_id"])

print(f"\n  Features: {X_feat.shape[1]}")
print(f"  Samples:  {len(X_feat)}")
print(f"  Churn rate: {y.mean():.1%}")

# 2. Load model and scaler
model = joblib.load(MODELS_DIR / "churn_model.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl") if (MODELS_DIR / "scaler.pkl").exists() else None
threshold_path = MODELS_DIR / "optimal_threshold.txt"
threshold = float(open(threshold_path).read()) if threshold_path.exists() else 0.5

print(f"\n  Model type: {type(model).__name__}")
print(f"  Scaler: {'Yes' if scaler else 'No'}")
print(f"  Optimal threshold: {threshold:.4f}")

# 3. Split to get test set (same as pipeline's 60/20/20 split)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_feat, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25,
    random_state=RANDOM_SEED, stratify=y_train_full,
)

print(f"\n  Test set size: {len(X_test)}")
print(f"  Test set churn rate: {y_test.mean():.1%}")

# 4. Predict on test set
if scaler:
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
else:
    y_proba_test = model.predict_proba(X_test)[:, 1]

# 5. Show probability distribution
print(f"\n{'=' * 70}")
print(f"  PROBABILITY DISTRIBUTION ON TEST SET ({len(X_test)} users)")
print(f"{'=' * 70}")

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(y_proba_test, p)
    print(f"  Percentile {p:2d}th: {val:.4f}")

print(f"\n  Min:     {y_proba_test.min():.6f}")
print(f"  Max:     {y_proba_test.max():.4f}")
print(f"  Mean:    {y_proba_test.mean():.4f}")
print(f"  Std:     {y_proba_test.std():.4f}")
print(f"  Range:   {y_proba_test.max() - y_proba_test.min():.4f}")

# 6. Split by actual label
churned_probas = y_proba_test[y_test.values == 1]
not_churned_probas = y_proba_test[y_test.values == 0]

print(f"\n  Churned users ({len(churned_probas)}):")
print(f"    Mean proba: {churned_probas.mean():.4f}")
print(f"    Median proba: {np.median(churned_probas):.4f}")
for p in [10, 25, 50, 75, 90]:
    print(f"    Percentile {p:2d}th: {np.percentile(churned_probas, p):.4f}")

print(f"\n  Non-churned users ({len(not_churned_probas)}):")
print(f"    Mean proba: {not_churned_probas.mean():.4f}")
print(f"    Median proba: {np.median(not_churned_probas):.4f}")
for p in [10, 25, 50, 75, 90]:
    print(f"    Percentile {p:2d}th: {np.percentile(not_churned_probas, p):.4f}")

# 7. How many exceed threshold?
above_threshold = (y_proba_test >= threshold).sum()
print(f"\n  Users above threshold ({threshold:.4f}): {above_threshold} / {len(X_test)} ({above_threshold/len(X_test)*100:.1f}%)")

# 8. ROC-AUC and PR-AUC to confirm model quality
roc_auc = roc_auc_score(y_test, y_proba_test)
pr_auc = average_precision_score(y_test, y_proba_test)
print(f"\n  ROC-AUC on test set: {roc_auc:.4f}")
print(f"  PR-AUC on test set:  {pr_auc:.4f}")

# 9. Now test the same manual curl data
print(f"\n{'=' * 70}")
print(f"  HAND-CRAFTED CURL DATA (vs model expectations)")
print(f"{'=' * 70}")

manual_users = [
    {"name": "User 101 (loyal)", "data": {
        "days_since_last_order": 2, "days_since_high_value": 2, "days_since_last_late": 999,
        "total_orders": 60, "unique_meal_plans": 3, "mean_days_between_orders": 5,
        "std_days_between_orders": 1, "weekend_order_ratio": 0.3, "order_frequency_per_month": 6,
        "total_spent": 2400, "avg_order_value": 40, "max_order_value": 80, "min_order_value": 20,
        "avg_rating": 4.9, "late_delivery_count": 0, "spending_trend": 0, "value_variability": 60,
        "first_half_avg_value": 38, "second_half_avg_value": 42,
        "n_plan_changes": 0, "monthly_price": 79.99, "subscription_tenure_days": 500,
        "tenure_days": 520,
        "avg_app_logins": 12, "avg_recipes_viewed": 20, "avg_meals_skipped": 0,
        "total_support_tickets": 0, "total_referral_clicks": 25, "avg_orders_per_week": 2,
        "recent_avg_logins": 11, "recent_avg_recipes": 18, "login_decline": 0, "recipe_decline": 0,
        "age": 45, "age_group_code": 2,
    }},
    {"name": "User 202 (disengaged)", "data": {
        "days_since_last_order": 60, "days_since_high_value": 999, "days_since_last_late": 15,
        "total_orders": 2, "unique_meal_plans": 1, "mean_days_between_orders": 30,
        "std_days_between_orders": 0, "weekend_order_ratio": 0, "order_frequency_per_month": 0.5,
        "total_spent": 45, "avg_order_value": 22.5, "max_order_value": 25, "min_order_value": 20,
        "avg_rating": 1.8, "late_delivery_count": 2, "spending_trend": 0, "value_variability": 5,
        "first_half_avg_value": 22, "second_half_avg_value": 23,
        "n_plan_changes": 3, "monthly_price": 29.99, "subscription_tenure_days": 30,
        "tenure_days": 40,
        "avg_app_logins": 0.3, "avg_recipes_viewed": 1, "avg_meals_skipped": 3.5,
        "total_support_tickets": 0, "total_referral_clicks": 0, "avg_orders_per_week": 0.1,
        "recent_avg_logins": 0.1, "recent_avg_recipes": 0.3, "login_decline": 8, "recipe_decline": 5,
        "age": 19, "age_group_code": 0,
    }},
]

# Load feature names
feature_names_path = MODELS_DIR / "feature_names.txt"
feature_names = []
if feature_names_path.exists():
    with open(feature_names_path) as f:
        feature_names = [line.strip() for line in f.readlines()]

print(f"\n  Model expects {len(feature_names)} features")

for user in manual_users:
    # Build feature vector
    feat_dict = {}
    missing = []
    for col in feature_names:
        if col in user["data"]:
            feat_dict[col] = user["data"][col]
        else:
            feat_dict[col] = 0
            missing.append(col)

    feat_df = pd.DataFrame([feat_dict])

    if scaler:
        feat_scaled = scaler.transform(feat_df[feature_names])
        feat_scaled = pd.DataFrame(feat_scaled, columns=feature_names)
        proba = model.predict_proba(feat_scaled)[:, 1][0]
    else:
        proba = model.predict_proba(feat_df[feature_names])[:, 1][0]

    print(f"\n  {user['name']}:")
    print(f"    churn_probability = {proba:.4f}")
    if missing:
        print(f"    Missing {len(missing)} features (filled with 0): {missing[:5]}...")

# 10. Test with REAL users from test set
print(f"\n{'=' * 70}")
print(f"  REAL USERS FROM TEST SET - Sample predictions")
print(f"{'=' * 70}")

# Pick some high-probability and low-probability users
indices_sorted = np.argsort(y_proba_test)
n_sample = 5

print(f"\n  Highest churn probability users:")
for idx in indices_sorted[-n_sample:]:
    actual = "CHURNED" if y_test.values[idx] == 1 else "stayed"
    print(f"    user_id={user_ids.values[idx]:6d}  proba={y_proba_test[idx]:.4f}  actual: {actual}")

print(f"\n  Lowest churn probability users:")
for idx in indices_sorted[:n_sample]:
    actual = "CHURNED" if y_test.values[idx] == 1 else "stayed"
    print(f"    user_id={user_ids.values[idx]:6d}  proba={y_proba_test[idx]:.4f}  actual: {actual}")

print(f"\n  Random sample from middle:")
for idx in indices_sorted[len(indices_sorted)//2 - 2:len(indices_sorted)//2 + 3]:
    actual = "CHURNED" if y_test.values[idx] == 1 else "stayed"
    print(f"    user_id={user_ids.values[idx]:6d}  proba={y_proba_test[idx]:.4f}  actual: {actual}")

print(f"\n{'=' * 70}")
if y_proba_test.max() > 0.3:
    print(f"  VERDICT: Model produces a GOOD spread of probabilities")
    print(f"           Range: {y_proba_test.min():.4f} to {y_proba_test.max():.4f}")
    print(f"           The manual curl test data was just not representative of real users.")
else:
    print(f"  VERDICT: Model probabilities ARE compressed")
    print(f"           Range: {y_proba_test.min():.4f} to {y_proba_test.max():.4f}")
    print(f"           Need to fix calibration or strengthen signal.")
print(f"{'=' * 70}")
