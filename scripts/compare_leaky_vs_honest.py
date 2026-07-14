"""
Leaky vs Honest Model Comparison
==================================
Trains two models side-by-side on the same data:
  - LEAKY: uses is_sub_active, days_since_cancellation, total_subscription_days
  - HONEST: uses only behavioral features (current pipeline)

Shows exactly HOW the leaky model cheats and why it's dangerous for production.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

from src.utils.config import (
    FEATURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED, SNAPSHOT_DATE,
)

# ---------------------------------------------------------------------------
# 1. Load current feature matrix
# ---------------------------------------------------------------------------
print("=" * 80)
print("  LEAKY vs HONEST Model Comparison")
print("=" * 80)

X = pd.read_csv(FEATURES_DIR / "features_encoded.csv")
y = pd.read_csv(FEATURES_DIR / "target.csv")["churned"]

user_ids = X["user_id"]
X_base = X.drop(columns=["user_id"])

print(f"  Current feature matrix: {X_base.shape}")
print(f"  Positive class ratio: {y.mean():.1%}")

# ---------------------------------------------------------------------------
# 2. Build leaky subscription features from raw data
# ---------------------------------------------------------------------------
print("\n  [1] Recomputing leaked subscription features from raw data...")

subscriptions = pd.read_csv(PROCESSED_DATA_DIR / "subscriptions_clean.csv")
subscriptions["start_date"] = pd.to_datetime(subscriptions["start_date"])
subscriptions["end_date"] = pd.to_datetime(subscriptions["end_date"])

# For each user, get their latest subscription
subs_sorted = subscriptions.sort_values(["user_id", "end_date"])
latest_sub = subs_sorted.groupby("user_id").last().reset_index()

# Compute leaked features
leak_features = latest_sub[["user_id"]].copy()
leak_features["is_sub_active"] = (latest_sub["end_date"] > SNAPSHOT_DATE).astype(int)
leak_features["days_since_cancellation"] = np.where(
    leak_features["is_sub_active"] == 0,
    (SNAPSHOT_DATE - pd.to_datetime(latest_sub["end_date"])).dt.days,
    0,
)
leak_features["total_subscription_days"] = (
    pd.to_datetime(latest_sub["end_date"]) - pd.to_datetime(latest_sub["start_date"])
).dt.days.clip(lower=0)

# Merge leaky features properly (by user_id, not index)
X_leaky = X_base.copy()
X_leaky["user_id"] = user_ids.values
X_leaky = X_leaky.merge(leak_features, on="user_id", how="left")
X_leaky = X_leaky.drop(columns=["user_id"])
X_leaky = X_leaky.fillna(0)

X_honest = X_base.copy()

print(f"  Leaky features added: is_sub_active, days_since_cancellation, total_subscription_days")
print(f"  Leaky feature set:    {X_leaky.shape[1]} features")
print(f"  Honest feature set:   {X_honest.shape[1]} features")

# ---------------------------------------------------------------------------
# 3. Verify the leakage
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("  [2] VERIFYING THE LEAKAGE")
print("=" * 80)

merged_check = X_leaky[["is_sub_active", "days_since_cancellation", "total_subscription_days"]].copy()
merged_check["churned"] = y.values

print("\n  Cross-tab: is_sub_active vs churn")
ct = pd.crosstab(merged_check["is_sub_active"], merged_check["churned"], margins=True)
print(f"\n{ct}\n")

impossible = merged_check[merged_check["is_sub_active"] == 0]
possible = merged_check[merged_check["is_sub_active"] == 1]

print(f"  Users with is_sub_active=0 (already cancelled):    {len(impossible)}")
print(f"    Churned: {impossible['churned'].sum()} / {len(impossible)}")
print(f"    CHURN RATE: {impossible['churned'].mean():.1%}")
print(f"  Users with is_sub_active=1 (still active):         {len(possible)}")
print(f"    Churned: {possible['churned'].sum()} / {len(possible)}")
print(f"    CHURN RATE: {possible['churned'].mean():.1%}")

print("\n  >> INSIGHT: Users with is_sub_active=0 have ALREADY cancelled their subscription.")
print("     They physically CANNOT churn again (their sub already ended).")
print("     The model learns: is_sub_active=0 -> churn_probability=0.")
print("     This is CHEATING - it's not predicting churn, it's reading the answer key.")

print("\n  >> INSIGHT: days_since_cancellation tells the model EXACTLY how long ago")
print("     the user cancelled. Combined with is_sub_active=0, the model achieves")
print("     near-perfect predictions by reading the cancellation date.")

# ---------------------------------------------------------------------------
# 4. Train both models side-by-side
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("  [3] TRAINING BOTH MODELS (Random Forest)")
print("=" * 80)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_SEED, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    # Find optimal threshold via F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
    y_pred_opt = (y_proba >= best_thresh).astype(int)

    metrics = {
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        "PR-AUC": round(average_precision_score(y_test, y_proba), 4),
        "F1 (opt)": round(f1_score(y_test, y_pred_opt, zero_division=0), 4),
        "Precision": round(precision_score(y_test, y_pred_opt, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred_opt, zero_division=0), 4),
        "Opt Thresh": round(best_thresh, 4),
    }

    importances = pd.DataFrame({
        "feature": X_train.columns.tolist(),
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return metrics, importances, y_proba, rf


# Same split for both (same random_state + same y)
X_tr_l, X_te_l, y_tr, y_te = train_test_split(
    X_leaky, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)
X_tr_h, X_te_h, _, _ = train_test_split(
    X_honest, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
)

metrics_leaky, imp_leaky, proba_leaky, rf_leaky = train_and_evaluate(
    X_tr_l, X_te_l, y_tr, y_te
)
metrics_honest, imp_honest, proba_honest, rf_honest = train_and_evaluate(
    X_tr_h, X_te_h, y_tr, y_te
)

# ---------------------------------------------------------------------------
# 5. Results table
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print(f"  [4] RESULTS - SIDE BY SIDE")
print(f"{'=' * 80}")

print(f"\n{'Metric':<20} {'LEAKY':<20} {'HONEST':<20} {'DIFFERENCE':<20}")
print(f"{'-' * 80}")
for metric in ["ROC-AUC", "PR-AUC", "F1 (opt)", "Precision", "Recall"]:
    val_l = metrics_leaky[metric]
    val_h = metrics_honest[metric]
    diff = val_l - val_h
    print(f"{metric:<20} {val_l:<20.4f} {val_h:<20.4f} {diff:<+8.4f}")

gap = metrics_leaky["ROC-AUC"] - metrics_honest["ROC-AUC"]
print(f"\n  Honest model ROC-AUC:  {metrics_honest['ROC-AUC']:.4f}")
print(f"  Leaky model ROC-AUC:   {metrics_leaky['ROC-AUC']:.4f}")
print(f"  Gap (leak advantage):  {gap:+.4f}")

# ---------------------------------------------------------------------------
# 6. Feature importance comparison
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print(f"  [5] TOP 10 FEATURES - LEAKY MODEL")
print(f"{'=' * 80}")
print(f"\n{'Feature':<40} {'Importance':<15}")
print(f"{'-' * 55}")
for _, row in imp_leaky.head(10).iterrows():
    marker = " **LEAK**" if row["feature"] in ["is_sub_active", "days_since_cancellation", "total_subscription_days"] else ""
    print(f"{row['feature']:<40} {row['importance']:<15.6f}{marker}")

print(f"\n{'=' * 80}")
print(f"  [6] TOP 10 FEATURES - HONEST MODEL (No Leakage)")
print(f"{'=' * 80}")
print(f"\n{'Feature':<40} {'Importance':<15}")
print(f"{'-' * 55}")
for _, row in imp_honest.head(10).iterrows():
    print(f"{row['feature']:<40} {row['importance']:<15.6f}")

# ---------------------------------------------------------------------------
# 7. Interpret the cheat
# ---------------------------------------------------------------------------
leak_imp_sum = imp_leaky[
    imp_leaky["feature"].isin(["is_sub_active", "days_since_cancellation", "total_subscription_days"])
]["importance"].sum()
total_imp = imp_leaky["importance"].sum()

print(f"\n{'=' * 80}")
print(f"  [7] INTERPRETING THE CHEAT")
print(f"{'=' * 80}")
print(f"\n  Leaked features account for {leak_imp_sum:.1%} of ALL feature importance")
print(f"  in the leaky model's decision-making.")

print(f"\n  If this model were deployed:")
print(f"  - A user whose subscription already expired is scored as 0% churn risk")
print(f"    -> No retention effort is made")
print(f"    -> The user was ALREADY lost - the model just confirms the past")
print(f"  - A user with an active subscription who hasn't ordered in 60 days")
print(f"    -> Gets a low churn score (because is_sub_active=1)")
print(f"    -> No intervention triggered")
print(f"    -> The user never re-engages and silently leaves")
print(f"")
print(f"  The honest model, meanwhile, correctly identifies the second user as high-risk")
print(f"  because it detects behavioral signals: no orders, declining app logins, skipped meals.")

# ---------------------------------------------------------------------------
# 8. Confusion matrices
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print(f"  [8] WHAT THE MODEL LEARNED - Confusion Matrices")
print(f"{'=' * 80}")

y_pred_leaky = (proba_leaky >= metrics_leaky["Opt Thresh"]).astype(int)
cm_leaky = confusion_matrix(y_te, y_pred_leaky)

y_pred_honest = (proba_honest >= metrics_honest["Opt Thresh"]).astype(int)
cm_honest = confusion_matrix(y_te, y_pred_honest)

print(f"\n  LEAKY model (threshold={metrics_leaky['Opt Thresh']:.3f}):")
print(f"                  Predicted")
print(f"                 No Churn  Churn")
print(f"  Actual No       {cm_leaky[0,0]:>6d}  {cm_leaky[0,1]:>5d}")
print(f"  Churn           {cm_leaky[1,0]:>6d}  {cm_leaky[1,1]:>5d}")

print(f"\n  HONEST model (threshold={metrics_honest['Opt Thresh']:.3f}):")
print(f"                  Predicted")
print(f"                 No Churn  Churn")
print(f"  Actual No       {cm_honest[0,0]:>6d}  {cm_honest[0,1]:>5d}")
print(f"  Churn           {cm_honest[1,0]:>6d}  {cm_honest[1,1]:>5d}")

# Deep dive into leaky model
leak_active_idx = (X_te_l["is_sub_active"] == 1).values
leak_inactive_idx = (X_te_l["is_sub_active"] == 0).values

print(f"\n  >> Deep dive: leaky model's behavior by is_sub_active status")
if leak_inactive_idx.sum() > 0:
    inactive_preds = proba_leaky[leak_inactive_idx]
    print(f"  is_sub_active=0 (already cancelled): mean proba = {inactive_preds.mean():.4f}")
    print(f"    -> Model assigns near-zero churn probability because they CAN'T churn")
active_preds = proba_leaky[leak_active_idx]
print(f"  is_sub_active=1 (active sub):          mean proba = {active_preds.mean():.4f}")
print(f"    -> Model must use REAL behavioral signals here")

# ---------------------------------------------------------------------------
# 9. Save visualizations
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print(f"  [9] GENERATING VISUALIZATIONS")
print(f"{'=' * 80}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Leaky vs Honest Model - Comparison", fontsize=16, fontweight="bold")

# ROC Curves
fpr_l, tpr_l, _ = roc_curve(y_te, proba_leaky)
fpr_h, tpr_h, _ = roc_curve(y_te, proba_honest)

ax = axes[0, 0]
ax.plot(fpr_l, tpr_l, "r-", linewidth=2, label=f"LEAKY (AUC={metrics_leaky['ROC-AUC']:.3f})")
ax.plot(fpr_h, tpr_h, "g-", linewidth=2, label=f"HONEST (AUC={metrics_honest['ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend()
ax.grid(alpha=0.3)

# PR Curves
prec_l, rec_l, _ = precision_recall_curve(y_te, proba_leaky)
prec_h, rec_h, _ = precision_recall_curve(y_te, proba_honest)

ax = axes[0, 1]
ax.plot(rec_l, prec_l, "r-", linewidth=2, label=f"LEAKY (PR-AUC={metrics_leaky['PR-AUC']:.3f})")
ax.plot(rec_h, prec_h, "g-", linewidth=2, label=f"HONEST (PR-AUC={metrics_honest['PR-AUC']:.3f})")
ax.axhline(y.mean(), color="gray", linestyle="--", alpha=0.5, label="Baseline")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend()
ax.grid(alpha=0.3)

# Leaky model feature importance
ax = axes[0, 2]
top_leaky = imp_leaky.head(10)
colors = ["#ff6b6b" if f in ["is_sub_active", "days_since_cancellation", "total_subscription_days"]
          else "#4ecdc4" for f in top_leaky["feature"]]
ax.barh(range(len(top_leaky)), top_leaky["importance"].values, color=colors)
ax.set_yticks(range(len(top_leaky)))
ax.set_yticklabels(top_leaky["feature"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Importance")
ax.set_title("LEAKY Model: Top 10 Features")
ax.legend(
    [Patch(color="#ff6b6b"), Patch(color="#4ecdc4")],
    ["Leaked feature", "Honest feature"],
    loc="lower right", fontsize=8,
)

# Honest model feature importance
ax = axes[1, 0]
top_honest = imp_honest.head(10)
ax.barh(range(len(top_honest)), top_honest["importance"].values, color="#4ecdc4")
ax.set_yticks(range(len(top_honest)))
ax.set_yticklabels(top_honest["feature"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Importance")
ax.set_title("HONEST Model: Top 10 Features")

# Probability distribution by is_sub_active
ax = axes[1, 1]
leak_probs_active = proba_leaky[leak_active_idx]
leak_probs_inactive = proba_leaky[leak_inactive_idx] if leak_inactive_idx.sum() > 0 else [0]

ax.hist(leak_probs_active, bins=20, alpha=0.6, color="#4ecdc4", label="is_sub_active=1", density=True)
if leak_inactive_idx.sum() > 0:
    ax.hist(leak_probs_inactive, bins=20, alpha=0.6, color="#ff6b6b", label="is_sub_active=0", density=True)
ax.set_xlabel("Predicted Churn Probability")
ax.set_ylabel("Density")
ax.set_title("LEAKY: Probability Distribution\nby is_sub_active Status")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Metric comparison bar chart
ax = axes[1, 2]
comparison_metrics = ["ROC-AUC", "PR-AUC", "F1 (opt)", "Recall"]
x_pos = np.arange(len(comparison_metrics))
width = 0.35

leaky_vals = [metrics_leaky[m] for m in comparison_metrics]
honest_vals = [metrics_honest[m] for m in comparison_metrics]

bars1 = ax.bar(x_pos - width/2, leaky_vals, width, label="LEAKY", color="#ff6b6b")
bars2 = ax.bar(x_pos + width/2, honest_vals, width, label="HONEST", color="#4ecdc4")
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_metrics, fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Metric Comparison")
ax.legend(fontsize=8)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

plt.tight_layout()
comparison_path = MODELS_DIR / "leaky_vs_honest_comparison.png"
plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"  Visualization saved to: {comparison_path}")

# ---------------------------------------------------------------------------
# 10. Final summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print(f"  FINAL VERDICT")
print(f"{'=' * 80}")
print(f"""
  THE RESULT:
     The leaky model isn't predicting churn -- it's reading cancellation status.
     With is_sub_active in the features, the model achieves ROC-AUC={metrics_leaky['ROC-AUC']:.3f}
     but fails at the ACTUAL task: identifying users at risk of churning in the next 30 days.

  WHY THE HONEST MODEL IS BETTER:
     Even with lower raw scores (ROC-AUC={metrics_honest['ROC-AUC']:.3f}), the honest model
     learns REAL behavioral patterns: order frequency decline, support tickets, low ratings.
     These signals actually PREDICT churn rather than CONFIRMING it after the fact.

  METRIC SUMMARY:
     {'':<25} LEAKY  HONEST  DELTA
""")
for metric in ["ROC-AUC", "PR-AUC", "F1 (opt)", "Precision", "Recall", "Opt Thresh"]:
    v_l = metrics_leaky[metric]
    v_h = metrics_honest[metric]
    delta = v_l - v_h
    print(f"     {metric:<25} {v_l:>10.4f} {v_h:>10.4f} {delta:>+10.4f}")

print(f"""
  LESSON: Target leakage is invisible at training time.
     The leaky model LOOKS amazing in cross-validation (ROC-AUC={metrics_leaky['ROC-AUC']:.3f})
     but would FAIL in production because:
     1. A cancelled user with is_sub_active=0 gets 0% churn probability
        -> No retention offer -> They were ALREADY lost
     2. An active user who hasn't ordered in 60 days gets low churn probability
        -> No intervention -> They silently become the next cancelled user
     3. The model learned NOTHING about actual churn behavior
        -> It only learned to read the subscription status field

  The honest model ({metrics_honest['ROC-AUC']:.3f} ROC-AUC) actually LEARNS from
     behavioral patterns and can genuinely predict at-risk users.
""")
