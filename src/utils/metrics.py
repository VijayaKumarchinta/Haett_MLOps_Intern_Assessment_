"""
Evaluation Metrics Module
Provides utilities for model evaluation including classification metrics and threshold tuning.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    """Compute comprehensive classification metrics.

    Returns:
        dict with accuracy, balanced_accuracy, precision, recall, f1_score,
        and (if y_proba provided) roc_auc, pr_auc, brier_score.
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }

    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        metrics["pr_auc"] = round(average_precision_score(y_true, y_proba), 4)
        # Brier score measures probability calibration (lower is better)
        # Clip probabilities to avoid extreme values in Brier computation
        y_proba_clipped = np.clip(y_proba, 0.001, 0.999)
        metrics["brier_score"] = round(brier_score_loss(y_true, y_proba_clipped), 4)

    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Find the optimal probability threshold maximizing F1 score.

    Uses precision-recall curve to find the threshold that gives the best
    trade-off between precision and recall.

    Returns:
        dict with optimal_threshold, max_f1, precision_at_optimal, recall_at_optimal.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Remove the last element of thresholds (thresholds has one less element)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return {
        "optimal_threshold": round(best_threshold, 4),
        "max_f1": round(f1_scores[best_idx], 4),
        "precision_at_optimal": round(precisions[best_idx], 4),
        "recall_at_optimal": round(recalls[best_idx], 4),
    }


def compute_lift_at_top_k(y_true: np.ndarray, y_proba: np.ndarray, top_k_percent: float = 0.1) -> float:
    """Compute the lift in recall when targeting the top-k% highest-risk users.

    This is a business-focused metric: if we intervene on the top 10% of users
    by churn probability, what fraction of actual churners do we capture?

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted churn probabilities.
        top_k_percent: Fraction of users to target (e.g., 0.1 = top 10%).

    Returns:
        Lift value (recall_at_top_k / baseline_recall).
    """
    n_top = max(1, int(len(y_true) * top_k_percent))
    top_k_idx = np.argsort(y_proba)[-n_top:]  # highest probabilities
    n_churners_in_top_k = y_true.iloc[top_k_idx].sum() if hasattr(y_true, 'iloc') else y_true[top_k_idx].sum()
    total_churners = y_true.sum()
    recall_at_top_k = n_churners_in_top_k / total_churners if total_churners > 0 else 0
    baseline = top_k_percent
    lift = recall_at_top_k / baseline if baseline > 0 else 0
    return round(float(lift), 4)


def assess_risk_level(probability: float) -> str:
    """Convert churn probability to risk level category."""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def get_business_recommendation(probability: float, features: dict | None = None) -> str:
    """Generate business recommendation based on churn probability and features."""
    if probability < 0.3:
        return "No action needed. User is at low risk of churning."

    elif probability < 0.6:
        reasons = []
        if features:
            if features.get("days_since_last_order", 0) > 14:
                reasons.append("user hasn't ordered recently")
            if features.get("avg_rating", 5) < 3:
                reasons.append("low satisfaction ratings")
            # Check both feature naming conventions
            tickets = features.get("total_support_tickets", features.get("support_tickets", 0))
            if tickets > 3:
                reasons.append("multiple support tickets submitted")

        base = "Medium risk detected. Recommend:"
        recs = [
            "Send a personalized re-engagement email with a 15% discount",
            "Offer a free recipe consultation to explore meal preferences",
            "Share new menu additions tailored to their dietary preference",
        ]
        if reasons:
            base += f" (Signals: {', '.join(reasons)})"
        return f"{base} {' | '.join(recs)}"

    else:  # High risk
        reasons = []
        if features:
            if features.get("days_since_last_order", 0) > 30:
                reasons.append("no orders in 30+ days")
            if features.get("avg_rating", 5) < 2.5:
                reasons.append("consistently low ratings")
            if features.get("is_sub_active", True) is False:
                reasons.append("subscription cancelled")
            if features.get("login_decline", 0) > 2:
                reasons.append("steep decline in app engagement")

        base = "HIGH RISK - Immediate retention action required. Recommend:"
        recs = [
            "Call or SMS the user with a personalized win-back offer (50% off next box)",
            "Assign a dedicated meal planner to address dietary concerns",
            "Offer a free week trial with new menu rotation",
            "Survey the user to understand specific dissatisfaction reasons",
        ]
        if reasons:
            base += f" (Signals: {', '.join(reasons)})"
        return f"{base} {' | '.join(recs)}"
