"""
Evaluation Metrics Module
Provides utilities for model evaluation including classification metrics, threshold tuning,
risk assessment, and SHAP-driven business recommendations.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)

# ─── Feature → Signal / Action Mapping ──────────────────────────────────────
# Each entry maps a feature name to:
#   - label: human-readable display name
#   - signal: template describing WHY this feature matters (uses SHAP sign)
#   - action: what to do about it (positive-impact = risk-increasing features)
#   - strength_action: what to do if this is a strength (negative-impact = protective features)

_FEATURE_SIGNAL_MAP = {
    "days_since_last_order": {
        "label": "Days since last order",
        "signal": "Hasn't ordered in {value:.0f} days — disengagement signal",
        "action": "Send personalized re-engagement offer with free delivery on next order",
        "strength_action": "Recently ordered — maintain engagement with loyalty rewards",
    },
    "tenure_days": {
        "label": "Account tenure",
        "signal": "Short account history of only {value:.0f} days — low loyalty",
        "action": "Welcome series: offer a discounted 4-week trial to build ordering habit",
        "strength_action": "Long-tenured user — leverage loyalty with VIP perks",
    },
    "total_orders": {
        "label": "Total orders",
        "signal": "Only {value:.0f} total orders placed — low engagement",
        "action": "Offer a 'complete 5 more orders for a free box' incentive",
        "strength_action": "High order volume — recognize with a loyalty badge",
    },
    "orders_last_30_days": {
        "label": "Orders in last 30 days",
        "signal": "Only {value:.0f} orders in the last month — declining activity",
        "action": "Send 'We miss you' email with a free meal upgrade on next order",
        "strength_action": "Recently active — encourage with referral bonus",
    },
    "avg_order_value": {
        "label": "Average order value",
        "signal": "Low average order value of ${value:.2f} — bargain-seeking behavior",
        "action": "Upsell with premium meal bundles or add-on items",
        "strength_action": "High-value customer — offer exclusive tasting menu preview",
    },
    "avg_rating": {
        "label": "Average rating",
        "signal": "Average rating of {value:.1f}/5 — below satisfaction threshold",
        "action": "Schedule a call with a meal planner to address recipe preferences",
        "strength_action": "High ratings — leverage as brand advocate (request review)",
    },
    "coupon_usage_count": {
        "label": "Coupon usage count",
        "signal": "Used coupons {value:.0f} times — price-sensitive behavior",
        "action": "Offer a subscription discount to lock in commitment",
        "strength_action": "Price-insensitive — focus on quality/premium messaging",
    },
    "coupon_usage_rate": {
        "label": "Coupon usage rate",
        "signal": "{value:.0%} of orders used coupons — deal-dependent",
        "action": "Convert to annual subscription with locked-in rate",
        "strength_action": "Low coupon reliance — stable revenue stream",
    },
    "n_plan_changes": {
        "label": "Plan changes",
        "signal": "Changed plans {value:.0f} times — indecision or dissatisfaction",
        "action": "Offer a personalized meal assessment to find the right plan fit",
        "strength_action": "Stable plan — user is satisfied with current offering",
    },
    "monthly_price": {
        "label": "Monthly price",
        "signal": "Spending ${value:.2f}/month — high price sensitivity risk",
        "action": "Check if pricing aligns with perceived value; offer mid-tier alternative",
        "strength_action": "Premium subscriber — ensure VIP service level",
    },
    "subscription_tenure_days": {
        "label": "Subscription tenure",
        "signal": "Only {value:.0f} days subscribed — early churn risk window",
        "action": "Strengthen onboarding with guided meal selection and tips",
        "strength_action": "Long-term subscriber — recognize milestone with a reward",
    },
    "avg_app_logins": {
        "label": "App logins per week",
        "signal": "Only {value:.1f} app logins per week — app disengagement",
        "action": "Send push notification with a surprise reward to re-engage",
        "strength_action": "Frequent app user — leverage app for exclusive offers",
    },
    "avg_meals_skipped": {
        "label": "Meals skipped per week",
        "signal": "Skipping {value:.1f} meals per week — meal fatigue or dissatisfaction",
        "action": "Offer menu customization consultation to refresh meal choices",
        "strength_action": "Low skip rate — meal plans are well-matched",
    },
    "total_support_tickets": {
        "label": "Support tickets",
        "signal": "Filed {value:.0f} support tickets — unresolved issues likely",
        "action": "Prioritize resolving all open tickets; offer goodwill credit",
        "strength_action": "No support issues — satisfied customer experience",
    },
    "age": {
        "label": "Age",
        "signal": "Age {value:.0f} — may need tailored meal offerings",
        "action": "Offer age-relevant meal plans (e.g., senior-friendly options)",
        "strength_action": "Demographic profile is well-served — maintain targeting",
    },
    "age_group_code": {
        "label": "Age group",
        "signal": "Age group may need different communication style",
        "action": "Adjust marketing channel to match age group preferences",
        "strength_action": "Age group well-engaged — continue current approach",
    },
    "std_days_between_orders": {
        "label": "Order consistency",
        "signal": "Irregular ordering pattern (std dev {value:.1f} days) — unpredictable",
        "action": "Offer a recurring subscription incentive to regularize order cadence",
        "strength_action": "Consistent ordering pattern — reliable engagement",
    },
}


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
    f1_scores = (
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return {
        "optimal_threshold": round(best_threshold, 4),
        "max_f1": round(f1_scores[best_idx], 4),
        "precision_at_optimal": round(precisions[best_idx], 4),
        "recall_at_optimal": round(recalls[best_idx], 4),
    }


def compute_lift_at_top_k(
    y_true: np.ndarray, y_proba: np.ndarray, top_k_percent: float = 0.1
) -> float:
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
    n_churners_in_top_k = (
        y_true.iloc[top_k_idx].sum()
        if hasattr(y_true, "iloc")
        else y_true[top_k_idx].sum()
    )
    total_churners = y_true.sum()
    recall_at_top_k = n_churners_in_top_k / total_churners if total_churners > 0 else 0
    baseline = top_k_percent
    lift = recall_at_top_k / baseline if baseline > 0 else 0
    return round(float(lift), 4)


def assess_risk_level(probability: float, optimal_threshold: float = 0.5) -> str:
    """Convert churn probability to risk level using model's decision threshold.

    Uses the model's optimal threshold (F1-maximizing, from training) as the
    reference point for classification:

        Low:    probability < 0.5 * optimal_threshold
        Medium: probability < 1.5 * optimal_threshold
        High:   otherwise

    For a model with optimal_threshold ≈ 0.18 (typical for XGB):
        Low:   < 0.09
        Medium: 0.09 - 0.27
        High:  > 0.27
    """
    low_boundary = 0.5 * optimal_threshold
    high_boundary = 1.5 * optimal_threshold

    if probability < low_boundary:
        return "Low"
    elif probability < high_boundary:
        return "Medium"
    else:
        return "High"


def get_business_recommendation(
    probability: float,
    risk_level: str | None = None,
    optimal_threshold: float = 0.5,
    shap_explanations: list[dict] | None = None,
) -> str:
    """Generate business recommendation based on churn probability and SHAP-driven risk signals.

    Uses SHAP feature contributions to explain WHY the model thinks a user is at risk,
    and provides targeted, actionable retention recommendations.

    Args:
        probability: Churn probability from the model (0-1).
        risk_level: Pre-computed risk level ("Low", "Medium", "High"). If None, computed here.
        optimal_threshold: The F1-maximizing threshold from training.
        shap_explanations: List of dicts with "feature", "value", "impact" keys from SHAP.
            Positive impact = feature increases churn risk.
            Negative impact = feature decreases churn risk.

    Returns:
        A formatted string with risk signals and actionable recommendations.
    """
    if risk_level is None:
        risk_level = assess_risk_level(probability, optimal_threshold)

    # ── LOW RISK ──────────────────────────────────────────────────────────
    if risk_level == "Low":
        # Even low-risk users get a proactive suggestion, not just "no action"
        if shap_explanations:
            # Find the feature that most DECREASES their risk (strength to leverage)
            decreasing = [
                (e["feature"], e["value"], abs(e["impact"]))
                for e in shap_explanations
                if e["impact"] < 0
            ]
            if decreasing:
                best_feature = max(decreasing, key=lambda x: x[2])
                feature_info = _FEATURE_SIGNAL_MAP.get(best_feature[0], None)
                if feature_info:
                    return (
                        f"Low risk (P={probability:.2%}). "
                        f"User is in good standing — no retention action needed. "
                        f"Strength: {feature_info['strength_action']}."
                    )

        return (
            f"Low risk (P={probability:.2%}). "
            f"No action needed. User shows healthy engagement patterns."
        )

    # ── MEDIUM RISK ───────────────────────────────────────────────────────
    if risk_level == "Medium":
        signals = _extract_risk_signals(shap_explanations, max_signals=3)
        actions = _extract_risk_actions(shap_explanations, n_actions=2)

        base = f"At-risk (P={probability:.2%}). Recommend proactive retention:"

        if signals:
            signal_text = "; ".join(signals[:2])
            base = f"At-risk (P={probability:.2%}). Signals: {signal_text}."

        if actions:
            action_lines = "\n  ".join(f"{i}. {a}" for i, a in enumerate(actions, 1))
            return f"{base}\n\nRecommended actions:\n  {action_lines}"

        return (
            f"{base}\n\n"
            f"Recommended actions:\n"
            f"  - Send a re-engagement email with a personalized discount\n"
            f"  - Offer a free recipe consultation"
        )

    # ── HIGH RISK ─────────────────────────────────────────────────────────
    # Always include SHAP-driven signals for high risk
    signals = _extract_risk_signals(shap_explanations, max_signals=5)
    strengths = _extract_strengths(shap_explanations, max_signals=2)
    actions = _extract_risk_actions(shap_explanations, n_actions=4)

    base = (
        f"⚠ HIGH CHURN RISK (P={probability:.2%}). Immediate retention action required."
    )

    if signals:
        base += "\n\nRisk signals:\n"
        for s in signals[:4]:
            base += f"  - {s}\n"

    if strengths:
        base += "\nStrengths to leverage:\n"
        for s in strengths:
            base += f"  - {s}\n"

    if actions:
        base += "\nRecommended actions:\n"
        for i, action in enumerate(actions[:4], 1):
            base += f"  {i}. {action}\n"
    else:
        base += (
            "\n\nRecommended actions:\n"
            "  1. Call or SMS the user with a personalized win-back offer\n"
            "  2. Offer a free week trial with new menu rotation\n"
            "  3. Survey the user to understand specific dissatisfaction reasons\n"
            "  4. Assign a dedicated meal planner to address dietary concerns\n"
        )

    return base.strip()


# ─── Internal Helpers ──────────────────────────────────────────────────────


def _extract_risk_signals(
    shap_explanations: list[dict] | None,
    max_signals: int = 3,
) -> list[str]:
    """Extract human-readable risk signals from SHAP explanations.

    Only considers features with POSITIVE impact (increase churn risk).
    Higher impact → more important signal.
    """
    if not shap_explanations:
        return []

    # Filter features that increase churn risk
    increasing = [
        e for e in shap_explanations if e["impact"] > 0.001  # Only meaningful impacts
    ]

    # Sort by absolute impact (most impactful first)
    increasing.sort(key=lambda x: abs(x["impact"]), reverse=True)

    signals = []
    for explanation in increasing[:max_signals]:
        feature_name = explanation["feature"]
        value = explanation["value"]
        impact = explanation["impact"]
        feature_info = _FEATURE_SIGNAL_MAP.get(feature_name)

        if feature_info:
            signal_text = feature_info["signal"].format(value=value)
            signals.append(f"{signal_text} (impact: +{impact:.3f})")
        else:
            signals.append(
                f"{feature_name} = {value:.2f} increases risk (impact: +{impact:.3f})"
            )

    return signals


def _extract_strengths(
    shap_explanations: list[dict] | None,
    max_signals: int = 2,
) -> list[str]:
    """Extract strengths (protective factors) from SHAP explanations.

    Only considers features with NEGATIVE impact (decrease churn risk).
    """
    if not shap_explanations:
        return []

    decreasing = [e for e in shap_explanations if e["impact"] < -0.001]

    decreasing.sort(key=lambda x: abs(x["impact"]), reverse=True)

    strengths = []
    for explanation in decreasing[:max_signals]:
        feature_name = explanation["feature"]
        value = explanation["value"]
        impact = abs(explanation["impact"])
        feature_info = _FEATURE_SIGNAL_MAP.get(feature_name)

        if feature_info:
            strengths.append(
                f"{feature_info['strength_action']} (impact: -{impact:.3f})"
            )
        else:
            strengths.append(
                f"{feature_name} = {value:.2f} reduces risk (impact: -{impact:.3f})"
            )

    return strengths


def _extract_risk_actions(
    shap_explanations: list[dict] | None,
    n_actions: int = 2,
) -> list[str]:
    """Generate actionable recommendations based on top risk-increasing features."""
    if not shap_explanations:
        return []

    increasing = [e for e in shap_explanations if e["impact"] > 0.001]
    increasing.sort(key=lambda x: abs(x["impact"]), reverse=True)

    actions = []
    for explanation in increasing[:n_actions]:
        feature_name = explanation["feature"]
        feature_info = _FEATURE_SIGNAL_MAP.get(feature_name)
        if feature_info:
            actions.append(feature_info["action"])

    return actions
