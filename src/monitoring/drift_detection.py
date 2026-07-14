"""
Data Drift Detection Module
Monitors feature distributions over time using statistical tests.

Drift detection methods:
- Continuous features: Kolmogorov-Smirnov test (detects distribution shifts)
- Categorical/binary features: Chi-squared test or Population Stability Index (PSI)
- Overall drift flag based on share of drifted features

Note: This implementation uses scipy (not Evidently AI) because Evidently 0.7.x
has an incompatible internal API. The statistical approach provides equivalent
functionality: per-feature drift scores, configurable alert thresholds, and
JSON reports for scheduled monitoring.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import MODELS_DIR

logger = logging.getLogger(__name__)

# Directories
MONITORING_DIR = MODELS_DIR.parent / "monitoring"
REFERENCE_DIR = MONITORING_DIR / "reference"
DRIFT_REPORTS_DIR = MONITORING_DIR / "reports"

# Drift thresholds
DRIFT_THRESHOLD = 0.15   # Share of drifted features → warning
CRITICAL_DRIFT_THRESHOLD = 0.30  # → critical alert
KS_PVALUE_THRESHOLD = 0.05  # p-value below this = significant drift
PSI_THRESHOLD = 0.2  # Population Stability Index above this = significant drift

# Key features to monitor
MONITORED_FEATURES = [
    "days_since_last_order", "total_orders", "total_spent",
    "avg_order_value", "avg_rating", "total_support_tickets",
    "is_sub_active", "avg_app_logins", "login_decline",
    "order_frequency_per_month", "subscription_tenure_days", "tenure_days",
]


def _ensure_dirs():
    """Create required directories if they don't exist."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_reference_data(
    features: pd.DataFrame | None = None,
    predictions: pd.Series | None = None,
    metadata: dict | None = None,
) -> Path:
    """Save current feature distributions as reference data for drift monitoring.

    Args:
        features: Feature matrix (e.g., training data for the best model).
        predictions: Model predictions/probabilities (saved for future use).
        metadata: Optional metadata dict (model name, date, etc.).

    Returns:
        Path to the saved reference data directory.
    """
    _ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_dir = REFERENCE_DIR / f"ref_{timestamp}"
    ref_dir.mkdir(parents=True, exist_ok=True)

    # Save monitored features only (keeps reference small)
    if features is not None:
        cols = [c for c in MONITORED_FEATURES if c in features.columns]
        if len(cols) < 5:
            cols = features.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in cols if c != "user_id"][:50]
        features[cols].to_parquet(ref_dir / "features.parquet")
        logger.info("Reference saved: %s (%d samples, %d features)",
                     ref_dir / "features.parquet", len(features), len(cols))

    if predictions is not None:
        pd.DataFrame({"prediction": predictions}).to_parquet(ref_dir / "predictions.parquet")

    meta = {
        "timestamp": timestamp,
        "n_samples": len(features) if features is not None else 0,
        "n_features": len(features.columns) if features is not None else 0,
        "model_name": (metadata or {}).get("model_name", "unknown"),
    }
    with open(ref_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    _cleanup_old_references()
    return ref_dir


def _cleanup_old_references(max_keep: int = 3):
    """Remove old reference datasets, keeping only the most recent ones."""
    ref_dirs = sorted(REFERENCE_DIR.glob("ref_*"))
    while len(ref_dirs) > max_keep:
        shutil.rmtree(ref_dirs.pop(0))


def _load_latest_reference() -> dict | None:
    """Load the most recent reference dataset."""
    ref_dirs = sorted(REFERENCE_DIR.glob("ref_*"))
    if not ref_dirs:
        return None
    latest = ref_dirs[-1]
    result = {"metadata": {}}
    meta_path = latest / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            result["metadata"] = json.load(f)
    feat_path = latest / "features.parquet"
    if feat_path.exists():
        result["features"] = pd.read_parquet(feat_path)
    pred_path = latest / "predictions.parquet"
    if pred_path.exists():
        result["predictions"] = pd.read_parquet(pred_path)["prediction"]
    return result


def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    Uses percentile-based bins from the combined data range.
    PSI < 0.1 = no shift, 0.1–0.2 = moderate shift, > 0.2 = significant shift.
    """
    combined = np.concatenate([expected, actual])
    bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    # Ensure unique bin edges (handle duplicate percentiles)
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    expected_pct = np.histogram(expected, bins=bins, density=False)[0].astype(float) + 1e-10
    actual_pct = np.histogram(actual, bins=bins, density=False)[0].astype(float) + 1e-10
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()
    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return float(psi)


def _detect_column_drift(
    ref_values: np.ndarray, cur_values: np.ndarray, col_name: str
) -> dict:
    """Detect drift for a single column using appropriate statistical test.

    Uses KS test for continuous distributions, PSI for all numeric.
    Returns drift score, p-value, and whether drift was detected.
    """
    ref_values = ref_values[~np.isnan(ref_values)]
    cur_values = cur_values[~np.isnan(cur_values)]

    if len(ref_values) < 5 or len(cur_values) < 5:
        return {"drift_score": 0.0, "p_value": 1.0, "drifted": False,
                "n_ref": len(ref_values), "n_cur": len(cur_values)}

    # Check if binary/categorical (few unique values)
    n_unique = len(np.unique(np.concatenate([ref_values, cur_values])))

    if n_unique <= 5:
        # Chi-squared test for categorical
        all_cats = np.unique(np.concatenate([ref_values, cur_values]))
        ref_counts = np.array([(ref_values == c).sum() for c in all_cats]) + 1e-10
        cur_counts = np.array([(cur_values == c).sum() for c in all_cats]) + 1e-10
        _, p_value = stats.chisquare(cur_counts, f_exp=ref_counts)
        drift_score = float(p_value)
        drifted = p_value < KS_PVALUE_THRESHOLD
    else:
        # KS test for continuous
        statistic, p_value = stats.ks_2samp(ref_values, cur_values)
        # Also compute PSI for additional signal
        psi = _compute_psi(ref_values, cur_values)
        drift_score = float(statistic)
        drifted = (p_value < KS_PVALUE_THRESHOLD) or (psi > PSI_THRESHOLD)

    return {
        "drift_score": round(drift_score, 4),
        "p_value": round(float(p_value), 4),
        "drifted": bool(drifted),
        "n_ref": int(len(ref_values)),
        "n_cur": int(len(cur_values)),
    }


def generate_drift_report(
    current_data: pd.DataFrame,
    report_name: str | None = None,
) -> dict:
    """Generate a data drift report comparing current data against the reference.

    Uses KS tests for continuous features and Chi-squared for categorical.
    Reports per-feature drift status, overall drift share, and alert level.

    Args:
        current_data: Current batch of feature data to check for drift.
        report_name: Optional name for the report file.

    Returns:
        dict with drift status, per-feature metrics, and paths to saved reports.
    """
    _ensure_dirs()

    reference = _load_latest_reference()
    if reference is None or "features" not in reference:
        return {"status": "error", "error": "No reference data available."}

    ref_features = reference["features"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = report_name or f"drift_report_{timestamp}"

    # Align columns
    common_cols = [c for c in ref_features.columns if c in current_data.columns]
    if not common_cols:
        return {"status": "error", "error": "No common columns between reference and current data."}

    # Detect drift per column
    feature_metrics = {}
    drifted_features = []

    for col in common_cols:
        ref_vals = ref_features[col].values
        cur_vals = current_data[col].values
        result = _detect_column_drift(ref_vals, cur_vals, col)
        feature_metrics[col] = result
        if result["drifted"]:
            drifted_features.append(col)

    n_total = len(common_cols)
    n_drifted = len(drifted_features)
    drift_share = n_drifted / n_total if n_total > 0 else 0

    if drift_share >= CRITICAL_DRIFT_THRESHOLD:
        status = "critical"
    elif drift_share >= DRIFT_THRESHOLD:
        status = "warning"
    else:
        status = "ok"

    report_dict = {
        "timestamp": timestamp,
        "status": status,
        "n_features": n_total,
        "n_drifted_features": n_drifted,
        "drift_share": round(drift_share, 4),
        "n_reference_samples": len(ref_features),
        "n_current_samples": len(current_data),
        "drifted_features": drifted_features,
        "feature_metrics": {k: v for k, v in feature_metrics.items()},
    }

    json_path = DRIFT_REPORTS_DIR / f"{report_name}.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)

    result = {
        "status": status,
        "timestamp": timestamp,
        "json_path": str(json_path),
        "drift_summary": {
            "n_drifted": n_drifted,
            "drifted_features": drifted_features,
            "dataset_drift": drift_share > DRIFT_THRESHOLD,
        },
        "n_drifted_features": n_drifted,
        "n_features": n_total,
        "drift_share": drift_share,
    }

    logger.info("Drift report: status=%s, drifted=%d/%d features", status, n_drifted, n_total)
    return result


def list_drift_reports(n_recent: int = 10) -> pd.DataFrame:
    """List recent drift reports with their status and drift share."""
    if not DRIFT_REPORTS_DIR.exists():
        return pd.DataFrame(columns=["report", "timestamp", "status", "drift_share"])

    reports = []
    for json_path in sorted(DRIFT_REPORTS_DIR.glob("*.json"), reverse=True)[:n_recent]:
        try:
            with open(json_path) as f:
                data = json.load(f)
            reports.append({
                "report": json_path.stem,
                "timestamp": data.get("timestamp", ""),
                "status": data.get("status", "unknown"),
                "drift_share": data.get("drift_share", 0),
                "n_drifted": data.get("n_drifted_features", 0),
                "n_features": data.get("n_features", 0),
            })
        except Exception:
            continue
    return pd.DataFrame(reports)


def run_drift_monitoring(current_features: pd.DataFrame, **kwargs) -> dict:
    """Run drift monitoring: compare current data against reference.

    Args:
        current_features: Current batch of feature data.

    Returns:
        dict with monitoring results.
    """
    return generate_drift_report(current_features)
