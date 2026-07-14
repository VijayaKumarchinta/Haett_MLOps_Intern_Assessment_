#!/usr/bin/env python3
"""
Drift Detection Runner
Compares current feature data against saved reference data using Evidently.

Usage:
    python scripts/check_drift.py                          # Compare latest reference against itself (baseline)
    python scripts/check_drift.py --data path/to/features.csv  # Compare against a specific CSV
    python scripts/check_drift.py --list                    # List all saved drift reports
    python scripts/check_drift.py --generate                # Generate synthetic current data for testing
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_synthetic_current_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic 'current' feature data for testing drift detection.

    Adds slight perturbation to reference data to simulate natural drift.
    If no reference exists, generates fresh synthetic data.
    """
    from src.monitoring.drift_detection import _load_latest_reference

    reference = _load_latest_reference()
    if reference and "features" in reference:
        ref = reference["features"]
        # Add noise to simulate drift
        noisy = ref.sample(min(n_samples, len(ref)), random_state=42).copy()
        for col in noisy.select_dtypes(include=[np.number]).columns:
            noise = np.random.normal(0, 0.15, len(noisy)) * noisy[col].std()
            noisy[col] = noisy[col] + noise
            noisy[col] = noisy[col].clip(0)  # No negative features
        return noisy
    else:
        # Generate synthetic data from scratch
        np.random.seed(42)
        return pd.DataFrame({
            "days_since_last_order": np.random.exponential(20, n_samples).astype(int),
            "total_orders": np.random.poisson(15, n_samples),
            "total_spent": np.random.exponential(300, n_samples).round(2),
            "avg_order_value": np.random.normal(35, 15, n_samples).clip(5).round(2),
            "avg_rating": np.random.uniform(1, 5, n_samples).round(1),
            "total_support_tickets": np.random.poisson(2, n_samples),
            "is_sub_active": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            "avg_app_logins": np.random.exponential(3, n_samples),
            "login_decline": np.random.exponential(1, n_samples),
            "order_frequency_per_month": np.random.exponential(4, n_samples),
            "subscription_tenure_days": np.random.exponential(200, n_samples).astype(int),
            "tenure_days": np.random.exponential(250, n_samples).astype(int),
        })


def main():
    parser = argparse.ArgumentParser(
        description="Haett MLOps - Data Drift Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=str, help="Path to CSV/parquet file with current features")
    parser.add_argument("--list", action="store_true", help="List recent drift reports")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic current data for testing")
    parser.add_argument("--output", type=str, help="Output path for drift report (optional)")

    args = parser.parse_args()

    # List reports
    if args.list:
        from src.monitoring.drift_detection import list_drift_reports
        reports = list_drift_reports()
        if reports.empty:
            print("No drift reports found.")
        else:
            print("\nRecent Drift Reports:")
            print(reports.to_string(index=False))
        return

    # Generate synthetic data
    if args.generate:
        print("[...] Generating synthetic current data...")
        data = generate_synthetic_current_data()
        print(f"   Generated {len(data)} samples with {len(data.columns)} features")
        current_features = data
        current_predictions = None
    # Load from file
    elif args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"[!] File not found: {data_path}")
            sys.exit(1)
        print(f"[...] Loading data from {data_path}...")
        if data_path.suffix == ".parquet":
            current_features = pd.read_parquet(data_path)
        else:
            current_features = pd.read_csv(data_path)
        current_predictions = None
    else:
        # Compare reference against itself (baseline - should show no drift)
        print("[...] No data provided. Running baseline check (reference vs itself)...")
        from src.monitoring.drift_detection import _load_latest_reference
        reference = _load_latest_reference()
        if reference and "features" in reference:
            current_features = reference["features"].copy()
            current_predictions = None
        else:
            print("[!] No reference data found. Use --generate or --data to provide current data.")
            sys.exit(1)

    # Run drift detection
    print("\n[...] Running drift detection...")
    from src.monitoring.drift_detection import run_drift_monitoring
    result = run_drift_monitoring(current_features, current_predictions)

    # Print results
    print(f"\nDrift Detection Result:")
    print(f"   Status: {result.get('status', 'unknown').upper()}")
    print(f"   Drifted features: {result.get('n_drifted_features', 0)} / {result.get('n_features', 0)}")
    print(f"   Drift share: {result.get('drift_share', 0):.1%}")
    print(f"   Report: {result.get('report_path', 'N/A')}")

    if result.get("drift_summary", {}).get("drifted_features"):
        print("\n   Drifted features:")
        for feat in result["drift_summary"]["drifted_features"]:
            print(f"      - {feat}")

    if result.get("status") == "critical":
        print("\n[ALERT] Critical drift detected! Model may need retraining.")
        sys.exit(2)
    elif result.get("status") == "warning":
        print("\n[WARNING] Drift above threshold. Monitor closely.")
        sys.exit(1)
    else:
        print("\n[OK] No significant drift detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
