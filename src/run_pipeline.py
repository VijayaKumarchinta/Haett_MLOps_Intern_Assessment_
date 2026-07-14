"""
Pipeline Orchestrator
Runs the entire MLOps pipeline end-to-end: data generation → preprocessing → feature engineering → training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate_data import generate_all_data
from src.data.preprocess import preprocess_all
from src.data.feature_engineering import build_feature_matrix
from src.models.train import train_all_models


def run_full_pipeline():
    """Execute the complete MLOps pipeline."""
    print("=" * 60)
    print("  Haett MLOps Churn Prediction Pipeline")
    print("=" * 60)

    # Step 1: Data Generation
    print("\n[1/4] STEP 1: Data Generation")
    print("-" * 40)
    generate_all_data()

    # Step 2: Data Preprocessing
    print("\n[2/4] STEP 2: Data Preprocessing")
    print("-" * 40)
    preprocess_all()

    # Step 3: Feature Engineering
    print("\n[3/4] STEP 3: Feature Engineering")
    print("-" * 40)
    X, y = build_feature_matrix()

    # Step 4: Model Training
    print("\n[4/4] STEP 4: Model Training")
    print("-" * 40)
    results, best_model_name = train_all_models()

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Best Model: {best_model_name}")
    best_metrics = results[best_model_name]["metrics"]
    for k, v in best_metrics.items():
        print(f"    {k}: {v}")
    print("\n  To start the API:")
    print("    uvicorn src.api.main:app --reload")
    print("\n  To view MLflow UI:")
    print("    mlflow ui --backend-store-uri mlruns")
    print()


if __name__ == "__main__":
    run_full_pipeline()
