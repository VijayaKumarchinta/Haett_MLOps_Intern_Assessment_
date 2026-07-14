#!/usr/bin/env python3
"""
Haett MLOps - One-command runner
Usage:
  python run.py                    # Run full pipeline (data gen → train)
  python run.py --fast             # Run pipeline without hyperparameter tuning
  python run.py --api              # Start the API server (requires trained model)
  python run.py --all              # Full pipeline + start API
  python run.py --api --no-train   # Start API without training (use existing model)
  python run.py --help             # Show this help
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_pipeline(fast: bool = False) -> bool:
    """Run the full pipeline (data generation → preprocessing → features → training).

    Returns True if successful, False otherwise.
    """
    print("=" * 60)
    print("  Haett MLOps - Running Pipeline")
    print("=" * 60)

    if fast:
        print("\n[Fast mode] Skipping hyperparameter tuning for speed.\n")

    env = dict(os.environ)
    if fast:
        env["N_HPARAM_ITER"] = "5"

    result = subprocess.run(
        [sys.executable, "src/run_pipeline.py"],
        cwd=PROJECT_ROOT,
        env=env,
    )
    return result.returncode == 0


def check_model_exists() -> bool:
    """Check if a trained model exists."""
    model_path = PROJECT_ROOT / "models" / "churn_model.pkl"
    return model_path.exists()


def start_api() -> bool:
    """Start the FastAPI server.

    Returns True if the server started successfully.
    """
    if not check_model_exists():
        print("[!] No trained model found. Run pipeline first:")
        print("    python run.py")
        print("    python run.py --api")
        return False

    print("\n" + "=" * 60)
    print("  Starting API Server")
    print("=" * 60)
    print(f"\n  Model: {PROJECT_ROOT / 'models' / 'churn_model.pkl'}")
    print("\n  API Docs:    http://localhost:8000/docs")
    print("  Health:      http://localhost:8000/health")
    print("  Predict:     POST http://localhost:8000/predict")
    print("\n  Press Ctrl+C to stop.\n")

    result = subprocess.run(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=PROJECT_ROOT,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Haett MLOps Churn Prediction - Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                        Full pipeline (slow, with tuning)
  python run.py --fast                 Quick pipeline (no tuning)
  python run.py --api                  Start API (model must exist)
  python run.py --all                  Pipeline + API
  python run.py --api --no-train       Start API without training
        """,
    )

    parser.add_argument(
        "--fast", action="store_true",
        help="Run pipeline without hyperparameter tuning (much faster)",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Start the FastAPI prediction server",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run full pipeline then start API",
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="When used with --api, skip training and use existing model",
    )

    args = parser.parse_args()

    # No args: run pipeline
    if not any([args.fast, args.api, args.all, args.no_train]):
        success = run_pipeline(fast=False)
        sys.exit(0 if success else 1)

    # --all: pipeline + api
    if args.all:
        success = run_pipeline(fast=args.fast)
        if not success:
            print("[FAIL] Pipeline failed. Aborting.")
            sys.exit(1)
        start_api()
        return

    # --api: just start the API
    if args.api:
        if not args.no_train and not check_model_exists():
            print("[...] No model found. Running pipeline first...")
            success = run_pipeline(fast=args.fast)
            if not success:
                print("[FAIL] Pipeline failed.")
                sys.exit(1)
        start_api()
        return

    # --fast only: run fast pipeline
    if args.fast:
        success = run_pipeline(fast=True)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
