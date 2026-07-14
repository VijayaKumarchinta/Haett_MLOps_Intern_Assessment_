"""
Haett MLOps Churn Prediction System

Usage:
    python -m src           Run the full pipeline
    python -m src --api     Start the API server
    python -m src --fast    Run pipeline without hyperparameter tuning
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    args = sys.argv[1:]

    if "--api" in args:
        # Start API server
        print("Starting Haett Churn Prediction API...")
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ]
        subprocess.run(cmd, cwd=ROOT)
    else:
        # Run pipeline
        env = dict(os.environ)
        if "--fast" in args:
            env["N_HPARAM_ITER"] = "5"
            print("[Fast mode] Using reduced hyperparameter iterations.")

        result = subprocess.run(
            [sys.executable, "src/run_pipeline.py"],
            cwd=ROOT,
            env=env,
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
