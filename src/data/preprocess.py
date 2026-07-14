"""
Data Preprocessing Module
Loads, cleans, and validates raw datasets for feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    SNAPSHOT_DATE,
)

np.random.seed(RANDOM_SEED)


def load_raw_data() -> dict:
    """Load all raw CSV datasets."""
    datasets = {}
    for fname in [
        "users.csv",
        "orders.csv",
        "subscriptions.csv",
        "engagement.csv",
        "churn_labels.csv",
    ]:
        path = RAW_DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {path}. Run generate_data.py first."
            )
        datasets[fname.replace(".csv", "")] = pd.read_csv(path)
    return datasets


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate users data."""
    df = users.copy()

    # Standardize date columns
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")

    # Remove invalid ages
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]

    # Fill missing cities with mode
    df["city"] = df["city"].fillna(
        df["city"].mode()[0] if not df["city"].mode().empty else "Unknown"
    )

    # Fill missing dietary preferences
    df["dietary_preference"] = df["dietary_preference"].fillna("balanced")

    return df.reset_index(drop=True)


def clean_orders(
    orders: pd.DataFrame, min_date: pd.Timestamp, max_date: pd.Timestamp
) -> pd.DataFrame:
    """Clean and validate orders data."""
    df = orders.copy()

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Remove orders outside valid date range
    df = df[(df["order_date"] >= min_date) & (df["order_date"] <= max_date)]

    # Remove negative or zero order values
    df = df[df["order_value"] > 0]

    # Fill missing ratings with user's average or global average
    df["rating"] = df.groupby("user_id")["rating"].transform(
        lambda x: x.fillna(x.mean())
    )
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # Fill missing delivery info
    df["on_time_delivery"] = df["on_time_delivery"].fillna(True)
    df["delivery_hour"] = df["delivery_hour"].fillna("afternoon")
    df["meal_plan"] = df["meal_plan"].fillna("weekly_classic")

    return df.reset_index(drop=True)


def clean_subscriptions(subscriptions: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate subscriptions data."""
    df = subscriptions.copy()

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    # Fill missing end dates with start date + 30 days (default minimum)
    df["end_date"] = df["end_date"].fillna(df["start_date"] + pd.Timedelta(days=30))

    # Ensure end_date >= start_date
    mask = df["end_date"] < df["start_date"]
    df.loc[mask, "end_date"] = df.loc[mask, "start_date"] + pd.Timedelta(days=30)

    # Fill missing prices with median
    df["monthly_price"] = df["monthly_price"].fillna(df["monthly_price"].median())

    # Fill missing status
    df["status"] = df["status"].fillna("active")
    df["cancellation_reason"] = df["cancellation_reason"].fillna("")

    return df.reset_index(drop=True)


def clean_engagement(engagement: pd.DataFrame) -> pd.DataFrame:
    """Clean engagement data."""
    df = engagement.copy()

    df["week_date"] = pd.to_datetime(df["week_date"], errors="coerce")

    # Clip negative values
    for col in [
        "app_logins",
        "recipes_viewed",
        "meals_skipped",
        "support_tickets",
        "referral_clicks",
    ]:
        df[col] = df[col].clip(0)

    return df.reset_index(drop=True)


def preprocess_all() -> dict:
    """Load, clean, and save all datasets."""
    print("[...] Loading raw data...")
    datasets = load_raw_data()

    print("  |-- Cleaning users...")
    datasets["users"] = clean_users(datasets["users"])

    print("  |-- Cleaning orders...")
    min_date = datasets["users"]["signup_date"].min()
    max_date = SNAPSHOT_DATE
    datasets["orders"] = clean_orders(datasets["orders"], min_date, max_date)

    print("  |-- Cleaning subscriptions...")
    datasets["subscriptions"] = clean_subscriptions(datasets["subscriptions"])

    print("  |-- Cleaning engagement...")
    datasets["engagement"] = clean_engagement(datasets["engagement"])

    print("  +-- Cleaning churn labels...")
    datasets["churn_labels"] = datasets["churn_labels"].dropna(subset=["churned"])

    # Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        df.to_csv(PROCESSED_DATA_DIR / f"{name}_clean.csv", index=False)
        print(f"    - {name}_clean.csv ({len(df)} records)")

    print(f"\n[OK] Preprocessing complete! Files saved to: {PROCESSED_DATA_DIR}")
    return datasets


if __name__ == "__main__":
    preprocess_all()
