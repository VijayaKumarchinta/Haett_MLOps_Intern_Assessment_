"""Validate online Feast feature retrieval."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[1]
REPO_PATH = ROOT / "feature_repo"
PARQUET_PATH = REPO_PATH / "data" / "churn_features.parquet"


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError("Feast source parquet does not exist.")

    dataframe = pd.read_parquet(PARQUET_PATH)

    feature_columns = [
        column
        for column in dataframe.columns
        if column
        not in {
            "user_id",
            "event_timestamp",
        }
    ]

    if not feature_columns:
        raise RuntimeError("No Feast features were generated.")

    user_id = int(dataframe.iloc[0]["user_id"])
    selected_features = feature_columns[:3]

    store = FeatureStore(repo_path=str(REPO_PATH))

    result = store.get_online_features(
        features=[f"haett_churn_features:{feature}" for feature in selected_features],
        entity_rows=[{"user_id": user_id}],
    ).to_dict()

    missing_features = [
        feature
        for feature in selected_features
        if not result.get(feature) or result[feature][0] is None
    ]

    if missing_features:
        raise RuntimeError(
            "Feast online store returned null values " f"for: {missing_features}"
        )

    print("Feast online feature retrieval successful:")
    print(result)


if __name__ == "__main__":
    main()
