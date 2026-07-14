"""Create a Feast-compatible local feature repository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_REPO = ROOT / "feature_repo"

CANDIDATE_INPUTS = [
    ROOT / "data" / "features" / "features_encoded.csv",
    ROOT / "data" / "features" / "feature_matrix.csv",
    ROOT / "data" / "features" / "features.csv",
]

FEAST_DATA_PATH = FEATURE_REPO / "data" / "churn_features.parquet"

DEFINITIONS_PATH = FEATURE_REPO / "feature_definitions.py"


def locate_feature_file() -> Path:
    for path in CANDIDATE_INPUTS:
        if path.exists():
            return path

    available = list((ROOT / "data" / "features").glob("*.csv"))

    if available:
        return available[0]

    raise FileNotFoundError(
        "No feature CSV was found. " "Run python src/run_pipeline.py first."
    )


def normalize_numeric_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    normalized = dataframe.copy()

    for column in normalized.columns:
        if column in {
            "user_id",
            "event_timestamp",
            "churned",
            "churn",
            "target",
        }:
            continue

        series = normalized[column]

        # Feast 0.64 materialization can reject numpy.bool_ when
        # the schema expects a numeric primitive. Convert every
        # boolean feature to an explicit int64.
        if is_bool_dtype(series.dtype):
            normalized[column] = series.fillna(False).astype("int64")

        elif is_integer_dtype(series.dtype):
            normalized[column] = series.fillna(0).astype("int64")

        elif is_float_dtype(series.dtype):
            normalized[column] = series.fillna(0.0).astype("float64")

        elif is_numeric_dtype(series.dtype):
            normalized[column] = (
                pd.to_numeric(
                    series,
                    errors="coerce",
                )
                .fillna(0.0)
                .astype("float64")
            )

        else:
            raise TypeError(
                f"Feature {column!r} is not numeric. " f"Found dtype: {series.dtype}"
            )

    return normalized


def feast_type(dtype) -> str:
    if is_bool_dtype(dtype):
        return "Int64"

    if is_integer_dtype(dtype):
        return "Int64"

    return "Float64"


def prepare_feast_repository() -> None:
    feature_input = locate_feature_file()
    dataframe = pd.read_csv(feature_input)

    target_columns = {
        "churned",
        "churn",
        "target",
    }

    if "user_id" not in dataframe.columns:
        dataframe.insert(
            0,
            "user_id",
            range(1, len(dataframe) + 1),
        )

    dataframe["user_id"] = (
        pd.to_numeric(
            dataframe["user_id"],
            errors="coerce",
        )
        .fillna(0)
        .astype("int64")
    )

    dataframe = normalize_numeric_features(dataframe)

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in target_columns | {"user_id"}
    ]

    # Exclude labels and unused source columns entirely.
    feast_dataframe = dataframe[["user_id", *feature_columns]].copy()

    feast_dataframe["event_timestamp"] = pd.Timestamp.now(tz="UTC")

    FEAST_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feast_dataframe.to_parquet(
        FEAST_DATA_PATH,
        index=False,
    )

    required_types = sorted(
        {feast_type(feast_dataframe[column].dtype) for column in feature_columns}
    )

    schema_lines = "\n".join(
        (
            f'        Field(name="{column}", '
            f"dtype={feast_type(feast_dataframe[column].dtype)}),"
        )
        for column in feature_columns
    )

    definitions = f"""
from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import {", ".join(required_types)}
from feast.value_type import ValueType

user = Entity(
    name="user",
    join_keys=["user_id"],
    value_type=ValueType.INT64,
    description="Haett customer identifier",
)

churn_feature_source = FileSource(
    name="haett_churn_source",
    path="data/churn_features.parquet",
    timestamp_field="event_timestamp",
)

churn_feature_view = FeatureView(
    name="haett_churn_features",
    entities=[user],
    ttl=timedelta(days=3650),
    schema=[
{schema_lines}
    ],
    source=churn_feature_source,
    online=True,
    description="Features used by the Haett churn model",
)

churn_feature_service = FeatureService(
    name="haett_churn_service",
    features=[churn_feature_view],
)
"""

    DEFINITIONS_PATH.write_text(
        definitions.strip() + "\n",
        encoding="utf-8",
    )

    print(f"Input feature file: {feature_input}")
    print("Feast source written to: " f"{FEAST_DATA_PATH}")
    print("Feature definitions written to: " f"{DEFINITIONS_PATH}")
    print("Features registered in schema: " f"{len(feature_columns)}")

    boolean_columns = [
        column
        for column in feature_columns
        if is_bool_dtype(feast_dataframe[column].dtype)
    ]

    if boolean_columns:
        raise RuntimeError(
            "Boolean columns remain after normalization: " f"{boolean_columns}"
        )


if __name__ == "__main__":
    prepare_feast_repository()
