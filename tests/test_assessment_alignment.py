"""Assessment-alignment regression tests."""

import pandas as pd

from src.data.feature_engineering import compute_demographic_features
from src.data.generate_data import generate_churn_labels
from src.utils.config import CHURN_LABEL_DAYS, SNAPSHOT_DATE


def test_churn_target_uses_active_future_cohort_only():
    users = pd.DataFrame({"user_id": [1, 2, 3]})
    subscriptions = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "start_date": [
                SNAPSHOT_DATE - pd.Timedelta(days=100),
                SNAPSHOT_DATE - pd.Timedelta(days=100),
                SNAPSHOT_DATE - pd.Timedelta(days=100),
            ],
            "end_date": [
                SNAPSHOT_DATE + pd.Timedelta(days=10),
                SNAPSHOT_DATE + pd.Timedelta(days=CHURN_LABEL_DAYS + 30),
                SNAPSHOT_DATE - pd.Timedelta(days=1),
            ],
            "status": ["active", "active", "cancelled"],
        }
    )

    labels = generate_churn_labels(users, subscriptions).set_index("user_id")

    assert set(labels.index) == {1, 2}
    assert labels.loc[1, "churned"] == 1
    assert labels.loc[2, "churned"] == 0


def test_demographic_features_match_api_inputs():
    users = pd.DataFrame(
        {
            "user_id": [1],
            "age": [30],
            "dietary_preference": ["vegan"],
            "referral_source": ["friend"],
            "city": ["Hyderabad"],
        }
    )

    features = compute_demographic_features(users)

    assert list(features.columns) == ["user_id", "age", "age_group_code"]
