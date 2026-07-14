from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float64, Int64
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
        Field(name="days_since_last_order", dtype=Float64),
        Field(name="tenure_days", dtype=Float64),
        Field(name="total_orders", dtype=Float64),
        Field(name="std_days_between_orders", dtype=Float64),
        Field(name="orders_last_30_days", dtype=Float64),
        Field(name="avg_order_value", dtype=Float64),
        Field(name="avg_rating", dtype=Float64),
        Field(name="coupon_usage_count", dtype=Float64),
        Field(name="coupon_usage_rate", dtype=Float64),
        Field(name="monthly_price", dtype=Float64),
        Field(name="subscription_tenure_days", dtype=Int64),
        Field(name="n_plan_changes", dtype=Int64),
        Field(name="avg_app_logins", dtype=Float64),
        Field(name="avg_meals_skipped", dtype=Float64),
        Field(name="total_support_tickets", dtype=Int64),
        Field(name="age", dtype=Int64),
        Field(name="age_group_code", dtype=Int64),
        Field(name="diet_balanced", dtype=Int64),
        Field(name="diet_keto", dtype=Int64),
        Field(name="diet_low_carb", dtype=Int64),
        Field(name="diet_mediterranean", dtype=Int64),
        Field(name="diet_paleo", dtype=Int64),
        Field(name="diet_vegan", dtype=Int64),
        Field(name="referral_blog", dtype=Int64),
        Field(name="referral_direct", dtype=Int64),
        Field(name="referral_facebook", dtype=Int64),
        Field(name="referral_friend", dtype=Int64),
        Field(name="referral_google", dtype=Int64),
        Field(name="referral_instagram", dtype=Int64),
        Field(name="referral_tiktok", dtype=Int64),
    ],
    source=churn_feature_source,
    online=True,
    description="Features used by the Haett churn model",
)

churn_feature_service = FeatureService(
    name="haett_churn_service",
    features=[churn_feature_view],
)
