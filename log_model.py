import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

# نمونه داده با همان featureها
data = [
    {
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "neighbourhood_name": "Centrum-West",
        "accommodates": 2,
        "bedrooms": 1.0,
        "beds": 1.0,
        "bathrooms": 1.0,
        "listing_price": 150.0,
        "minimum_nights": 2,
        "maximum_nights": 365,
        "instant_bookable": True,
        "host_is_superhost": False,
        "host_listing_count": 1,
        "total_reviews_before_cutoff": 10.0,
        "unique_reviewers_before_cutoff": 9.0,
        "avg_comment_len_before_cutoff": 120.0,
        "max_comment_len_before_cutoff": 300.0,
        "days_since_last_review": 30.0,
        "available_days_last_90d": 45,
        "available_rate_last_90d": 0.5,
        "avg_minimum_nights_calendar_last_90d": 2.0,
        "avg_maximum_nights_calendar_last_90d": 365.0,
        "available_days_last_30d": 15,
        "available_rate_last_30d": 0.5,
        "avg_minimum_nights_calendar_last_30d": 2.0,
        "avg_maximum_nights_calendar_last_30d": 365.0,
    },
    {
        "room_type": "Private room",
        "property_type": "Private room in rental unit",
        "neighbourhood_name": "De Baarsjes - Oud-West",
        "accommodates": 1,
        "bedrooms": 1.0,
        "beds": 1.0,
        "bathrooms": 1.0,
        "listing_price": 80.0,
        "minimum_nights": 1,
        "maximum_nights": 30,
        "instant_bookable": False,
        "host_is_superhost": True,
        "host_listing_count": 2,
        "total_reviews_before_cutoff": 25.0,
        "unique_reviewers_before_cutoff": 20.0,
        "avg_comment_len_before_cutoff": 90.0,
        "max_comment_len_before_cutoff": 220.0,
        "days_since_last_review": 10.0,
        "available_days_last_90d": 20,
        "available_rate_last_90d": 0.22,
        "avg_minimum_nights_calendar_last_90d": 1.0,
        "avg_maximum_nights_calendar_last_90d": 30.0,
        "available_days_last_30d": 8,
        "available_rate_last_30d": 0.27,
        "avg_minimum_nights_calendar_last_30d": 1.0,
        "avg_maximum_nights_calendar_last_30d": 30.0,
    },
]

df = pd.DataFrame(data)
y = [1, 0]

cat_cols = ["room_type", "property_type", "neighbourhood_name"]
num_cols = [c for c in df.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression()),
    ]
)

model.fit(df, y)

mlflow.set_experiment("airbnb-serving")

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, artifact_path="model")
    print("RUN_ID:", run.info.run_id)
