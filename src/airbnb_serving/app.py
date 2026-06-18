from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import mlflow.pyfunc

app = FastAPI(title="Airbnb Price Prediction API")

model = None

class AirbnbFeatures(BaseModel):

    room_type: str
    property_type: str
    neighbourhood_name: str

    accommodates: int
    bedrooms: float
    beds: float
    bathrooms: float

    number_of_reviews: float

    review_scores_rating: float
    review_scores_accuracy: float
    review_scores_cleanliness: float
    review_scores_checkin: float
    review_scores_communication: float
    review_scores_location: float
    review_scores_value: float

    minimum_nights: int
    maximum_nights: int

    instant_bookable: bool
    host_is_superhost: bool
    host_listing_count: int

    total_reviews_before_cutoff: float
    unique_reviewers_before_cutoff: float

    avg_comment_len_before_cutoff: float
    max_comment_len_before_cutoff: float

    days_since_last_review: float

    available_days_last_90d: int
    available_rate_last_90d: float

    avg_minimum_nights_calendar_last_90d: float
    avg_maximum_nights_calendar_last_90d: float

    available_days_last_30d: int
    available_rate_last_30d: float

    avg_minimum_nights_calendar_last_30d: float
    avg_maximum_nights_calendar_last_30d: float

    listing_price: float

@app.on_event("startup")
def load_model():

    global model

    try:

        model = mlflow.pyfunc.load_model(
            "/app/mlruns/1/models/m-dfb803ba756347abb649a67b04d7c3e7/artifacts"
        )

        print(" MLflow model loaded successfully")

    except Exception as e:

        print(" Failed to load MLflow model:", e)

        model = None


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(features: AirbnbFeatures):

    if model is None:
        return {"error": "model not loaded"}

    df = pd.DataFrame([features.dict()])

    prediction = model.predict(df)

    return {
        "prediction": float(prediction[0])
    }


@app.post("/predict_batch")
def predict_batch(features: List[AirbnbFeatures]):

    if model is None:
        return {"error": "model not loaded"}

    rows = [f.dict() for f in features]

    df = pd.DataFrame(rows)

    predictions = model.predict(df)

    return {
        "predictions": predictions.tolist()
    }