from pydantic import BaseModel
from typing import Optional


class ListingFeatures(BaseModel):
    room_type: str
    property_type: str
    neighbourhood_name: str
    accommodates: int
    bedrooms: float
    beds: float
    bathrooms: float
    listing_price: float
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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_run_id: str


class PredictionResponse(BaseModel):
    listing_id: Optional[str] = None
    prediction: int
    probability_high_demand: float
    model_run_id: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    model_run_id: str