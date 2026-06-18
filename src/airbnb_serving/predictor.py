import pandas as pd

from airbnb_serving.schema import (
    ListingFeatures,
    PredictionResponse,
    BatchPredictionResponse,
)


def _to_row_dict(features: ListingFeatures | dict) -> dict:
    """
    Convert either a Pydantic model or dict into a plain Python dict.
    Compatible with both Pydantic v1 and v2.
    """
    if hasattr(features, "model_dump"):  # Pydantic v2
        return features.model_dump()
    if hasattr(features, "dict"):  # Pydantic v1
        return features.dict()
    return features


def predict_single(features: ListingFeatures | dict, model, run_id: str) -> PredictionResponse:
    """
    Run prediction for a single listing.
    Assumes the loaded model supports predict() and predict_proba().
    """
    row_dict = _to_row_dict(features)
    X = pd.DataFrame([row_dict])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    return PredictionResponse(
        listing_id=None,
        prediction=int(pred),
        probability_high_demand=float(proba),
        model_run_id=run_id,
    )


def predict_batch(
    features_list: list[ListingFeatures | dict],
    model,
    run_id: str,
) -> BatchPredictionResponse:
    """
    Run prediction for a batch of listings.
    """
    rows = [_to_row_dict(features) for features in features_list]
    X = pd.DataFrame(rows)

    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    predictions = [
        PredictionResponse(
            listing_id=None,
            prediction=int(pred),
            probability_high_demand=float(proba),
            model_run_id=run_id,
        )
        for pred, proba in zip(preds, probas)
    ]

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        model_run_id=run_id,
    )