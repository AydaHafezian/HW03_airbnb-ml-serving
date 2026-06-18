# Airbnb Price Prediction API (MLOps)

A production-style machine learning service for predicting Airbnb listing prices using a trained model served through a FastAPI REST API and containerized with Docker.

This project demonstrates core MLOps practices including model serving, reproducible deployment, containerized inference, MLflow model loading, and batch prediction benchmarking.

---

## Project Overview

The goal of this project is to deploy a trained machine learning model as an API service. The model predicts Airbnb listing prices based on property information, host attributes, review scores, availability statistics, and historical review features.

The service provides endpoints for:

- API health checking
- Single listing price prediction
- Batch listing price prediction
- Interactive API testing through Swagger UI

---

## Tech Stack

- Python
- FastAPI
- Uvicorn
- MLflow
- Pandas
- Docker
- Docker Compose
- Jupyter Notebook

---

## Features

- FastAPI-based model serving API
- MLflow model loading from saved artifacts
- Dockerized deployment
- Health check endpoint
- Single prediction endpoint
- Batch prediction endpoint
- Swagger UI documentation
- Benchmark comparison between single and batch inference

---

## Project Structure

```text
01_model_serving_student/
│
├── src/
│   └── airbnb_serving/
│       └── app.py                  # FastAPI application
│
├── mlruns/                         # MLflow model artifacts
├── notebooks/                      # Notebooks for testing and benchmarking
│
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker Compose service configuration
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## Model Serving

The trained model is loaded using MLflow when the FastAPI application starts.

Inside the Docker container, the model is loaded from the MLflow artifact path:

```text
/app/mlruns/1/models/m-dfb803ba756347abb649a67b04d7c3e7/artifacts
```

The `/health` endpoint confirms whether the model has been loaded successfully.

---

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/airbnb-ml-serving.git
cd airbnb-ml-serving
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### 2. Build and Start the Docker Container

```bash
docker compose up --build
```

Or run it in detached mode:

```bash
docker compose up -d --build
```

The API will be available at:

```text
http://localhost:8000
```

---

### 3. Check Container Status

```bash
docker ps
```

You should see the `airbnb-serving` container running and mapped to port `8000`.

---

## API Documentation

FastAPI automatically provides Swagger UI documentation.

Open this URL in your browser:

```text
http://localhost:8000/docs
```

The following endpoints should be available:

- `GET /health`
- `POST /predict`
- `POST /predict_batch`

---

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### Single Prediction

```http
POST /predict
```

Example request body:

```json
{
  "room_type": "Entire home/apt",
  "property_type": "Apartment",
  "neighbourhood_name": "Manhattan",
  "accommodates": 2,
  "bedrooms": 1,
  "beds": 1,
  "bathrooms": 1,
  "number_of_reviews": 10,
  "review_scores_rating": 90,
  "review_scores_accuracy": 9,
  "review_scores_cleanliness": 9,
  "review_scores_checkin": 9,
  "review_scores_communication": 9,
  "review_scores_location": 9,
  "review_scores_value": 9,
  "minimum_nights": 2,
  "maximum_nights": 30,
  "instant_bookable": true,
  "host_is_superhost": false,
  "host_listing_count": 1,
  "total_reviews_before_cutoff": 10,
  "unique_reviewers_before_cutoff": 10,
  "avg_comment_len_before_cutoff": 50,
  "max_comment_len_before_cutoff": 100,
  "days_since_last_review": 30,
  "available_days_last_90d": 60,
  "available_rate_last_90d": 0.7,
  "avg_minimum_nights_calendar_last_90d": 2,
  "avg_maximum_nights_calendar_last_90d": 30,
  "available_days_last_30d": 20,
  "available_rate_last_30d": 0.66,
  "avg_minimum_nights_calendar_last_30d": 2,
  "avg_maximum_nights_calendar_last_30d": 30,
  "listing_price": 120
}
```

Example response:

```json
{
  "prediction": 132.5
}
```

---

### Batch Prediction

```http
POST /predict_batch
```

The batch endpoint accepts a list of listing feature objects and returns a list of predictions.

Example response:

```json
{
  "predictions": [132.5, 148.2, 101.7]
}
```

Batch prediction is useful when many listings need to be scored efficiently.

---

## Benchmarking

The project includes benchmark testing to compare:

1. Sending many individual requests to `/predict`
2. Sending one batch request to `/predict_batch`

Typical results show that batch inference is significantly faster.

Example benchmark output:

```text
Method             Total Time       Per Prediction
Single requests       3.20s              32 ms
Batch request         0.45s             4.5 ms

Speedup: 7x
```

This demonstrates the performance benefit of batch model inference.

---

## Useful Commands

Start the service:

```bash
docker compose up --build
```

Start the service in background:

```bash
docker compose up -d --build
```

View logs:

```bash
docker compose logs -f airbnb-serving
```

Check running containers:

```bash
docker ps
```

Stop the service:

```bash
docker compose down
```

Rebuild without cache:

```bash
docker compose build --no-cache
```

---

## Expected Submission Screenshots

Recommended screenshots for project submission:

- Swagger UI at `http://localhost:8000/docs`
- `/health` response showing `model_loaded: true`
- Successful `/predict` request and response
- Benchmark output showing batch speedup
- `docker ps` showing the running `airbnb-serving` container

---

## Future Improvements

Possible future enhancements:

- Add model monitoring
- Add structured logging
- Add automated tests
- Add CI/CD pipeline with GitHub Actions
- Deploy the API to a cloud platform
- Register and version models with MLflow Model Registry
- Add authentication for production use

---

## Author

This project was developed as part of an MLOps model-serving workflow to demonstrate deployment of a machine learning model as a containerized API service.
