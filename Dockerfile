FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY pyproject.toml .
COPY src ./src

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install -e .
RUN pip install --no-cache-dir --prefix=/install scikit-learn==1.6.1

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY --from=builder /install /usr/local
COPY src ./src
COPY mlruns ./mlruns
COPY mlflow.db ./mlflow.db

EXPOSE 8000

CMD ["uvicorn", "airbnb_serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
