# Disease Outbreak Early Warning System - API Documentation

This document provides comprehensive documentation for the Disease Outbreak Early Warning System API, built with FastAPI and following OpenAPI 3.0 specification.

## Table of Contents
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Get Predictions](#get-predictions)
  - [Batch Predictions](#batch-predictions)
  - [Get Model Information](#get-model-information)
  - [Get Model Metrics](#get-model-metrics)
  - [Get Data Statistics](#get-data-statistics)
  - [Trigger Model Retraining](#trigger-model-retraining)

## Base URL

All API endpoints are relative to the base URL:

```
https://api.disease-outbreak.example.com/v1
```

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer <your_jwt_token>
```

## Error Handling

### Error Response Format

```json
{
  "detail": [
    {
      "loc": ["string"],
      "msg": "string",
      "type": "string"
    }
  ]
}
```

### Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | OK - Request successful |
| 201 | Created - Resource created |
| 400 | Bad Request - Invalid request format |
| 401 | Unauthorized - Invalid or missing authentication |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 422 | Unprocessable Entity - Validation error |
| 500 | Internal Server Error - Server error |

## Endpoints

### Health Check

Check if the API is running and its dependencies are available.

**Endpoint:** `GET /health`

#### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-08-17T12:00:00Z",
  "services": {
    "database": true,
    "cache": true,
    "ml_model": true
  }
}
```

### Get Predictions

Get predictions for a single input.

**Endpoint:** `POST /predict`

#### Request Body

```json
{
  "features": {
    "temperature": 28.5,
    "humidity": 75.2,
    "rainfall": 15.0,
    "population_density": 1200.5,
    "avg_historical_cases": 25.3,
    "case_trend": 1.2,
    "social_media_mentions": 45
  },
  "model_version": "1.0.0"
}
```

#### Response

```json
{
  "prediction": 0.85,
  "risk_level": "high",
  "model_version": "1.0.0",
  "timestamp": "2025-08-17T12:00:00Z"
}
```

### Batch Predictions

Get predictions for multiple inputs.

**Endpoint:** `POST /predict/batch`

#### Request Body

```json
{
  "instances": [
    {
      "temperature": 28.5,
      "humidity": 75.2,
      "rainfall": 15.0,
      "population_density": 1200.5,
      "avg_historical_cases": 25.3,
      "case_trend": 1.2,
      "social_media_mentions": 45
    },
    {
      "temperature": 22.1,
      "humidity": 65.8,
      "rainfall": 5.0,
      "population_density": 800.2,
      "avg_historical_cases": 12.1,
      "case_trend": 0.8,
      "social_media_mentions": 22
    }
  ],
  "model_version": "1.0.0"
}
```

#### Response

```json
{
  "predictions": [
    {
      "prediction": 0.85,
      "risk_level": "high"
    },
    {
      "prediction": 0.42,
      "risk_level": "medium"
    }
  ],
  "model_version": "1.0.0",
  "timestamp": "2025-08-17T12:00:00Z"
}
```

### Get Model Information

Get information about the currently deployed model.

**Endpoint:** `GET /model`

#### Response

```json
{
  "name": "disease_outbreak_predictor",
  "version": "1.0.0",
  "stage": "production",
  "description": "Random Forest model for predicting disease outbreak risks",
  "created_at": "2025-08-15T10:30:00Z",
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1": 0.92,
    "roc_auc": 0.96
  },
  "features": [
    "temperature",
    "humidity",
    "rainfall",
    "population_density",
    "avg_historical_cases",
    "case_trend",
    "social_media_mentions"
  ]
}
```

### Get Model Metrics

Get detailed metrics for a specific model version.

**Endpoint:** `GET /model/metrics`

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| version | string | No | Model version (default: latest) |
| start_date | string | No | Start date for metrics (ISO 8601) |
| end_date | string | No | End date for metrics (ISO 8601) |

#### Response

```json
{
  "model_version": "1.0.0",
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1": 0.92,
    "roc_auc": 0.96,
    "confusion_matrix": {
      "true_positive": 120,
      "false_positive": 10,
      "true_negative": 180,
      "false_negative": 8
    },
    "feature_importance": {
      "case_trend": 0.28,
      "avg_historical_cases": 0.25,
      "social_media_mentions": 0.18,
      "temperature": 0.15,
      "population_density": 0.08,
      "humidity": 0.04,
      "rainfall": 0.02
    }
  },
  "evaluation_date": "2025-08-16T15:30:00Z"
}
```

### Get Data Statistics

Get statistics about the training data.

**Endpoint:** `GET /data/statistics`

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| feature | string | No | Specific feature to get statistics for |
| start_date | string | No | Start date filter (ISO 8601) |
| end_date | string | No | End date filter (ISO 8601) |

#### Response

```json
{
  "statistics": {
    "temperature": {
      "count": 1000,
      "mean": 26.5,
      "std": 5.2,
      "min": 15.0,
      "25%": 22.8,
      "50%": 26.3,
      "75%": 30.1,
      "max": 38.5
    },
    "humidity": {
      "count": 1000,
      "mean": 68.2,
      "std": 12.5,
      "min": 30.1,
      "25%": 58.7,
      "50%": 67.9,
      "75%": 77.3,
      "max": 95.0
    },
    "outbreak_risk": {
      "count": 1000,
      "mean": 0.32,
      "std": 0.28,
      "min": 0.0,
      "25%": 0.1,
      "50%": 0.25,
      "75%": 0.48,
      "max": 1.0
    }
  },
  "correlation_matrix": {
    "temperature": {
      "humidity": 0.45,
      "outbreak_risk": 0.32
    },
    "humidity": {
      "outbreak_risk": 0.51
    }
  },
  "data_points": 1000,
  "time_range": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-08-17T00:00:00Z"
  }
}
```

### Trigger Model Retraining

Trigger a new model training job.

**Endpoint:** `POST /model/retrain`

#### Request Body

```json
{
  "training_data_path": "s3://disease-outbreak-data/training/v2/train.parquet",
  "validation_data_path": "s3://disease-outbreak-data/validation/v2/validation.parquet",
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5
  },
  "notify_email": "ml-team@example.com"
}
```

#### Response

```json
{
  "training_id": "train_abc123xyz",
  "status": "started",
  "message": "Model training job started",
  "start_time": "2025-08-17T12:00:00Z",
  "monitor_url": "https://mlflow.example.com/#/experiments/1/runs/abc123xyz"
}
```

## Rate Limiting

The API is rate limited to prevent abuse. The current limits are:

- 100 requests per minute per IP for anonymous users
- 1000 requests per minute per authenticated user

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1629198000
```

## Versioning

API versioning is handled through the URL path. The current version is `v1`.

Example: `https://api.disease-outbreak.example.com/v1/health`

## SDKs and Client Libraries

### Python Client

```python
from disease_outbreak_client import DiseaseOutbreakClient

# Initialize client
client = DiseaseOutbreakClient(
    api_key="your_api_key",
    base_url="https://api.disease-outbreak.example.com/v1"
)

# Get prediction
prediction = client.predict(
    temperature=28.5,
    humidity=75.2,
    rainfall=15.0,
    population_density=1200.5,
    avg_historical_cases=25.3,
    case_trend=1.2,
    social_media_mentions=45
)

print(f"Outbreak risk: {prediction['risk_level']} ({prediction['prediction']:.2f})")
```

## WebSocket API

For real-time predictions and monitoring, a WebSocket API is available at:

```
wss://api.disease-outbreak.example.com/v1/ws
```

### Events

#### Subscribe to Predictions

```json
{
  "type": "subscribe",
  "channel": "predictions",
  "region": "south-asia"
}
```

#### Prediction Update

```json
{
  "type": "prediction_update",
  "region": "south-asia",
  "prediction": 0.78,
  "risk_level": "high",
  "timestamp": "2025-08-17T12:00:00Z"
}
```

## Webhook Integration

You can configure webhooks to receive notifications about important events.

### Events

- `model.deployed`: Triggered when a new model is deployed to production
- `prediction.anomaly`: Triggered when an anomalous prediction is detected
- `data.drift`: Triggered when data drift is detected
- `model.retrained`: Triggered when model retraining is complete

### Example Payload

```json
{
  "event": "model.deployed",
  "model": {
    "name": "disease_outbreak_predictor",
    "version": "1.1.0",
    "metrics": {
      "accuracy": 0.93,
      "precision": 0.92,
      "recall": 0.94,
      "f1": 0.93
    }
  },
  "timestamp": "2025-08-17T12:00:00Z"
}
```

## Support

For support, please contact `support@disease-outbreak.example.com` or open an issue in our [GitHub repository](https://github.com/your-org/disease-outbreak-mlops).
