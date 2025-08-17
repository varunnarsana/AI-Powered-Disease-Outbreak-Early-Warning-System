#!/usr/bin/env python3
"""
FastAPI Service for Disease Outbreak Early Warning System
Provides REST endpoints for predictions, monitoring, and alerting
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, BaseSettings, ValidationError
import uvicorn
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry, push_to_gateway
from dotenv import load_dotenv
import traceback
import json

# Custom Exceptions
class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass

class ModelLoadingError(Exception):
    """Raised when ML models fail to load"""
    pass

class PredictionError(Exception):
    """Raised when prediction fails"""
    pass

class ValidationError(Exception):
    """Raised when input validation fails"""
    pass

# Error Response Model
class ErrorResponse(BaseModel):
    status_code: int
    error: str
    message: str
    timestamp: str
    path: str
    details: Optional[Dict[str, Any]] = None

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "insecure-secret-key")
    API_DEBUG: bool = os.getenv("API_DEBUG", "False").lower() == "true"
    API_ALLOWED_ORIGINS: list = os.getenv("API_ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
    
    # Database Settings
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "disease_outbreak")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "db")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    
    # Redis Settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")

# Initialize settings
settings = Settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request latency')
ACTIVE_ALERTS = Gauge('active_alerts_total', 'Total active alerts')
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'Model prediction accuracy')

# Initialize FastAPI app
app = FastAPI(
    title="Disease Outbreak Early Warning System API",
    description="AI-powered system for predicting disease outbreaks using real-time health data",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status_code=exc.status_code,
            error=exc.__class__.__name__,
            message=str(exc.detail) if hasattr(exc, 'detail') else str(exc),
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="ValidationError",
            message="Invalid request data",
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path,
            details={"errors": exc.errors()}
        ).dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    # Log the full exception
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error=exc.__class__.__name__,
            message="An unexpected error occurred",
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path,
            details={"error_details": str(exc)}
        ).dict()
    )

# Custom exception handlers
@app.exception_handler(DatabaseConnectionError)
async def db_connection_error_handler(request: Request, exc: DatabaseConnectionError):
    """Handle database connection errors"""
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="DatabaseConnectionError",
            message="Unable to connect to the database",
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path,
            details={"error_details": str(exc)}
        ).dict()
    )

@app.exception_handler(ModelLoadingError)
async def model_loading_error_handler(request: Request, exc: ModelLoadingError):
    """Handle model loading errors"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error="ModelLoadingError",
            message="Failed to load ML models",
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path,
            details={"error_details": str(exc)}
        ).dict()
    )

@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc: PredictionError):
    """Handle prediction errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="PredictionError",
            message="Prediction failed",
            timestamp=datetime.utcnow().isoformat(),
            path=request.url.path,
            details={"error_details": str(exc)}
        ).dict()
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Length", "X-Request-ID"],
    max_age=600  # 10 minutes
)

# Security
security = HTTPBearer()

# Redis connection
redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0, decode_responses=True)

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed"
        )

# Pydantic models
class PredictionRequest(BaseModel):
    city: str = Field(..., description="City name")
    region: str = Field(..., description="Region/State")
    post_count: int = Field(..., ge=0, description="Number of social media posts")
    avg_risk_score: float = Field(..., ge=0, le=5, description="Average risk score")
    avg_engagement: float = Field(..., ge=0, le=5, description="Average engagement score")
    unique_users: int = Field(..., ge=0, description="Number of unique users")
    admission_count: int = Field(..., ge=0, description="Number of hospital admissions")
    avg_severity: float = Field(..., ge=0, le=5, description="Average severity score")
    avg_length_of_stay: float = Field(..., ge=0, description="Average length of hospital stay")
    disease_variety: int = Field(..., ge=0, description="Variety of diseases")
    avg_temperature: float = Field(..., description="Average temperature in Celsius")
    avg_humidity: float = Field(..., ge=0, le=100, description="Average humidity percentage")
    avg_rainfall: float = Field(..., ge=0, description="Average rainfall in mm")
    mosquito_risk_level: str = Field(..., description="Mosquito risk level (low/medium/high)")

class PredictionResponse(BaseModel):
    prediction_id: str
    city: str
    region: str
    outbreak_risk_score: float
    risk_level: str
    confidence: float
    timestamp: datetime
    recommendations: List[str]
    model_used: str

class AlertRequest(BaseModel):
    city: str
    region: str
    risk_level: str
    message: str
    threshold: float

class AlertResponse(BaseModel):
    alert_id: str
    city: str
    region: str
    risk_level: str
    message: str
    timestamp: datetime
    status: str

class HealthMetrics(BaseModel):
    city: str
    region: str
    timestamp: datetime
    post_count: int
    admission_count: int
    avg_temperature: float
    avg_humidity: float
    outbreak_risk_score: float
    risk_level: str

class DiseaseOutbreakAPI:
    """Main API class for disease outbreak prediction and monitoring"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.load_models()
        
    def load_models(self):
        """Load trained ML models"""
        try:
            # Load XGBoost model
            import xgboost as xgb
            if os.path.exists("models/xgboost_model.json"):
                self.models['xgboost'] = xgb.XGBRegressor()
                self.models['xgboost'].load_model("models/xgboost_model.json")
            
            # Load preprocessors
            import joblib
            if os.path.exists("models/scaler.pkl"):
                self.scaler = joblib.load("models/scaler.pkl")
            if os.path.exists("models/label_encoder.pkl"):
                self.label_encoder = joblib.load("models/label_encoder.pkl")
                
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def preprocess_input(self, request: PredictionRequest) -> np.ndarray:
        """Preprocess input data for prediction"""
        # Encode mosquito risk level
        mosquito_risk_encoded = 0  # default
        if self.label_encoder:
            try:
                mosquito_risk_encoded = self.label_encoder.transform([request.mosquito_risk_level])[0]
            except:
                # Fallback encoding
                risk_mapping = {'low': 0, 'medium': 1, 'high': 2}
                mosquito_risk_encoded = risk_mapping.get(request.mosquito_risk_level, 0)
        
        # Create feature vector
        features = np.array([
            request.post_count, request.avg_risk_score, request.avg_engagement,
            request.unique_users, request.admission_count, request.avg_severity,
            request.avg_length_of_stay, request.disease_variety, request.avg_temperature,
            request.avg_humidity, request.avg_rainfall, mosquito_risk_encoded
        ]).reshape(1, -1)
        
        # Scale features
        if self.scaler:
            features = self.scaler.transform(features)
        
        return features
    
    def predict_outbreak_risk(self, request: PredictionRequest) -> Dict[str, Any]:
        """Predict disease outbreak risk"""
        try:
            # Preprocess input
            features = self.preprocess_input(request)
            
            # Make prediction
            prediction = 0.0
            model_used = "ensemble"
            
            if 'xgboost' in self.models:
                prediction = self.models['xgboost'].predict(features)[0]
                model_used = "xgboost"
            
            # Calculate risk level
            if prediction > 8:
                risk_level = "critical"
            elif prediction > 6:
                risk_level = "high"
            elif prediction > 4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Generate recommendations
            recommendations = self.generate_recommendations(risk_level, request)
            
            return {
                "outbreak_risk_score": float(prediction),
                "risk_level": risk_level,
                "confidence": 0.85,  # Placeholder confidence score
                "recommendations": recommendations,
                "model_used": model_used
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def generate_recommendations(self, risk_level: str, request: PredictionRequest) -> List[str]:
        """Generate recommendations based on risk level and data"""
        recommendations = []
        
        if risk_level in ["high", "critical"]:
            recommendations.extend([
                "Increase medical staff deployment",
                "Stock up on essential medicines",
                "Implement emergency protocols",
                "Coordinate with neighboring regions"
            ])
        
        if request.avg_temperature > 30 and request.avg_humidity > 70:
            recommendations.append("Implement mosquito control measures")
        
        if request.admission_count > 10:
            recommendations.append("Prepare additional hospital beds")
        
        if request.post_count > 20:
            recommendations.append("Monitor social media for public sentiment")
        
        if not recommendations:
            recommendations.append("Continue monitoring current situation")
        
        return recommendations

# Initialize API
api = DiseaseOutbreakAPI()

@app.get("/")
async def root():
    """Root endpoint"""
    REQUEST_COUNT.labels(endpoint="/", method="GET").inc()
    return {"message": "Disease Outbreak Early Warning System API", "status": "active"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict_outbreak_risk(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Predict disease outbreak risk for a given location"""
    start_time = datetime.now()
    
    try:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
        
        # Make prediction
        prediction_result = api.predict_outbreak_risk(request)
        
        # Create response
        response = PredictionResponse(
            prediction_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.city)}",
            city=request.city,
            region=request.region,
            outbreak_risk_score=prediction_result["outbreak_risk_score"],
            risk_level=prediction_result["risk_level"],
            confidence=prediction_result["confidence"],
            timestamp=datetime.now(),
            recommendations=prediction_result["recommendations"],
            model_used=prediction_result["model_used"]
        )
        
        # Store prediction in Redis for caching
        background_tasks.add_task(
            store_prediction_cache, 
            response.prediction_id, 
            response.dict()
        )
        
        # Update metrics
        PREDICTION_ACCURACY.set(prediction_result["confidence"])
        
        # Log request latency
        latency = (datetime.now() - start_time).total_seconds()
        REQUEST_LATENCY.observe(latency)
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts", response_model=AlertResponse)
async def create_alert(request: AlertRequest):
    """Create a new alert for high-risk situations"""
    REQUEST_COUNT.labels(endpoint="/alerts", method="POST").inc()
    
    try:
        alert = AlertResponse(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(request.city)}",
            city=request.city,
            region=request.region,
            risk_level=request.risk_level,
            message=request.message,
            timestamp=datetime.now(),
            status="active"
        )
        
        # Store alert in Redis
        redis_client.hset(
            f"alert:{alert.alert_id}",
            mapping=alert.dict()
        )
        
        # Update active alerts metric
        active_count = len(redis_client.keys("alert:*"))
        ACTIVE_ALERTS.set(active_count)
        
        # Send notification (background task)
        asyncio.create_task(send_alert_notification(alert))
        
        return alert
        
    except Exception as e:
        logger.error(f"Alert creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts():
    """Get all active alerts"""
    REQUEST_COUNT.labels(endpoint="/alerts", method="GET").inc()
    
    try:
        alerts = []
        alert_keys = redis_client.keys("alert:*")
        
        for key in alert_keys:
            alert_data = redis_client.hgetall(key)
            if alert_data.get("status") == "active":
                alerts.append(AlertResponse(**alert_data))
        
        return alerts
        
    except Exception as e:
        logger.error(f"Get alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return generate_latest()

@app.get("/dashboard/risk-map")
async def get_risk_map():
    """Get current risk map data for dashboard"""
    REQUEST_COUNT.labels(endpoint="/dashboard/risk-map", method="GET").inc()
    
    try:
        # Get recent predictions from cache
        risk_data = []
        prediction_keys = redis_client.keys("prediction:*")
        
        for key in prediction_keys[:100]:  # Limit to recent 100
            pred_data = redis_client.hgetall(key)
            if pred_data:
                risk_data.append({
                    "city": pred_data.get("city"),
                    "region": pred_data.get("region"),
                    "risk_score": float(pred_data.get("outbreak_risk_score", 0)),
                    "risk_level": pred_data.get("risk_level"),
                    "timestamp": pred_data.get("timestamp")
                })
        
        return {"risk_map_data": risk_data}
        
    except Exception as e:
        logger.error(f"Risk map error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/trends")
async def get_trends(city: Optional[str] = None, days: int = 30):
    """Get trend data for dashboard"""
    REQUEST_COUNT.labels(endpoint="/dashboard/trends", method="GET").inc()
    
    try:
        # Generate mock trend data (in real implementation, fetch from database)
        trends = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        current_date = start_date
        while current_date <= end_date:
            trends.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "risk_score": np.random.uniform(2, 8),
                "post_count": np.random.poisson(15),
                "admission_count": np.random.poisson(5),
                "temperature": np.random.uniform(20, 35)
            })
            current_date += timedelta(days=1)
        
        return {"trends": trends}
        
    except Exception as e:
        logger.error(f"Trends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def store_prediction_cache(prediction_id: str, prediction_data: Dict[str, Any]):
    """Store prediction in Redis cache"""
    try:
        redis_client.hset(
            f"prediction:{prediction_id}",
            mapping=prediction_data
        )
        # Set expiration (24 hours)
        redis_client.expire(f"prediction:{prediction_id}", 86400)
    except Exception as e:
        logger.error(f"Cache storage error: {e}")

async def send_alert_notification(alert: AlertResponse):
    """Send alert notification (placeholder for actual notification system)"""
    try:
        logger.info(f"ALERT: {alert.risk_level.upper()} risk in {alert.city}, {alert.region}")
        logger.info(f"Message: {alert.message}")
        
        # In real implementation, send to:
        # - Email/SMS systems
        # - Slack/Discord
        # - Emergency response systems
        
    except Exception as e:
        logger.error(f"Alert notification error: {e}")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
