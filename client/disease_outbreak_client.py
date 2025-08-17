"""
Python client for the Disease Outbreak Early Warning System API.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field, HttpUrl, validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class APIConfig(BaseModel):
    """Configuration for the API client."""
    
    api_key: str = Field(..., description="API key for authentication")
    base_url: HttpUrl = Field(
        default="https://api.disease-outbreak.example.com/v1",
        description="Base URL of the API"
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FeatureVector(BaseModel):
    """Feature vector for prediction."""
    
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Relative humidity percentage")
    rainfall: float = Field(..., description="Rainfall in mm")
    population_density: float = Field(..., description="Population density (people per km²)")
    avg_historical_cases: float = Field(..., description="Average historical cases")
    case_trend: float = Field(..., description="Trend in cases (e.g., 1.2 for 20% increase)")
    social_media_mentions: int = Field(..., description="Number of relevant social media mentions")


class PredictionResult(BaseModel):
    """Prediction result."""
    
    prediction: float = Field(..., description="Predicted probability of outbreak (0-1)")
    risk_level: str = Field(..., description="Risk level category (low/medium/high)")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class BatchPredictionResult(BaseModel):
    """Batch prediction result."""
    
    predictions: List[Dict[str, Any]] = Field(..., description="List of prediction results")
    model_version: str = Field(..., description="Model version used for predictions")
    timestamp: datetime = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Model information."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    stage: str = Field(..., description="Deployment stage (staging/production)")
    description: str = Field(..., description="Model description")
    created_at: datetime = Field(..., description="When the model was created")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    features: List[str] = Field(..., description="List of feature names")


class TrainingJob(BaseModel):
    """Training job information."""
    
    training_id: str = Field(..., description="Unique training job ID")
    status: str = Field(..., description="Job status (started/running/completed/failed)")
    message: str = Field(..., description="Status message")
    start_time: datetime = Field(..., description="When the job started")
    end_time: Optional[datetime] = Field(None, description="When the job completed")
    monitor_url: Optional[HttpUrl] = Field(None, description="URL to monitor training progress")


class DiseaseOutbreakClient:
    """Client for interacting with the Disease Outbreak Early Warning System API."""
    
    def __init__(self, api_key: str, **kwargs):
        """Initialize the client.
        
        Args:
            api_key: API key for authentication
            **kwargs: Additional configuration (base_url, timeout, etc.)
        """
        self.config = APIConfig(api_key=api_key, **kwargs)
        self._client = httpx.AsyncClient(
            base_url=str(self.config.base_url),
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "DiseaseOutbreakPythonClient/1.0.0"
            }
        )
        logger.info(f"Initialized client for {self.config.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
        logger.info("Client closed")
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request with retries."""
        url = endpoint.lstrip("/")
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.config.max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_retries} failed: {e}. Retrying..."
                    )
                    continue
                logger.error(f"HTTP error: {e}")
                try:
                    error_detail = e.response.json()
                except json.JSONDecodeError:
                    error_detail = e.response.text
                raise Exception(f"API request failed: {error_detail}") from e
            except httpx.RequestError as e:
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"Request failed: {e}. Retrying..."
                    )
                    continue
                logger.error(f"Request error: {e}")
                raise Exception(f"Request failed: {str(e)}") from e
    
    async def health(self) -> Dict[str, Any]:
        """Check API health.
        
        Returns:
            Health status information
        """
        return await self._request("GET", "/health")
    
    async def predict(
        self,
        features: Union[Dict[str, Any], FeatureVector],
        model_version: str = "latest"
    ) -> PredictionResult:
        """Get a prediction for a single input.
        
        Args:
            features: Input features for prediction
            model_version: Model version to use (default: latest)
            
        Returns:
            Prediction result
        """
        if isinstance(features, FeatureVector):
            features = features.dict()
            
        payload = {
            "features": features,
            "model_version": model_version
        }
        
        response = await self._request("POST", "/predict", json=payload)
        return PredictionResult(**response)
    
    async def predict_batch(
        self,
        instances: List[Union[Dict[str, Any], FeatureVector]],
        model_version: str = "latest"
    ) -> BatchPredictionResult:
        """Get predictions for multiple inputs.
        
        Args:
            instances: List of input features
            model_version: Model version to use (default: latest)
            
        Returns:
            Batch prediction results
        """
        # Convert FeatureVector objects to dicts if needed
        instances_data = [
            instance.dict() if isinstance(instance, FeatureVector) else instance
            for instance in instances
        ]
        
        payload = {
            "instances": instances_data,
            "model_version": model_version
        }
        
        response = await self._request("POST", "/predict/batch", json=payload)
        return BatchPredictionResult(**response)
    
    async def get_model_info(self, version: str = "latest") -> ModelInfo:
        """Get information about a model.
        
        Args:
            version: Model version (default: latest)
            
        Returns:
            Model information
        """
        params = {"version": version} if version != "latest" else {}
        response = await self._request("GET", "/model", params=params)
        return ModelInfo(**response)
    
    async def get_model_metrics(
        self,
        version: str = "latest",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get metrics for a specific model version.
        
        Args:
            version: Model version (default: latest)
            start_date: Start date for metrics
            end_date: End date for metrics
            
        Returns:
            Model metrics
        """
        params = {"version": version}
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
            
        return await self._request("GET", "/model/metrics", params=params)
    
    async def get_data_statistics(
        self,
        feature: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get statistics about the training data.
        
        Args:
            feature: Specific feature to get statistics for
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Data statistics
        """
        params = {}
        
        if feature:
            params["feature"] = feature
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
            
        return await self._request("GET", "/data/statistics", params=params)
    
    async def trigger_retraining(
        self,
        training_data_path: str,
        validation_data_path: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        notify_email: Optional[str] = None
    ) -> TrainingJob:
        """Trigger a model retraining job.
        
        Args:
            training_data_path: Path to training data
            validation_data_path: Path to validation data
            hyperparameters: Optional hyperparameters for training
            notify_email: Email to notify when training completes
            
        Returns:
            Training job information
        """
        payload = {
            "training_data_path": training_data_path,
            "validation_data_path": validation_data_path
        }
        
        if hyperparameters:
            payload["hyperparameters"] = hyperparameters
        if notify_email:
            payload["notify_email"] = notify_email
            
        response = await self._request("POST", "/model/retrain", json=payload)
        return TrainingJob(**response)


# Example usage
async def example_usage():
    """Example usage of the DiseaseOutbreakClient."""
    # Initialize client
    async with DiseaseOutbreakClient(
        api_key="your_api_key_here",
        base_url="http://localhost:8000/v1"  # Update with your API URL
    ) as client:
        # Check API health
        health = await client.health()
        print(f"API Health: {health['status']}")
        
        # Get model info
        model_info = await client.get_model_info()
        print(f"Model: {model_info.name} v{model_info.version}")
        
        # Make a prediction
        features = {
            "temperature": 28.5,
            "humidity": 75.2,
            "rainfall": 15.0,
            "population_density": 1200.5,
            "avg_historical_cases": 25.3,
            "case_trend": 1.2,
            "social_media_mentions": 45
        }
        
        prediction = await client.predict(features)
        print(f"Prediction: {prediction.risk_level} ({prediction.prediction:.2f})")
        
        # Get data statistics
        stats = await client.get_data_statistics()
        print(f"Data statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
