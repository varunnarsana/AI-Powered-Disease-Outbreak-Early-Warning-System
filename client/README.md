# Disease Outbreak API Client

Python client for interacting with the Disease Outbreak Early Warning System API.

## Features

- Async/await support for high performance
- Type hints and data validation with Pydantic
- Retry mechanism for failed requests
- Comprehensive error handling
- Full API coverage

## Installation

Install using pip:

```bash
pip install git+https://github.com/your-org/disease-outbreak-mlops.git#subdirectory=client
```

Or install from source:

```bash
git clone https://github.com/your-org/disease-outbreak-mlops.git
cd disease-outbreak-mlops/client
pip install .
```

## Usage

### Basic Example

```python
import asyncio
from disease_outbreak_client import DiseaseOutbreakClient, FeatureVector

async def main():
    # Initialize client
    async with DiseaseOutbreakClient(
        api_key="your_api_key_here",
        base_url="https://api.disease-outbreak.example.com/v1"
    ) as client:
        
        # Check API health
        health = await client.health()
        print(f"API Health: {health['status']}")
        
        # Create a feature vector
        features = FeatureVector(
            temperature=28.5,
            humidity=75.2,
            rainfall=15.0,
            population_density=1200.5,
            avg_historical_cases=25.3,
            case_trend=1.2,
            social_media_mentions=45
        )
        
        # Get prediction
        prediction = await client.predict(features)
        print(f"Outbreak risk: {prediction.risk_level} ({prediction.prediction:.2f})")

# Run the async function
asyncio.run(main())
```

### Batch Prediction

```python
# Create multiple feature vectors
features_list = [
    FeatureVector(temperature=28.5, humidity=75.2, rainfall=15.0, ...),
    FeatureVector(temperature=22.1, humidity=65.8, rainfall=5.0, ...),
]

# Get batch predictions
results = await client.predict_batch(features_list)
for i, pred in enumerate(results.predictions):
    print(f"Prediction {i+1}: {pred['risk_level']} ({pred['prediction']:.2f})")
```

### Get Model Information

```python
# Get information about the current model
model_info = await client.get_model_info()
print(f"Model: {model_info.name} v{model_info.version}")
print(f"Stage: {model_info.stage}")
print(f"Metrics: {model_info.metrics}")
```

### Trigger Model Retraining

```python
# Trigger a new training job
training_job = await client.trigger_retraining(
    training_data_path="s3://disease-outbreak-data/training/v2/train.parquet",
    validation_data_path="s3://disease-outbreak-data/validation/v2/validation.parquet",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5
    },
    notify_email="your.email@example.com"
)

print(f"Training job started: {training_job.training_id}")
print(f"Monitor at: {training_job.monitor_url}")
```

## Configuration

The client can be configured with the following environment variables:

- `DISEASE_OUTBREAK_API_KEY`: Your API key
- `DISEASE_OUTBREAK_BASE_URL`: Base URL of the API (default: `https://api.disease-outbreak.example.com/v1`)
- `DISEASE_OUTBREAK_TIMEOUT`: Request timeout in seconds (default: 30)
- `DISEASE_OUTBREAK_MAX_RETRIES`: Maximum number of retries for failed requests (default: 3)

Or pass these values directly to the client constructor:

```python
client = DiseaseOutbreakClient(
    api_key="your_api_key",
    base_url="https://api.disease-outbreak.example.com/v1",
    timeout=30,
    max_retries=3
)
```

## Error Handling

The client raises exceptions for HTTP errors and failed requests:

```python
try:
    await client.predict(invalid_features)
except Exception as e:
    print(f"Error: {str(e)}")
```

## Development

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/disease-outbreak-mlops.git
   cd disease-outbreak-mlops/client
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest
   ```

4. Format code:
   ```bash
   black .
   isort .
   ```

## License

MIT

## Support

For support, please open an issue in the [GitHub repository](https://github.com/your-org/disease-outbreak-mlops/issues).
