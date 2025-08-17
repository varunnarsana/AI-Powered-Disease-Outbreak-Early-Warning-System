# AI-Powered Disease Outbreak Early Warning System

A comprehensive MLOps pipeline that collects real-time health signals, processes data, and predicts potential disease outbreak hotspots before they escalate.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Problem Statement

Outbreaks of diseases like dengue, malaria, and influenza often spread silently before authorities detect them. This delays medical response, overwhelms hospitals, and risks thousands of lives. Traditional monitoring relies heavily on hospital data alone, which is often delayed.

## Features

- **Real-time Monitoring**: Track disease indicators across multiple data sources
- **Predictive Analytics**: Machine learning models to forecast outbreak risks
- **Interactive Dashboard**: Visualize disease trends and predictions
- **Alert System**: Get notified about potential outbreaks
- **Data Versioning**: Track changes in data and models over time
- **Scalable Infrastructure**: Deploy on-premises or in the cloud

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Data Pipeline  │    │  ML Pipeline    │
│                 │    │                 │    │                 │
│ • Social Media  │───▶│ • Kafka         │───▶│ • MLflow        │
│ • Hospital Logs │    │ • Spark         │    │ • XGBoost       │
│ • Weather Data  │    │ • DBT           │    │ • LSTM          │
│ • Health APIs   │    │ • Redshift      │    │ • Prophet       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  FastAPI API    │    │  Monitoring     │
                       │                 │    │                 │
                       │ • Predictions   │    │ • Prometheus    │
                       │ • Alerts        │    │ • Grafana       │
                       │ • Dashboard     │    │ • Alerting      │
                       └─────────────────┘    └─────────────────┘
```

## Project Structure

```
disease-mlops-project/
├── data/                      # Data storage and samples
│   ├── raw/                  # Raw data files
│   └── processed/            # Processed data files
├── infrastructure/           # Terraform and Kubernetes configs
├── ml/                       # Machine learning models and pipelines
│   ├── train_models.py       # Model training script
│   └── monitor_model.py      # Model monitoring
├── pipelines/                # Data processing pipelines
├── api/                      # FastAPI service
│   ├── main.py               # API entry point
│   └── models.py             # ML model definitions
├── dashboard/                # Streamlit dashboard
│   └── app.py               # Dashboard application
├── frontend/                 # Modern web interface (HTML/CSS/JS)
├── monitoring/               # Prometheus and Grafana configs
├── docker/                   # Docker configurations
│   └── docker-compose.yml    # Service definitions
├── scripts/                  # Utility scripts
│   └── load_sample_data.py   # Sample data loader
└── docs/                     # Documentation
    └── DATA_VERSIONING.md    # Data versioning strategy
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- PostgreSQL 13+
- Redis 6+

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/disease-mlops-project.git
   cd disease-mlops-project
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install Python dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running with Docker (Recommended)

1. **Start all services**
   ```bash
   docker-compose up -d
   ```

   This will start:
   - API service (port 8000)
   - PostgreSQL database
   - Redis cache
   - MLflow tracking server (port 5000)
   - Prometheus (port 9090)
   - Grafana (port 3000)

2. **Verify services**
   - API Docs: http://localhost:8000/api/docs
   - MLflow UI: http://localhost:5000
   - Grafana: http://localhost:3000 (admin/admin)

3. **Load sample data**
   ```bash
   docker-compose exec api python scripts/load_sample_data.py
   ```

### Running Locally

1. **Start database services**
   ```bash
   docker-compose up -d postgres redis
   ```

2. **Run the API**
   ```bash
   cd api
   uvicorn main:app --reload
   ```

3. **Run the dashboard**
   ```bash
   cd dashboard
   streamlit run app.py
   ```

## Using the System

### Making Predictions

```python
import requests

url = "http://localhost:8000/api/predict"
data = {
    "location": {"city": "Mumbai", "region": "Maharashtra"},
    "disease_name": "Dengue",
    "weather_conditions": {
        "temperature": 30.5,
        "humidity": 75.0,
        "rainfall": 10.2
    },
    "population_density": 20000,
    "historical_cases": [
        {"date": "2023-01-01", "cases": 5},
        {"date": "2023-01-02", "cases": 7}
    ],
    "social_media_mentions": 45
}

response = requests.post(url, json=data)
print(response.json())
```

API Endpoints

- `GET /` - Health check
- `POST /api/predict` - Get outbreak risk prediction
- `GET /api/data/cases` - Get historical case data
- `GET /api/data/weather` - Get weather data
- `GET /api/metrics` - Prometheus metrics

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage report
pytest --cov=api --cov=ml tests/
```

## Monitoring and Logging

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **MLflow**: http://localhost:5000

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Documentation

For detailed documentation, please see the [docs](docs/) directory.

##  Acknowledgments

- Healthcare workers and data scientists working on disease prevention
- Open-source communities for their invaluable tools and libraries
- Docker and Docker Compose
- Python 3.9+
- Terraform
- kubectl (for Kubernetes deployment)

### 1. Start Local Development Environment
```bash
docker-compose up -d
```

### 2. Initialize Infrastructure
```bash
cd infrastructure
terraform init
terraform apply
```

### 3. Start Data Pipeline
```bash
cd pipelines
python start_kafka_producer.py
python start_spark_streaming.py
```

### 4. Train ML Models
```bash
cd ml
python train_models.py
```

### 5. Start API Service
```bash
cd api
uvicorn main:app --reload
```

### 6. Launch Dashboard
```bash
cd dashboard
streamlit run app.py
```

### 7. Launch Modern Web Frontend (Optional)
```bash
cd frontend
python server.py
```

Then open your browser to: **http://localhost:3000**

The frontend provides a professional web interface with:
-  Interactive dashboard with real-time metrics and risk maps
-  AI prediction interface for outbreak risk assessment  
-  Alert management with severity classification
-  Advanced analytics with customizable time ranges
-  Responsive design for all devices

##  Features

- **Real-time Data Ingestion**: Kafka streams for social media, hospital logs, and weather data
- **Data Processing**: Apache Spark for streaming data and DBT for batch transformations
- **ML Pipeline**: XGBoost, LSTM, and Prophet models with MLflow tracking
- **API Service**: FastAPI endpoints for predictions and alerts
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Alerting**: Automated notifications for outbreak risks and system issues
- **Dashboard**: Real-time outbreak risk visualization by region

##  Technologies Used

- **Data Pipeline**: Apache Kafka, Apache Spark, DBT, Amazon Redshift
- **ML**: MLflow, XGBoost, TensorFlow, Prophet
- **API**: FastAPI, Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana, AlertManager
- **Infrastructure**: Terraform, AWS
- **Dashboard**: Streamlit
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js, Leaflet Maps

##  Impact

This system enables:
- Early detection of disease outbreaks
- Proactive resource allocation
- Faster medical response
- Prevention of healthcare system overload
- Data-driven public health decisions

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

##  License

MIT License - see LICENSE file for details
