# Data Versioning Strategy

This document outlines the data versioning strategy for the Disease Outbreak Early Warning System.

## Overview

We follow a hybrid versioning approach that combines:
- **Raw Data Versioning**: Original data as received from sources
- **Processed Data Versioning**: Cleaned and transformed data ready for analysis
- **Database Schema Versioning**: Versioning of database structure and migrations

## Directory Structure

```
data/
├── raw/                   # Original, immutable data snapshots
│   ├── {source_name}/     # Source-specific directories
│   │   ├── {date}/        # Date-based versioning (YYYY-MM-DD)
│   │   │   └── data.{csv,json,parquet}
│   │   └── latest -> {date}/  # Symlink to latest version
│   └── README.md          # Data dictionary for raw data
│
└── processed/            # Processed and cleaned data
    ├── {model_name}/     # Model-specific processing
    │   ├── {version}/    # Versioned processed data
    │   │   └── data.parquet
    │   └── latest -> {version}/
    └── README.md         # Processing documentation
```

## Versioning Scheme

### Raw Data
- **Naming Convention**: `{source}-{YYYYMMDD}-{description}.{ext}`
- **Storage**: Immutable, append-only
- **Metadata**: Each raw data file should include a JSON sidecar file with:
  - Source information
  - Collection date and time
  - Data schema
  - Any known issues or anomalies

### Processed Data
- **Versioning**: Semantic versioning (MAJOR.MINOR.PATCH)
  - **MAJOR**: Breaking changes in schema or processing logic
  - **MINOR**: Backward-compatible additions
  - **PATCH**: Bug fixes and minor improvements
- **Storage**: Stored in columnar format (Parquet) for efficient querying
- **Metadata**: Each processed dataset includes:
  - Raw data versions used
  - Processing parameters and scripts
  - Data quality metrics
  - Schema definition

## Database Schema Versioning

We use Flyway for database schema versioning. All schema changes are stored as SQL migration scripts in:

```
db/migrations/
├── V1__initial_schema.sql
├── V2__add_indexes.sql
└── ...
```

## Sample Data Loading

The `scripts/load_sample_data.py` script populates the database with realistic sample data for development and testing.

### Prerequisites

1. Python 3.8+
2. PostgreSQL database
3. Required Python packages (install with `pip install -r scripts/requirements.txt`)

### Usage

1. Configure your database connection in `.env`:
   ```
   POSTGRES_DB=disease_outbreak
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   ```

2. Run the script:
   ```bash
   python scripts/load_sample_data.py
   ```

### Generated Data

The script generates:
- Location data for 8 major Indian cities
- 90 days of weather data
- Disease case reports for 10 different diseases
- Social media mentions with sentiment scores

## Data Retention Policy

- **Raw Data**: Retained for 1 year
- **Processed Data**: Retained for 6 months
- **Backups**: Nightly database backups retained for 30 days

## Data Quality

We implement the following data quality checks:

1. **Schema Validation**: All data must conform to defined schemas
2. **Completeness Checks**: Required fields must be present
3. **Validity Checks**: Data must fall within expected ranges
4. **Timeliness**: Data freshness is monitored
5. **Consistency**: Cross-validated against other data sources

## Access Control

- Raw data access is restricted to data engineers
- Processed data is available to data scientists and analysts
- Production database access is restricted to authorized personnel only

## Data Lineage

We track data lineage using MLflow to maintain a complete audit trail of:
- Data sources and versions used
- Processing steps applied
- Models trained on the data
- Predictions made using the data

## Backup and Recovery

- **Daily Backups**: Full database dump at 2 AM UTC
- **Point-in-Time Recovery**: WAL archiving enabled
- **Disaster Recovery**: Cross-region replication for critical data

## Compliance

- All data handling follows GDPR and other relevant regulations
- PII is hashed or removed from processed data
- Data retention and deletion policies are enforced

## Monitoring and Alerting

- Data quality metrics are monitored in real-time
- Alerts are triggered for:
  - Data ingestion failures
  - Schema validation errors
  - Unusual data patterns
  - Processing delays

## Documentation

- Data dictionary is maintained in `docs/data_dictionary.md`
- API documentation includes data schemas
- All data processing steps are documented in Jupyter notebooks or Python scripts
