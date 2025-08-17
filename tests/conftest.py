"""
Pytest configuration and fixtures for testing the Disease Outbreak Early Warning System.
"""
import os
import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Load test environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env.test")

# Import the FastAPI app after environment is loaded
from api.main import app, get_db
from api.database import Base, get_db as get_db_override

# Test database configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/test_disease_outbreak"
)

# Create test database engine
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create test database tables
@pytest.fixture(scope="session", autouse=True)
def create_test_database():
    """Create test database tables at the start of the test session."""
    Base.metadata.create_all(bind=engine)
    yield
    # Clean up after tests
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session():
    ""
    Create a new database session with a rollback at the end of the test.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    
    # Begin a nested transaction (using SAVEPOINT).
    nested = connection.begin_nested()
    
    # If the application code calls session.commit, it will end the nested
    # transaction. We need to start a new one when that happens.
    @event.listens_for(session, 'after_transaction_end')
    def end_savepoint(session, transaction):
        nonlocal nested
        if not nested.is_active:
            nested = connection.begin_nested()
    
    yield session
    
    # Cleanup
    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture(scope="function")
def client(db_session):
    ""
    Create a test client that uses the override_get_db fixture to return a test database session.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def sample_prediction_data():
    """Sample prediction request data for testing."""
    return {
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

@pytest.fixture(scope="function")
def sample_weather_data():
    """Sample weather data for testing."""
    return {
        "temperature": 28.5,
        "humidity": 65.0,
        "rainfall": 5.2,
        "report_date": "2023-06-15"
    }
