"""
Tests for the Disease Outbreak Early Warning System API endpoints.
"""
import pytest
from fastapi import status
from datetime import date, timedelta

# Test the root endpoint
def test_read_root(client):
    """Test the root endpoint returns a welcome message."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert "Disease Outbreak Early Warning System" in response.json()["message"]

# Test prediction endpoint
class TestPredictionEndpoints:
    """Test cases for prediction-related endpoints."""
    
    def test_predict_outbreak_risk_valid(self, client, sample_prediction_data):
        ""Test prediction with valid data.""
        response = client.post("/api/predict", json=sample_prediction_data)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "risk_level" in data
        assert "probability" in data
        assert "recommendations" in data
        assert isinstance(data["probability"], float)
        assert 0 <= data["probability"] <= 1
    
    def test_predict_outbreak_risk_invalid_data(self, client):
        ""Test prediction with invalid data.""
        invalid_data = {"invalid": "data"}
        response = client.post("/api/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_outbreak_risk_missing_required_field(self, client, sample_prediction_data):
        ""Test prediction with missing required field."""
        del sample_prediction_data["disease_name"]
        response = client.post("/api/predict", json=sample_prediction_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

# Test data retrieval endpoints
class TestDataEndpoints:
    """Test cases for data retrieval endpoints."""
    
    def test_get_disease_cases(self, client, db_session):
        ""Test retrieving disease cases."""
        # Add test data
        db_session.execute(
            """
            INSERT INTO locations (city, region, population)
            VALUES ('TestCity', 'TestRegion', 100000)
            RETURNING location_id
            """
        )
        location_id = db_session.scalar("SELECT location_id FROM locations LIMIT 1")
        
        db_session.execute(
            """
            INSERT INTO disease_cases 
            (location_id, disease_name, case_count, report_date, severity_index)
            VALUES (:loc_id, 'Dengue', 10, CURRENT_DATE, 0.5)
            """,
            {"loc_id": location_id}
        )
        db_session.commit()
        
        # Test with location filter
        response = client.get(f"/api/data/cases?location_id={location_id}")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        if data:  # If there are results
            assert "case_id" in data[0]
            assert "disease_name" in data[0]
    
    def test_get_weather_data(self, client, db_session, sample_weather_data):
        ""Test retrieving weather data."""
        # Add test location
        db_session.execute(
            """
            INSERT INTO locations (city, region, population)
            VALUES ('WeatherCity', 'TestRegion', 50000)
            RETURNING location_id
            """
        )
        location_id = db_session.scalar("SELECT location_id FROM locations WHERE city = 'WeatherCity'")
        
        # Add test weather data
        db_session.execute(
            """
            INSERT INTO weather_data 
            (location_id, temperature, humidity, rainfall, report_date)
            VALUES (:loc_id, :temp, :humidity, :rainfall, :report_date)
            """,
            {
                "loc_id": location_id,
                "temp": sample_weather_data["temperature"],
                "humidity": sample_weather_data["humidity"],
                "rainfall": sample_weather_data["rainfall"],
                "report_date": sample_weather_data["report_date"]
            }
        )
        db_session.commit()
        
        # Test with date range
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=7)).isoformat()
        
        response = client.get(
            f"/api/data/weather?location_id={location_id}"
            f"&start_date={start_date}&end_date={end_date}"
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        if data:  # If there are results
            assert "temperature" in data[0]
            assert "humidity" in data[0]
            assert "rainfall" in data[0]

# Test authentication
class TestAuthentication:
    """Test cases for authentication."""
    
    def test_protected_endpoint_without_auth(self, client):
        ""Test accessing a protected endpoint without authentication."""
        response = client.get("/api/admin/status")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_with_invalid_auth(self, client):
        ""Test accessing a protected endpoint with invalid credentials."""
        response = client.get(
            "/api/admin/status",
            headers={"Authorization": "Bearer invalidtoken"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

# Test error handling
class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_nonexistent_endpoint(self, client):
        ""Test accessing a non-existent endpoint."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Not Found" in response.json()["detail"]
    
    def test_internal_server_error_handling(self, client, mocker):
        ""Test handling of internal server errors."""
        # Mock a function to raise an exception
        def mock_fail():
            raise ValueError("Something went wrong")
            
        # Apply the mock to the app
        original_route = app.router.routes[0].endpoint
        app.router.routes[0].endpoint = mock_fail
        
        try:
            response = client.get("/")
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            assert "internal server error" in response.json()["message"].lower()
        finally:
            # Restore the original route
            app.router.routes[0].endpoint = original_route
