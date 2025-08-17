"""
Tests for the ML model components of the Disease Outbreak Early Warning System.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Test model loading and prediction
class TestDiseaseOutbreakModel:
    """Test cases for the DiseaseOutbreakModel class."""
    
    @pytest.fixture(autouse=True)
    def setup_model(self):
        ""Set up the model with mock components for testing."""
        with patch('api.models.load') as mock_load, \
             patch('api.models.pickle.load') as mock_pickle_load:
            
            # Mock the model and scaler
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% risk
            mock_scaler = MagicMock()
            
            # Mock the file loading
            mock_pickle_load.side_effect = [mock_model, mock_scaler]
            
            # Import after patching to ensure mocks are in place
            from api.models import DiseaseOutbreakModel
            
            self.model = DiseaseOutbreakModel()
            self.model.model = mock_model
            self.model.scaler = mock_scaler
            
            yield
    
    def test_predict_risk_level(self):
        ""Test risk level prediction based on probability thresholds."""
        assert self.model._get_risk_level(0.1) == "low"
        assert self.model._get_risk_level(0.3) == "medium"
        assert self.model._get_risk_level(0.6) == "high"
        assert self.model._get_risk_level(0.9) == "critical"
    
    def test_preprocess_features(self):
        ""Test feature preprocessing."""
        sample_data = {
            'temperature': 30.5,
            'humidity': 75.0,
            'rainfall': 10.2,
            'population_density': 20000,
            'historical_cases': [5, 7, 10],
            'social_media_mentions': 45
        }
        
        processed = self.model._preprocess_features(sample_data)
        
        assert isinstance(processed, dict)
        assert 'avg_historical_cases' in processed
        assert 'case_trend' in processed
        assert processed['temperature'] == 30.5
    
    @patch('api.models.DiseaseOutbreakModel._preprocess_features')
    def test_predict(self, mock_preprocess):
        ""Test the predict method with mock data."""
        # Mock preprocessing
        mock_preprocess.return_value = {
            'temperature': 0.8,
            'humidity': 0.7,
            'rainfall': 0.6,
            'population_density': 0.9,
            'avg_historical_cases': 0.4,
            'case_trend': 0.5,
            'social_media_mentions': 0.7
        }
        
        # Mock model prediction
        self.model.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # Test prediction
        result = self.model.predict({
            'temperature': 30.5,
            'humidity': 75.0,
            'rainfall': 10.2,
            'population_density': 20000,
            'historical_cases': [5, 7, 10],
            'social_media_mentions': 45
        })
        
        assert 'risk_level' in result
        assert 'probability' in result
        assert result['risk_level'] == 'high'
        assert 0 <= result['probability'] <= 1
        assert 'features' in result

# Test model training pipeline
class TestModelTraining:
    """Test cases for the model training pipeline."""
    
    @patch('ml.train_models.pd.read_csv')
    @patch('ml.train_models.train_test_split')
    @patch('ml.train_models.RandomForestClassifier')
    @patch('ml.train_models.dump')
    @patch('ml.train_models.log_metric')
    def test_train_model(self, mock_log_metric, mock_dump, mock_rf, 
                        mock_split, mock_read_csv):
        ""Test the model training pipeline with mock data."""
        # Mock data loading
        mock_df = pd.DataFrame({
            'temperature': [25, 30, 35, 20],
            'humidity': [60, 70, 80, 50],
            'rainfall': [5, 10, 15, 2],
            'population_density': [10000, 20000, 15000, 5000],
            'avg_historical_cases': [2, 5, 8, 1],
            'case_trend': [0.1, 0.3, 0.5, -0.2],
            'social_media_mentions': [10, 30, 50, 5],
            'outbreak_risk': [0, 1, 1, 0]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock train-test split
        mock_split.return_value = (
            mock_df.drop('outbreak_risk', axis=1).iloc[:3],  # X_train
            mock_df.drop('outbreak_risk', axis=1).iloc[3:],  # X_test
            mock_df['outbreak_risk'].iloc[:3],  # y_train
            mock_df['outbreak_risk'].iloc[3:]   # y_test
        )
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0, 1, 1]
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
        mock_rf.return_value = mock_model
        
        # Import and run training
        from ml.train_models import train_model
        model_path = 'test_model.pkl'
        metrics = train_model('dummy_data.csv', model_path)
        
        # Verify metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        # Verify model was saved
        mock_dump.assert_called_once()
        
        # Verify metrics were logged
        assert mock_log_metric.call_count >= 5  # At least 5 metrics should be logged

# Test model validation
class TestModelValidation:
    """Test cases for model validation."""
    
    def test_calculate_metrics(self):
        ""Test calculation of evaluation metrics."""
        from ml.train_models import calculate_metrics
        
        y_true = [0, 1, 1, 0, 1, 0, 0, 1]
        y_pred = [0, 1, 0, 0, 1, 1, 0, 1]
        y_proba = [0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.95]
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1

# Test model monitoring
class TestModelMonitoring:
    """Test cases for model monitoring."""
    
    @patch('ml.monitor_model.DataQualityMonitor')
    @patch('ml.monitor_model.ModelPerformanceMonitor')
    def test_monitor_model(self, mock_perf_monitor, mock_data_monitor):
        ""Test the model monitoring setup."""
        from ml.monitor_model import monitor_model
        
        # Mock model and data
        mock_model = MagicMock()
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_test = [0, 1, 0]
        
        # Run monitoring
        monitor = monitor_model(mock_model, X_test, y_test)
        
        # Verify monitors were created
        mock_data_monitor.assert_called_once()
        mock_perf_monitor.assert_called_once()
        
        # Verify monitor has required methods
        assert hasattr(monitor, 'check_data_quality')
        assert hasattr(monitor, 'check_model_performance')
        assert hasattr(monitor, 'generate_report')
