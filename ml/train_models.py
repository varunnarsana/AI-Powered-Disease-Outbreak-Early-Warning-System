#!/usr/bin/env python3
"""
Machine Learning Pipeline for Disease Outbreak Early Warning System
Trains multiple models using MLflow for experiment tracking
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.prophet
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiseaseOutbreakMLPipeline:
    """Machine learning pipeline for disease outbreak prediction"""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Model parameters
        self.xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        self.lstm_params = {
            'units': 64,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2
        }
        
        self.prophet_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative'
        }
        
        # Feature columns
        self.feature_columns = [
            'post_count', 'avg_risk_score', 'avg_engagement', 'unique_users',
            'admission_count', 'avg_severity', 'avg_length_of_stay', 'disease_variety',
            'avg_temperature', 'avg_humidity', 'avg_rainfall', 'mosquito_risk_level'
        ]
        
        # Target column
        self.target_column = 'outbreak_risk_score'

    def load_simulated_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """Load or generate simulated training data"""
        logger.info(f"Generating {num_samples} simulated training samples...")
        
        np.random.seed(42)
        
        # Generate synthetic data
        data = []
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
        regions = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal', 'Telangana', 'Maharashtra', 'Gujarat']
        
        for i in range(num_samples):
            city_idx = i % len(cities)
            date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            
            # Generate realistic features
            post_count = np.random.poisson(15)  # Social media posts
            avg_risk_score = np.random.uniform(1, 3)
            avg_engagement = np.random.uniform(1, 3)
            unique_users = np.random.poisson(8)
            
            admission_count = np.random.poisson(5)  # Hospital admissions
            avg_severity = np.random.uniform(1, 4)
            avg_length_of_stay = np.random.uniform(1, 21)
            disease_variety = np.random.randint(1, 8)
            
            # Weather features with seasonal patterns
            month = date.month
            if month in [6, 7, 8, 9]:  # Monsoon
                avg_temperature = np.random.uniform(25, 35)
                avg_humidity = np.random.uniform(70, 95)
                avg_rainfall = np.random.uniform(10, 100)
                mosquito_risk = np.random.choice(['low', 'medium', 'high'], p=[0.1, 0.3, 0.6])
            elif month in [12, 1, 2]:  # Winter
                avg_temperature = np.random.uniform(15, 25)
                avg_humidity = np.random.uniform(40, 70)
                avg_rainfall = np.random.uniform(0, 20)
                mosquito_risk = np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.2, 0.1])
            else:  # Summer/Spring
                avg_temperature = np.random.uniform(30, 40)
                avg_humidity = np.random.uniform(50, 80)
                avg_rainfall = np.random.uniform(0, 50)
                mosquito_risk = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.4, 0.3])
            
            # Calculate outbreak risk score
            mosquito_multiplier = {'low': 1.0, 'medium': 1.2, 'high': 1.5}[mosquito_risk]
            outbreak_risk_score = (
                (post_count * 0.3 + 
                 admission_count * 0.4 + 
                 avg_severity * 0.3) * mosquito_multiplier
            )
            
            # Add some noise
            outbreak_risk_score += np.random.normal(0, 0.5)
            outbreak_risk_score = max(0, outbreak_risk_score)
            
            data.append({
                'timestamp': date,
                'city': cities[city_idx],
                'region': regions[city_idx],
                'post_count': post_count,
                'avg_risk_score': avg_risk_score,
                'avg_engagement': avg_engagement,
                'unique_users': unique_users,
                'admission_count': admission_count,
                'avg_severity': avg_severity,
                'avg_length_of_stay': avg_length_of_stay,
                'disease_variety': disease_variety,
                'avg_temperature': avg_temperature,
                'avg_humidity': avg_humidity,
                'avg_rainfall': avg_rainfall,
                'mosquito_risk_level': mosquito_risk,
                'outbreak_risk_score': outbreak_risk_score
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated dataset with shape: {df.shape}")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
        """Preprocess data for training"""
        logger.info("Preprocessing data...")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Encode categorical variables
        le = LabelEncoder()
        df['mosquito_risk_encoded'] = le.fit_transform(df['mosquito_risk_level'])
        
        # Select features and target
        feature_cols = self.feature_columns + ['mosquito_risk_encoded']
        X = df[feature_cols].values
        y = df[self.target_column].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Preprocessed data shape: X={X_scaled.shape}, y={y.shape}")
        return X_scaled, y, scaler, le

    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost model with MLflow tracking"""
        logger.info("Training XGBoost model...")
        
        with mlflow.start_run(run_name="xgboost_outbreak_prediction"):
            # Log parameters
            mlflow.log_params(self.xgb_params)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "xgboost_model")
            
            logger.info(f"XGBoost - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            return model

    def train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> keras.Model:
        """Train LSTM model with MLflow tracking"""
        logger.info("Training LSTM model...")
        
        with mlflow.start_run(run_name="lstm_outbreak_prediction"):
            # Log parameters
            mlflow.log_params(self.lstm_params)
            
            # Reshape data for LSTM (samples, timesteps, features)
            # For this example, we'll use a simple approach with 1 timestep
            X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_reshaped, y, test_size=0.2, random_state=42
            )
            
            # Build LSTM model
            model = keras.Sequential([
                keras.layers.LSTM(
                    self.lstm_params['units'],
                    return_sequences=True,
                    input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])
                ),
                keras.layers.Dropout(self.lstm_params['dropout']),
                keras.layers.LSTM(self.lstm_params['units']),
                keras.layers.Dropout(self.lstm_params['dropout']),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1)
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=self.lstm_params['epochs'],
                batch_size=self.lstm_params['batch_size'],
                validation_split=self.lstm_params['validation_split'],
                verbose=1
            )
            
            # Make predictions
            y_pred = model.predict(X_test).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.pytorch.log_model(model, "lstm_model")
            
            logger.info(f"LSTM - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            return model

    def train_prophet_model(self, df: pd.DataFrame) -> Prophet:
        """Train Prophet model for time series forecasting"""
        logger.info("Training Prophet model...")
        
        with mlflow.start_run(run_name="prophet_outbreak_prediction"):
            # Log parameters
            mlflow.log_params(self.prophet_params)
            
            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_df = df.groupby('timestamp')[self.target_column].mean().reset_index()
            prophet_df.columns = ['ds', 'y']
            
            # Add city as additional regressor
            city_df = df.groupby('timestamp')['city'].first().reset_index()
            prophet_df = prophet_df.merge(city_df, on='timestamp')
            
            # Train Prophet model
            model = Prophet(**self.prophet_params)
            
            # Add city as additional regressor
            model.add_country_holidays(country_name='IN')
            
            # Fit model
            model.fit(prophet_df)
            
            # Make predictions for evaluation
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Calculate metrics on historical data
            y_true = prophet_df['y'].values
            y_pred = forecast['yhat'][:len(y_true)].values
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Log model
            mlflow.prophet.log_model(model, "prophet_model")
            
            logger.info(f"Prophet - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            return model

    def create_ensemble_model(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Create ensemble predictions from multiple models"""
        logger.info("Creating ensemble predictions...")
        
        predictions = {}
        
        # XGBoost predictions
        if 'xgboost' in models:
            predictions['xgboost'] = models['xgboost'].predict(X)
        
        # LSTM predictions
        if 'lstm' in models:
            X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
            predictions['lstm'] = models['lstm'].predict(X_reshaped).flatten()
        
        # Calculate ensemble (simple average)
        if len(predictions) > 1:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
            # Calculate ensemble metrics
            mse = mean_squared_error(y, ensemble_pred)
            mae = mean_absolute_error(y, ensemble_pred)
            r2 = r2_score(y, ensemble_pred)
            
            logger.info(f"Ensemble - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'predictions': ensemble_pred
            }
        
        return {}

    def save_models(self, models: Dict[str, Any], scaler: StandardScaler, 
                   le: LabelEncoder, output_dir: str = "models"):
        """Save trained models and preprocessors"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save XGBoost model
        if 'xgboost' in models:
            models['xgboost'].save_model(f"{output_dir}/xgboost_model.json")
        
        # Save LSTM model
        if 'lstm' in models:
            models['lstm'].save(f"{output_dir}/lstm_model")
        
        # Save Prophet model
        if 'prophet' in models:
            with open(f"{output_dir}/prophet_model.json", 'w') as f:
                json.dump(self.prophet_params, f)
        
        # Save preprocessors
        import joblib
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(le, f"{output_dir}/label_encoder.pkl")
        
        logger.info(f"Models saved to {output_dir}")

    def run_training_pipeline(self):
        """Run the complete ML training pipeline"""
        logger.info("Starting ML training pipeline...")
        
        try:
            # Load data
            df = self.load_simulated_data(num_samples=10000)
            
            # Preprocess data
            X, y, scaler, le = self.preprocess_data(df)
            
            # Train models
            models = {}
            
            # Train XGBoost
            models['xgboost'] = self.train_xgboost_model(X, y)
            
            # Train LSTM
            models['lstm'] = self.train_lstm_model(X, y)
            
            # Train Prophet
            models['prophet'] = self.train_prophet_model(df)
            
            # Create ensemble
            ensemble_results = self.create_ensemble_model(models, X, y)
            
            # Save models
            self.save_models(models, scaler, le)
            
            logger.info("ML training pipeline completed successfully!")
            
            return models, ensemble_results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise

def main():
    """Main function to run the ML training pipeline"""
    pipeline = DiseaseOutbreakMLPipeline()
    
    try:
        models, ensemble_results = pipeline.run_training_pipeline()
        logger.info("Training completed successfully!")
        
        # Print final results
        if ensemble_results:
            logger.info(f"Final Ensemble Results:")
            logger.info(f"MSE: {ensemble_results['mse']:.4f}")
            logger.info(f"MAE: {ensemble_results['mae']:.4f}")
            logger.info(f"R2: {ensemble_results['r2']:.4f}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
