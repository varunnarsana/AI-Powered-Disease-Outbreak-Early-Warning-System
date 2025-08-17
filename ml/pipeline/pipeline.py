"""
Model versioning and validation pipeline for the Disease Outbreak Early Warning System.
"""
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import PipelineConfig, load_config

logger = logging.getLogger(__name__)


class ModelVersioningPipeline:
    """Pipeline for model versioning and validation."""

    def __init__(self, config_path: Union[str, Path]):
        """Initialize the pipeline with configuration.

        Args:
            config_path: Path to the YAML configuration file.
        ""
        self.config = load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
        self.model: Optional[BaseEstimator] = None
        self.metrics: Dict[str, float] = {}

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=self.config.log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking."""
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        if self.config.mlflow.registry_uri:
            mlflow.set_registry_uri(self.config.mlflow.registry_uri)
        
        # Create experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(self.config.mlflow.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                self.config.mlflow.experiment_name,
                artifact_location=self.config.mlflow.artifact_location,
            )
        mlflow.set_experiment(self.config.mlflow.experiment_name)

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and prepare training and test data.

        Returns:
            Tuple containing X_train, y_train, X_test, y_test
        """
        logger.info("Loading training and test data")
        
        # Load training data
        train_df = pd.read_parquet(self.config.data.train_path)
        X_train = train_df[self.config.data.feature_columns]
        y_train = train_df[self.config.data.target_column]
        
        # Load test data
        test_df = pd.read_parquet(self.config.data.test_path)
        X_test = test_df[self.config.data.feature_columns]
        y_test = test_df[self.config.data.target_column]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> BaseEstimator:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Training model")
        
        model = RandomForestClassifier(
            n_estimators=self.config.model.hyperparameters["n_estimators"],
            max_depth=self.config.model.hyperparameters["max_depth"],
            min_samples_split=self.config.model.hyperparameters["min_samples_split"],
            min_samples_leaf=self.config.model.hyperparameters["min_samples_leaf"],
            random_state=self.config.model.hyperparameters["random_state"],
            n_jobs=-1,
        )
        
        model.fit(X_train, y_train)
        self.model = model
        return model

    def evaluate_model(
        self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, prefix: str = ""
    ) -> Dict[str, float]:
        """Evaluate the model and return metrics.

        Args:
            model: Trained model
            X: Features
            y: True labels
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating model on {prefix} data")
        
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            f"{prefix}_accuracy": accuracy_score(y, y_pred),
            f"{prefix}_precision": precision_score(y, y_pred, average="weighted"),
            f"{prefix}_recall": recall_score(y, y_pred, average="weighted"),
            f"{prefix}_f1": f1_score(y, y_pred, average="weighted"),
            f"{prefix}_roc_auc": roc_auc_score(y, y_pred_proba),
        }
        
        # Log metrics
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
        
        return metrics

    def validate_model(self, metrics: Dict[str, float]) -> bool:
        """Validate the model against predefined thresholds.

        Args:
            metrics: Dictionary of metrics

        Returns:
            bool: True if model passes validation, False otherwise
        """
        logger.info("Validating model")
        
        validation_passed = True
        
        # Check accuracy
        if metrics["test_accuracy"] < self.config.validation.min_accuracy:
            logger.warning(
                f"Accuracy {metrics['test_accuracy']:.4f} is below threshold "
                f"{self.config.validation.min_accuracy}"
            )
            validation_passed = False
        
        # Check false positive rate
        if metrics.get("test_false_positive_rate", 0) > self.config.validation.max_false_positive_rate:
            logger.warning(
                f"False positive rate {metrics.get('test_false_positive_rate', 0):.4f} "
                f"exceeds threshold {self.config.validation.max_false_positive_rate}"
            )
            validation_passed = False
        
        if validation_passed:
            logger.info("Model validation passed")
        else:
            logger.warning("Model validation failed")
        
        return validation_passed

    def log_to_mlflow(
        self, model: BaseEstimator, metrics: Dict[str, float], params: Dict[str, Any]
    ) -> str:
        """Log model and metrics to MLflow.

        Args:
            model: Trained model
            metrics: Dictionary of metrics
            params: Dictionary of parameters

        Returns:
            str: Run ID
        """
        logger.info("Logging to MLflow")
        
        with mlflow.start_run(run_name=self.config.training.run_name) as run:
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=self.config.model.name,
                input_example=self.config.model.input_example,
                signature=self.config.model.signature,
            )
            
            # Log artifacts
            if self.config.model.tags:
                mlflow.set_tags(self.config.model.tags)
            
            # Log config file
            mlflow.log_artifact(self.config_path, "config")
            
            return run.info.run_id

    def promote_model(self, run_id: str, stage: str) -> None:
        """Promote model to a specific stage in the model registry.

        Args:
            run_id: MLflow run ID
            stage: Target stage (Staging, Production, Archived)
        """
        logger.info(f"Promoting model to {stage} stage")
        
        client = mlflow.tracking.MlflowClient()
        model_uri = f"runs:/{run_id}/model"
        
        # Register the model if not already registered
        model_name = self.config.model.name
        try:
            client.get_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            client.create_registered_model(model_name)
        
        # Create a new model version
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
        )
        
        # Transition model to target stage
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
            archive_existing_versions=True,
        )
        
        logger.info(f"Model {model_name} version {mv.version} promoted to {stage}")

    def save_model(self, model: BaseEstimator, path: Union[str, Path]) -> None:
        """Save the model to disk.

        Args:
            model: Trained model
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {path}")

    def run(self) -> bool:
        """Run the complete pipeline.

        Returns:
            bool: True if pipeline completed successfully, False otherwise
        """
        try:
            # Load data
            X_train, y_train, X_test, y_test = self.load_data()
            
            # Train model
            model = self.train_model(X_train, y_train)
            
            # Evaluate model
            train_metrics = self.evaluate_model(model, X_train, y_train, "train")
            test_metrics = self.evaluate_model(model, X_test, y_test, "test")
            metrics = {**train_metrics, **test_metrics}
            
            # Validate model
            if not self.validate_model(metrics):
                logger.error("Model validation failed")
                return False
            
            # Log to MLflow
            params = self.config.model.hyperparameters.copy()
            params.update({
                "model_name": self.config.model.name,
                "model_version": self.config.model.version,
            })
            
            run_id = self.log_to_mlflow(model, metrics, params)
            
            # Promote model to staging
            self.promote_model(run_id, self.config.model.stage)
            
            # Save model locally
            model_path = Path(self.config.output_dir) / "model.pkl"
            self.save_model(model, model_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False
