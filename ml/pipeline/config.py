""
Configuration settings for the model versioning and validation pipeline.
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union


class ModelStage(str, Enum):
    """Model stages in the deployment lifecycle."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass
class ModelConfig:
    """Configuration for a machine learning model."""
    name: str
    version: str
    stage: ModelStage = ModelStage.NONE
    description: str = ""
    tags: Optional[Dict[str, str]] = None
    hyperparameters: Optional[Dict] = None
    metrics: Optional[Dict[str, float]] = None
    signature: Optional[Dict] = None
    input_example: Optional[Dict] = None


@dataclass
class DataConfig:
    """Configuration for data sources and validation."""
    train_path: Union[str, Path]
    test_path: Union[str, Path]
    validation_path: Optional[Union[str, Path]] = None
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    validation_expectations_path: Optional[Union[str, Path]] = None


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    experiment_name: str
    run_name: Optional[str] = None
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    validation_metric: str = "val_loss"
    validation_mode: str = "min"


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    min_accuracy: float = 0.7
    max_false_positive_rate: float = 0.3
    max_drift_score: float = 0.2
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking."""
    tracking_uri: str = "http://localhost:5000"
    registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    experiment_name: str = "disease-outbreak"


@dataclass
class PipelineConfig:
    """Main configuration for the model pipeline."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    validation: ValidationConfig
    mlflow: MLflowConfig
    output_dir: Union[str, Path] = "output"
    log_level: str = "INFO"

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.output_dir = Path(self.output_dir)
        self.data.train_path = Path(self.data.train_path)
        self.data.test_path = Path(self.data.test_path)
        
        if self.data.validation_path:
            self.data.validation_path = Path(self.data.validation_path)
        if self.data.validation_expectations_path:
            self.data.validation_expectations_path = Path(self.data.validation_expectations_path)


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        PipelineConfig: Loaded configuration.
    """
    import yaml
    from pathlib import Path
    
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dictionaries to config objects
    model_config = ModelConfig(**config_dict['model'])
    data_config = DataConfig(**config_dict['data'])
    training_config = TrainingConfig(**config_dict['training'])
    validation_config = ValidationConfig(**config_dict['validation'])
    mlflow_config = MLflowConfig(**config_dict['mlflow'])
    
    return PipelineConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        validation=validation_config,
        mlflow=mlflow_config,
        output_dir=config_dict.get('output_dir', 'output'),
        log_level=config_dict.get('log_level', 'INFO')
    )
