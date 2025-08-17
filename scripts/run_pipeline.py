#!/usr/bin/env python3
"""
Script to run the model versioning and validation pipeline.
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ml.pipeline.pipeline import ModelVersioningPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run model versioning and validation pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


def main():
    """Run the pipeline."""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log"),
        ],
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting pipeline with config: {args.config}")
        
        # Initialize and run pipeline
        pipeline = ModelVersioningPipeline(args.config)
        success = pipeline.run()
        
        if success:
            logger.info("Pipeline completed successfully")
            return 0
        else:
            logger.error("Pipeline failed")
            return 1
            
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
