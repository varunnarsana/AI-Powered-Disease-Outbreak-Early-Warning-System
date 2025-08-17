#!/usr/bin/env python3
"""
Startup Script for Disease Outbreak Early Warning System
Orchestrates the entire system startup and health checks
"""

import os
import sys
import time
import subprocess
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import signal
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DiseaseOutbreakSystem:
    """Main system orchestrator for disease outbreak early warning system"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config = self.load_config(config_file)
        self.services = {}
        self.health_status = {}
        self.running = False
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                return {
                    "services": {
                        "docker": {
                            "enabled": True,
                            "command": "docker-compose up -d",
                            "health_check": "docker ps",
                            "wait_time": 30
                        },
                        "kafka": {
                            "enabled": True,
                            "url": "http://localhost:9092",
                            "health_check": "http://localhost:9092",
                            "wait_time": 60
                        },
                        "redis": {
                            "enabled": True,
                            "url": "http://localhost:6379",
                            "health_check": "redis-cli ping",
                            "wait_time": 30
                        },
                        "postgresql": {
                            "enabled": True,
                            "url": "postgresql://admin:password123@localhost:5432/disease_mlops",
                            "health_check": "pg_isready -h localhost -U admin -d disease_mlops",
                            "wait_time": 45
                        },
                        "mlflow": {
                            "enabled": True,
                            "url": "http://localhost:5000",
                            "health_check": "http://localhost:5000",
                            "wait_time": 30
                        },
                        "prometheus": {
                            "enabled": True,
                            "url": "http://localhost:9090",
                            "health_check": "http://localhost:9090",
                            "wait_time": 30
                        },
                        "grafana": {
                            "enabled": True,
                            "url": "http://localhost:3000",
                            "health_check": "http://localhost:3000",
                            "wait_time": 30
                        },
                        "elasticsearch": {
                            "enabled": True,
                            "url": "http://localhost:9200",
                            "health_check": "http://localhost:9200",
                            "wait_time": 45
                        },
                        "kibana": {
                            "enabled": True,
                            "url": "http://localhost:5601",
                            "health_check": "http://localhost:5601",
                            "wait_time": 30
                        }
                    },
                    "pipelines": {
                        "kafka_producer": {
                            "enabled": True,
                            "script": "pipelines/kafka_producer.py",
                            "wait_time": 10
                        },
                        "spark_streaming": {
                            "enabled": True,
                            "script": "pipelines/spark_streaming.py",
                            "wait_time": 10
                        }
                    },
                    "ml": {
                        "training": {
                            "enabled": True,
                            "script": "ml/train_models.py",
                            "wait_time": 60
                        }
                    },
                    "api": {
                        "fastapi": {
                            "enabled": True,
                            "script": "api/main.py",
                            "port": 8000,
                            "wait_time": 30
                        }
                    },
                    "dashboard": {
                        "streamlit": {
                            "enabled": True,
                            "script": "dashboard/app.py",
                            "port": 8501,
                            "wait_time": 30
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.shutdown()
        sys.exit(0)
    
    def check_docker(self) -> bool:
        """Check if Docker is running"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Docker check failed: {e}")
            return False
    
    def start_docker_services(self) -> bool:
        """Start Docker services using docker-compose"""
        try:
            logger.info("Starting Docker services...")
            
            # Check if docker-compose file exists
            if not os.path.exists("docker-compose.yml"):
                logger.error("docker-compose.yml not found")
                return False
            
            # Start services
            result = subprocess.run(['docker-compose', 'up', '-d'], 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info("Docker services started successfully")
                return True
            else:
                logger.error(f"Docker services failed to start: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting Docker services: {e}")
            return False
    
    def wait_for_service(self, service_name: str, health_check: str, wait_time: int) -> bool:
        """Wait for a service to become healthy"""
        logger.info(f"Waiting for {service_name} to become healthy...")
        
        start_time = time.time()
        while time.time() - start_time < wait_time:
            try:
                if health_check.startswith("http"):
                    # HTTP health check
                    response = requests.get(health_check, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"{service_name} is healthy")
                        return True
                else:
                    # Command-based health check
                    result = subprocess.run(health_check.split(), 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        logger.info(f"{service_name} is healthy")
                        return True
                
                time.sleep(5)
                
            except Exception as e:
                logger.debug(f"Health check failed for {service_name}: {e}")
                time.sleep(5)
        
        logger.error(f"{service_name} failed to become healthy within {wait_time} seconds")
        return False
    
    def start_service(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """Start a specific service"""
        try:
            if not service_config.get("enabled", True):
                logger.info(f"{service_name} is disabled, skipping...")
                return True
            
            logger.info(f"Starting {service_name}...")
            
            if service_name == "docker":
                return self.start_docker_services()
            
            # For other services, wait for them to become healthy
            health_check = service_config.get("health_check", "")
            wait_time = service_config.get("wait_time", 30)
            
            if health_check:
                return self.wait_for_service(service_name, health_check, wait_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting {service_name}: {e}")
            return False
    
    def start_pipeline(self, pipeline_name: str, pipeline_config: Dict[str, Any]) -> bool:
        """Start a data pipeline"""
        try:
            if not pipeline_config.get("enabled", True):
                logger.info(f"{pipeline_name} is disabled, skipping...")
                return True
            
            logger.info(f"Starting {pipeline_name}...")
            
            script_path = pipeline_config.get("script", "")
            if not script_path or not os.path.exists(script_path):
                logger.error(f"Script not found: {script_path}")
                return False
            
            # Start pipeline in background
            process = subprocess.Popen([sys.executable, script_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            
            self.services[pipeline_name] = process
            
            # Wait for pipeline to start
            wait_time = pipeline_config.get("wait_time", 10)
            time.sleep(wait_time)
            
            if process.poll() is None:
                logger.info(f"{pipeline_name} started successfully")
                return True
            else:
                logger.error(f"{pipeline_name} failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting {pipeline_name}: {e}")
            return False
    
    def start_ml_training(self, training_config: Dict[str, Any]) -> bool:
        """Start ML model training"""
        try:
            if not training_config.get("enabled", True):
                logger.info("ML training is disabled, skipping...")
                return True
            
            logger.info("Starting ML model training...")
            
            script_path = training_config.get("script", "")
            if not script_path or not os.path.exists(script_path):
                logger.error(f"Training script not found: {script_path}")
                return False
            
            # Start training in background
            process = subprocess.Popen([sys.executable, script_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            
            self.services["ml_training"] = process
            
            # Wait for training to start
            wait_time = training_config.get("wait_time", 60)
            time.sleep(wait_time)
            
            if process.poll() is None:
                logger.info("ML training started successfully")
                return True
            else:
                logger.error("ML training failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting ML training: {e}")
            return False
    
    def start_api_service(self, api_config: Dict[str, Any]) -> bool:
        """Start API service"""
        try:
            if not api_config.get("enabled", True):
                logger.info("API service is disabled, skipping...")
                return True
            
            logger.info("Starting API service...")
            
            script_path = api_config.get("script", "")
            if not script_path or not os.path.exists(script_path):
                logger.error(f"API script not found: {script_path}")
                return False
            
            # Start API service in background
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", 
                                      "--host", "0.0.0.0", "--port", "8000"],
                                     cwd="api",
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            
            self.services["api"] = process
            
            # Wait for API to start
            wait_time = api_config.get("wait_time", 30)
            time.sleep(wait_time)
            
            # Check if API is responding
            try:
                response = requests.get("http://localhost:8000/health", timeout=10)
                if response.status_code == 200:
                    logger.info("API service started successfully")
                    return True
                else:
                    logger.error("API service health check failed")
                    return False
            except Exception as e:
                logger.error(f"API health check failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting API service: {e}")
            return False
    
    def start_dashboard(self, dashboard_config: Dict[str, Any]) -> bool:
        """Start dashboard service"""
        try:
            if not dashboard_config.get("enabled", True):
                logger.info("Dashboard is disabled, skipping...")
                return True
            
            logger.info("Starting dashboard...")
            
            script_path = dashboard_config.get("script", "")
            if not script_path or not os.path.exists(script_path):
                logger.error(f"Dashboard script not found: {script_path}")
                return False
            
            # Start dashboard in background
            process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py"],
                                     cwd="dashboard",
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            
            self.services["dashboard"] = process
            
            # Wait for dashboard to start
            wait_time = dashboard_config.get("wait_time", 30)
            time.sleep(wait_time)
            
            # Check if dashboard is responding
            try:
                response = requests.get("http://localhost:8501", timeout=10)
                if response.status_code == 200:
                    logger.info("Dashboard started successfully")
                    return True
                else:
                    logger.error("Dashboard health check failed")
                    return False
            except Exception as e:
                logger.error(f"Dashboard health check failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            return False
    
    def run_health_monitoring(self):
        """Run continuous health monitoring"""
        while self.running:
            try:
                # Check service health
                for service_name, service_config in self.config.get("services", {}).items():
                    if service_config.get("enabled", True):
                        health_check = service_config.get("health_check", "")
                        if health_check:
                            try:
                                if health_check.startswith("http"):
                                    response = requests.get(health_check, timeout=5)
                                    self.health_status[service_name] = response.status_code == 200
                                else:
                                    result = subprocess.run(health_check.split(), 
                                                          capture_output=True, text=True, timeout=10)
                                    self.health_status[service_name] = result.returncode == 0
                            except Exception as e:
                                self.health_status[service_name] = False
                                logger.debug(f"Health check failed for {service_name}: {e}")
                
                # Check pipeline health
                for pipeline_name in self.config.get("pipelines", {}):
                    if pipeline_name in self.services:
                        process = self.services[pipeline_name]
                        self.health_status[pipeline_name] = process.poll() is None
                
                # Log health status
                unhealthy_services = [name for name, healthy in self.health_status.items() if not healthy]
                if unhealthy_services:
                    logger.warning(f"Unhealthy services: {unhealthy_services}")
                else:
                    logger.info("All services are healthy")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(30)
    
    def startup(self) -> bool:
        """Start the entire system"""
        logger.info("Starting Disease Outbreak Early Warning System...")
        
        try:
            # Check prerequisites
            if not self.check_docker():
                logger.error("Docker is not running. Please start Docker first.")
                return False
            
            # Start infrastructure services
            services_config = self.config.get("services", {})
            for service_name, service_config in services_config.items():
                if not self.start_service(service_name, service_config):
                    logger.error(f"Failed to start {service_name}")
                    return False
            
            # Start data pipelines
            pipelines_config = self.config.get("pipelines", {})
            for pipeline_name, pipeline_config in pipelines_config.items():
                if not self.start_pipeline(pipeline_name, pipeline_config):
                    logger.error(f"Failed to start {pipeline_name}")
                    return False
            
            # Start ML training
            ml_config = self.config.get("ml", {})
            training_config = ml_config.get("training", {})
            if not self.start_ml_training(training_config):
                logger.error("Failed to start ML training")
                return False
            
            # Start API service
            api_config = self.config.get("api", {})
            fastapi_config = api_config.get("fastapi", {})
            if not self.start_api_service(fastapi_config):
                logger.error("Failed to start API service")
                return False
            
            # Start dashboard
            dashboard_config = self.config.get("dashboard", {})
            streamlit_config = dashboard_config.get("streamlit", {})
            if not self.start_dashboard(streamlit_config):
                logger.error("Failed to start dashboard")
                return False
            
            logger.info("System startup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down system...")
        
        try:
            # Stop all services
            for service_name, process in self.services.items():
                if process and process.poll() is None:
                    logger.info(f"Stopping {service_name}...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
            
            # Stop Docker services
            try:
                subprocess.run(['docker-compose', 'down'], 
                              capture_output=True, text=True, timeout=30)
                logger.info("Docker services stopped")
            except Exception as e:
                logger.error(f"Error stopping Docker services: {e}")
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def run(self):
        """Run the system"""
        try:
            if not self.startup():
                logger.error("System startup failed")
                return False
            
            self.running = True
            
            # Start health monitoring in background
            health_thread = threading.Thread(target=self.run_health_monitoring, daemon=True)
            health_thread.start()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            return False
        except Exception as e:
            logger.error(f"System error: {e}")
            return False
        finally:
            self.shutdown()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Disease Outbreak Early Warning System")
    parser.add_argument("--config", "-c", default="config.json", 
                       help="Configuration file path")
    parser.add_argument("--daemon", "-d", action="store_true",
                       help="Run in daemon mode")
    parser.add_argument("--log-level", "-l", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run system
    system = DiseaseOutbreakSystem(args.config)
    
    if args.daemon:
        # Run in background
        import daemon
        with daemon.DaemonContext():
            success = system.run()
            sys.exit(0 if success else 1)
    else:
        # Run in foreground
        success = system.run()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
