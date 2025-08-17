#!/usr/bin/env python3
"""
Script to load sample data into the Disease Outbreak Early Warning System database.
This script populates the database with realistic sample data for development and testing.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd
from faker import Faker
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_batch
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'disease_outbreak'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

# Initialize Faker
fake = Faker()

# Sample disease data
DISEASES = [
    'Influenza', 'Dengue', 'Malaria', 'Cholera', 'COVID-19',
    'Tuberculosis', 'Hepatitis A', 'Measles', 'Zika', 'Ebola'
]

# Sample cities and regions
CITIES = [
    ('Mumbai', 'Maharashtra'), ('Delhi', 'Delhi'), ('Bangalore', 'Karnataka'),
    ('Chennai', 'Tamil Nadu'), ('Kolkata', 'West Bengal'), ('Hyderabad', 'Telangana'),
    ('Pune', 'Maharashtra'), ('Ahmedabad', 'Gujarat')
]

class DataLoader:
    """Class to handle data loading operations"""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            logger.info("Connected to the database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Create locations table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    location_id SERIAL PRIMARY KEY,
                    city VARCHAR(100) NOT NULL,
                    region VARCHAR(100) NOT NULL,
                    latitude FLOAT,
                    longitude FLOAT,
                    population INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(city, region)
                )
            """)
            
            # Create disease_cases table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS disease_cases (
                    case_id SERIAL PRIMARY KEY,
                    location_id INTEGER REFERENCES locations(location_id),
                    disease_name VARCHAR(100) NOT NULL,
                    case_count INTEGER NOT NULL,
                    report_date DATE NOT NULL,
                    severity_index FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create social_media_mentions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS social_media_mentions (
                    mention_id SERIAL PRIMARY KEY,
                    location_id INTEGER REFERENCES locations(location_id),
                    disease_name VARCHAR(100) NOT NULL,
                    mention_count INTEGER NOT NULL,
                    sentiment_score FLOAT,
                    report_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create weather_data table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS weather_data (
                    weather_id SERIAL PRIMARY KEY,
                    location_id INTEGER REFERENCES locations(location_id),
                    temperature FLOAT NOT NULL,
                    humidity FLOAT NOT NULL,
                    rainfall FLOAT NOT NULL,
                    report_date DATE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(location_id, report_date)
                )
            """)
            
            self.conn.commit()
            logger.info("Tables created successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error creating tables: {e}")
            raise
    
    def generate_sample_data(self, days=90):
        """Generate sample data for the specified number of days"""
        try:
            # Insert locations if they don't exist
            location_ids = {}
            for city, region in CITIES:
                self.cursor.execute(
                    """
                    INSERT INTO locations (city, region, latitude, longitude, population)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (city, region) DO NOTHING
                    RETURNING location_id
                    """,
                    (city, region, 
                     round(random.uniform(8.4, 28.6), 4),  # India latitude range
                     round(random.uniform(68.7, 97.25), 4),  # India longitude range
                     random.randint(500000, 20000000)  # Population
                    )
                )
                result = self.cursor.fetchone()
                if result:
                    location_ids[(city, region)] = result[0]
                else:
                    # Get existing location ID
                    self.cursor.execute(
                        "SELECT location_id FROM locations WHERE city = %s AND region = %s",
                        (city, region)
                    )
                    location_ids[(city, region)] = self.cursor.fetchone()[0]
            
            # Generate data for each day
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            current_date = start_date
            
            while current_date <= end_date:
                for (city, region), loc_id in location_ids.items():
                    # Generate weather data
                    temp = round(random.uniform(15, 45), 1)  # Temperature in Celsius
                    humidity = round(random.uniform(30, 95), 1)  # Humidity percentage
                    rainfall = round(random.uniform(0, 50), 1)  # Rainfall in mm
                    
                    self.cursor.execute(
                        """
                        INSERT INTO weather_data 
                        (location_id, temperature, humidity, rainfall, report_date)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (location_id, report_date) DO NOTHING
                        """,
                        (loc_id, temp, humidity, rainfall, current_date)
                    )
                    
                    # Generate disease cases and social media mentions
                    for disease in DISEASES:
                        # Disease cases
                        case_count = random.randint(0, 50)
                        if case_count > 0:
                            severity = round(random.uniform(0.1, 1.0), 2)
                            self.cursor.execute(
                                """
                                INSERT INTO disease_cases
                                (location_id, disease_name, case_count, report_date, severity_index)
                                VALUES (%s, %s, %s, %s, %s)
                                """,
                                (loc_id, disease, case_count, current_date, severity)
                            )
                        
                        # Social media mentions
                        mention_count = random.randint(0, 100)
                        if mention_count > 0:
                            sentiment = round(random.uniform(-1.0, 1.0), 2)  # -1 to 1 sentiment score
                            self.cursor.execute(
                                """
                                INSERT INTO social_media_mentions
                                (location_id, disease_name, mention_count, sentiment_score, report_date)
                                VALUES (%s, %s, %s, %s, %s)
                                """,
                                (loc_id, disease, mention_count, sentiment, current_date)
                            )
                
                current_date += timedelta(days=1)
                
                # Commit every 7 days to avoid large transactions
                if (current_date - start_date).days % 7 == 0:
                    self.conn.commit()
                    logger.info(f"Committed data up to {current_date}")
            
            self.conn.commit()
            logger.info("Sample data generation completed successfully")
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error generating sample data: {e}")
            raise

def main():
    """Main function to run the data loading process"""
    loader = None
    try:
        logger.info("Starting sample data loading process")
        
        # Initialize data loader
        loader = DataLoader()
        loader.connect()
        
        # Create tables if they don't exist
        loader.create_tables()
        
        # Generate and load sample data
        loader.generate_sample_data(days=90)  # Last 90 days of data
        
        logger.info("Sample data loading completed successfully")
        
    except Exception as e:
        logger.error(f"Sample data loading failed: {e}")
        return 1
    finally:
        if loader:
            loader.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
