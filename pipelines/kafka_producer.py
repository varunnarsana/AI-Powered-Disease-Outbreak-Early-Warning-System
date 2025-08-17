#!/usr/bin/env python3
"""
Kafka Producer for Disease Outbreak Early Warning System
Simulates real-time data streams from various health data sources
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError
import faker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthDataProducer:
    """Produces simulated health data streams to Kafka topics"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.fake = faker.Faker()
        
        # Disease symptoms for social media simulation
        self.symptoms = [
            'fever', 'headache', 'cough', 'fatigue', 'body aches',
            'nausea', 'vomiting', 'diarrhea', 'rash', 'sore throat',
            'runny nose', 'congestion', 'chills', 'sweating', 'dizziness'
        ]
        
        # Diseases to simulate
        self.diseases = [
            'dengue', 'malaria', 'influenza', 'covid19', 'chikungunya',
            'zika', 'typhoid', 'hepatitis', 'measles', 'chickenpox'
        ]
        
        # Cities for geolocation
        self.cities = [
            {'name': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'region': 'Maharashtra'},
            {'name': 'Delhi', 'lat': 28.7041, 'lon': 77.1025, 'region': 'Delhi'},
            {'name': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946, 'region': 'Karnataka'},
            {'name': 'Chennai', 'lat': 13.0827, 'lon': 80.2707, 'region': 'Tamil Nadu'},
            {'name': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639, 'region': 'West Bengal'},
            {'name': 'Hyderabad', 'lat': 17.3850, 'lon': 78.4867, 'region': 'Telangana'},
            {'name': 'Pune', 'lat': 18.5204, 'lon': 73.8567, 'region': 'Maharashtra'},
            {'name': 'Ahmedabad', 'lat': 23.0225, 'lon': 72.5714, 'region': 'Gujarat'}
        ]

    def generate_social_media_post(self) -> Dict[str, Any]:
        """Generate a simulated social media post mentioning health symptoms"""
        city = random.choice(self.cities)
        symptom = random.choice(self.symptoms)
        
        # Generate realistic post content
        post_templates = [
            f"Feeling terrible today with {symptom} 😷",
            f"Anyone else experiencing {symptom}? This is awful",
            f"Day 3 of {symptom} and still not better",
            f"Think I caught something - {symptom} is killing me",
            f"Local clinic is packed with people having {symptom}",
            f"Stay safe everyone, {symptom} is going around",
            f"Not feeling well, {symptom} and fatigue",
            f"Emergency room visit for severe {symptom}"
        ]
        
        return {
            'id': self.fake.uuid4(),
            'platform': random.choice(['twitter', 'facebook', 'instagram']),
            'user_id': self.fake.user_name(),
            'content': random.choice(post_templates),
            'symptoms': [symptom],
            'location': {
                'city': city['name'],
                'region': city['region'],
                'lat': city['lat'] + random.uniform(-0.01, 0.01),
                'lon': city['lon'] + random.uniform(-0.01, 0.01)
            },
            'timestamp': datetime.now().isoformat(),
            'sentiment': random.choice(['negative', 'neutral', 'positive']),
            'engagement': random.randint(0, 1000)
        }

    def generate_hospital_log(self) -> Dict[str, Any]:
        """Generate a simulated hospital admission log"""
        city = random.choice(self.cities)
        disease = random.choice(self.diseases)
        
        # Simulate seasonal patterns
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:  # Monsoon season
            mosquito_diseases = ['dengue', 'malaria', 'chikungunya', 'zika']
            if disease in mosquito_diseases:
                disease = random.choice(mosquito_diseases)
        
        return {
            'id': self.fake.uuid4(),
            'hospital_id': f"HOSP_{random.randint(1000, 9999)}",
            'patient_id': self.fake.uuid4(),
            'admission_date': datetime.now().isoformat(),
            'discharge_date': (datetime.now() + timedelta(days=random.randint(1, 14))).isoformat(),
            'diagnosis': disease,
            'severity': random.choice(['mild', 'moderate', 'severe', 'critical']),
            'age_group': random.choice(['0-5', '6-18', '19-30', '31-50', '51-70', '70+']),
            'gender': random.choice(['male', 'female', 'other']),
            'location': {
                'city': city['name'],
                'region': city['region'],
                'lat': city['lat'] + random.uniform(-0.01, 0.01),
                'lon': city['lon'] + random.uniform(-0.01, 0.01)
            },
            'symptoms': random.sample(self.symptoms, random.randint(1, 4)),
            'outcome': random.choice(['recovered', 'discharged', 'transferred', 'deceased']),
            'length_of_stay': random.randint(1, 21)
        }

    def generate_weather_data(self) -> Dict[str, Any]:
        """Generate simulated weather data for disease correlation"""
        city = random.choice(self.cities)
        
        # Simulate realistic weather patterns
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:  # Monsoon
            temperature = random.uniform(25, 35)
            humidity = random.uniform(70, 95)
            rainfall = random.uniform(10, 100)
        elif current_month in [12, 1, 2]:  # Winter
            temperature = random.uniform(15, 25)
            humidity = random.uniform(40, 70)
            rainfall = random.uniform(0, 20)
        else:  # Summer/Spring
            temperature = random.uniform(30, 40)
            humidity = random.uniform(50, 80)
            rainfall = random.uniform(0, 50)
        
        return {
            'id': self.fake.uuid4(),
            'station_id': f"WX_{random.randint(1000, 9999)}",
            'timestamp': datetime.now().isoformat(),
            'location': {
                'city': city['name'],
                'region': city['region'],
                'lat': city['lat'],
                'lon': city['lon']
            },
            'temperature_celsius': round(temperature, 1),
            'humidity_percent': round(humidity, 1),
            'rainfall_mm': round(rainfall, 1),
            'wind_speed_kmh': random.uniform(0, 25),
            'pressure_hpa': random.uniform(1000, 1020),
            'uv_index': random.uniform(0, 11),
            'air_quality_index': random.randint(1, 500)
        }

    def generate_public_health_report(self) -> Dict[str, Any]:
        """Generate simulated public health reports from government sources"""
        city = random.choice(self.cities)
        disease = random.choice(self.diseases)
        
        # Simulate weekly/monthly reports
        report_date = datetime.now() - timedelta(days=random.randint(0, 7))
        
        return {
            'id': self.fake.uuid4(),
            'report_id': f"PHR_{random.randint(10000, 99999)}",
            'source': random.choice(['WHO', 'CDC', 'Ministry of Health', 'State Health Department']),
            'report_date': report_date.isoformat(),
            'location': {
                'city': city['name'],
                'region': city['region'],
                'lat': city['lat'],
                'lon': city['lon']
            },
            'disease': disease,
            'total_cases': random.randint(10, 1000),
            'new_cases': random.randint(0, 100),
            'deaths': random.randint(0, 50),
            'recovered': random.randint(0, 800),
            'active_cases': random.randint(5, 200),
            'testing_rate': random.uniform(0.1, 0.8),
            'vaccination_rate': random.uniform(0.1, 0.9),
            'risk_level': random.choice(['low', 'medium', 'high', 'critical']),
            'recommendations': [
                'Maintain social distancing',
                'Wear masks in crowded areas',
                'Get vaccinated',
                'Monitor symptoms',
                'Seek medical attention if severe'
            ]
        }

    def send_data(self, topic: str, data: Dict[str, Any], key: str = None):
        """Send data to Kafka topic"""
        try:
            future = self.producer.send(topic, key=key, value=data)
            record_metadata = future.get(timeout=10)
            logger.info(f"Data sent to {topic} partition {record_metadata.partition} offset {record_metadata.offset}")
        except KafkaError as e:
            logger.error(f"Failed to send data to {topic}: {e}")

    def run_streaming_producer(self, duration_minutes: int = 60):
        """Run the producer for a specified duration"""
        logger.info(f"Starting health data producer for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                # Generate and send social media posts (every 30 seconds)
                if int(time.time()) % 30 == 0:
                    social_data = self.generate_social_media_post()
                    self.send_data('health_social_media', social_data, social_data['id'])
                
                # Generate and send hospital logs (every 2 minutes)
                if int(time.time()) % 120 == 0:
                    hospital_data = self.generate_hospital_log()
                    self.send_data('health_hospital_logs', hospital_data, hospital_data['id'])
                
                # Generate and send weather data (every 5 minutes)
                if int(time.time()) % 300 == 0:
                    weather_data = self.generate_weather_data()
                    self.send_data('health_weather_data', weather_data, weather_data['id'])
                
                # Generate and send public health reports (every 10 minutes)
                if int(time.time()) % 600 == 0:
                    health_report = self.generate_public_health_report()
                    self.send_data('health_public_reports', health_report, health_report['id'])
                
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Producer stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in producer: {e}")
                time.sleep(5)
        
        self.producer.close()
        logger.info("Producer stopped")

def main():
    """Main function to run the producer"""
    producer = HealthDataProducer()
    
    try:
        # Run for 1 hour by default
        producer.run_streaming_producer(duration_minutes=60)
    except KeyboardInterrupt:
        logger.info("Producer stopped by user")
    finally:
        producer.producer.close()

if __name__ == "__main__":
    main()
