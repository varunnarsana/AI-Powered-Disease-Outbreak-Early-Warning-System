#!/usr/bin/env python3
"""
Apache Spark Structured Streaming Pipeline for Disease Outbreak Early Warning System
Processes real-time health data streams and performs data cleaning and feature engineering
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_json, struct, lit, udf, window, count, 
    avg, sum as spark_sum, max as spark_max, min as spark_min,
    expr, when, regexp_replace, lower, split, explode, array_contains
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, 
    BooleanType, ArrayType, TimestampType, MapType
)
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthDataStreamingPipeline:
    """Spark Structured Streaming pipeline for health data processing"""
    
    def __init__(self):
        self.spark = self._create_spark_session()
        self._define_schemas()
        
    def _create_spark_session(self) -> SparkSession:
        """Create and configure Spark session"""
        return (SparkSession.builder
                .appName("HealthDataStreamingPipeline")
                .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.adaptive.skewJoin.enabled", "true")
                .config("spark.sql.streaming.schemaInference", "true")
                .getOrCreate())
    
    def _define_schemas(self):
        """Define schemas for different data types"""
        
        # Social media schema
        self.social_media_schema = StructType([
            StructField("id", StringType(), False),
            StructField("platform", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("content", StringType(), True),
            StructField("symptoms", ArrayType(StringType()), True),
            StructField("location", StructType([
                StructField("city", StringType(), True),
                StructField("region", StringType(), True),
                StructField("lat", DoubleType(), True),
                StructField("lon", DoubleType(), True)
            ]), True),
            StructField("timestamp", StringType(), True),
            StructField("sentiment", StringType(), True),
            StructField("engagement", IntegerType(), True)
        ])
        
        # Hospital logs schema
        self.hospital_schema = StructType([
            StructField("id", StringType(), False),
            StructField("hospital_id", StringType(), True),
            StructField("patient_id", StringType(), True),
            StructField("admission_date", StringType(), True),
            StructField("discharge_date", StringType(), True),
            StructField("diagnosis", StringType(), True),
            StructField("severity", StringType(), True),
            StructField("age_group", StringType(), True),
            StructField("gender", StringType(), True),
            StructField("location", StructType([
                StructField("city", StringType(), True),
                StructField("region", StringType(), True),
                StructField("lat", DoubleType(), True),
                StructField("lon", DoubleType(), True)
            ]), True),
            StructField("symptoms", ArrayType(StringType()), True),
            StructField("outcome", StringType(), True),
            StructField("length_of_stay", IntegerType(), True)
        ])
        
        # Weather data schema
        self.weather_schema = StructType([
            StructField("id", StringType(), False),
            StructField("station_id", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("location", StructType([
                StructField("city", StringType(), True),
                StructField("region", StringType(), True),
                StructField("lat", DoubleType(), True),
                StructField("lon", DoubleType(), True)
            ]), True),
            StructField("temperature_celsius", DoubleType(), True),
            StructField("humidity_percent", DoubleType(), True),
            StructField("rainfall_mm", DoubleType(), True),
            StructField("wind_speed_kmh", DoubleType(), True),
            StructField("pressure_hpa", DoubleType(), True),
            StructField("uv_index", DoubleType(), True),
            StructField("air_quality_index", IntegerType(), True)
        ])
        
        # Public health reports schema
        self.health_report_schema = StructType([
            StructField("id", StringType(), False),
            StructField("report_id", StringType(), True),
            StructField("source", StringType(), True),
            StructField("report_date", StringType(), True),
            StructField("location", StructType([
                StructField("city", StringType(), True),
                StructField("region", StringType(), True),
                StructField("lat", DoubleType(), True),
                StructField("lon", DoubleType(), True)
            ]), True),
            StructField("disease", StringType(), True),
            StructField("total_cases", IntegerType(), True),
            StructField("new_cases", IntegerType(), True),
            StructField("deaths", IntegerType(), True),
            StructField("recovered", IntegerType(), True),
            StructField("active_cases", IntegerType(), True),
            StructField("testing_rate", DoubleType(), True),
            StructField("vaccination_rate", DoubleType(), True),
            StructField("risk_level", StringType(), True),
            StructField("recommendations", ArrayType(StringType()), True)
        ])

    def create_social_media_stream(self) -> "DataFrame":
        """Create streaming DataFrame for social media data"""
        return (self.spark
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "health_social_media")
                .option("startingOffsets", "latest")
                .load()
                .selectExpr("CAST(value AS STRING) as json")
                .select(from_json(col("json"), self.social_media_schema).alias("data"))
                .select("data.*")
                .withColumn("timestamp", F.to_timestamp(col("timestamp")))
                .withWatermark("timestamp", "10 minutes"))

    def create_hospital_logs_stream(self) -> "DataFrame":
        """Create streaming DataFrame for hospital logs"""
        return (self.spark
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "health_hospital_logs")
                .option("startingOffsets", "latest")
                .load()
                .selectExpr("CAST(value AS STRING) as json")
                .select(from_json(col("json"), self.hospital_schema).alias("data"))
                .select("data.*")
                .withColumn("admission_date", F.to_timestamp(col("admission_date")))
                .withColumn("discharge_date", F.to_timestamp(col("discharge_date")))
                .withWatermark("admission_date", "10 minutes"))

    def create_weather_data_stream(self) -> "DataFrame":
        """Create streaming DataFrame for weather data"""
        return (self.spark
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "health_weather_data")
                .option("startingOffsets", "latest")
                .load()
                .selectExpr("CAST(value AS STRING) as json")
                .select(from_json(col("json"), self.weather_schema).alias("data"))
                .select("data.*")
                .withColumn("timestamp", F.to_timestamp(col("timestamp")))
                .withWatermark("timestamp", "10 minutes"))

    def create_health_reports_stream(self) -> "DataFrame":
        """Create streaming DataFrame for public health reports"""
        return (self.spark
                .readStream
                .readStream
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("subscribe", "health_public_reports")
                .option("startingOffsets", "latest")
                .load()
                .selectExpr("CAST(value AS STRING) as json")
                .select(from_json(col("json"), self.health_report_schema).alias("data"))
                .select("data.*")
                .withColumn("report_date", F.to_timestamp(col("report_date")))
                .withWatermark("report_date", "10 minutes"))

    def process_social_media_data(self, df: "DataFrame") -> "DataFrame":
        """Process and clean social media data"""
        return (df
                .filter(col("content").isNotNull())
                .filter(col("location.city").isNotNull())
                .withColumn("content_clean", 
                           regexp_replace(lower(col("content")), r'[^\w\s]', ''))
                .withColumn("symptom_count", F.size(col("symptoms")))
                .withColumn("has_health_keywords", 
                           array_contains(col("symptoms"), "fever") | 
                           array_contains(col("symptoms"), "cough") |
                           array_contains(col("symptoms"), "headache"))
                .withColumn("risk_score", 
                           when(col("sentiment") == "negative", 3)
                           .when(col("sentiment") == "neutral", 2)
                           .otherwise(1))
                .withColumn("engagement_score", 
                           when(col("engagement") > 100, 3)
                           .when(col("engagement") > 50, 2)
                           .otherwise(1)))

    def process_hospital_data(self, df: "DataFrame") -> "DataFrame":
        """Process and clean hospital data"""
        return (df
                .filter(col("diagnosis").isNotNull())
                .filter(col("location.city").isNotNull())
                .withColumn("severity_score", 
                           when(col("severity") == "critical", 4)
                           .when(col("severity") == "severe", 3)
                           .when(col("severity") == "moderate", 2)
                           .otherwise(1))
                .withColumn("age_score", 
                           when(col("age_group") == "0-5", 3)
                           .when(col("age_group") == "70+", 3)
                           .when(col("age_group") == "6-18", 2)
                           .otherwise(1))
                .withColumn("outcome_score", 
                           when(col("outcome") == "deceased", 4)
                           .when(col("outcome") == "transferred", 3)
                           .otherwise(1)))

    def process_weather_data(self, df: "DataFrame") -> "DataFrame":
        """Process and clean weather data"""
        return (df
                .filter(col("temperature_celsius").isNotNull())
                .filter(col("humidity_percent").isNotNull())
                .withColumn("mosquito_risk", 
                           when((col("temperature_celsius") >= 25) & 
                                (col("humidity_percent") >= 70) & 
                                (col("rainfall_mm") > 0), "high")
                           .when((col("temperature_celsius") >= 20) & 
                                 (col("humidity_percent") >= 60), "medium")
                           .otherwise("low"))
                .withColumn("weather_severity", 
                           when(col("air_quality_index") > 300, 3)
                           .when(col("air_quality_index") > 150, 2)
                           .otherwise(1)))

    def create_aggregated_metrics(self, social_df: "DataFrame", 
                                 hospital_df: "DataFrame", 
                                 weather_df: "DataFrame") -> "DataFrame":
        """Create aggregated metrics for outbreak detection"""
        
        # Social media metrics by city and time window
        social_metrics = (social_df
                         .groupBy(
                             window("timestamp", "5 minutes"),
                             "location.city",
                             "location.region"
                         )
                         .agg(
                             count("*").alias("post_count"),
                             avg("risk_score").alias("avg_risk_score"),
                             avg("engagement_score").alias("avg_engagement"),
                             F.countDistinct("user_id").alias("unique_users")
                         ))
        
        # Hospital metrics by city and time window
        hospital_metrics = (hospital_df
                           .groupBy(
                               window("admission_date", "5 minutes"),
                               "location.city",
                               "location.region"
                           )
                           .agg(
                               count("*").alias("admission_count"),
                               avg("severity_score").alias("avg_severity"),
                               avg("length_of_stay").alias("avg_length_of_stay"),
                               F.countDistinct("diagnosis").alias("disease_variety")
                           ))
        
        # Weather metrics by city and time window
        weather_metrics = (weather_df
                          .groupBy(
                              window("timestamp", "5 minutes"),
                              "location.city",
                              "location.region"
                          )
                          .agg(
                              avg("temperature_celsius").alias("avg_temperature"),
                              avg("humidity_percent").alias("avg_humidity"),
                              avg("rainfall_mm").alias("avg_rainfall"),
                              F.countDistinct("mosquito_risk").alias("mosquito_risk_level")
                          ))
        
        # Join all metrics
        return (social_metrics
                .join(hospital_metrics, 
                      ["window", "location.city", "location.region"], "outer")
                .join(weather_metrics, 
                      ["window", "location.city", "location.region"], "outer")
                .withColumn("outbreak_risk_score", 
                           (col("post_count") * 0.3 + 
                            col("admission_count") * 0.4 + 
                            col("avg_severity") * 0.3) * 
                           when(col("mosquito_risk_level") == "high", 1.5)
                           .when(col("mosquito_risk_level") == "medium", 1.2)
                           .otherwise(1.0))
                .withColumn("risk_level", 
                           when(col("outbreak_risk_score") > 8, "critical")
                           .when(col("outbreak_risk_score") > 6, "high")
                           .when(col("outbreak_risk_score") > 4, "medium")
                           .otherwise("low")))

    def write_to_console(self, df: "DataFrame", output_mode: str = "append"):
        """Write streaming DataFrame to console for debugging"""
        return (df.writeStream
                .outputMode(output_mode)
                .format("console")
                .option("truncate", False)
                .option("numRows", 20)
                .start())

    def write_to_kafka(self, df: "DataFrame", topic: str, output_mode: str = "append"):
        """Write streaming DataFrame back to Kafka"""
        return (df.select(to_json(struct("*")).alias("value"))
                .writeStream
                .outputMode(output_mode)
                .format("kafka")
                .option("kafka.bootstrap.servers", "localhost:9092")
                .option("topic", topic)
                .option("checkpointLocation", f"/tmp/checkpoint/{topic}")
                .start())

    def run_pipeline(self):
        """Run the complete streaming pipeline"""
        logger.info("Starting health data streaming pipeline...")
        
        try:
            # Create streaming DataFrames
            social_stream = self.create_social_media_stream()
            hospital_stream = self.create_hospital_logs_stream()
            weather_stream = self.create_weather_data_stream()
            
            # Process data
            processed_social = self.process_social_media_data(social_stream)
            processed_hospital = self.process_hospital_data(hospital_stream)
            processed_weather = self.process_weather_data(weather_stream)
            
            # Create aggregated metrics
            aggregated_metrics = self.create_aggregated_metrics(
                processed_social, processed_hospital, processed_weather
            )
            
            # Start streaming queries
            queries = []
            
            # Write aggregated metrics to console for monitoring
            queries.append(
                self.write_to_console(aggregated_metrics, "complete")
            )
            
            # Write processed data back to Kafka for ML pipeline
            queries.append(
                self.write_to_kafka(aggregated_metrics, "health_aggregated_metrics")
            )
            
            # Write individual processed streams
            queries.append(
                self.write_to_kafka(processed_social, "health_processed_social")
            )
            
            queries.append(
                self.write_to_kafka(processed_hospital, "health_processed_hospital")
            )
            
            queries.append(
                self.write_to_kafka(processed_weather, "health_processed_weather")
            )
            
            # Wait for all queries to terminate
            self.spark.streams.awaitAnyTermination()
            
        except Exception as e:
            logger.error(f"Error in streaming pipeline: {e}")
            raise
        finally:
            self.spark.stop()

def main():
    """Main function to run the streaming pipeline"""
    pipeline = HealthDataStreamingPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()
