-- Database initialization script for Disease Outbreak Early Warning System
-- This script creates the necessary tables and schemas

-- Create database if it doesn't exist
-- CREATE DATABASE disease_mlops;

-- Connect to the database
\c disease_mlops;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS health_data;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS alerts;

-- Create tables for health data
CREATE TABLE IF NOT EXISTS health_data.social_media_posts (
    id UUID PRIMARY KEY,
    platform VARCHAR(50) NOT NULL,
    user_id VARCHAR(100),
    content TEXT,
    symptoms TEXT[],
    city VARCHAR(100),
    region VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sentiment VARCHAR(20),
    engagement INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS health_data.hospital_logs (
    id UUID PRIMARY KEY,
    hospital_id VARCHAR(50) NOT NULL,
    patient_id UUID,
    admission_date TIMESTAMP WITH TIME ZONE,
    discharge_date TIMESTAMP WITH TIME ZONE,
    diagnosis VARCHAR(200),
    severity VARCHAR(20),
    age_group VARCHAR(20),
    gender VARCHAR(20),
    city VARCHAR(100),
    region VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    symptoms TEXT[],
    outcome VARCHAR(50),
    length_of_stay INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS health_data.weather_data (
    id UUID PRIMARY KEY,
    station_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    city VARCHAR(100),
    region VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    temperature_celsius DECIMAL(5, 2),
    humidity_percent DECIMAL(5, 2),
    rainfall_mm DECIMAL(8, 2),
    wind_speed_kmh DECIMAL(5, 2),
    pressure_hpa DECIMAL(7, 2),
    uv_index DECIMAL(4, 2),
    air_quality_index INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS health_data.public_health_reports (
    id UUID PRIMARY KEY,
    report_id VARCHAR(50) NOT NULL,
    source VARCHAR(100),
    report_date TIMESTAMP WITH TIME ZONE,
    city VARCHAR(100),
    region VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    disease VARCHAR(100),
    total_cases INTEGER,
    new_cases INTEGER,
    deaths INTEGER,
    recovered INTEGER,
    active_cases INTEGER,
    testing_rate DECIMAL(5, 4),
    vaccination_rate DECIMAL(5, 4),
    risk_level VARCHAR(20),
    recommendations TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for aggregated metrics
CREATE TABLE IF NOT EXISTS health_data.aggregated_metrics (
    id UUID PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    region VARCHAR(100) NOT NULL,
    time_window TIMESTAMP WITH TIME ZONE NOT NULL,
    post_count INTEGER DEFAULT 0,
    avg_risk_score DECIMAL(5, 2) DEFAULT 0,
    avg_engagement DECIMAL(5, 2) DEFAULT 0,
    unique_users INTEGER DEFAULT 0,
    admission_count INTEGER DEFAULT 0,
    avg_severity DECIMAL(5, 2) DEFAULT 0,
    avg_length_of_stay DECIMAL(5, 2) DEFAULT 0,
    disease_variety INTEGER DEFAULT 0,
    avg_temperature DECIMAL(5, 2) DEFAULT 0,
    avg_humidity DECIMAL(5, 2) DEFAULT 0,
    avg_rainfall DECIMAL(8, 2) DEFAULT 0,
    mosquito_risk_level VARCHAR(20),
    outbreak_risk_score DECIMAL(8, 4) DEFAULT 0,
    risk_level VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for ML models
CREATE TABLE IF NOT EXISTS ml_models.model_registry (
    id UUID PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    model_path TEXT NOT NULL,
    training_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    performance_metrics JSONB,
    feature_columns TEXT[],
    target_column VARCHAR(100),
    hyperparameters JSONB,
    training_data_info JSONB,
    model_artifacts JSONB,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id UUID PRIMARY KEY,
    prediction_id VARCHAR(100) UNIQUE NOT NULL,
    city VARCHAR(100) NOT NULL,
    region VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    input_features JSONB,
    prediction_value DECIMAL(8, 4),
    confidence_score DECIMAL(5, 4),
    risk_level VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_models.model_performance (
    id UUID PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    mse DECIMAL(10, 6),
    mae DECIMAL(10, 6),
    r2_score DECIMAL(5, 4),
    accuracy DECIMAL(5, 4),
    precision DECIMAL(5, 4),
    recall DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    confusion_matrix JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for monitoring
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6),
    metric_unit VARCHAR(20),
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitoring.api_metrics (
    id UUID PRIMARY KEY,
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    user_agent TEXT,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monitoring.pipeline_metrics (
    id UUID PRIMARY KEY,
    pipeline_name VARCHAR(100) NOT NULL,
    stage VARCHAR(100),
    status VARCHAR(20),
    records_processed INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create tables for alerts
CREATE TABLE IF NOT EXISTS alerts.alert_rules (
    id UUID PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
    rule_description TEXT,
    condition_expression TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50),
    enabled BOOLEAN DEFAULT true,
    cooldown_minutes INTEGER DEFAULT 0,
    notification_channels TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts.active_alerts (
    id UUID PRIMARY KEY,
    alert_id VARCHAR(100) UNIQUE NOT NULL,
    rule_id UUID REFERENCES alerts.alert_rules(id),
    city VARCHAR(100),
    region VARCHAR(100),
    risk_level VARCHAR(20),
    message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts.alert_history (
    id UUID PRIMARY KEY,
    alert_id VARCHAR(100) NOT NULL,
    rule_id UUID REFERENCES alerts.alert_rules(id),
    city VARCHAR(100),
    region VARCHAR(100),
    risk_level VARCHAR(20),
    message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(50),
    status VARCHAR(20),
    lifecycle JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_social_media_city_time ON health_data.social_media_posts(city, timestamp);
CREATE INDEX IF NOT EXISTS idx_social_media_symptoms ON health_data.social_media_posts USING GIN(symptoms);
CREATE INDEX IF NOT EXISTS idx_hospital_city_time ON health_data.hospital_logs(city, admission_date);
CREATE INDEX IF NOT EXISTS idx_hospital_diagnosis ON health_data.hospital_logs(diagnosis);
CREATE INDEX IF NOT EXISTS idx_weather_city_time ON health_data.weather_data(city, timestamp);
CREATE INDEX IF NOT EXISTS idx_aggregated_metrics_city_time ON health_data.aggregated_metrics(city, time_window);
CREATE INDEX IF NOT EXISTS idx_predictions_city_time ON ml_models.predictions(city, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_city_status ON alerts.active_alerts(city, status);

-- Create views for common queries
CREATE OR REPLACE VIEW health_data.city_risk_summary AS
SELECT 
    city,
    region,
    COUNT(*) as total_predictions,
    AVG(outbreak_risk_score) as avg_risk_score,
    MAX(outbreak_risk_score) as max_risk_score,
    COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_count,
    COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_count,
    COUNT(CASE WHEN risk_level = 'medium' THEN 1 END) as medium_count,
    COUNT(CASE WHEN risk_level = 'low' THEN 1 END) as low_count,
    MAX(timestamp) as last_updated
FROM health_data.aggregated_metrics
GROUP BY city, region;

CREATE OR REPLACE VIEW monitoring.system_health_summary AS
SELECT 
    metric_name,
    AVG(metric_value) as avg_value,
    MAX(metric_value) as max_value,
    MIN(metric_value) as min_value,
    COUNT(*) as data_points,
    MAX(timestamp) as last_updated
FROM monitoring.system_metrics
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY metric_name;

-- Insert sample alert rules
INSERT INTO alerts.alert_rules (rule_name, rule_description, condition_expression, severity, category, notification_channels) VALUES
('High Outbreak Risk', 'Alert when outbreak risk score exceeds threshold', 'outbreak_risk_score > 7', 'warning', 'outbreak', ARRAY['email', 'slack']),
('Critical Outbreak Risk', 'Alert when outbreak risk score is critical', 'outbreak_risk_score > 8.5', 'critical', 'outbreak', ARRAY['email', 'slack', 'sms']),
('High Social Media Activity', 'Alert when social media posts exceed threshold', 'post_count > 50', 'warning', 'social_media', ARRAY['email', 'slack']),
('High Hospital Admissions', 'Alert when hospital admissions exceed threshold', 'admission_count > 15', 'warning', 'hospital', ARRAY['email', 'slack']),
('Favorable Weather Conditions', 'Alert when weather conditions favor disease spread', 'avg_temperature > 30 AND avg_humidity > 80', 'info', 'weather', ARRAY['email']);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_alerts_alert_rules_updated_at BEFORE UPDATE ON alerts.alert_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_alerts_active_alerts_updated_at BEFORE UPDATE ON alerts.active_alerts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA health_data TO admin;
GRANT USAGE ON SCHEMA ml_models TO admin;
GRANT USAGE ON SCHEMA monitoring TO admin;
GRANT USAGE ON SCHEMA alerts TO admin;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA health_data TO admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_models TO admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA alerts TO admin;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA health_data TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_models TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA alerts TO admin;

-- Create a function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Clean up old social media posts (keep last 30 days)
    DELETE FROM health_data.social_media_posts WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Clean up old hospital logs (keep last 90 days)
    DELETE FROM health_data.hospital_logs WHERE admission_date < NOW() - INTERVAL '90 days';
    
    -- Clean up old weather data (keep last 60 days)
    DELETE FROM health_data.weather_data WHERE timestamp < NOW() - INTERVAL '60 days';
    
    -- Clean up old predictions (keep last 30 days)
    DELETE FROM ml_models.predictions WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Clean up old system metrics (keep last 7 days)
    DELETE FROM monitoring.system_metrics WHERE timestamp < NOW() - INTERVAL '7 days';
    
    -- Clean up old API metrics (keep last 7 days)
    DELETE FROM monitoring.api_metrics WHERE timestamp < NOW() - INTERVAL '7 days';
    
    -- Clean up old pipeline metrics (keep last 30 days)
    DELETE FROM monitoring.pipeline_metrics WHERE timestamp < NOW() - INTERVAL '30 days';
    
    -- Clean up resolved alerts older than 90 days
    DELETE FROM alerts.active_alerts WHERE status = 'resolved' AND resolved_at < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- Create a scheduled job to run cleanup (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');

COMMENT ON DATABASE disease_mlops IS 'Database for Disease Outbreak Early Warning System';
COMMENT ON SCHEMA health_data IS 'Schema for health-related data including social media, hospital logs, and weather data';
COMMENT ON SCHEMA ml_models IS 'Schema for machine learning models and predictions';
COMMENT ON SCHEMA monitoring IS 'Schema for system monitoring and metrics';
COMMENT ON SCHEMA alerts IS 'Schema for alerting and notification system';
