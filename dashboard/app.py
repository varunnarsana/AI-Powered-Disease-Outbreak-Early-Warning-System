#!/usr/bin/env python3
"""
Streamlit Dashboard for Disease Outbreak Early Warning System
Provides real-time visualization and monitoring of outbreak risks
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional
import altair as alt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DashboardConfig:
    """Configuration for the dashboard"""
    
    def __init__(self):
        # API Configuration
        self.API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
        
        # Dashboard Configuration
        self.REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "30"))  # seconds
        self.THEME = os.getenv("DASHBOARD_THEME", "light")
        self.DEFAULT_CITY = os.getenv("DEFAULT_CITY", "New York")
        self.DEFAULT_REGION = os.getenv("DEFAULT_REGION", "New York")
        
        # Authentication (if needed)
        self.API_KEY = os.getenv("API_KEY", "")
        
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if self.API_KEY:
            headers["Authorization"] = f"Bearer {self.API_KEY}"
        return headers

# Initialize config
config = DashboardConfig()

# Configure Streamlit page
st.set_page_config(
    page_title="Disease Outbreak Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-medium {
        background-color: #fff8e1;
        border-left: 4px solid #ffc107;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class DiseaseOutbreakDashboard:
    """Main dashboard class for disease outbreak monitoring"""
    
    def __init__(self):
        self.config = config
        self.api_base_url = config.API_BASE_URL
        self.refresh_interval = config.REFRESH_INTERVAL
        
    def get_api_data(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Fetch data from API endpoint with improved error handling
        
        Args:
            endpoint: API endpoint path (e.g., "/api/risk")
            params: Optional query parameters
            
        Returns:
            Dict containing the API response data or error information
        """
        url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self.config.get_headers()
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=self.config.API_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching data from {url}: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg += f"\nDetails: {error_details.get('detail', 'No details')}"
                except:
                    error_msg += f"\nStatus code: {e.response.status_code}"
            
            st.error(error_msg)
            logging.error(error_msg, exc_info=True)
            return {"error": error_msg, "status": "error"}
    
    def create_sample_data(self) -> Dict[str, Any]:
        """Create sample data for demonstration when API is not available"""
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
        regions = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal', 'Telangana', 'Maharashtra', 'Gujarat']
        
        # Generate sample risk map data
        risk_data = []
        for i, city in enumerate(cities):
            risk_score = np.random.uniform(2, 9)
            risk_level = "low" if risk_score < 4 else "medium" if risk_score < 6 else "high" if risk_score < 8 else "critical"
            
            risk_data.append({
                "city": city,
                "region": regions[i],
                "risk_score": round(risk_score, 2),
                "risk_level": risk_level,
                "timestamp": datetime.now().isoformat()
            })
        
        # Generate sample trend data
        trends = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        current_date = start_date
        while current_date <= end_date:
            trends.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "risk_score": np.random.uniform(2, 8),
                "post_count": np.random.poisson(15),
                "admission_count": np.random.poi(5),
                "temperature": np.random.uniform(20, 35)
            })
            current_date += timedelta(days=1)
        
        return {
            "risk_map_data": risk_data,
            "trends": trends
        }
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🏥 Disease Outbreak Early Warning System</h1>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_metrics_overview(self, risk_data: List[Dict[str, Any]]):
        """Render key metrics overview"""
        st.subheader("📊 System Overview")
        
        if not risk_data:
            return
        
        # Calculate metrics
        total_cities = len(risk_data)
        critical_risk = len([r for r in risk_data if r['risk_level'] == 'critical'])
        high_risk = len([r for r in risk_data if r['risk_level'] == 'high'])
        avg_risk_score = np.mean([r['risk_score'] for r in risk_data])
        
        # Create metric columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cities Monitored", total_cities)
        
        with col2:
            st.metric("Critical Risk Areas", critical_risk, delta=critical_risk)
        
        with col3:
            st.metric("High Risk Areas", high_risk, delta=high_risk)
        
        with col4:
            st.metric("Average Risk Score", f"{avg_risk_score:.2f}")
    
    def render_risk_map(self, risk_data: List[Dict[str, Any]]):
        """Render interactive risk map"""
        st.subheader("🗺️ Outbreak Risk Map")
        
        if not risk_data:
            st.warning("No risk data available")
            return
        
        # Create DataFrame for visualization
        df = pd.DataFrame(risk_data)
        
        # Color mapping for risk levels
        color_map = {
            'low': '#4caf50',
            'medium': '#ffc107',
            'high': '#ff9800',
            'critical': '#f44336'
        }
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='city',
            y='risk_score',
            color='risk_level',
            size='risk_score',
            hover_data=['region', 'timestamp'],
            color_discrete_map=color_map,
            title="Disease Outbreak Risk by City"
        )
        
        fig.update_layout(
            xaxis_title="City",
            yaxis_title="Risk Score",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk level breakdown
        st.subheader("Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_map=color_map
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Level Counts",
                color=risk_counts.index,
                color_discrete_map=color_map
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def render_trends_analysis(self, trends_data: List[Dict[str, Any]]):
        """Render trends analysis charts"""
        st.subheader("📈 Trends Analysis")
        
        if not trends_data:
            st.warning("No trends data available")
            return
        
        # Create DataFrame
        df_trends = pd.DataFrame(trends_data)
        df_trends['date'] = pd.to_datetime(df_trends['date'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score Trend', 'Social Media Posts', 'Hospital Admissions', 'Temperature Trend'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Risk Score Trend
        fig.add_trace(
            go.Scatter(x=df_trends['date'], y=df_trends['risk_score'], 
                      mode='lines+markers', name='Risk Score'),
            row=1, col=1
        )
        
        # Social Media Posts
        fig.add_trace(
            go.Scatter(x=df_trends['date'], y=df_trends['post_count'], 
                      mode='lines+markers', name='Posts'),
            row=1, col=2
        )
        
        # Hospital Admissions
        fig.add_trace(
            go.Scatter(x=df_trends['date'], y=df_trends['admission_count'], 
                      mode='lines+markers', name='Admissions'),
            row=2, col=1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(x=df_trends['date'], y=df_trends['temperature'], 
                      mode='lines+markers', name='Temperature'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_panel(self):
        """Render alerts and notifications panel"""
        st.subheader("🚨 Active Alerts")
        
        # Try to get alerts from API
        alerts_data = self.get_api_data("/alerts")
        
        if not alerts_data:
            # Create sample alerts
            sample_alerts = [
                {
                    "alert_id": "alert_001",
                    "city": "Mumbai",
                    "region": "Maharashtra",
                    "risk_level": "critical",
                    "message": "High dengue cases detected in western suburbs",
                    "timestamp": datetime.now().isoformat(),
                    "status": "active"
                },
                {
                    "alert_id": "alert_002",
                    "city": "Delhi",
                    "region": "Delhi",
                    "risk_level": "high",
                    "message": "Increasing respiratory illness reports",
                    "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "status": "active"
                }
            ]
            alerts_data = sample_alerts
        
        # Display alerts
        for alert in alerts_data:
            alert_class = f"alert-{alert['risk_level']}"
            
            with st.container():
                st.markdown(f"""
                <div class="metric-card {alert_class}">
                    <h4>🚨 {alert['risk_level'].upper()} RISK - {alert['city']}, {alert['region']}</h4>
                    <p><strong>Message:</strong> {alert['message']}</p>
                    <p><strong>Time:</strong> {alert['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
    
    def render_prediction_form(self):
        """Render prediction form for manual risk assessment"""
        st.subheader("🔮 Manual Risk Prediction")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                city = st.text_input("City", "Mumbai")
                region = st.text_input("Region/State", "Maharashtra")
                post_count = st.number_input("Social Media Posts", min_value=0, value=15)
                avg_risk_score = st.slider("Average Risk Score", 1.0, 5.0, 2.5)
                avg_engagement = st.slider("Average Engagement", 1.0, 5.0, 2.5)
                unique_users = st.number_input("Unique Users", min_value=0, value=8)
            
            with col2:
                admission_count = st.number_input("Hospital Admissions", min_value=0, value=5)
                avg_severity = st.slider("Average Severity", 1.0, 5.0, 2.5)
                avg_length_of_stay = st.number_input("Avg Length of Stay (days)", min_value=0, value=7)
                disease_variety = st.number_input("Disease Variety", min_value=1, value=3)
                avg_temperature = st.number_input("Temperature (°C)", min_value=0, value=30)
                avg_humidity = st.slider("Humidity (%)", 0, 100, 70)
                avg_rainfall = st.number_input("Rainfall (mm)", min_value=0, value=20)
                mosquito_risk_level = st.selectbox("Mosquito Risk Level", ["low", "medium", "high"])
            
            submitted = st.form_submit_button("Predict Risk")
            
            if submitted:
                # Create prediction request
                prediction_data = {
                    "city": city,
                    "region": region,
                    "post_count": post_count,
                    "avg_risk_score": avg_risk_score,
                    "avg_engagement": avg_engagement,
                    "unique_users": unique_users,
                    "admission_count": admission_count,
                    "avg_severity": avg_severity,
                    "avg_length_of_stay": avg_length_of_stay,
                    "disease_variety": disease_variety,
                    "avg_temperature": avg_temperature,
                    "avg_humidity": avg_humidity,
                    "avg_rainfall": avg_rainfall,
                    "mosquito_risk_level": mosquito_risk_level
                }
                
                # Try to get prediction from API
                try:
                    response = requests.post(
                        f"{self.api_base_url}/predict",
                        json=prediction_data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display prediction result
                        st.success("Prediction completed!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Risk Score", f"{result['outbreak_risk_score']:.2f}")
                        with col2:
                            st.metric("Risk Level", result['risk_level'].upper())
                        with col3:
                            st.metric("Confidence", f"{result['confidence']:.2f}")
                        
                        # Display recommendations
                        st.subheader("Recommendations")
                        for rec in result['recommendations']:
                            st.write(f"• {rec}")
                    
                    else:
                        st.error("API prediction failed")
                        
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    
    def render_system_status(self):
        """Render system status and health metrics"""
        st.subheader("⚙️ System Status")
        
        # Check API health
        try:
            health_response = requests.get(f"{self.api_base_url}/health", timeout=5)
            api_status = "🟢 Healthy" if health_response.status_code == 200 else "🔴 Unhealthy"
        except:
            api_status = "🔴 Unreachable"
        
        # Check MLflow
        try:
            mlflow_response = requests.get("http://localhost:5000", timeout=5)
            mlflow_status = "🟢 Active" if mlflow_response.status_code == 200 else "🔴 Inactive"
        except:
            mlflow_status = "🔴 Unreachable"
        
        # Display status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Status", api_status)
        
        with col2:
            st.metric("MLflow Status", mlflow_status)
        
        with col3:
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    
    def run_dashboard(self):
        """Run the main dashboard"""
        self.render_header()
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Overview", "Risk Map", "Trends", "Alerts", "Prediction", "System Status"]
        )
        
        # Get data
        try:
            risk_data = self.get_api_data("/dashboard/risk-map")
            if risk_data and "risk_map_data" in risk_data:
                risk_data = risk_data["risk_map_data"]
            else:
                risk_data = []
            
            trends_data = self.get_api_data("/dashboard/trends")
            if trends_data and "trends" in trends_data:
                trends_data = trends_data["trends"]
            else:
                trends_data = []
                
        except:
            # Use sample data if API is not available
            sample_data = self.create_sample_data()
            risk_data = sample_data["risk_map_data"]
            trends_data = sample_data["trends"]
        
        # Render selected page
        if page == "Overview":
            self.render_metrics_overview(risk_data)
            self.render_risk_map(risk_data)
            
        elif page == "Risk Map":
            self.render_risk_map(risk_data)
            
        elif page == "Trends":
            self.render_trends_analysis(trends_data)
            
        elif page == "Alerts":
            self.render_alerts_panel()
            
        elif page == "Prediction":
            self.render_prediction_form()
            
        elif page == "System Status":
            self.render_system_status()
        
        # Auto-refresh
        if st.button("🔄 Refresh Data"):
            st.rerun()

def main():
    """Main function to run the dashboard"""
    dashboard = DiseaseOutbreakDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
