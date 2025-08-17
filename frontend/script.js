// Disease Outbreak Early Warning System - Frontend JavaScript
class DiseaseOutbreakDashboard {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.currentSection = 'dashboard';
        this.charts = {};
        this.map = null;
        this.refreshInterval = null;
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupEventListeners();
        this.initializeCharts();
        this.initializeMap();
        this.loadDashboardData();
        this.startAutoRefresh();
    }

    // Navigation Setup
    setupNavigation() {
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');
        const navLinks = document.querySelectorAll('.nav-link');

        // Mobile menu toggle
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('active');
        });

        // Navigation links
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetSection = link.getAttribute('href').substring(1);
                this.navigateToSection(targetSection);
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.navbar')) {
                navMenu.classList.remove('active');
            }
        });
    }

    navigateToSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });

        // Remove active class from all nav links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        // Show target section
        const targetSection = document.getElementById(sectionName);
        if (targetSection) {
            targetSection.classList.add('active');
        }

        // Update active nav link
        const activeLink = document.querySelector(`[href="#${sectionName}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }

        this.currentSection = sectionName;

        // Load section-specific data
        switch (sectionName) {
            case 'dashboard':
                this.loadDashboardData();
                break;
            case 'alerts':
                this.loadAlerts();
                break;
            case 'analytics':
                this.loadAnalyticsData();
                break;
        }
    }

    // Event Listeners
    setupEventListeners() {
        // Prediction form submission
        const predictionForm = document.getElementById('prediction-form');
        if (predictionForm) {
            predictionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitPrediction();
            });
        }

        // Time range selector for analytics
        const timeRangeSelector = document.getElementById('time-range');
        if (timeRangeSelector) {
            timeRangeSelector.addEventListener('change', () => {
                this.loadAnalyticsData();
            });
        }

        // Filter selectors for alerts
        const severityFilter = document.getElementById('severity-filter');
        const regionFilter = document.getElementById('region-filter');
        
        if (severityFilter) {
            severityFilter.addEventListener('change', () => {
                this.filterAlerts();
            });
        }
        
        if (regionFilter) {
            regionFilter.addEventListener('change', () => {
                this.filterAlerts();
            });
        }
    }

    // Charts Initialization
    initializeCharts() {
        this.initializeRiskTrendsChart();
        this.initializeDiseaseDistributionChart();
        this.initializeHistoricalChart();
        this.initializeWeatherCorrelationChart();
        this.initializeSentimentChart();
    }

    initializeRiskTrendsChart() {
        const ctx = document.getElementById('risk-trends-chart');
        if (!ctx) return;

        this.charts.riskTrends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Risk Score',
                    data: [],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 10,
                        ticks: {
                            stepSize: 2
                        }
                    }
                }
            }
        });
    }

    initializeDiseaseDistributionChart() {
        const ctx = document.getElementById('disease-distribution-chart');
        if (!ctx) return;

        this.charts.diseaseDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Dengue', 'Malaria', 'Influenza', 'COVID-19', 'Other'],
                datasets: [{
                    data: [30, 25, 20, 15, 10],
                    backgroundColor: [
                        '#ef4444',
                        '#f59e0b',
                        '#3b82f6',
                        '#10b981',
                        '#8b5cf6'
                    ],
                    borderWidth: 2,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    initializeHistoricalChart() {
        const ctx = document.getElementById('historical-chart');
        if (!ctx) return;

        this.charts.historical = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Outbreak Cases',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'Risk Score',
                    data: [],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    tension: 0.4,
                    fill: false,
                    yAxisID: 'y1'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        beginAtZero: true
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        beginAtZero: true,
                        max: 10,
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }

    initializeWeatherCorrelationChart() {
        const ctx = document.getElementById('weather-correlation-chart');
        if (!ctx) return;

        this.charts.weatherCorrelation = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Temperature vs Risk',
                    data: [],
                    backgroundColor: 'rgba(37, 99, 235, 0.6)',
                    borderColor: '#2563eb'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Temperature (°C)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Risk Score'
                        },
                        beginAtZero: true,
                        max: 10
                    }
                }
            }
        });
    }

    initializeSentimentChart() {
        const ctx = document.getElementById('sentiment-chart');
        if (!ctx) return;

        this.charts.sentiment = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'],
                datasets: [{
                    label: 'Sentiment Distribution',
                    data: [15, 25, 35, 20, 5],
                    backgroundColor: [
                        '#ef4444',
                        '#f59e0b',
                        '#64748b',
                        '#10b981',
                        '#059669'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    // Map Initialization
    initializeMap() {
        const mapContainer = document.getElementById('risk-map');
        if (!mapContainer) return;

        // Initialize Leaflet map
        this.map = L.map('risk-map').setView([20.5937, 78.9629], 5); // India coordinates

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);

        // Add sample risk markers
        this.addSampleRiskMarkers();
    }

    addSampleRiskMarkers() {
        const sampleData = [
            { lat: 19.0760, lng: 72.8777, city: 'Mumbai', risk: 8.5, disease: 'Dengue' },
            { lat: 28.7041, lng: 77.1025, city: 'Delhi', risk: 7.2, disease: 'Malaria' },
            { lat: 12.9716, lng: 77.5946, city: 'Bangalore', risk: 6.8, disease: 'Influenza' },
            { lat: 22.5726, lng: 88.3639, city: 'Kolkata', risk: 9.1, disease: 'Dengue' },
            { lat: 13.0827, lng: 80.2707, city: 'Chennai', risk: 5.4, disease: 'Malaria' }
        ];

        sampleData.forEach(location => {
            const riskColor = this.getRiskColor(location.risk);
            const marker = L.circleMarker([location.lat, location.lng], {
                radius: 15,
                fillColor: riskColor,
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);

            const popupContent = `
                <div style="text-align: center;">
                    <h4 style="margin: 0 0 10px 0; color: #1e293b;">${location.city}</h4>
                    <p style="margin: 5px 0; font-weight: 600;">Risk Score: <span style="color: ${riskColor};">${location.risk}</span></p>
                    <p style="margin: 5px 0;">Disease: ${location.disease}</p>
                </div>
            `;
            marker.bindPopup(popupContent);
        });
    }

    getRiskColor(risk) {
        if (risk >= 8) return '#ef4444'; // High risk - red
        if (risk >= 6) return '#f59e0b'; // Medium risk - orange
        return '#10b981'; // Low risk - green
    }

    // Data Loading Functions
    async loadDashboardData() {
        try {
            this.showLoading();
            
            // Load metrics
            await this.loadMetrics();
            
            // Load recent alerts
            await this.loadRecentAlerts();
            
            // Update charts with real data
            this.updateChartsWithRealData();
            
            // Update last update time
            this.updateLastUpdateTime();
            
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showToast('Error loading dashboard data', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadMetrics() {
        try {
            // Simulate API call for metrics
            const metrics = await this.fetchMetrics();
            
            // Update metric cards
            document.getElementById('active-alerts-count').textContent = metrics.activeAlerts;
            document.getElementById('high-risk-areas').textContent = metrics.highRiskAreas;
            document.getElementById('avg-risk-score').textContent = metrics.avgRiskScore.toFixed(1);
            
        } catch (error) {
            console.error('Error loading metrics:', error);
        }
    }

    async loadRecentAlerts() {
        try {
            const alerts = await this.fetchRecentAlerts();
            this.renderRecentAlerts(alerts);
        } catch (error) {
            console.error('Error loading recent alerts:', error);
        }
    }

    async loadAlerts() {
        try {
            this.showLoading();
            const alerts = await this.fetchAllAlerts();
            this.renderAlerts(alerts);
        } catch (error) {
            console.error('Error loading alerts:', error);
            this.showToast('Error loading alerts', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadAnalyticsData() {
        try {
            this.showLoading();
            const timeRange = document.getElementById('time-range')?.value || 30;
            const analyticsData = await this.fetchAnalyticsData(timeRange);
            this.updateAnalyticsCharts(analyticsData);
        } catch (error) {
            console.error('Error loading analytics data:', error);
            this.showToast('Error loading analytics data', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // API Functions
    async fetchMetrics() {
        // Simulate API call
        return new Promise(resolve => {
            setTimeout(() => {
                resolve({
                    activeAlerts: Math.floor(Math.random() * 10) + 5,
                    highRiskAreas: Math.floor(Math.random() * 8) + 3,
                    avgRiskScore: (Math.random() * 4) + 3
                });
            }, 500);
        });
    }

    async fetchRecentAlerts() {
        // Simulate API call
        return new Promise(resolve => {
            setTimeout(() => {
                resolve([
                    {
                        id: 1,
                        title: 'High Dengue Risk Detected',
                        severity: 'high',
                        city: 'Mumbai',
                        region: 'Maharashtra',
                        description: 'Elevated dengue risk due to increased mosquito breeding conditions',
                        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
                    },
                    {
                        id: 2,
                        title: 'Malaria Outbreak Warning',
                        severity: 'critical',
                        city: 'Kolkata',
                        region: 'West Bengal',
                        description: 'Critical malaria risk detected in multiple districts',
                        timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString()
                    }
                ]);
            }, 300);
        });
    }

    async fetchAllAlerts() {
        // Simulate API call
        return new Promise(resolve => {
            setTimeout(() => {
                resolve([
                    {
                        id: 1,
                        title: 'High Dengue Risk Detected',
                        severity: 'high',
                        city: 'Mumbai',
                        region: 'Maharashtra',
                        description: 'Elevated dengue risk due to increased mosquito breeding conditions',
                        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
                    },
                    {
                        id: 2,
                        title: 'Malaria Outbreak Warning',
                        severity: 'critical',
                        city: 'Kolkata',
                        region: 'West Bengal',
                        description: 'Critical malaria risk detected in multiple districts',
                        timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString()
                    },
                    {
                        id: 3,
                        title: 'Influenza Season Alert',
                        severity: 'medium',
                        city: 'Delhi',
                        region: 'Delhi',
                        description: 'Moderate influenza risk as seasonal patterns emerge',
                        timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString()
                    },
                    {
                        id: 4,
                        title: 'Low Risk Monitoring',
                        severity: 'low',
                        city: 'Chennai',
                        region: 'Tamil Nadu',
                        description: 'Low disease risk, continue routine monitoring',
                        timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString()
                    }
                ]);
            }, 500);
        });
    }

    async fetchAnalyticsData(timeRange) {
        // Simulate API call
        return new Promise(resolve => {
            setTimeout(() => {
                const days = parseInt(timeRange);
                const labels = [];
                const cases = [];
                const risks = [];
                const weatherData = [];
                
                for (let i = days - 1; i >= 0; i--) {
                    const date = new Date();
                    date.setDate(date.getDate() - i);
                    labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
                    
                    cases.push(Math.floor(Math.random() * 100) + 20);
                    risks.push((Math.random() * 4) + 3);
                    weatherData.push({
                        temp: (Math.random() * 20) + 15,
                        risk: (Math.random() * 4) + 3
                    });
                }
                
                resolve({
                    labels,
                    cases,
                    risks,
                    weatherData
                });
            }, 600);
        });
    }

    // Rendering Functions
    renderRecentAlerts(alerts) {
        const container = document.getElementById('recent-alerts');
        if (!container) return;

        container.innerHTML = alerts.map(alert => `
            <div class="alert-item">
                <div class="alert-header">
                    <div>
                        <h4 class="alert-title">${alert.title}</h4>
                        <div class="alert-meta">
                            <span>${alert.city}, ${alert.region}</span>
                            <span>${this.formatTimestamp(alert.timestamp)}</span>
                        </div>
                    </div>
                    <span class="alert-severity ${alert.severity}">${alert.severity.toUpperCase()}</span>
                </div>
                <p class="alert-description">${alert.description}</p>
            </div>
        `).join('');
    }

    renderAlerts(alerts) {
        const container = document.getElementById('alerts-grid');
        if (!container) return;

        container.innerHTML = alerts.map(alert => `
            <div class="alert-item">
                <div class="alert-header">
                    <div>
                        <h4 class="alert-title">${alert.title}</h4>
                        <div class="alert-meta">
                            <span>${alert.city}, ${alert.region}</span>
                            <span>${this.formatTimestamp(alert.timestamp)}</span>
                        </div>
                    </div>
                    <span class="alert-severity ${alert.severity}">${alert.severity.toUpperCase()}</span>
                </div>
                <p class="alert-description">${alert.description}</p>
            </div>
        `).join('');
    }

    updateChartsWithRealData() {
        // Update risk trends chart with sample data
        if (this.charts.riskTrends) {
            const labels = [];
            const data = [];
            
            for (let i = 6; i >= 0; i--) {
                const date = new Date();
                date.setDate(date.getDate() - i);
                labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
                data.push((Math.random() * 4) + 3);
            }
            
            this.charts.riskTrends.data.labels = labels;
            this.charts.riskTrends.data.datasets[0].data = data;
            this.charts.riskTrends.update();
        }
    }

    updateAnalyticsCharts(data) {
        // Update historical chart
        if (this.charts.historical) {
            this.charts.historical.data.labels = data.labels;
            this.charts.historical.data.datasets[0].data = data.cases;
            this.charts.historical.data.datasets[1].data = data.risks;
            this.charts.historical.update();
        }

        // Update weather correlation chart
        if (this.charts.weatherCorrelation) {
            const weatherData = data.weatherData.map(w => ({
                x: w.temp,
                y: w.risk
            }));
            this.charts.weatherCorrelation.data.datasets[0].data = weatherData;
            this.charts.weatherCorrelation.update();
        }
    }

    // Prediction Functions
    async submitPrediction() {
        try {
            this.showLoading();
            
            const formData = new FormData(document.getElementById('prediction-form'));
            const predictionData = Object.fromEntries(formData.entries());
            
            // Simulate API call
            const prediction = await this.makePrediction(predictionData);
            
            // Display results
            this.displayPredictionResults(prediction);
            
            this.showToast('Prediction completed successfully', 'success');
            
        } catch (error) {
            console.error('Error making prediction:', error);
            this.showToast('Error making prediction', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async makePrediction(data) {
        // Simulate API call
        return new Promise(resolve => {
            setTimeout(() => {
                const riskScore = (Math.random() * 6) + 2; // 2-8 range
                let riskLevel = 'low';
                if (riskScore >= 6) riskLevel = 'high';
                else if (riskScore >= 4) riskLevel = 'medium';
                
                resolve({
                    riskScore: riskScore.toFixed(1),
                    riskLevel,
                    recommendations: [
                        'Monitor local health reports closely',
                        'Implement mosquito control measures',
                        'Prepare healthcare resources',
                        'Increase public awareness campaigns'
                    ]
                });
            }, 1000);
        });
    }

    displayPredictionResults(prediction) {
        const container = document.getElementById('prediction-results');
        if (!container) return;

        container.innerHTML = `
            <div class="prediction-result ${prediction.riskLevel}-risk">
                <div class="risk-score">${prediction.riskScore}</div>
                <span class="risk-level ${prediction.riskLevel}">${prediction.riskLevel.toUpperCase()} RISK</span>
                <div class="recommendations">
                    <h4>Recommendations:</h4>
                    <ul>
                        ${prediction.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
    }

    // Utility Functions
    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / (1000 * 60));
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

        if (diffMins < 60) return `${diffMins} minutes ago`;
        if (diffHours < 24) return `${diffHours} hours ago`;
        return `${diffDays} days ago`;
    }

    updateLastUpdateTime() {
        const lastUpdateElement = document.getElementById('last-update');
        if (lastUpdateElement) {
            lastUpdateElement.textContent = 'Just now';
        }
    }

    filterAlerts() {
        // Implementation for filtering alerts based on severity and region
        console.log('Filtering alerts...');
    }

    // Loading and Toast Functions
    showLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('show');
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('show');
        }
    }

    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        if (!toast) return;

        toast.textContent = message;
        toast.className = `toast ${type} show`;

        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }

    // Auto-refresh functionality
    startAutoRefresh() {
        this.refreshInterval = setInterval(() => {
            if (this.currentSection === 'dashboard') {
                this.loadDashboardData();
            }
        }, 30000); // Refresh every 30 seconds
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }

    // Public methods for external calls
    refreshMap() {
        if (this.map) {
            this.map.invalidateSize();
            this.showToast('Map refreshed', 'success');
        }
    }

    refreshAlerts() {
        this.loadAlerts();
        this.showToast('Alerts refreshed', 'success');
    }
}

// Global functions for onclick handlers
function refreshMap() {
    if (window.dashboard) {
        window.dashboard.refreshMap();
    }
}

function refreshAlerts() {
    if (window.dashboard) {
        window.dashboard.refreshAlerts();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DiseaseOutbreakDashboard();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        if (window.dashboard) {
            window.dashboard.stopAutoRefresh();
        }
    } else {
        if (window.dashboard) {
            window.dashboard.startAutoRefresh();
        }
    }
});

// Handle window resize for responsive charts
window.addEventListener('resize', () => {
    if (window.dashboard && window.dashboard.charts) {
        Object.values(window.dashboard.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
});
