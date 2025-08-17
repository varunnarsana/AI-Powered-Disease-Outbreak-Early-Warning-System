# 🌐 Disease Outbreak Early Warning System - Frontend

A modern, responsive web interface for the AI-powered disease outbreak early warning system. Built with vanilla HTML, CSS, and JavaScript for maximum compatibility and performance.

## ✨ Features

### 🎯 **Dashboard**
- **Real-time Metrics**: Active alerts, high-risk areas, average risk scores, and last update time
- **Interactive Risk Map**: Geographic visualization of outbreak risks using Leaflet maps
- **Trend Charts**: Risk trends and disease distribution using Chart.js
- **Recent Alerts**: Live feed of the latest disease outbreak alerts

### 🔮 **Predictions**
- **Risk Assessment Form**: Input city, region, weather conditions, and health metrics
- **AI-Powered Analysis**: Get outbreak risk predictions with actionable recommendations
- **Real-time Results**: Instant feedback with risk levels and prevention strategies

### 🚨 **Alerts & Notifications**
- **Comprehensive Alert Management**: View all active alerts with filtering options
- **Severity Classification**: Color-coded alerts (Critical, High, Medium, Low)
- **Geographic Context**: City and region information for each alert
- **Real-time Updates**: Automatic refresh and live data

### 📊 **Advanced Analytics**
- **Historical Analysis**: Long-term outbreak patterns and trends
- **Weather Correlation**: Temperature and risk score relationships
- **Social Media Sentiment**: Public health sentiment analysis
- **Customizable Time Ranges**: 7 days to 1 year analysis periods

### ℹ️ **About & Technology**
- **System Overview**: Learn about the AI-powered detection capabilities
- **Technology Stack**: Python, Kafka, Spark, MLflow, FastAPI, Prometheus
- **Real-time Processing**: Streaming data pipeline information
- **Early Warning Benefits**: 2-3 week advance detection capabilities

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ (for the simple server)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Backend API running on `http://localhost:8000`

### Running the Frontend

#### Option 1: Python HTTP Server (Recommended)
```bash
cd frontend
python server.py
```

Then open your browser and navigate to: **http://localhost:3000**

#### Option 2: Any HTTP Server
You can use any HTTP server to serve the files:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 3000

# Using Node.js (if you have it installed)
cd frontend
npx http-server -p 3000

# Using PHP (if you have it installed)
cd frontend
php -S localhost:3000
```

#### Option 3: Direct File Opening
Simply open `index.html` in your browser (some features may not work due to CORS restrictions).

## 🔧 Configuration

### API Endpoint
The frontend is configured to connect to the backend API at `http://localhost:8000`. To change this:

1. Edit `script.js`
2. Modify the `apiBaseUrl` property in the `DiseaseOutbreakDashboard` class:
   ```javascript
   this.apiBaseUrl = 'http://your-api-url:port';
   ```

### Port Configuration
To change the frontend server port:

1. Edit `server.py`
2. Modify the `PORT` variable:
   ```python
   PORT = 8080  # or any port you prefer
   ```

## 📱 Responsive Design

The website is fully responsive and works on:
- **Desktop**: Full-featured dashboard with side-by-side layouts
- **Tablet**: Optimized layouts for medium screens
- **Mobile**: Touch-friendly interface with collapsible navigation

## 🎨 Customization

### Colors and Themes
The website uses CSS custom properties (variables) for easy theming:

```css
:root {
    --primary-color: #2563eb;      /* Main brand color */
    --secondary-color: #64748b;    /* Secondary text */
    --success-color: #10b981;      /* Success states */
    --warning-color: #f59e0b;      /* Warning states */
    --danger-color: #ef4444;       /* Error states */
}
```

### Charts and Visualizations
All charts are built with Chart.js and can be customized:

```javascript
// Example: Customizing chart colors
this.charts.riskTrends = new Chart(ctx, {
    data: {
        datasets: [{
            borderColor: '#your-color',
            backgroundColor: 'rgba(your-color, 0.1)'
        }]
    }
});
```

## 🔌 Integration with Backend

### API Endpoints Used
The frontend expects these backend endpoints:

- `GET /health` - Health check
- `GET /dashboard/risk-map` - Risk map data
- `GET /dashboard/trends` - Trend analysis data
- `POST /predict` - Risk prediction
- `GET /alerts` - Active alerts
- `GET /metrics` - System metrics

### Data Format
The frontend expects JSON responses with specific structures. See the JavaScript code for expected data formats.

## 🧪 Development

### File Structure
```
frontend/
├── index.html          # Main HTML file
├── styles.css          # All CSS styles
├── script.js           # JavaScript functionality
├── server.py           # Development server
└── README.md           # This file
```

### Adding New Features
1. **New Section**: Add HTML in `index.html`, CSS in `styles.css`, and JavaScript in `script.js`
2. **New Charts**: Use Chart.js library and add to the `initializeCharts()` method
3. **New API Calls**: Add methods to the `DiseaseOutbreakDashboard` class

### Testing
- Test on multiple browsers (Chrome, Firefox, Safari, Edge)
- Test responsive design on different screen sizes
- Test with backend API running and stopped
- Test form validation and error handling

## 🚨 Troubleshooting

### Common Issues

#### **Charts Not Displaying**
- Check browser console for JavaScript errors
- Ensure Chart.js library is loaded
- Verify canvas elements exist in HTML

#### **Map Not Loading**
- Check if Leaflet library is loaded
- Verify internet connection (map tiles are external)
- Check browser console for CORS errors

#### **API Connection Issues**
- Ensure backend is running on `http://localhost:8000`
- Check browser console for network errors
- Verify CORS headers are properly set

#### **Styling Issues**
- Clear browser cache
- Check CSS file path
- Verify CSS custom properties are supported

### Debug Mode
Enable debug logging by opening browser console and checking for:
- JavaScript errors
- Network request failures
- Chart.js warnings
- Leaflet map errors

## 🌟 Browser Support

- **Chrome**: 80+ (Full support)
- **Firefox**: 75+ (Full support)
- **Safari**: 13+ (Full support)
- **Edge**: 80+ (Full support)
- **Mobile browsers**: iOS Safari 13+, Chrome Mobile 80+

## 📈 Performance

- **Lightweight**: No heavy frameworks, fast loading
- **Optimized**: Efficient DOM manipulation and event handling
- **Caching**: Automatic data refresh with intelligent caching
- **Responsive**: Smooth animations and transitions

## 🔒 Security

- **CORS Enabled**: Cross-origin requests allowed for development
- **Input Validation**: Form validation on both client and server side
- **XSS Protection**: Sanitized HTML rendering
- **HTTPS Ready**: Configured for secure connections

## 🤝 Contributing

To contribute to the frontend:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Disease Outbreak Early Warning System and follows the same license terms.

---

**🎯 Ready to monitor disease outbreaks? Start the frontend server and open your browser!**
