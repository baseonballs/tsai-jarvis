#!/bin/bash

# TSAI Jarvis - Advanced Analytics API Startup Script
# Phase 2.7: Advanced Analytics Implementation

echo "📊 Starting TSAI Jarvis Advanced Analytics API..."
echo "Phase 2.7: Advanced Analytics"
echo "============================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Additional advanced analytics requirements
echo "Installing advanced analytics requirements..."
pip install scipy matplotlib seaborn plotly scikit-learn

# Set environment variables
export ADVANCED_ANALYTICS_PORT=8011
export ADVANCED_ANALYTICS_HOST=0.0.0.0
export ADVANCED_ANALYTICS_LOG_LEVEL=info

# Create database directory if it doesn't exist
mkdir -p data/advanced_analytics

# Start the Advanced Analytics API
echo "Starting Advanced Analytics API on port 8011..."
echo "API Documentation: http://localhost:8011/docs"
echo "WebSocket: ws://localhost:8011/ws/analytics"
echo ""

python hockey-analytics/advanced_analytics_api.py
