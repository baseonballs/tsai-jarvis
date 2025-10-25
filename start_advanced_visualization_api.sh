#!/bin/bash

# TSAI Jarvis - Advanced Visualization API Startup Script
# Phase 2.4: Advanced Visualization Implementation

echo "🎮 Starting TSAI Jarvis Advanced Visualization API..."
echo "Phase 2.4: Advanced Visualization"
echo "=================================="

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

# Additional advanced visualization requirements
echo "Installing advanced visualization requirements..."
pip install matplotlib seaborn plotly

# Set environment variables
export ADVANCED_VISUALIZATION_PORT=8008
export ADVANCED_VISUALIZATION_HOST=0.0.0.0
export ADVANCED_VISUALIZATION_LOG_LEVEL=info

# Create database directory if it doesn't exist
mkdir -p data/advanced_visualization

# Start the Advanced Visualization API
echo "Starting Advanced Visualization API on port 8008..."
echo "API Documentation: http://localhost:8008/docs"
echo "WebSocket: ws://localhost:8008/ws/visualization"
echo ""

python hockey-analytics/advanced_visualization_api.py
