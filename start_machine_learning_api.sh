#!/bin/bash

# TSAI Jarvis - Machine Learning API Startup Script
# Phase 2.5: Machine Learning Implementation

echo "🤖 Starting TSAI Jarvis Machine Learning API..."
echo "Phase 2.5: Machine Learning"
echo "============================"

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

# Additional machine learning requirements
echo "Installing machine learning requirements..."
pip install scikit-learn torch torchvision

# Set environment variables
export MACHINE_LEARNING_PORT=8009
export MACHINE_LEARNING_HOST=0.0.0.0
export MACHINE_LEARNING_LOG_LEVEL=info

# Create database directory if it doesn't exist
mkdir -p data/machine_learning

# Start the Machine Learning API
echo "Starting Machine Learning API on port 8009..."
echo "API Documentation: http://localhost:8009/docs"
echo "WebSocket: ws://localhost:8009/ws/ml"
echo ""

python hockey-analytics/machine_learning_api.py
