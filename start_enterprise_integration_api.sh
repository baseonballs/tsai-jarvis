#!/bin/bash

# TSAI Jarvis - Enterprise Integration API Startup Script
# Phase 2.3: Enterprise Integration Implementation

echo "🏢 Starting TSAI Jarvis Enterprise Integration API..."
echo "Phase 2.3: Enterprise Integration"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Additional enterprise integration requirements
echo "Installing enterprise integration requirements..."
pip install fastapi uvicorn websockets aiofiles sqlite3

# Set environment variables
export ENTERPRISE_INTEGRATION_PORT=8007
export ENTERPRISE_INTEGRATION_HOST=0.0.0.0
export ENTERPRISE_INTEGRATION_LOG_LEVEL=info

# Create database directory if it doesn't exist
mkdir -p data/enterprise_integration

# Start the Enterprise Integration API
echo "Starting Enterprise Integration API on port 8007..."
echo "API Documentation: http://localhost:8007/docs"
echo "WebSocket: ws://localhost:8007/ws/enterprise"
echo ""

python hockey-analytics/enterprise_integration_api.py
