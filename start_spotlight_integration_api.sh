#!/bin/bash

# TSAI Jarvis - Spotlight Integration API Startup Script
# Video processing and event detection for hockey analytics

# Source the virtual environment
source .venv/bin/activate

# Define the API file and port
API_FILE="hockey-analytics/spotlight_integration_api.py"
PORT=8015

echo "Starting TSAI Jarvis Spotlight Integration API on port $PORT..."
echo "API Documentation: http://localhost:$PORT/docs"
echo "WebSocket (General): ws://localhost:$PORT/ws/spotlight"
echo "WebSocket (Live Stream): ws://localhost:$PORT/ws/spotlight/live-stream"
echo "WebSocket (Analytics): ws://localhost:$PORT/ws/spotlight/analytics"
echo ""
echo "Spotlight Integration Features:"
echo "  - Video stream processing"
echo "  - Real-time event detection"
echo "  - Highlight generation"
echo "  - Video analytics"
echo "  - Live stream monitoring"
echo "  - Event classification"
echo "  - Player tracking"
echo "  - Team activity analysis"
echo ""

# Start the FastAPI application using uvicorn
# The --reload flag is useful for development, remove for production
cd hockey-analytics && uvicorn spotlight_integration_api:app --host 0.0.0.0 --port "$PORT" --reload
