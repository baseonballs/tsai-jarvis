#!/bin/bash

# Source the virtual environment
source venv/bin/activate

# Define the API file and port
API_FILE="hockey-analytics/main_api.py"
PORT=8001

echo "Starting TSAI Jarvis Main Hockey Analytics API on port $PORT..."
echo "API Documentation: http://localhost:$PORT/docs"
echo "WebSocket: ws://localhost:$PORT/ws/analytics"

# Start the FastAPI application using uvicorn
# The --reload flag is useful for development, remove for production
cd hockey-analytics && uvicorn main_api:app --host 0.0.0.0 --port "$PORT" --reload
