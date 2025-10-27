#!/bin/bash

# TSAI Jarvis - Autopilot Integration API Startup Script
# Autonomous ML lifecycle and model management for hockey analytics

# Source the virtual environment
source .venv/bin/activate

# Define the API file and port
API_FILE="hockey-analytics/autopilot_integration_api.py"
PORT=8014

echo "Starting TSAI Jarvis Autopilot Integration API on port $PORT..."
echo "API Documentation: http://localhost:$PORT/docs"
echo "WebSocket (General): ws://localhost:$PORT/ws/autopilot"
echo "WebSocket (Monitoring): ws://localhost:$PORT/ws/autopilot/monitoring"
echo ""
echo "Autopilot Integration Features:"
echo "  - ML experiment management"
echo "  - Model training orchestration"
echo "  - Hyperparameter optimization"
echo "  - Model deployment automation"
echo "  - Performance monitoring"
echo "  - A/B testing and model versioning"
echo "  - Automated retraining workflows"
echo ""

# Start the FastAPI application using uvicorn
# The --reload flag is useful for development, remove for production
cd hockey-analytics && uvicorn autopilot_integration_api:app --host 0.0.0.0 --port "$PORT" --reload
