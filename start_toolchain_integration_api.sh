#!/bin/bash

# TSAI Jarvis - Toolchain Integration API Startup Script
# Human-driven hockey analytics and AI detection workflows

# Source the virtual environment
source .venv/bin/activate

# Define the API file and port
API_FILE="hockey-analytics/toolchain_integration_api.py"
PORT=8013

echo "Starting TSAI Jarvis Toolchain Integration API on port $PORT..."
echo "API Documentation: http://localhost:$PORT/docs"
echo "WebSocket (General): ws://localhost:$PORT/ws/toolchain"
echo "WebSocket (Human): ws://localhost:$PORT/ws/toolchain/human"
echo ""
echo "Toolchain Integration Features:"
echo "  - Human-driven analytics workflows"
echo "  - Interactive parameter tuning"
echo "  - Human approval gates"
echo "  - Annotation and labeling tools"
echo "  - Quality assurance workflows"
echo "  - Real-time human feedback"
echo ""

# Start the FastAPI application using uvicorn
# The --reload flag is useful for development, remove for production
cd hockey-analytics && uvicorn toolchain_integration_api:app --host 0.0.0.0 --port "$PORT" --reload
