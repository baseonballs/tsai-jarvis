#!/bin/bash

# TSAI Jarvis - Service Mesh API Startup Script
# Inter-service communication and orchestration for TSAI ecosystem

# Source the virtual environment
source .venv/bin/activate

# Define the API file and port
API_FILE="hockey-analytics/service_mesh_api.py"
PORT=8018

echo "Starting TSAI Jarvis Service Mesh API on port $PORT..."
echo "API Documentation: http://localhost:$PORT/docs"
echo "WebSocket (Service): ws://localhost:$PORT/ws/service-mesh/{service_name}"
echo "WebSocket (Monitoring): ws://localhost:$PORT/ws/service-mesh/monitoring"
echo "WebSocket (Orchestration): ws://localhost:$PORT/ws/service-mesh/orchestration"
echo ""
echo "Service Mesh Features:"
echo "  - Service discovery and registration"
echo "  - Inter-service communication"
echo "  - Workflow orchestration"
echo "  - Circuit breakers and fault tolerance"
echo "  - Load balancing"
echo "  - Health monitoring"
echo "  - Message routing"
echo "  - Service mesh coordination"
echo ""

# Start the FastAPI application using uvicorn
# The --reload flag is useful for development, remove for production
cd hockey-analytics && uvicorn service_mesh_api:app --host 0.0.0.0 --port "$PORT" --reload
