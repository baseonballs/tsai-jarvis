#!/bin/bash

# TSAI Jarvis - Holmes Integration API Startup Script
# Media curation and asset management for hockey analytics

# Source the virtual environment
source venv/bin/activate

# Define the API file and port
API_FILE="hockey-analytics/holmes_integration_api.py"
PORT=8017

echo "Starting TSAI Jarvis Holmes Integration API on port $PORT..."
echo "API Documentation: http://localhost:$PORT/docs"
echo "WebSocket (General): ws://localhost:$PORT/ws/holmes"
echo "WebSocket (Curation): ws://localhost:$PORT/ws/holmes/curation"
echo "WebSocket (Analytics): ws://localhost:$PORT/ws/holmes/analytics"
echo ""
echo "Holmes Integration Features:"
echo "  - Media Asset Management"
echo "  - Content Curation"
echo "  - Playlist Management"
echo "  - Asset Processing"
echo "  - Search & Discovery"
echo "  - Analytics & Insights"
echo "  - Content Recommendations"
echo "  - Version Control"
echo "  - Workflow Automation"
echo "  - Media Analytics"
echo ""

# Start the FastAPI application using uvicorn
# The --reload flag is useful for development, remove for production
cd hockey-analytics && uvicorn holmes_integration_api:app --host 0.0.0.0 --port "$PORT" --reload
