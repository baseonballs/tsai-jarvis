#!/bin/bash

# TSAI Jarvis - Watson Integration API Startup Script
# NLP reasoning and text analysis for hockey analytics

# Source the virtual environment
source venv/bin/activate

# Define the API file and port
API_FILE="hockey-analytics/watson_integration_api.py"
PORT=8016

echo "Starting TSAI Jarvis Watson Integration API on port $PORT..."
echo "API Documentation: http://localhost:$PORT/docs"
echo "WebSocket (General): ws://localhost:$PORT/ws/watson"
echo "WebSocket (Analysis): ws://localhost:$PORT/ws/watson/analysis"
echo "WebSocket (Reasoning): ws://localhost:$PORT/ws/watson/reasoning"
echo ""
echo "Watson Integration Features:"
echo "  - Natural Language Processing"
echo "  - Sentiment Analysis"
echo "  - Entity Recognition"
echo "  - Topic Modeling"
echo "  - Text Summarization"
echo "  - Intent Classification"
echo "  - Emotion Analysis"
echo "  - Keyword Extraction"
echo "  - Logical Reasoning"
echo "  - Causal Reasoning"
echo "  - Temporal Reasoning"
echo "  - Comparative Analysis"
echo "  - Predictive Reasoning"
echo ""

# Start the FastAPI application using uvicorn
# The --reload flag is useful for development, remove for production
cd hockey-analytics && uvicorn watson_integration_api:app --host 0.0.0.0 --port "$PORT" --reload
