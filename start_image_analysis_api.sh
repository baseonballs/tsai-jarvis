#!/bin/bash

# Source the virtual environment
source .venv/bin/activate

# Define the API file and port
API_FILE="hockey-analytics/image_analysis_api.py"
PORT=8012

echo "Starting TSAI Jarvis Image Analysis API on port $PORT..."
echo "API Documentation: http://localhost:$PORT/docs"
echo "Image Analysis Features:"
echo "  - Object Detection & Recognition"
echo "  - Image Enhancement & Processing"
echo "  - Image Segmentation"
echo "  - Computer Vision Analytics"
echo "  - Target Detection & Highlighting"

# Start the FastAPI application using uvicorn
# The --reload flag is useful for development, remove for production
cd hockey-analytics && uvicorn image_analysis_api:app --host 0.0.0.0 --port "$PORT" --reload
