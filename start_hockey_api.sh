#!/bin/bash

# TSAI Jarvis - Hockey Analytics API Service Startup Script
# This script starts the hockey analytics API service for real-time processing

echo "🏒 Starting TSAI Jarvis Hockey Analytics API Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
print_status "Checking Python dependencies..."

# Check for required packages
python3 -c "import fastapi, uvicorn, websockets, aiohttp" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Required Python packages not found. Installing dependencies..."
    
    # Install required packages
    pip3 install fastapi uvicorn websockets aiohttp aiofiles ultralytics torch torchvision opencv-python numpy
    if [ $? -ne 0 ]; then
        print_error "Failed to install required packages"
        exit 1
    fi
fi

print_success "Python dependencies are ready"

# Create logs directory
mkdir -p logs

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export API_HOST="0.0.0.0"
export API_PORT="8001"

print_status "Starting Hockey Analytics API Service..."

# Start the API service
cd hockey-analytics

# Start the service in background
python3 hockey_api_service.py &
API_PID=$!

# Wait a moment for the service to start
sleep 3

# Check if the service is running
if ps -p $API_PID > /dev/null; then
    print_success "🏒 Hockey Analytics API Service is running!"
    echo ""
    echo "📊 API Endpoints:"
    echo "  - API Base: http://localhost:8001"
    echo "  - Health Check: http://localhost:8001/health"
    echo "  - Game State: http://localhost:8001/api/game/state"
    echo "  - Player Stats: http://localhost:8001/api/players/stats"
    echo "  - Live Events: http://localhost:8001/api/events/live"
    echo "  - Analytics: http://localhost:8001/api/analytics/metrics"
    echo "  - WebSocket: ws://localhost:8001/ws/analytics"
    echo ""
    echo "🎯 Features Available:"
    echo "  ✅ Real-time hockey analytics API"
    echo "  ✅ YOLO model integration"
    echo "  ✅ Live event detection"
    echo "  ✅ Player tracking and metrics"
    echo "  ✅ WebSocket real-time updates"
    echo ""
    echo "📚 API Documentation:"
    echo "  - Swagger UI: http://localhost:8001/docs"
    echo "  - ReDoc: http://localhost:8001/redoc"
    echo ""
    print_status "API Service is ready for hockey analytics! 🚀"
    
    # Keep the script running and show logs
    echo ""
    print_status "Press Ctrl+C to stop the service"
    
    # Wait for the process
    wait $API_PID
else
    print_error "Failed to start Hockey Analytics API Service"
    exit 1
fi
