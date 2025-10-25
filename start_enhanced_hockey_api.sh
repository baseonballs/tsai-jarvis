#!/bin/bash

# TSAI Jarvis - Enhanced Hockey Analytics API Service Startup Script
# Phase 1.2: Enhanced API Service for Live Analytics

echo "🏒 Starting TSAI Jarvis Enhanced Hockey Analytics API Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_enhanced() {
    echo -e "${PURPLE}[ENHANCED]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
print_status "Checking Python dependencies..."

# Check for required packages
python3 -c "import fastapi, uvicorn, websockets, aiohttp, numpy" 2>/dev/null
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
export API_PORT="8002"

print_enhanced "Starting Enhanced Hockey Analytics API Service..."

# Start the enhanced API service
cd hockey-analytics

# Start the service in background
python3 enhanced_hockey_api.py &
API_PID=$!

# Wait a moment for the service to start
sleep 5

# Check if the service is running
if ps -p $API_PID > /dev/null; then
    print_success "🏒 Enhanced Hockey Analytics API Service is running!"
    echo ""
    print_enhanced "📊 Enhanced API Endpoints:"
    echo "  - API Base: http://localhost:8002"
    echo "  - Health Check: http://localhost:8002/health"
    echo "  - Game State: http://localhost:8002/api/game/state"
    echo "  - Player Stats: http://localhost:8002/api/players/stats"
    echo "  - Live Events: http://localhost:8002/api/events/live"
    echo "  - Analytics: http://localhost:8002/api/analytics/metrics"
    echo "  - Momentum: http://localhost:8002/api/analytics/momentum"
    echo "  - Pressure: http://localhost:8002/api/analytics/pressure"
    echo "  - WebSocket: ws://localhost:8002/ws/analytics"
    echo ""
    print_enhanced "🎯 Enhanced Features Available:"
    echo "  ✅ Real-time momentum tracking"
    echo "  ✅ Advanced pressure analysis"
    echo "  ✅ Performance prediction algorithms"
    echo "  ✅ Enhanced event detection"
    echo "  ✅ Advanced player analytics"
    echo "  ✅ Team analytics with zone control"
    echo "  ✅ WebSocket real-time updates"
    echo ""
    print_enhanced "📚 Enhanced API Documentation:"
    echo "  - Swagger UI: http://localhost:8002/docs"
    echo "  - ReDoc: http://localhost:8002/redoc"
    echo ""
    print_enhanced "🚀 Phase 1.2: Live Analytics Features:"
    echo "  🎯 Momentum Tracking: Real-time game momentum analysis"
    echo "  📊 Pressure Analysis: Advanced pressure metrics"
    echo "  🔮 Performance Prediction: AI-powered predictions"
    echo "  🎪 Enhanced Events: Advanced event detection"
    echo "  📈 Player Analytics: Comprehensive player metrics"
    echo "  🏒 Team Analytics: Team performance analysis"
    echo ""
    print_status "Enhanced API Service is ready for advanced hockey analytics! 🚀"
    
    # Keep the script running and show logs
    echo ""
    print_status "Press Ctrl+C to stop the service"
    
    # Wait for the process
    wait $API_PID
else
    print_error "Failed to start Enhanced Hockey Analytics API Service"
    exit 1
fi
