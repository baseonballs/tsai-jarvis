#!/bin/bash

# TSAI Jarvis - Advanced Hockey Analytics API Service Startup Script
# Phase 1.3: Advanced Analytics API Service

echo "🏒 Starting TSAI Jarvis Advanced Hockey Analytics API Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

print_advanced() {
    echo -e "${PURPLE}[ADVANCED]${NC} $1"
}

print_analytics() {
    echo -e "${CYAN}[ANALYTICS]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
print_status "Checking Python dependencies..."

# Check for required packages
python3 -c "import fastapi, uvicorn, websockets, aiohttp, numpy, asyncio" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Required Python packages not found. Installing dependencies..."
    
    # Install required packages
    pip3 install fastapi uvicorn websockets aiohttp aiofiles ultralytics torch torchvision opencv-python numpy asyncio
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
export API_PORT="8003"

print_advanced "Starting Advanced Hockey Analytics API Service..."

# Start the advanced API service
cd hockey-analytics

# Start the service in background
python3 advanced_hockey_api.py &
API_PID=$!

# Wait a moment for the service to start
sleep 5

# Check if the service is running
if ps -p $API_PID > /dev/null; then
    print_success "🏒 Advanced Hockey Analytics API Service is running!"
    echo ""
    print_advanced "📊 Advanced API Endpoints:"
    echo "  - API Base: http://localhost:8003"
    echo "  - Health Check: http://localhost:8003/health"
    echo "  - Game State: http://localhost:8003/api/game/state"
    echo "  - Player Stats: http://localhost:8003/api/players/stats"
    echo "  - Speed Analytics: http://localhost:8003/api/analytics/speed"
    echo "  - Shot Analytics: http://localhost:8003/api/analytics/shots"
    echo "  - Formation Analytics: http://localhost:8003/api/analytics/formations"
    echo "  - Strategy Analytics: http://localhost:8003/api/analytics/strategy"
    echo "  - WebSocket: ws://localhost:8003/ws/analytics"
    echo ""
    print_analytics "🎯 Advanced Analytics Features:"
    echo "  ✅ Player speed tracking and analysis"
    echo "  ✅ Shot quality and trajectory analysis"
    echo "  ✅ Team formation analysis and tracking"
    echo "  ✅ Game strategy insights and recommendations"
    echo "  ✅ Advanced performance metrics"
    echo "  ✅ Real-time analytics processing"
    echo "  ✅ WebSocket streaming for live updates"
    echo ""
    print_analytics "📈 Advanced Analytics Capabilities:"
    echo "  🏃 Player Speed: Real-time speed tracking with zone analysis"
    echo "  🎯 Shot Analysis: Quality, velocity, and accuracy metrics"
    echo "  🏒 Formations: Team positioning and tactical analysis"
    echo "  📊 Strategy: Possession, zone control, and game insights"
    echo "  🔮 Predictions: AI-powered performance predictions"
    echo "  📈 Trends: Historical analysis and pattern recognition"
    echo ""
    print_analytics "🚀 Phase 1.3: Advanced Analytics Features:"
    echo "  🏃 Speed Tracking: Real-time player speed and movement analysis"
    echo "  🎯 Shot Analysis: Comprehensive shot quality and trajectory analysis"
    echo "  🏒 Formation Analysis: Team positioning and tactical insights"
    echo "  📊 Strategy Insights: Game strategy and possession analytics"
    echo "  🔮 Performance Prediction: AI-powered performance forecasting"
    echo "  📈 Advanced Metrics: Comprehensive analytics and insights"
    echo ""
    print_status "Advanced Analytics API Service is ready for comprehensive hockey analysis! 🚀"
    
    # Keep the script running and show logs
    echo ""
    print_status "Press Ctrl+C to stop the service"
    
    # Wait for the process
    wait $API_PID
else
    print_error "Failed to start Advanced Hockey Analytics API Service"
    exit 1
fi
