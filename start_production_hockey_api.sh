#!/bin/bash

# TSAI Jarvis - Production Hockey Analytics API Service Startup Script
# Phase 1.4: Production Analytics API Service

echo "🏒 Starting TSAI Jarvis Production Hockey Analytics API Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
ORANGE='\033[0;33m'
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

print_production() {
    echo -e "${PURPLE}[PRODUCTION]${NC} $1"
}

print_analytics() {
    echo -e "${CYAN}[ANALYTICS]${NC} $1"
}

print_enterprise() {
    echo -e "${ORANGE}[ENTERPRISE]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
print_status "Checking Python dependencies..."

# Check for required packages
python3 -c "import fastapi, uvicorn, websockets, aiohttp, numpy, sqlite3, pandas, asyncio" 2>/dev/null
if [ $? -ne 0 ]; then
    print_warning "Required Python packages not found. Installing dependencies..."
    
    # Install required packages
    pip3 install fastapi uvicorn websockets aiohttp aiofiles ultralytics torch torchvision opencv-python numpy pandas sqlite3
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
export API_PORT="8004"

print_production "Starting Production Hockey Analytics API Service..."

# Start the production API service
cd hockey-analytics

# Start the service in background
python3 production_hockey_api.py &
API_PID=$!

# Wait a moment for the service to start
sleep 5

# Check if the service is running
if ps -p $API_PID > /dev/null; then
    print_success "🏒 Production Hockey Analytics API Service is running!"
    echo ""
    print_production "📊 Production API Endpoints:"
    echo "  - API Base: http://localhost:8004"
    echo "  - Health Check: http://localhost:8004/health"
    echo "  - Game Sessions: http://localhost:8004/api/games/sessions"
    echo "  - Historical Data: http://localhost:8004/api/analytics/historical"
    echo "  - Team Analysis: http://localhost:8004/api/analytics/teams"
    echo "  - API Integration: http://localhost:8004/api/integration"
    echo "  - Mobile Optimization: http://localhost:8004/api/mobile"
    echo "  - Production Metrics: http://localhost:8004/api/analytics/production"
    echo "  - WebSocket: ws://localhost:8004/ws/analytics"
    echo ""
    print_analytics "🎯 Production Analytics Features:"
    echo "  ✅ Multi-game processing and concurrent analysis"
    echo "  ✅ Historical performance tracking and analysis"
    echo "  ✅ API integration with external data sources"
    echo "  ✅ Mobile optimization and responsive analytics"
    echo "  ✅ Production deployment and scaling infrastructure"
    echo "  ✅ Performance optimization for production workloads"
    echo "  ✅ Enterprise-grade analytics and insights"
    echo ""
    print_enterprise "📈 Production Capabilities:"
    echo "  🏒 Multi-Game: Concurrent processing of multiple games"
    echo "  📊 Historical: Season-long performance tracking and analysis"
    echo "  🔗 API Integration: External data source integration"
    echo "  📱 Mobile: Mobile-optimized APIs and responsive dashboards"
    echo "  🚀 Production: Enterprise-grade deployment and scaling"
    echo "  ⚡ Performance: Optimized for production workloads"
    echo ""
    print_enterprise "🚀 Phase 1.4: Production Analytics Features:"
    echo "  🏒 Multi-Game Processing: Concurrent game analysis and processing"
    echo "  📊 Historical Analysis: Season-long performance tracking and trends"
    echo "  🔗 API Integration: NHL API and third-party data source integration"
    echo "  📱 Mobile Support: Mobile-optimized APIs and responsive dashboards"
    echo "  🚀 Production Deployment: Enterprise-grade deployment and scaling"
    echo "  ⚡ Performance Optimization: Optimized for production workloads"
    echo ""
    print_status "Production Analytics API Service is ready for enterprise hockey analysis! 🚀"
    
    # Keep the script running and show logs
    echo ""
    print_status "Press Ctrl+C to stop the service"
    
    # Wait for the process
    wait $API_PID
else
    print_error "Failed to start Production Hockey Analytics API Service"
    exit 1
fi
