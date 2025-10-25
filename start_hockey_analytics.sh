#!/bin/bash

# TSAI Jarvis - Hockey Analytics Platform Startup Script
# This script starts the complete hockey analytics platform

echo "🏒 Starting TSAI Jarvis Hockey Analytics Platform..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

print_status "Starting TSAI Jarvis infrastructure..."

# Start the infrastructure services
print_status "Starting PostgreSQL, Redis, and Temporal AI..."
docker-compose up -d postgresql redis temporal temporal-web

# Wait for services to be ready
print_status "Waiting for services to initialize..."
sleep 10

# Check if services are running
print_status "Checking service health..."

# Check PostgreSQL
if docker-compose exec postgresql pg_isready -U temporal > /dev/null 2>&1; then
    print_success "PostgreSQL is ready"
else
    print_error "PostgreSQL failed to start"
    exit 1
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is ready"
else
    print_error "Redis failed to start"
    exit 1
fi

# Check Temporal
if curl -s http://localhost:7233 > /dev/null 2>&1; then
    print_success "Temporal AI is ready"
else
    print_warning "Temporal AI may still be starting up..."
fi

print_status "Starting TSAI Jarvis Core Service..."
docker-compose up -d jarvis-core

# Wait for Jarvis to be ready
print_status "Waiting for TSAI Jarvis to initialize..."
sleep 15

# Check if Jarvis is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "TSAI Jarvis Core is running on port 8000"
else
    print_warning "TSAI Jarvis Core may still be starting up..."
fi

print_status "Starting monitoring services..."
docker-compose up -d prometheus grafana

# Start the dashboard
print_status "Starting Hockey Analytics Dashboard..."
cd dashboard

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if pnpm is available
if ! command -v pnpm &> /dev/null; then
    print_warning "pnpm not found, using npm instead..."
    npm install
    npm run dev &
else
    pnpm install
    pnpm dev &
fi

cd ..

# Wait a moment for the dashboard to start
sleep 5

print_success "🏒 TSAI Jarvis Hockey Analytics Platform is starting up!"
echo ""
echo "📊 Dashboard: http://localhost:3000"
echo "🧠 Jarvis API: http://localhost:8000"
echo "⚡ Temporal UI: http://localhost:8080"
echo "📈 Grafana: http://localhost:3000 (admin/admin)"
echo "🔍 Prometheus: http://localhost:9090"
echo ""
echo "🎯 Features Available:"
echo "  ✅ Real-time hockey game analysis"
echo "  ✅ AI-powered player detection"
echo "  ✅ Live event recognition"
echo "  ✅ Performance analytics"
echo "  ✅ Video processing pipeline"
echo ""
echo "📚 Documentation:"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/health"
echo ""
print_status "Platform is ready for hockey analytics! 🚀"
