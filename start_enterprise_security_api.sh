#!/bin/bash

# TSAI Jarvis - Enterprise Security API Startup Script
# Phase 2.6: Enterprise Security Implementation

echo "🔒 Starting TSAI Jarvis Enterprise Security API..."
echo "Phase 2.6: Enterprise Security"
echo "============================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Additional security requirements
echo "Installing enterprise security requirements..."
pip install pyjwt bcrypt pyotp qrcode cryptography email-validator

# Set environment variables
export ENTERPRISE_SECURITY_PORT=8010
export ENTERPRISE_SECURITY_HOST=0.0.0.0
export ENTERPRISE_SECURITY_LOG_LEVEL=info
export JWT_SECRET_KEY="your-jwt-secret-key-change-in-production"
export ENCRYPTION_KEY="your-encryption-key-change-in-production"

# Create database directory if it doesn't exist
mkdir -p data/enterprise_security

# Start the Enterprise Security API
echo "Starting Enterprise Security API on port 8010..."
echo "API Documentation: http://localhost:8010/docs"
echo "WebSocket: ws://localhost:8010/ws/security"
echo ""

python hockey-analytics/enterprise_security_api.py
