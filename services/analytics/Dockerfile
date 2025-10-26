# Jarvis Analytics Service Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY service.py .
COPY config.yaml ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/config /app/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV JARVIS_CONFIG_PATH=/app/config/jarvis-core.yaml
ENV PROMETHEUS_ENDPOINT=http://prometheus:9090
ENV GRAFANA_ENDPOINT=http://grafana:3000
ENV LOG_LEVEL=INFO
ENV CACHE_DIR=/app/cache
ENV LOGS_DIR=/app/logs

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the service
CMD ["python", "service.py"]
