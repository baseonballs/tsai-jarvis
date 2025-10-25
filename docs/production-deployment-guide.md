# TSAI Jarvis - Production Deployment Guide

## Executive Summary

This guide provides comprehensive instructions for deploying the TSAI Jarvis Core Intelligence Platform to production environments. The platform includes Temporal AI orchestration, PostgreSQL database, Redis caching, and a unified API Gateway for the entire TSAI ecosystem.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Requirements](#infrastructure-requirements)
3. [Production Configuration](#production-configuration)
4. [Deployment Methods](#deployment-methods)
5. [Security Configuration](#security-configuration)
6. [Monitoring & Observability](#monitoring--observability)
7. [Backup & Disaster Recovery](#backup--disaster-recovery)
8. [Scaling & Performance](#scaling--performance)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

**Minimum Production Specs:**
- **CPU**: 8 cores (16 cores recommended)
- **RAM**: 32GB (64GB recommended)
- **Storage**: 500GB SSD (1TB recommended)
- **Network**: 1Gbps (10Gbps recommended)
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+

**Software Dependencies:**
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.10+
- Node.js 18+ (for monitoring)
- Git 2.30+

### Network Requirements

**Ports:**
- `8000`: TSAI Jarvis API Gateway
- `7233`: Temporal Server
- `8081`: Temporal Web UI
- `5432`: PostgreSQL Database
- `6379`: Redis Cache
- `9090`: Prometheus (monitoring)
- `3000`: Grafana (monitoring)

**Firewall Rules:**
```bash
# Allow internal communication
ufw allow from 10.0.0.0/8 to any port 8000
ufw allow from 10.0.0.0/8 to any port 7233
ufw allow from 10.0.0.0/8 to any port 5432
ufw allow from 10.0.0.0/8 to any port 6379

# Allow external API access
ufw allow 443/tcp  # HTTPS
ufw allow 80/tcp   # HTTP (redirect to HTTPS)
```

## Infrastructure Requirements

### Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (HAProxy/Nginx)           │
│                         Port 443/80                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              TSAI Jarvis API Gateway                        │
│                    Port 8000                               │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Temporal AI Orchestration                    │
│                    Port 7233                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              PostgreSQL Database (Primary)                 │
│                    Port 5432                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Redis Cache (Primary)                         │
│                    Port 6379                              │
└─────────────────────────────────────────────────────────────┘
```

### High Availability Setup

**Database Cluster:**
- Primary PostgreSQL instance
- Read replica for analytics
- Automated failover with Patroni

**Cache Cluster:**
- Redis primary instance
- Redis replica for read operations
- Redis Sentinel for failover

**Temporal Cluster:**
- Temporal server (frontend)
- Temporal worker nodes
- Temporal history service

## Production Configuration

### Environment Variables

Create `.env.production`:

```bash
# TSAI Jarvis Configuration
TSAI_JARVIS_VERSION=1.0.0
TSAI_JARVIS_ENV=production
TSAI_JARVIS_LOG_LEVEL=INFO

# Database Configuration
POSTGRES_HOST=postgresql-primary
POSTGRES_PORT=5432
POSTGRES_USER=tsai_jarvis
POSTGRES_PASSWORD=SECURE_PASSWORD_HERE
POSTGRES_DB=tsai_jarvis
POSTGRES_SSL_MODE=require

# Redis Configuration
REDIS_HOST=redis-primary
REDIS_PORT=6379
REDIS_PASSWORD=SECURE_REDIS_PASSWORD
REDIS_DB=0

# Temporal Configuration
TEMPORAL_SERVER_URL=temporal:7233
TEMPORAL_NAMESPACE=tsai-jarvis
TEMPORAL_TASK_QUEUE=jarvis-tasks

# Security
JWT_SECRET=SECURE_JWT_SECRET_256_BITS
ENCRYPTION_KEY=SECURE_ENCRYPTION_KEY_256_BITS
API_KEY_SECRET=SECURE_API_KEY_SECRET

# Monitoring
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_ENDPOINT=http://grafana:3000

# TSAI Services
AUTOPILOT_ENDPOINT=http://autopilot:4001
SPOTLIGHT_ENDPOINT=http://spotlight:4002
TOOLCHAIN_ENDPOINT=http://toolchain:4003
WATSON_ENDPOINT=http://watson:4004
HOLMES_ENDPOINT=http://holmes:4005
```

### Production Docker Compose

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  # TSAI Jarvis API Gateway
  jarvis-api:
    image: tsai-jarvis:latest
    ports:
      - "8000:8000"
    environment:
      - TSAI_JARVIS_ENV=production
      - POSTGRES_HOST=postgresql-primary
      - REDIS_HOST=redis-primary
      - TEMPORAL_SERVER_URL=temporal:7233
    depends_on:
      - postgresql-primary
      - redis-primary
      - temporal
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - tsai-network

  # PostgreSQL Primary
  postgresql-primary:
    image: postgres:15
    environment:
      - POSTGRES_USER=tsai_jarvis
      - POSTGRES_PASSWORD=SECURE_PASSWORD_HERE
      - POSTGRES_DB=tsai_jarvis
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/database/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tsai_jarvis"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - tsai-network

  # Redis Primary
  redis-primary:
    image: redis:7-alpine
    command: redis-server --requirepass SECURE_REDIS_PASSWORD
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - tsai-network

  # Temporal Server
  temporal:
    image: temporalio/auto-setup:latest
    environment:
      - DB=postgres15
      - DB_PORT=5432
      - POSTGRES_USER=tsai_jarvis
      - POSTGRES_PWD=SECURE_PASSWORD_HERE
      - POSTGRES_SEEDS=postgresql-primary
    ports:
      - "7233:7233"
      - "8081:8080"
    depends_on:
      - postgresql-primary
    restart: unless-stopped
    networks:
      - tsai-network

  # Monitoring - Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped
    networks:
      - tsai-network

  # Monitoring - Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=SECURE_GRAFANA_PASSWORD
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    restart: unless-stopped
    networks:
      - tsai-network

networks:
  tsai-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

## Deployment Methods

### Method 1: Docker Compose (Recommended for Small-Medium Deployments)

```bash
# 1. Clone repository
git clone https://github.com/your-org/tsai-jarvis.git
cd tsai-jarvis

# 2. Configure environment
cp .env.example .env.production
# Edit .env.production with production values

# 3. Build and deploy
docker-compose -f docker-compose.production.yml up -d

# 4. Verify deployment
curl http://localhost:8000/health
```

### Method 2: Kubernetes (Recommended for Large Deployments)

```bash
# 1. Create namespace
kubectl create namespace tsai-jarvis

# 2. Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgresql.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/temporal.yaml
kubectl apply -f k8s/jarvis-api.yaml
kubectl apply -f k8s/monitoring.yaml

# 3. Verify deployment
kubectl get pods -n tsai-jarvis
kubectl get services -n tsai-jarvis
```

### Method 3: Cloud Deployment (AWS/GCP/Azure)

**AWS ECS:**
```bash
# 1. Create ECS cluster
aws ecs create-cluster --cluster-name tsai-jarvis

# 2. Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# 3. Create service
aws ecs create-service --cluster tsai-jarvis --service-name jarvis-api --task-definition jarvis-api:1
```

## Security Configuration

### SSL/TLS Setup

```bash
# 1. Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# 2. Configure Nginx reverse proxy
server {
    listen 443 ssl;
    server_name tsai-jarvis.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Database Security

```sql
-- Create production user
CREATE USER tsai_jarvis_prod WITH PASSWORD 'SECURE_PASSWORD';
GRANT CONNECT ON DATABASE tsai_jarvis TO tsai_jarvis_prod;
GRANT USAGE ON SCHEMA public TO tsai_jarvis_prod;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO tsai_jarvis_prod;

-- Enable SSL
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/server.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/server.key';
```

### API Security

```python
# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/analytics/hockey")
@limiter.limit("10/minute")
async def hockey_analytics(request: Request):
    # Implementation
    pass

# API Key authentication
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        api_key = request.headers.get("X-API-Key")
        if not validate_api_key(api_key):
            return JSONResponse({"error": "Invalid API key"}, status_code=401)
    return await call_next(request)
```

## Monitoring & Observability

### Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "tsai-jarvis-rules.yml"

scrape_configs:
  - job_name: 'tsai-jarvis'
    static_configs:
      - targets: ['jarvis-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgresql'
    static_configs:
      - targets: ['postgresql-primary:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-primary:6379']

  - job_name: 'temporal'
    static_configs:
      - targets: ['temporal:7233']
```

### Grafana Dashboards

**Key Metrics to Monitor:**
- API response times
- Database connection pool
- Redis cache hit ratio
- Temporal workflow execution
- System resource usage
- Error rates

### Alerting Rules

```yaml
# tsai-jarvis-rules.yml
groups:
  - name: tsai-jarvis
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
      
      - alert: DatabaseDown
        expr: up{job="postgresql"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL database is down"
```

## Backup & Disaster Recovery

### Database Backup

```bash
#!/bin/bash
# backup-database.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgresql"
DB_NAME="tsai_jarvis"

# Create backup
pg_dump -h postgresql-primary -U tsai_jarvis $DB_NAME > $BACKUP_DIR/tsai_jarvis_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/tsai_jarvis_$DATE.sql

# Upload to S3
aws s3 cp $BACKUP_DIR/tsai_jarvis_$DATE.sql.gz s3://tsai-jarvis-backups/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

### Automated Backup Schedule

```bash
# Add to crontab
0 2 * * * /opt/tsai-jarvis/scripts/backup-database.sh
0 3 * * * /opt/tsai-jarvis/scripts/backup-redis.sh
```

### Disaster Recovery Plan

1. **RTO (Recovery Time Objective)**: 4 hours
2. **RPO (Recovery Point Objective)**: 1 hour
3. **Backup Strategy**: Daily full backups + hourly incremental
4. **Recovery Process**:
   - Restore database from latest backup
   - Restore Redis from snapshot
   - Restart all services
   - Verify functionality

## Scaling & Performance

### Horizontal Scaling

**API Gateway Scaling:**
```yaml
# k8s/jarvis-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jarvis-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jarvis-api
  template:
    spec:
      containers:
      - name: jarvis-api
        image: tsai-jarvis:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

**Database Scaling:**
- Read replicas for analytics queries
- Connection pooling with PgBouncer
- Partitioning for large tables

### Performance Optimization

**Database Optimization:**
```sql
-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_hockey_analytics_game_id ON hockey_analytics(game_id);
CREATE INDEX CONCURRENTLY idx_hockey_analytics_timestamp ON hockey_analytics(timestamp);
CREATE INDEX CONCURRENTLY idx_workflows_status ON workflows(status);

-- Analyze tables for query optimization
ANALYZE hockey_analytics;
ANALYZE workflows;
```

**Redis Optimization:**
```bash
# Redis configuration
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

## Troubleshooting

### Common Issues

**1. Database Connection Issues:**
```bash
# Check database connectivity
docker exec -it tsai-jarvis-postgresql-1 psql -U tsai_jarvis -d tsai_jarvis -c "SELECT 1;"

# Check connection pool
curl http://localhost:8000/api/test -d '{"test_type": "database"}'
```

**2. Temporal Connection Issues:**
```bash
# Check Temporal server
curl http://localhost:7233/api/v1/namespaces

# Check Temporal Web UI
curl http://localhost:8081/
```

**3. Redis Connection Issues:**
```bash
# Check Redis connectivity
docker exec -it tsai-jarvis-redis-1 redis-cli ping

# Check Redis memory usage
docker exec -it tsai-jarvis-redis-1 redis-cli info memory
```

### Log Analysis

```bash
# View API Gateway logs
docker logs tsai-jarvis-jarvis-api-1 -f

# View database logs
docker logs tsai-jarvis-postgresql-1 -f

# View Temporal logs
docker logs tsai-jarvis-temporal-1 -f
```

### Performance Monitoring

```bash
# Check system resources
docker stats

# Check database performance
docker exec -it tsai-jarvis-postgresql-1 psql -U tsai_jarvis -d tsai_jarvis -c "
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;
"
```

## Maintenance

### Regular Maintenance Tasks

**Daily:**
- Monitor system health
- Check backup completion
- Review error logs

**Weekly:**
- Update dependencies
- Analyze performance metrics
- Review security logs

**Monthly:**
- Database maintenance (VACUUM, ANALYZE)
- Update SSL certificates
- Review and update documentation

### Update Procedures

```bash
# 1. Backup current deployment
docker-compose -f docker-compose.production.yml down
cp -r /opt/tsai-jarvis /opt/tsai-jarvis-backup-$(date +%Y%m%d)

# 2. Update code
git pull origin main
docker-compose -f docker-compose.production.yml build

# 3. Deploy update
docker-compose -f docker-compose.production.yml up -d

# 4. Verify deployment
curl http://localhost:8000/health
```

### Health Checks

```bash
#!/bin/bash
# health-check.sh

# Check API Gateway
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ API Gateway is down"
    exit 1
fi

# Check Database
if ! docker exec tsai-jarvis-postgresql-1 pg_isready -U tsai_jarvis > /dev/null 2>&1; then
    echo "❌ Database is down"
    exit 1
fi

# Check Redis
if ! docker exec tsai-jarvis-redis-1 redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is down"
    exit 1
fi

# Check Temporal
if ! curl -f http://localhost:7233/api/v1/namespaces > /dev/null 2>&1; then
    echo "❌ Temporal is down"
    exit 1
fi

echo "✅ All services are healthy"
```

## Conclusion

This production deployment guide provides comprehensive instructions for deploying TSAI Jarvis in production environments. The platform is designed for high availability, scalability, and security, making it suitable for enterprise-grade hockey analytics and AI orchestration.

For additional support or questions, please refer to the TSAI Jarvis documentation or contact the development team.

---

**Last Updated**: October 2024  
**Version**: 1.0.0  
**Platform**: TSAI Jarvis Core Intelligence Platform
