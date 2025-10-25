# TSAI Jarvis Platform Setup Guide

## Overview

This guide will help you set up the complete TSAI Jarvis platform, including all foundation services, core intelligence, and infrastructure components.

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **PostgreSQL**: v15 or higher
- **Redis**: 6.0 or higher
- **Temporal Server**: 1.20 or higher
- **Docker**: 20.10 or higher (optional, for containerized setup)

### Required Services
- PostgreSQL database server
- Redis server
- Temporal server
- Python virtual environment

## Quick Start

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url> tsai-jarvis
cd tsai-jarvis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Database Configuration
DATABASE_URL=postgresql://temporal:temporal@localhost:5432/tsai_platform

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Temporal Configuration
TEMPORAL_URL=localhost:7233
TEMPORAL_NAMESPACE=tsai-jarvis

# Security Configuration
MASTER_KEY=tsai-jarvis-master-key-2024
JWT_SECRET=tsai-jwt-secret-key-2024

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

### 3. Infrastructure Setup

#### Option A: Docker Compose (Recommended)

```bash
# Start infrastructure services
docker-compose up -d

# Wait for services to be ready
docker-compose ps
```

#### Option B: Manual Setup

**PostgreSQL Setup:**
```bash
# Install PostgreSQL 15
# Create database and user
sudo -u postgres psql
CREATE DATABASE tsai_platform;
CREATE USER temporal WITH PASSWORD 'temporal';
GRANT ALL PRIVILEGES ON DATABASE tsai_platform TO temporal;
\q
```

**Redis Setup:**
```bash
# Install Redis
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**Temporal Setup:**
```bash
# Install Temporal CLI
curl -sSf https://temporal.download/cli.sh | sh

# Start Temporal server
temporal server start-dev
```

### 4. Platform Setup

Run the comprehensive setup script:

```bash
# Run the setup script
python setup_jarvis.py
```

This will:
- Set up the PostgreSQL database with enterprise configuration
- Initialize all foundation services
- Configure core intelligence services
- Set up the API gateway
- Run integration tests

### 5. Verify Setup

Check the setup status:

```bash
# Check database
psql -h localhost -U temporal -d tsai_platform -c "SELECT COUNT(*) FROM platform.system_config;"

# Check Redis
redis-cli ping

# Check Temporal
temporal workflow list

# Check API Gateway
curl http://localhost:8000/health
```

## Detailed Setup

### Foundation Services

#### 1. User Management Service
- **Purpose**: User lifecycle management, authentication, entitlements
- **Features**: Registration, login, role-based access, subscription management
- **Database**: `users` schema with tables for users, sessions, entitlements, subscriptions

#### 2. Session Management Service
- **Purpose**: Cross-application session management with SSO
- **Features**: Session creation, validation, refresh, termination
- **Storage**: Redis for fast access, PostgreSQL for persistence

#### 3. Unified State Management Service
- **Purpose**: Centralized state management with caching and failover
- **Features**: State storage, retrieval, synchronization, cluster management
- **Storage**: Multi-tier caching (Redis, PostgreSQL)

#### 4. Key Management Service
- **Purpose**: Encryption key management and security
- **Features**: Key generation, rotation, encryption, signing, verification
- **Security**: AES-256, RSA, ECDSA, HMAC support

#### 5. Asset Management Service
- **Purpose**: Media asset management and storage
- **Features**: Asset storage, metadata, optimization, lifecycle management
- **Storage**: Multi-tier storage (hot, warm, cold)

### Core Intelligence Services

#### 1. Jarvis Brain (Decision Engine)
- **Purpose**: AI-powered decision making
- **Features**: Data analysis, pattern recognition, decision making, learning

#### 2. Jarvis Orchestrator (Temporal AI)
- **Purpose**: Workflow orchestration and coordination
- **Features**: Temporal workflows, service coordination, state management

### API Gateway

#### Endpoints
- `GET /health` - Health check
- `POST /api/intelligence/analyze` - Data analysis
- `POST /api/intelligence/decide` - Decision making
- `POST /api/workflows/start` - Start workflow
- `GET /api/workflows/{id}/status` - Workflow status

## Configuration

### Database Configuration

The setup script configures PostgreSQL for enterprise use:

```sql
-- Key settings
max_connections = 1000
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 64MB
```

### Redis Configuration

```yaml
# Redis settings
maxmemory: 2gb
maxmemory-policy: allkeys-lru
save: "900 1 300 10 60 10000"
```

### Temporal Configuration

```yaml
# Temporal settings
server:
  persistence:
    defaultStore: "postgresql"
    sql:
      driver: "postgres"
      connectAddr: "localhost:5432"
      database: "temporal"
      user: "temporal"
      password: "temporal"
```

## Monitoring and Observability

### Database Monitoring

```sql
-- Check database stats
SELECT * FROM platform.database_stats;

-- Check table stats
SELECT * FROM platform.table_stats;

-- Check index stats
SELECT * FROM platform.index_stats;

-- Check query stats
SELECT * FROM platform.query_stats;
```

### Service Health Checks

```bash
# Check all services
curl http://localhost:8000/health

# Check specific service
curl http://localhost:8000/api/services/status
```

### Logs

```bash
# View application logs
tail -f logs/jarvis.log

# View database logs
tail -f /var/log/postgresql/postgresql-15-main.log

# View Redis logs
tail -f /var/log/redis/redis-server.log

# View Temporal logs
temporal workflow list --namespace tsai-jarvis
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -h localhost -U temporal -d tsai_platform
```

#### 2. Redis Connection Failed
```bash
# Check Redis status
sudo systemctl status redis-server

# Test connection
redis-cli ping
```

#### 3. Temporal Connection Failed
```bash
# Check Temporal status
temporal server start-dev

# Test connection
temporal workflow list
```

#### 4. API Gateway Not Starting
```bash
# Check port availability
netstat -tlnp | grep 8000

# Check logs
tail -f logs/jarvis.log
```

### Performance Issues

#### Database Performance
```sql
-- Check slow queries
SELECT * FROM platform.query_stats ORDER BY total_time DESC LIMIT 10;

-- Check index usage
SELECT * FROM platform.index_stats WHERE idx_scan = 0;
```

#### Memory Usage
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head
```

### Security Issues

#### Check Security Events
```sql
SELECT * FROM security.security_events 
WHERE severity IN ('error', 'critical') 
ORDER BY created_at DESC;
```

#### Check Failed Logins
```sql
SELECT * FROM security.security_events 
WHERE event_type = 'failed_auth' 
ORDER BY created_at DESC;
```

## Development

### Running in Development Mode

```bash
# Start with hot reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Run linting
black .
isort .
flake8 .
```

### Adding New Services

1. Create service in `foundation-services/` or `services/`
2. Add to `setup_jarvis.py`
3. Update `requirements.txt` if needed
4. Add tests in `tests/`

## Production Deployment

### Docker Deployment

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods
kubectl get services
```

## Support

- **Documentation**: `/docs/` directory
- **Issues**: GitHub Issues
- **Logs**: `/logs/` directory
- **Health**: `http://localhost:8000/health`

## Next Steps

1. **Configure TSAI Services**: Set up Autopilot, Spotlight, Toolchain, Watson, Holmes
2. **Create Workflows**: Define business pipelines
3. **Set up Monitoring**: Configure Prometheus, Grafana
4. **Security Hardening**: SSL certificates, firewall rules
5. **Backup Strategy**: Database backups, asset backups
6. **Scaling**: Load balancing, horizontal scaling

---

**TSAI Jarvis Platform is now ready for development and production use!** 🚀
