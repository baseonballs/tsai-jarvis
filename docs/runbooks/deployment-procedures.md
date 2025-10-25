# Deployment Procedures

This runbook provides step-by-step procedures for deploying the TSAI Jarvis Hockey Analytics platform across different environments.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Development Deployment](#3-development-deployment)
4. [Staging Deployment](#4-staging-deployment)
5. [Production Deployment](#5-production-deployment)
6. [Rollback Procedures](#6-rollback-procedures)
7. [Post-Deployment Verification](#7-post-deployment-verification)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

### 1.1 System Requirements
- **Operating System**: Linux/macOS/Windows
- **Python**: 3.9+ with virtual environment
- **Node.js**: 18+ with npm/pnpm
- **Docker**: 20.10+ (optional)
- **Git**: 2.30+
- **Database**: PostgreSQL 13+ or SQLite 3.35+

### 1.2 Access Requirements
- **Repository Access**: Read access to `tsai-jarvis` repository
- **Environment Access**: Appropriate environment credentials
- **Database Access**: Database connection credentials
- **Monitoring Access**: Monitoring system access

### 1.3 Tools Required
```bash
# Verify required tools
python3 --version
node --version
npm --version
git --version
docker --version  # Optional
```

---

## 2. Environment Setup

### 2.1 Clone Repository
```bash
# Clone the repository
git clone https://github.com/baseonballs/tsai-jarvis.git
cd tsai-jarvis

# Checkout target branch
git checkout main  # or specific release tag
```

### 2.2 Environment Configuration
```bash
# Copy environment configuration
cp config/environments/development.yml config/local.yml
cp config/environments/staging.yml config/local.yml
cp config/environments/production.yml config/local.yml

# Update configuration values
vim config/local.yml
```

### 2.3 Dependencies Installation
```bash
# Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Node.js dependencies
cd dashboard
npm install  # or pnpm install
cd ..
```

---

## 3. Development Deployment

### 3.1 Local Development Setup
```bash
# Start all services
./jarvis.sh start --all

# Verify services
./jarvis.sh status

# Access dashboard
open http://localhost:8000
```

### 3.2 Development Database Setup
```bash
# Initialize development database
./scripts/init-dev-db.sh

# Run migrations
./scripts/migrate-db.sh --env development

# Seed test data
./scripts/seed-test-data.sh
```

### 3.3 Development Verification
```bash
# Run tests
./run-tests.sh

# Check API endpoints
curl http://localhost:8001/health
curl http://localhost:8007/health
curl http://localhost:8008/health
curl http://localhost:8009/health
curl http://localhost:8010/health
curl http://localhost:8011/health

# Check dashboard
curl http://localhost:8000
```

---

## 4. Staging Deployment

### 4.1 Staging Environment Preparation
```bash
# Set environment variables
export DEPLOY_ENV=staging
export DB_HOST=staging-db.example.com
export DB_NAME=jarvis_staging
export REDIS_HOST=staging-redis.example.com

# Update configuration
cp config/environments/staging.yml config/local.yml
```

### 4.2 Staging Deployment Process
```bash
# Build application
./scripts/build.sh --env staging

# Run pre-deployment tests
./scripts/pre-deploy-tests.sh --env staging

# Deploy to staging
./scripts/deploy.sh --env staging

# Verify deployment
./scripts/verify-deployment.sh --env staging
```

### 4.3 Staging Verification
```bash
# Health checks
curl https://staging-jarvis.example.com/health
curl https://staging-jarvis.example.com/api/health

# Functional tests
./scripts/staging-tests.sh

# Performance tests
./scripts/performance-tests.sh --env staging
```

---

## 5. Production Deployment

### 5.1 Production Environment Preparation
```bash
# Set environment variables
export DEPLOY_ENV=production
export DB_HOST=prod-db.example.com
export DB_NAME=jarvis_production
export REDIS_HOST=prod-redis.example.com

# Update configuration
cp config/environments/production.yml config/local.yml
```

### 5.2 Pre-Deployment Checklist
- [ ] Code review completed
- [ ] All tests passing
- [ ] Security scan passed
- [ ] Performance tests passed
- [ ] Database migrations ready
- [ ] Configuration updated
- [ ] Monitoring configured
- [ ] Rollback plan ready
- [ ] Team notified
- [ ] Maintenance window scheduled

### 5.3 Production Deployment Process
```bash
# Create deployment branch
git checkout -b release/v1.2.3
git push origin release/v1.2.3

# Build production artifacts
./scripts/build.sh --env production

# Run comprehensive tests
./scripts/full-test-suite.sh

# Deploy to production
./scripts/deploy.sh --env production --confirm

# Verify deployment
./scripts/verify-deployment.sh --env production
```

### 5.4 Production Verification
```bash
# Health checks
curl https://jarvis.example.com/health
curl https://jarvis.example.com/api/health

# Smoke tests
./scripts/smoke-tests.sh --env production

# Load tests
./scripts/load-tests.sh --env production

# Security tests
./scripts/security-tests.sh --env production
```

---

## 6. Rollback Procedures

### 6.1 Automatic Rollback
```bash
# Rollback to previous version
./scripts/rollback.sh --env production --version v1.2.2

# Verify rollback
./scripts/verify-rollback.sh --env production
```

### 6.2 Manual Rollback
```bash
# Stop current services
./jarvis.sh stop

# Restore previous version
git checkout v1.2.2
./jarvis.sh start --all

# Verify rollback
./jarvis.sh status
curl http://localhost:8000
```

### 6.3 Database Rollback
```bash
# Rollback database migrations
./scripts/rollback-db.sh --env production --version v1.2.2

# Verify database state
./scripts/verify-db.sh --env production
```

---

## 7. Post-Deployment Verification

### 7.1 Service Health Checks
```bash
# Check all services
./jarvis.sh status

# Detailed health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8007/health
curl http://localhost:8008/health
curl http://localhost:8009/health
curl http://localhost:8010/health
curl http://localhost:8011/health
```

### 7.2 Functional Verification
```bash
# Test dashboard functionality
curl http://localhost:8000
curl http://localhost:8000/api/game/state
curl http://localhost:8000/api/players/stats

# Test API endpoints
curl http://localhost:8001/api/game/state
curl http://localhost:8007/api/enterprise/teams
curl http://localhost:8008/api/visualization/3d-rink
curl http://localhost:8009/api/ml/predictions
curl http://localhost:8010/api/security/auth
curl http://localhost:8011/api/analytics/statistical
```

### 7.3 Performance Verification
```bash
# Load testing
./scripts/load-test.sh --duration 5m --users 100

# Performance monitoring
./scripts/performance-monitor.sh --duration 10m

# Resource usage
./scripts/resource-check.sh
```

### 7.4 Security Verification
```bash
# Security scan
./scripts/security-scan.sh

# Vulnerability check
./scripts/vulnerability-check.sh

# Access control test
./scripts/access-control-test.sh
```

---

## 8. Troubleshooting

### 8.1 Common Deployment Issues

#### Issue: Services not starting
```bash
# Check service status
./jarvis.sh status

# Check logs
./jarvis.sh logs

# Restart services
./jarvis.sh stop
./jarvis.sh start --all
```

#### Issue: Database connection errors
```bash
# Check database connectivity
./scripts/check-db-connection.sh

# Verify database credentials
./scripts/verify-db-credentials.sh

# Test database operations
./scripts/test-db-operations.sh
```

#### Issue: Port conflicts
```bash
# Check port usage
lsof -i :8000-8011

# Kill conflicting processes
./scripts/kill-conflicting-processes.sh

# Restart services
./jarvis.sh start --all
```

#### Issue: Configuration errors
```bash
# Validate configuration
./scripts/validate-config.sh

# Check environment variables
./scripts/check-env-vars.sh

# Update configuration
./scripts/update-config.sh
```

### 8.2 Emergency Procedures

#### Emergency Rollback
```bash
# Immediate rollback
./scripts/emergency-rollback.sh

# Notify team
./scripts/notify-team.sh --message "Emergency rollback initiated"

# Document incident
./scripts/document-incident.sh
```

#### Emergency Stop
```bash
# Stop all services
./jarvis.sh stop

# Kill all processes
./scripts/kill-all-processes.sh

# Notify team
./scripts/notify-team.sh --message "All services stopped"
```

### 8.3 Recovery Procedures

#### Service Recovery
```bash
# Restart services
./jarvis.sh start --all

# Verify recovery
./jarvis.sh status

# Test functionality
./scripts/test-functionality.sh
```

#### Data Recovery
```bash
# Restore from backup
./scripts/restore-from-backup.sh

# Verify data integrity
./scripts/verify-data-integrity.sh

# Test data operations
./scripts/test-data-operations.sh
```

---

## 9. Monitoring and Alerting

### 9.1 Deployment Monitoring
```bash
# Monitor deployment progress
./scripts/monitor-deployment.sh

# Check deployment metrics
./scripts/check-deployment-metrics.sh

# Monitor service health
./scripts/monitor-service-health.sh
```

### 9.2 Alert Configuration
```bash
# Configure alerts
./scripts/configure-alerts.sh

# Test alert system
./scripts/test-alerts.sh

# Verify alert delivery
./scripts/verify-alerts.sh
```

---

## 10. Documentation

### 10.1 Deployment Documentation
```bash
# Generate deployment report
./scripts/generate-deployment-report.sh

# Update deployment logs
./scripts/update-deployment-logs.sh

# Document changes
./scripts/document-changes.sh
```

### 10.2 Post-Deployment Documentation
```bash
# Create deployment summary
./scripts/create-deployment-summary.sh

# Update runbooks
./scripts/update-runbooks.sh

# Share deployment notes
./scripts/share-deployment-notes.sh
```

---

## 11. Best Practices

### 11.1 Deployment Best Practices
- **Automated Testing**: Run comprehensive tests before deployment
- **Gradual Rollout**: Use blue-green or canary deployments
- **Monitoring**: Monitor deployment progress and system health
- **Documentation**: Document all deployment steps and changes
- **Communication**: Keep team informed of deployment status

### 11.2 Risk Mitigation
- **Backup**: Always backup before deployment
- **Rollback Plan**: Have rollback procedures ready
- **Testing**: Test in staging environment first
- **Monitoring**: Monitor system health during deployment
- **Communication**: Communicate with stakeholders

### 11.3 Quality Assurance
- **Code Review**: Ensure code review completion
- **Testing**: Run all test suites
- **Security**: Perform security scans
- **Performance**: Run performance tests
- **Documentation**: Update documentation

---

## 12. Support

### 12.1 Deployment Support
- **Slack**: #tsai-jarvis-deployment
- **Email**: deployment@tsai-jarvis.com
- **Office Hours**: Tuesday 2-4 PM
- **Documentation**: This runbook

### 12.2 Emergency Support
- **On-call**: 24/7 emergency support
- **Escalation**: Escalation procedures
- **Communication**: Emergency communication channels
- **Documentation**: Emergency procedures
