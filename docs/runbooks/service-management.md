# Service Management Runbook

Comprehensive guide for managing TSAI Jarvis services in production.

## 🎯 Overview

This runbook covers all aspects of service management for the TSAI Jarvis platform, including startup, shutdown, health checks, and operational procedures.

## 🚀 Service Startup Procedures

### Starting All Services
```bash
# Start all services with CLI
./jarvis.sh start --all

# Start in background mode
./jarvis.sh start --all --background

# Force restart (cleanup + start)
./jarvis.sh start --all --force
```

### Starting Individual Services
```bash
# Start only dashboard
./jarvis.sh start --dashboard

# Manual service startup
cd /path/to/tsai-jarvis
source venv/bin/activate

# Start specific API
uvicorn hockey-analytics.enterprise_integration_api:app --host 0.0.0.0 --port 8007 --reload
```

### Service Startup Order
1. **Core API** (port 8001) - Foundation service
2. **Enterprise Integration** (port 8007) - Professional features
3. **Advanced Visualization** (port 8008) - 3D visualization
4. **Machine Learning** (port 8009) - AI/ML models
5. **Enterprise Security** (port 8010) - Authentication
6. **Advanced Analytics** (port 8011) - Statistical analysis
7. **Dashboard** (port 8000) - Frontend interface

## 🛑 Service Shutdown Procedures

### Graceful Shutdown
```bash
# Stop all services gracefully
./jarvis.sh stop

# Manual shutdown
# Send SIGTERM to processes
kill -TERM <PID>

# Wait for graceful shutdown (10 seconds)
sleep 10

# Force kill if needed
kill -9 <PID>
```

### Emergency Shutdown
```bash
# Force stop all services
pkill -f "tsai-jarvis"
pkill -f "hockey-analytics"
pkill -f "uvicorn.*jarvis"

# Clean up ports
for port in 8000 8001 8007 8008 8009 8010 8011; do
  lsof -ti:$port | xargs kill -9 2>/dev/null || true
done
```

## 🔍 Health Check Procedures

### Automated Health Checks
```bash
# Check all services
./jarvis.sh status

# Individual service checks
curl -f http://localhost:8000 > /dev/null && echo "Dashboard: OK" || echo "Dashboard: FAIL"
curl -f http://localhost:8001/health > /dev/null && echo "Core API: OK" || echo "Core API: FAIL"
curl -f http://localhost:8007/health > /dev/null && echo "Enterprise: OK" || echo "Enterprise: FAIL"
curl -f http://localhost:8008/health > /dev/null && echo "Visualization: OK" || echo "Visualization: FAIL"
curl -f http://localhost:8009/health > /dev/null && echo "ML: OK" || echo "ML: FAIL"
curl -f http://localhost:8010/health > /dev/null && echo "Security: OK" || echo "Security: FAIL"
curl -f http://localhost:8011/health > /dev/null && echo "Analytics: OK" || echo "Analytics: FAIL"
```

### Health Check Script
```bash
#!/bin/bash
# health-check.sh

SERVICES=(
  "8000:Dashboard"
  "8001:Core API"
  "8007:Enterprise Integration"
  "8008:Advanced Visualization"
  "8009:Machine Learning"
  "8010:Enterprise Security"
  "8011:Advanced Analytics"
)

for service in "${SERVICES[@]}"; do
  port="${service%%:*}"
  name="${service##*:}"
  
  if curl -s --connect-timeout 5 --max-time 10 "http://localhost:$port" > /dev/null 2>&1; then
    echo "✅ $name (port $port): HEALTHY"
  else
    echo "❌ $name (port $port): UNHEALTHY"
  fi
done
```

### Detailed Health Checks
```bash
# Check service response times
for port in 8000 8001 8007 8008 8009 8010 8011; do
  echo "Port $port response time:"
  time curl -s http://localhost:$port > /dev/null
done

# Check service logs for errors
grep -i "error\|exception\|failed" logs/jarvis-*.log | tail -10

# Check system resources
echo "CPU Usage: $(top -l 1 | grep "CPU usage" | awk '{print $3}')"
echo "Memory Usage: $(top -l 1 | grep "PhysMem" | awk '{print $2}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{print $5}')"
```

## 📊 Service Monitoring

### Process Monitoring
```bash
# Check running processes
ps aux | grep -E "(jarvis|hockey-analytics|uvicorn)" | grep -v grep

# Check port usage
lsof -i :8000-8011

# Check PID files
ls -la /tmp/tsai-jarvis-*.pid
```

### Resource Monitoring
```bash
# CPU and memory usage
top -l 1 | grep -E "(CPU usage|PhysMem)"

# Disk usage
df -h

# Network connections
netstat -an | grep -E ":(8000|8001|8007|8008|8009|8010|8011)"

# File descriptors
lsof | grep -E "(jarvis|hockey-analytics)" | wc -l
```

### Log Monitoring
```bash
# Real-time log monitoring
tail -f logs/jarvis-*.log

# Error log analysis
grep -i "error\|exception\|failed" logs/jarvis-*.log | tail -20

# Performance log analysis
grep -i "slow\|timeout\|latency" logs/jarvis-*.log | tail -10

# Access log analysis
grep "GET\|POST" logs/jarvis-*.log | tail -20
```

## 🔧 Service Configuration

### Environment Variables
```bash
# Service ports
export DASHBOARD_PORT=8000
export CORE_API_PORT=8001
export ENTERPRISE_INTEGRATION_PORT=8007
export ADVANCED_VISUALIZATION_PORT=8008
export MACHINE_LEARNING_PORT=8009
export ENTERPRISE_SECURITY_PORT=8010
export ADVANCED_ANALYTICS_PORT=8011

# Production settings
export PRODUCTION_MODE=true
export LOG_LEVEL=info
export MAX_RETRIES=3
export STARTUP_TIMEOUT=30
export SHUTDOWN_TIMEOUT=15
```

### Service Configuration Files
```bash
# Dashboard configuration
cat dashboard/next.config.ts
cat dashboard/package.json

# API configuration
cat hockey-analytics/*_api.py | grep -E "(host|port|reload)"

# Database configuration
cat hockey-analytics/*.db
```

## 🚨 Troubleshooting Common Issues

### Service Won't Start
```bash
# Check for port conflicts
lsof -i :8000-8011

# Check for process conflicts
ps aux | grep -E "(jarvis|hockey-analytics)"

# Check system resources
free -h
df -h

# Check logs
tail -f logs/jarvis-*.log
```

### Service Crashes
```bash
# Check crash logs
grep -i "crash\|segfault\|killed" logs/jarvis-*.log

# Check system logs
dmesg | tail -20

# Check resource limits
ulimit -a

# Check memory usage
free -h
```

### Performance Issues
```bash
# Check CPU usage
top -l 1

# Check memory usage
ps aux --sort=-%mem | head -10

# Check disk I/O
iostat -x 1 5

# Check network usage
netstat -i
```

## 📋 Service Maintenance

### Daily Maintenance
```bash
# Health check all services
./jarvis.sh status

# Review logs for errors
grep -i "error\|exception" logs/jarvis-*.log | tail -10

# Check system resources
htop
df -h

# Verify backups
ls -la backups/
```

### Weekly Maintenance
```bash
# Update dependencies
pip install --upgrade -r requirements.txt
npm update

# Clean up logs
find logs/ -name "*.log" -mtime +7 -delete

# Database maintenance
sqlite3 hockey-analytics/*.db "VACUUM;"

# Security updates
pip list --outdated
npm audit
```

### Monthly Maintenance
```bash
# Full system backup
tar -czf backup-$(date +%Y%m%d).tar.gz /path/to/tsai-jarvis

# Security audit
pip audit
npm audit

# Performance analysis
grep -i "slow\|timeout" logs/jarvis-*.log | wc -l

# Capacity planning
df -h
free -h
```

## 🔄 Service Updates

### Rolling Updates
```bash
# 1. Backup current state
./jarvis.sh stop
cp -r /path/to/tsai-jarvis /path/to/tsai-jarvis-backup

# 2. Update code
git pull origin main

# 3. Update dependencies
pip install -r requirements.txt
npm install

# 4. Start services
./jarvis.sh start --all

# 5. Verify health
./jarvis.sh status
```

### Blue-Green Deployment
```bash
# 1. Prepare new environment
cp -r /path/to/tsai-jarvis /path/to/tsai-jarvis-new

# 2. Update new environment
cd /path/to/tsai-jarvis-new
git pull origin main
pip install -r requirements.txt

# 3. Test new environment
./jarvis.sh start --all
./jarvis.sh status

# 4. Switch traffic (update load balancer)
# 5. Stop old environment
cd /path/to/tsai-jarvis
./jarvis.sh stop
```

## 📊 Service Metrics

### Key Performance Indicators
- **Uptime**: > 99.9%
- **Response Time**: < 200ms
- **Error Rate**: < 1%
- **Throughput**: > 1000 req/min
- **CPU Usage**: < 80%
- **Memory Usage**: < 85%
- **Disk Usage**: < 90%

### Service-Specific Metrics
```bash
# Dashboard metrics
curl -s http://localhost:8000 | wc -c

# API response times
for port in 8001 8007 8008 8009 8010 8011; do
  echo "Port $port:"
  time curl -s http://localhost:$port/health > /dev/null
done

# Database performance
sqlite3 hockey-analytics/*.db "PRAGMA table_info(analytics);"
```

## 🚨 Emergency Procedures

### Service Recovery
```bash
# 1. Identify failed service
./jarvis.sh status

# 2. Check logs
tail -f logs/jarvis-*.log

# 3. Restart service
./jarvis.sh stop
./jarvis.sh start --all

# 4. Verify recovery
./jarvis.sh status
```

### Data Recovery
```bash
# 1. Stop all services
./jarvis.sh stop

# 2. Restore from backup
cp /path/to/backup/*.db hockey-analytics/

# 3. Start services
./jarvis.sh start --all

# 4. Verify data integrity
sqlite3 hockey-analytics/*.db "SELECT COUNT(*) FROM analytics;"
```

### Complete System Recovery
```bash
# 1. Stop all services
./jarvis.sh stop

# 2. Restore from backup
tar -xzf backup-*.tar.gz

# 3. Update dependencies
pip install -r requirements.txt
npm install

# 4. Start services
./jarvis.sh start --all

# 5. Verify all services
./jarvis.sh status
```

## 📞 Escalation Procedures

### Level 1: On-Call Engineer
- Service restart procedures
- Basic troubleshooting
- Log analysis
- Health check verification

### Level 2: Senior Engineer
- Complex troubleshooting
- Performance analysis
- System configuration
- Database issues

### Level 3: Engineering Manager
- Architecture decisions
- Major system changes
- Security incidents
- Business impact assessment

### Level 4: Director
- Business continuity
- Resource allocation
- Strategic decisions
- External communication

## 📋 Checklists

### Pre-Deployment Checklist
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance tests passed
- [ ] Documentation updated
- [ ] Rollback plan prepared
- [ ] Team notified
- [ ] Monitoring configured
- [ ] Alerts set up

### Post-Deployment Checklist
- [ ] All services running
- [ ] Health checks passing
- [ ] Performance metrics normal
- [ ] Logs clean
- [ ] Monitoring active
- [ ] Team notified
- [ ] Documentation updated

### Incident Response Checklist
- [ ] Incident identified
- [ ] Impact assessed
- [ ] Team notified
- [ ] Troubleshooting started
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Services restored
- [ ] Post-incident review scheduled
