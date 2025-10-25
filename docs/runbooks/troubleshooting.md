# Troubleshooting Runbook

Comprehensive troubleshooting guide for TSAI Jarvis platform issues.

## 🎯 Overview

This runbook provides step-by-step troubleshooting procedures for common issues in the TSAI Jarvis platform.

## 🚨 Emergency Response

### Critical Issues (P0)
- **Service Down**: Complete service failure
- **Data Loss**: Data corruption or loss
- **Security Breach**: Unauthorized access
- **Performance Degradation**: > 50% performance drop

### High Priority Issues (P1)
- **Partial Service Failure**: Some services down
- **Performance Issues**: 20-50% performance drop
- **Authentication Issues**: Login problems
- **API Errors**: High error rates

### Medium Priority Issues (P2)
- **Minor Performance Issues**: < 20% performance drop
- **UI Issues**: Frontend problems
- **Feature Issues**: Non-critical features
- **Monitoring Issues**: Alert problems

## 🔍 Diagnostic Procedures

### System Health Check
```bash
# 1. Check overall system status
./jarvis.sh status

# 2. Check system resources
htop
df -h
free -h

# 3. Check network connectivity
ping -c 3 localhost
netstat -an | grep -E ":(8000|8001|8007|8008|8009|8010|8011)"

# 4. Check process status
ps aux | grep -E "(jarvis|hockey-analytics|uvicorn)" | grep -v grep
```

### Service-Specific Diagnostics
```bash
# Dashboard (port 8000)
curl -v http://localhost:8000
curl -v http://localhost:8000/health

# Core API (port 8001)
curl -v http://localhost:8001/health
curl -v http://localhost:8001/docs

# Enterprise Integration (port 8007)
curl -v http://localhost:8007/health
curl -v http://localhost:8007/docs

# Advanced Visualization (port 8008)
curl -v http://localhost:8008/health
curl -v http://localhost:8008/docs

# Machine Learning (port 8009)
curl -v http://localhost:8009/health
curl -v http://localhost:8009/docs

# Enterprise Security (port 8010)
curl -v http://localhost:8010/health
curl -v http://localhost:8010/docs

# Advanced Analytics (port 8011)
curl -v http://localhost:8011/health
curl -v http://localhost:8011/docs
```

## 🐛 Common Issues & Solutions

### Service Won't Start

#### Symptoms
- Service fails to start
- Port already in use errors
- Import errors
- Permission errors

#### Diagnosis
```bash
# Check for port conflicts
lsof -i :8000-8011

# Check for process conflicts
ps aux | grep -E "(jarvis|hockey-analytics|uvicorn)"

# Check system resources
free -h
df -h

# Check logs
tail -f logs/jarvis-*.log
```

#### Solutions
```bash
# 1. Kill conflicting processes
for port in 8000 8001 8007 8008 8009 8010 8011; do
  lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

# 2. Clean up PID files
rm -f /tmp/tsai-jarvis-*.pid

# 3. Restart services
./jarvis.sh start --all --force

# 4. Check system resources
free -h
df -h
```

### Service Crashes

#### Symptoms
- Service stops unexpectedly
- Error logs show crashes
- High memory usage
- CPU spikes

#### Diagnosis
```bash
# Check crash logs
grep -i "crash\|segfault\|killed\|exception" logs/jarvis-*.log

# Check system logs
dmesg | tail -20

# Check resource usage
ps aux --sort=-%mem | head -10
ps aux --sort=-%cpu | head -10

# Check file descriptors
lsof | grep -E "(jarvis|hockey-analytics)" | wc -l
```

#### Solutions
```bash
# 1. Restart crashed service
./jarvis.sh stop
./jarvis.sh start --all

# 2. Check resource limits
ulimit -a

# 3. Increase limits if needed
ulimit -n 65536
ulimit -u 32768

# 4. Monitor resource usage
htop
```

### Performance Issues

#### Symptoms
- Slow response times
- High CPU usage
- High memory usage
- Timeout errors

#### Diagnosis
```bash
# Check response times
for port in 8000 8001 8007 8008 8009 8010 8011; do
  echo "Port $port response time:"
  time curl -s http://localhost:$port > /dev/null
done

# Check resource usage
top -l 1
htop

# Check slow queries
grep -i "slow\|timeout\|latency" logs/jarvis-*.log

# Check database performance
sqlite3 hockey-analytics/*.db "PRAGMA table_info(analytics);"
```

#### Solutions
```bash
# 1. Restart services
./jarvis.sh stop
./jarvis.sh start --all

# 2. Check system resources
free -h
df -h

# 3. Optimize database
sqlite3 hockey-analytics/*.db "VACUUM;"
sqlite3 hockey-analytics/*.db "ANALYZE;"

# 4. Check for memory leaks
ps aux --sort=-%mem | head -10
```

### Database Issues

#### Symptoms
- Database connection errors
- Data corruption
- Slow queries
- Lock timeouts

#### Diagnosis
```bash
# Check database files
ls -la hockey-analytics/*.db

# Check database integrity
sqlite3 hockey-analytics/*.db "PRAGMA integrity_check;"

# Check database size
du -h hockey-analytics/*.db

# Check database locks
sqlite3 hockey-analytics/*.db "PRAGMA database_list;"
```

#### Solutions
```bash
# 1. Backup database
cp hockey-analytics/*.db hockey-analytics/*.db.backup

# 2. Repair database
sqlite3 hockey-analytics/*.db "VACUUM;"
sqlite3 hockey-analytics/*.db "REINDEX;"

# 3. Check integrity
sqlite3 hockey-analytics/*.db "PRAGMA integrity_check;"

# 4. Restart services
./jarvis.sh stop
./jarvis.sh start --all
```

### Authentication Issues

#### Symptoms
- Login failures
- Token errors
- Permission denied
- Session timeouts

#### Diagnosis
```bash
# Check authentication service
curl -v http://localhost:8010/health

# Check authentication logs
grep -i "auth\|login\|token" logs/jarvis-*.log

# Check database
sqlite3 hockey-analytics/*.db "SELECT * FROM users LIMIT 5;"

# Check JWT tokens
grep -i "jwt\|token" logs/jarvis-*.log
```

#### Solutions
```bash
# 1. Restart authentication service
./jarvis.sh stop
./jarvis.sh start --all

# 2. Check database
sqlite3 hockey-analytics/*.db "PRAGMA integrity_check;"

# 3. Clear expired tokens
# (Implementation depends on token storage)

# 4. Verify authentication configuration
grep -i "jwt\|secret" hockey-analytics/enterprise_security_api.py
```

### API Errors

#### Symptoms
- High error rates
- 500 errors
- Timeout errors
- Connection refused

#### Diagnosis
```bash
# Check API health
for port in 8001 8007 8008 8009 8010 8011; do
  echo "Port $port:"
  curl -s http://localhost:$port/health || echo "FAILED"
done

# Check error logs
grep -i "error\|exception\|failed" logs/jarvis-*.log | tail -20

# Check API response times
for port in 8001 8007 8008 8009 8010 8011; do
  echo "Port $port response time:"
  time curl -s http://localhost:$port/health > /dev/null
done
```

#### Solutions
```bash
# 1. Restart API services
./jarvis.sh stop
./jarvis.sh start --all

# 2. Check system resources
free -h
df -h

# 3. Check for memory leaks
ps aux --sort=-%mem | head -10

# 4. Verify API configuration
grep -i "host\|port" hockey-analytics/*_api.py
```

## 🔧 Advanced Troubleshooting

### Memory Issues

#### Symptoms
- High memory usage
- Out of memory errors
- Slow performance
- Service crashes

#### Diagnosis
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check memory leaks
valgrind --tool=memcheck python hockey-analytics/*_api.py

# Check swap usage
swapon -s
```

#### Solutions
```bash
# 1. Restart services
./jarvis.sh stop
./jarvis.sh start --all

# 2. Check for memory leaks
ps aux --sort=-%mem | head -10

# 3. Increase memory limits
ulimit -v 2097152  # 2GB

# 4. Monitor memory usage
htop
```

### Network Issues

#### Symptoms
- Connection refused
- Timeout errors
- Network unreachable
- Port conflicts

#### Diagnosis
```bash
# Check network connectivity
ping -c 3 localhost
netstat -an | grep -E ":(8000|8001|8007|8008|8009|8010|8011)"

# Check port usage
lsof -i :8000-8011

# Check firewall
sudo iptables -L
```

#### Solutions
```bash
# 1. Check port conflicts
lsof -i :8000-8011

# 2. Kill conflicting processes
for port in 8000 8001 8007 8008 8009 8010 8011; do
  lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

# 3. Restart services
./jarvis.sh start --all

# 4. Check firewall rules
sudo iptables -L
```

### Disk Issues

#### Symptoms
- Disk full errors
- Slow I/O
- Permission denied
- File system errors

#### Diagnosis
```bash
# Check disk usage
df -h
du -h logs/
du -h hockey-analytics/

# Check disk I/O
iostat -x 1 5

# Check file system
fsck /dev/sda1
```

#### Solutions
```bash
# 1. Clean up logs
find logs/ -name "*.log" -mtime +7 -delete

# 2. Clean up temporary files
rm -rf /tmp/tsai-jarvis-*

# 3. Compress old logs
gzip logs/jarvis-*.log

# 4. Check disk space
df -h
```

## 📊 Performance Analysis

### CPU Analysis
```bash
# Check CPU usage
top -l 1
htop

# Check CPU per process
ps aux --sort=-%cpu | head -10

# Check CPU load
uptime
```

### Memory Analysis
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10

# Check memory per process
pmap -x <PID>

# Check swap usage
swapon -s
```

### I/O Analysis
```bash
# Check disk I/O
iostat -x 1 5

# Check network I/O
netstat -i

# Check file descriptors
lsof | grep -E "(jarvis|hockey-analytics)" | wc -l
```

## 🚨 Emergency Procedures

### Complete System Failure
```bash
# 1. Stop all services
./jarvis.sh stop

# 2. Check system resources
free -h
df -h

# 3. Restart services
./jarvis.sh start --all

# 4. Verify all services
./jarvis.sh status
```

### Data Corruption
```bash
# 1. Stop all services
./jarvis.sh stop

# 2. Backup current data
cp hockey-analytics/*.db hockey-analytics/*.db.corrupted

# 3. Restore from backup
cp /path/to/backup/*.db hockey-analytics/

# 4. Start services
./jarvis.sh start --all

# 5. Verify data integrity
sqlite3 hockey-analytics/*.db "PRAGMA integrity_check;"
```

### Security Incident
```bash
# 1. Stop all services
./jarvis.sh stop

# 2. Check logs for suspicious activity
grep -i "unauthorized\|hack\|breach" logs/jarvis-*.log

# 3. Check system for intrusions
ps aux | grep -v grep
netstat -an | grep ESTABLISHED

# 4. Contact security team
# 5. Document incident
```

## 📋 Troubleshooting Checklists

### Pre-Troubleshooting Checklist
- [ ] Document the issue
- [ ] Check system status
- [ ] Review recent changes
- [ ] Check logs
- [ ] Verify backups

### During Troubleshooting Checklist
- [ ] Isolate the problem
- [ ] Check system resources
- [ ] Review error logs
- [ ] Test solutions
- [ ] Document findings

### Post-Troubleshooting Checklist
- [ ] Verify fix
- [ ] Monitor system
- [ ] Update documentation
- [ ] Schedule follow-up
- [ ] Learn from incident

## 📞 Escalation Procedures

### Level 1: On-Call Engineer
- Basic troubleshooting
- Service restart
- Log analysis
- Health checks

### Level 2: Senior Engineer
- Complex troubleshooting
- Performance analysis
- System configuration
- Database issues

### Level 3: Engineering Manager
- Architecture decisions
- Major system changes
- Security incidents
- Business impact

### Level 4: Director
- Business continuity
- Resource allocation
- Strategic decisions
- External communication

## 📚 Additional Resources

- [Service Management Runbook](./service-management.md)
- [Monitoring & Alerting Runbook](./monitoring-alerting.md)
- [Security Operations Runbook](./security-operations.md)
- [TSAI Jarvis CLI Documentation](../README-JARVIS-CLI.md)
