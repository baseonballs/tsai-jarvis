# TSAI Jarvis CLI - Production-Ready Service Management

A robust, production-ready command-line interface for managing TSAI Jarvis Hockey Analytics services.

## 🚀 Quick Start

```bash
# Start all services
./jarvis.sh start --all

# Check service status
./jarvis.sh status

# Stop all services
./jarvis.sh stop

# Get help
./jarvis.sh help
```

## 📋 Commands

### Start Services
```bash
# Start dashboard and backend services
./jarvis.sh start --dashboard

# Start all services
./jarvis.sh start --all

# Start in background mode
./jarvis.sh start --all --background

# Force restart (stop all, then start)
./jarvis.sh start --all --force
```

### Stop Services
```bash
# Stop all services
./jarvis.sh stop
```

### Status & Help
```bash
# Check service status
./jarvis.sh status

# Show help
./jarvis.sh help
```

## 🏗️ Architecture

The CLI manages the following services:

| Service | Port | Description |
|---------|------|-------------|
| Dashboard | 8000 | Next.js frontend interface |
| Core API | 8001 | Main hockey analytics API |
| Enterprise Integration | 8007 | Professional dashboard & NHL integration |
| Advanced Visualization | 8008 | 3D ice rink & player tracking |
| Machine Learning | 8009 | AI/ML models & predictions |
| Enterprise Security | 8010 | Authentication & authorization |
| Advanced Analytics | 8011 | Statistical analysis & insights |

## 🔧 Features

### Production-Ready
- **Robust Error Handling**: Comprehensive error detection and recovery
- **Process Management**: PID tracking and graceful shutdown
- **Port Conflict Resolution**: Automatic cleanup of conflicting processes
- **Health Checks**: Service startup verification and monitoring
- **Resource Monitoring**: System resource validation

### User-Friendly
- **Beautiful Output**: Colorized, structured logging with ASCII art
- **Progress Tracking**: Real-time startup progress and status updates
- **Comprehensive Help**: Detailed documentation and examples
- **Background Mode**: Run services in background for production use

### Advanced Management
- **Dependency Checking**: Validates Python, Node.js, and system requirements
- **Virtual Environment**: Automatic Python venv creation and activation
- **Service Dependencies**: Intelligent startup order based on service dependencies
- **Cleanup**: Comprehensive process and port cleanup on shutdown

## 🌐 Service URLs

When services are running, access them at:

- **🎨 Dashboard**: http://localhost:8000
- **⚙️ Core API**: http://localhost:8001
- **🏢 Enterprise Integration**: http://localhost:8007
- **📊 Advanced Visualization**: http://localhost:8008
- **🤖 Machine Learning**: http://localhost:8009
- **🔒 Enterprise Security**: http://localhost:8010
- **📈 Advanced Analytics**: http://localhost:8011

## 🔧 Configuration

### Environment Variables

Customize service ports using environment variables:

```bash
export DASHBOARD_PORT=8000
export CORE_API_PORT=8001
export ENTERPRISE_INTEGRATION_PORT=8007
export ADVANCED_VISUALIZATION_PORT=8008
export MACHINE_LEARNING_PORT=8009
export ENTERPRISE_SECURITY_PORT=8010
export ADVANCED_ANALYTICS_PORT=8011
```

### Production Settings

```bash
export PRODUCTION_MODE=true
export LOG_LEVEL=info
export MAX_RETRIES=3
export STARTUP_TIMEOUT=30
export SHUTDOWN_TIMEOUT=15
```

## 📊 Service Management

### Process Tracking
The CLI uses PID files to track running services:
- `/tmp/tsai-jarvis-dashboard.pid`
- `/tmp/tsai-jarvis-core-api.pid`
- `/tmp/tsai-jarvis-enterprise-integration.pid`
- `/tmp/tsai-jarvis-advanced-visualization.pid`
- `/tmp/tsai-jarvis-machine-learning.pid`
- `/tmp/tsai-jarvis-enterprise-security.pid`
- `/tmp/tsai-jarvis-advanced-analytics.pid`

### Health Checks
Services are verified through:
- **Process Validation**: PID file existence and process health
- **Port Binding**: Confirmation that services are listening on assigned ports
- **HTTP Health Checks**: API endpoint availability verification
- **Startup Timeout**: Configurable startup verification periods

## 🛠️ Troubleshooting

### Common Issues

**Port Conflicts**
```bash
# Check what's using a port
lsof -i :8000

# Force cleanup
./jarvis.sh stop
```

**Service Won't Start**
```bash
# Check dependencies
./jarvis.sh help  # Shows dependency check

# Force restart
./jarvis.sh start --all --force
```

**Background Services**
```bash
# Start in background
./jarvis.sh start --all --background

# Check status
./jarvis.sh status
```

### Manual Cleanup

If services get stuck, manually clean up:

```bash
# Kill all Jarvis processes
pkill -f "tsai-jarvis"
pkill -f "hockey-analytics"
pkill -f "uvicorn.*jarvis"

# Remove PID files
rm -f /tmp/tsai-jarvis-*.pid

# Check ports
lsof -i :8000-8011
```

## 🔒 Security

The CLI includes enterprise-grade security features:

- **Process Isolation**: Each service runs in its own process space
- **Port Security**: Automatic port conflict resolution
- **Resource Limits**: System resource monitoring and warnings
- **Clean Shutdown**: Graceful process termination with fallback to force kill

## 📈 Monitoring

### Service Health
- **Startup Verification**: Confirms services are running and healthy
- **Port Monitoring**: Tracks port usage and conflicts
- **Process Tracking**: Monitors service PIDs and health
- **Resource Monitoring**: Checks system resources and disk space

### Status Reporting
```bash
# Detailed status report
./jarvis.sh status

# Shows:
# - Service status (running/stopped)
# - Process IDs
# - Service URLs
# - Port usage
# - Health status
```

## 🚀 Production Deployment

### Background Mode
```bash
# Start all services in background
./jarvis.sh start --all --background

# Services continue running after terminal closes
# Use status to check health
./jarvis.sh status
```

### Service Management
```bash
# Stop all services
./jarvis.sh stop

# Restart with force cleanup
./jarvis.sh start --all --force
```

### Monitoring
```bash
# Regular health checks
./jarvis.sh status

# Check specific services
curl http://localhost:8000  # Dashboard
curl http://localhost:8001/health  # Core API
curl http://localhost:8007/health  # Enterprise Integration
```

## 📚 Examples

### Development Workflow
```bash
# Start development environment
./jarvis.sh start --all

# Check status
./jarvis.sh status

# Stop when done
./jarvis.sh stop
```

### Production Deployment
```bash
# Start production services
./jarvis.sh start --all --background

# Monitor health
./jarvis.sh status

# Graceful shutdown
./jarvis.sh stop
```

### Service-Specific Management
```bash
# Start only dashboard
./jarvis.sh start --dashboard

# Check dashboard status
curl http://localhost:8000
```

## 🔧 Development

### Adding New Services

1. **Add Service Configuration**:
   ```bash
   readonly NEW_SERVICE_PORT="${NEW_SERVICE_PORT:-8012}"
   readonly NEW_SERVICE_PID_FILE="/tmp/tsai-jarvis-new-service.pid"
   ```

2. **Add Service Function**:
   ```bash
   start_new_service() {
       local background_mode="${1:-false}"
       # Service startup logic
   }
   ```

3. **Update Service Lists**:
   ```bash
   # Add to services array
   "New Service:$NEW_SERVICE_PORT:start_new_service"
   ```

4. **Add to Status Check**:
   ```bash
   # Add status checking logic
   log_step "Checking New Service..."
   if is_port_in_use "$NEW_SERVICE_PORT"; then
       log_success "New Service: Running"
   fi
   ```

### Customization

The CLI is designed to be easily customizable:

- **Service Ports**: Modify port assignments in configuration section
- **Startup Order**: Adjust service startup sequence in `start_all()`
- **Health Checks**: Customize health check logic per service
- **Logging**: Modify log output format and verbosity

## 📄 License

This CLI is part of the TSAI Jarvis Hockey Analytics Platform.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review service logs
- Use `./jarvis.sh help` for command reference
- Check service status with `./jarvis.sh status`
