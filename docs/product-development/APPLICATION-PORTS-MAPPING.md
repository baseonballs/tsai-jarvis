# TSAI Jarvis - Application Ports Mapping

## Overview

This document defines the port allocation strategy for the TSAI Jarvis Hockey Analytics platform and ensures no conflicts with other TSAI ecosystem services.

## Port Allocation Strategy

### TSAI Jarvis Dedicated Port Range: **8000-8099**

TSAI Jarvis has been assigned the **8000-8099** port range to avoid conflicts with other TSAI services:

- **8000-8099**: TSAI Jarvis Hockey Analytics Platform (Reserved)
- **8100-8199**: Future TSAI Jarvis extensions (Reserved)

## Current TSAI Jarvis Service Ports

### Core Services
| Service | Port | URL | Description | Status |
|---------|------|-----|-------------|--------|
| **Main Dashboard** | 8000 | http://localhost:8000 | Next.js Frontend Dashboard | ✅ **Active** |
| **Core API** | 8001 | http://localhost:8001 | TSAI Jarvis Core API | 🎯 **Target** |
| **API Gateway** | 8002 | http://localhost:8002 | Main API Gateway | 🎯 **Target** |
| **Health Check** | 8000 | http://localhost:8000/health | Service Health | ✅ **Active** |

### Phase 2.0 Enterprise Services
| Service | Port | URL | Description | Status |
|---------|------|-----|-------------|--------|
| **Advanced AI Models** | 8003 | http://localhost:8003 | Phase 2.1: AI Models API | ✅ **Active** |
| **Real-time Streaming** | 8004 | http://localhost:8004 | Phase 2.2: Streaming API | ✅ **Active** |
| **Enterprise Integration** | 8007 | http://localhost:8007 | Phase 2.3: Enterprise API | ✅ **Active** |
| **Advanced Visualization** | 8008 | http://localhost:8008 | Phase 2.4: Visualization API | ✅ **Active** |
| **Machine Learning** | 8009 | http://localhost:8009 | Phase 2.5: ML API | ✅ **Active** |
| **Enterprise Security** | 8010 | http://localhost:8010 | Phase 2.6: Security API | ✅ **Active** |
| **Advanced Analytics** | 8011 | http://localhost:8011 | Phase 2.7: Analytics API | ✅ **Active** |
| **Image Analysis** | 8012 | http://localhost:8012 | Phase 2.8: Image Analysis API | ✅ **Active** |

### Reserved Ports
| Port Range | Purpose | Status |
|------------|---------|--------|
| 8005-8006 | Future Phase 2.x services | 🔒 **Reserved** |
| 8013-8099 | Future TSAI Jarvis services | 🔒 **Reserved** |

## TSAI Ecosystem Port Conflicts Analysis

### TSAI Autopilot (4000-4999 Range)
| Service | Port | Description | Conflict Status |
|---------|------|-------------|----------------|
| Frontend Dashboard | 4000 | Next.js UI | ✅ **No Conflict** |
| Backend API | 4001 | FastAPI Service | ✅ **No Conflict** |
| AI Pipeline | 4002 | AI Processing | ✅ **No Conflict** |
| Model Management | 4003 | Model Services | ✅ **No Conflict** |
| Grafana Dashboard | 3000 | Monitoring | ⚠️ **Conflicts with Jarvis Dashboard** |

### TSAI Sherlock (17000-17999 Range)
| Service | Port | Description | Conflict Status |
|---------|------|-------------|----------------|
| Main API | 17501 | Flask/FastAPI | ✅ **No Conflict** |
| Detection API | 5001 | Detection Service | ✅ **No Conflict** |
| Frontend | 3000 | Next.js UI | ⚠️ **Conflicts with Jarvis Dashboard** |
| FastAPI | 17601 | Enhanced API | ✅ **No Conflict** |

### TSAI Spotlight (8000-8999 Range)
| Service | Port | Description | Conflict Status |
|---------|------|-------------|----------------|
| Dashboard | 8080 | Analytics Dashboard | ⚠️ **Conflicts with Jarvis Range** |
| Dashboard Alt | 8081 | Alternative Port | ⚠️ **Conflicts with Jarvis Range** |
| Dashboard Alt | 8082 | Alternative Port | ⚠️ **Conflicts with Jarvis Range** |

### TSAI Watson & Holmes
| Service | Port | Description | Conflict Status |
|---------|------|-------------|----------------|
| Watson | N/A | No active services | ✅ **No Conflict** |
| Holmes | N/A | No active services | ✅ **No Conflict** |

### TSAI Toolchain
| Service | Port | Description | Conflict Status |
|---------|------|-------------|----------------|
| Tools | N/A | No web services | ✅ **No Conflict** |

## Port Conflict Resolution

### Current Conflicts Identified

1. **Port 3000**: Grafana (Docker) vs TSAI Jarvis Dashboard
   - **Issue**: `http://localhost:3000` shows Grafana instead of Jarvis
   - **Solution**: Move Jarvis Dashboard to port 8000
   - **Action**: Update Next.js configuration

2. **Port 8080**: TSAI Spotlight vs TSAI Jarvis
   - **Issue**: Spotlight uses 8080, conflicts with Jarvis range
   - **Solution**: Move Spotlight to 9000+ range
   - **Action**: Update Spotlight configuration

### Recommended Port Reassignments

#### TSAI Spotlight (Move to 9000+ Range)
| Current Port | New Port | Service | Action Required |
|--------------|----------|---------|----------------|
| 8080 | 9000 | Main Dashboard | Update `launch_dashboard.py` |
| 8081 | 9001 | Alt Dashboard | Update configuration |
| 8082 | 9002 | Alt Dashboard | Update configuration |

#### TSAI Jarvis (Use 8000+ Range)
| Service | Port | Action Required |
|---------|------|----------------|
| Main Dashboard | 8000 | Update Next.js config |
| API Gateway | 8001 | Already configured |
| All Phase APIs | 8002-8011 | Already configured |

## Implementation Plan

### Phase 1: Immediate Fixes
1. **Stop Grafana on port 3000**
   ```bash
   # Stop Docker Grafana
   docker stop $(docker ps -q --filter "publish=3000")
   ```

2. **Update TSAI Jarvis Dashboard to port 8000**
   ```bash
   # Update Next.js configuration
   # File: next.config.js
   const nextConfig = {
     port: 8000
   }
   ```

3. **Update TSAI Spotlight to port 9000**
   ```bash
   # Update launch_dashboard.py
   # Change default port from 8080 to 9000
   ```

### Phase 2: Service Verification
1. **Verify all TSAI Jarvis services on 8000+ range**
2. **Test dashboard accessibility on port 8000**
3. **Confirm no port conflicts**

### Phase 3: Documentation Update
1. **Update all service documentation**
2. **Update startup scripts**
3. **Update health check endpoints**

## Service Health Check Commands

### TSAI Jarvis Services
```bash
# Main Dashboard
curl http://localhost:8000

# API Gateway
curl http://localhost:8001/health

# Phase 2.1: Advanced AI Models
curl http://localhost:8002/health

# Phase 2.2: Real-time Streaming
curl http://localhost:8003/health

# Phase 2.3: Enterprise Integration
curl http://localhost:8007/health

# Phase 2.4: Advanced Visualization
curl http://localhost:8008/health

# Phase 2.5: Machine Learning
curl http://localhost:8009/health

# Phase 2.6: Enterprise Security
curl http://localhost:8010/health

# Phase 2.7: Advanced Analytics
curl http://localhost:8011/health
```

### Port Conflict Detection
```bash
# Check what's running on specific ports
lsof -i :3000  # Should be empty after fixes
lsof -i :8000  # Should show TSAI Jarvis Dashboard
lsof -i :8080  # Should be empty after Spotlight move
```

## Port Management Best Practices

### Development Environment
1. **Always check port availability before starting services**
2. **Use dedicated port ranges for each TSAI service**
3. **Document port assignments in this file**
4. **Update this document when adding new services**

### Production Environment
1. **Use reverse proxy (nginx) for port management**
2. **Map internal ports to standard HTTP/HTTPS ports**
3. **Use environment variables for port configuration**
4. **Implement port health checks**

## Future Port Planning

### TSAI Jarvis Extensions (8012-8099)
- **8012-8019**: Additional Phase 2.x services
- **8020-8029**: Phase 3.0 services
- **8030-8039**: Integration services
- **8040-8049**: Monitoring services
- **8050-8059**: Development services
- **8060-8099**: Future services

### TSAI Ecosystem Port Ranges
- **TSAI Autopilot**: 4000-4999
- **TSAI Sherlock**: 17000-17999
- **TSAI Jarvis**: 8000-8099
- **TSAI Spotlight**: 9000-9099 (after move)
- **TSAI Watson**: 10000-10099 (reserved)
- **TSAI Holmes**: 10100-10199 (reserved)
- **TSAI Toolchain**: 10200-10299 (reserved)

## Troubleshooting

### Common Issues
1. **Port already in use**: Check `lsof -i :PORT` and kill conflicting processes
2. **Service not accessible**: Verify port configuration and firewall settings
3. **Dashboard shows wrong service**: Check port conflicts and service priorities

### Debug Commands
```bash
# Check all listening ports
netstat -tulpn | grep LISTEN

# Check specific port
lsof -i :PORT_NUMBER

# Kill process on port
sudo kill -9 $(lsof -t -i:PORT_NUMBER)

# Check Docker containers
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

---

**Last Updated**: January 2025  
**Maintainer**: TSAI Development Team  
**Version**: 1.0
