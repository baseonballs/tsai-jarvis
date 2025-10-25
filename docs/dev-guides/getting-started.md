# Getting Started Guide

Complete guide for new developers joining the TSAI Jarvis project.

## 🎯 Overview

This guide will help you set up your development environment and get started with the TSAI Jarvis Hockey Analytics Platform.

## 📋 Prerequisites

### Required Software
- **Python 3.14+**: [Download](https://python.org/downloads/)
- **Node.js 22+**: [Download](https://nodejs.org/downloads/)
- **Git**: [Download](https://git-scm.com/downloads)
- **Code Editor**: VS Code, PyCharm, or similar

### Optional Software
- **Docker**: [Download](https://docker.com/get-started)
- **Postman**: [Download](https://postman.com/downloads/)
- **Database Browser**: [DB Browser for SQLite](https://sqlitebrowser.org/)

### System Requirements
- **OS**: macOS, Linux, or Windows
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ free space
- **CPU**: 4+ cores recommended

## 🚀 Quick Start

### 1. Clone Repository
```bash
# Clone the repository
git clone https://github.com/baseonballs/tsai-jarvis.git
cd tsai-jarvis

# Check out main branch
git checkout main
```

### 2. Python Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Node.js Environment Setup
```bash
# Navigate to dashboard directory
cd dashboard

# Install dependencies
npm install

# Return to project root
cd ..
```

### 4. Start Development Services
```bash
# Start all services
./jarvis.sh start --all

# Check service status
./jarvis.sh status

# View logs
tail -f logs/jarvis-*.log
```

### 5. Verify Installation
```bash
# Check dashboard
open http://localhost:8000

# Check API documentation
open http://localhost:8001/docs
open http://localhost:8007/docs
open http://localhost:8008/docs
open http://localhost:8009/docs
open http://localhost:8010/docs
open http://localhost:8011/docs
```

## 🏗️ Project Structure

```
tsai-jarvis/
├── dashboard/                 # Next.js frontend
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── next.config.ts
├── hockey-analytics/         # Python APIs
│   ├── main_api.py
│   ├── enterprise_integration_api.py
│   ├── advanced_visualization_api.py
│   ├── machine_learning_api.py
│   ├── enterprise_security_api.py
│   ├── advanced_analytics_api.py
│   └── *.db
├── docs/                     # Documentation
├── logs/                     # Log files
├── requirements.txt          # Python dependencies
├── jarvis.sh                # CLI management script
└── README.md
```

## 🔧 Development Environment

### IDE Setup

#### VS Code (Recommended)
```bash
# Install VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-python.flake8
code --install-extension ms-python.black
code --install-extension bradlc.vscode-tailwindcss
code --install-extension esbenp.prettier-vscode
```

#### PyCharm
1. Open project in PyCharm
2. Configure Python interpreter to use `venv/bin/python`
3. Install plugins: Python, SQLite, Docker

### Environment Variables
```bash
# Create .env file
cat > .env << EOF
# Development settings
NODE_ENV=development
PYTHON_ENV=development

# Service ports
DASHBOARD_PORT=8000
CORE_API_PORT=8001
ENTERPRISE_INTEGRATION_PORT=8007
ADVANCED_VISUALIZATION_PORT=8008
MACHINE_LEARNING_PORT=8009
ENTERPRISE_SECURITY_PORT=8010
ADVANCED_ANALYTICS_PORT=8011

# Database settings
DATABASE_URL=sqlite:///hockey-analytics/analytics.db

# Security settings
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=info
LOG_FILE=logs/jarvis.log
EOF
```

## 🧪 Development Workflow

### Daily Development
```bash
# 1. Start your day
cd /path/to/tsai-jarvis
source venv/bin/activate

# 2. Check service status
./jarvis.sh status

# 3. Start services if needed
./jarvis.sh start --all

# 4. Make your changes
# Edit code, add features, fix bugs

# 5. Test your changes
python -m pytest tests/
npm test

# 6. End your day
./jarvis.sh stop
```

### Feature Development
```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes
# Edit code, add tests, update documentation

# 3. Test locally
./jarvis.sh start --all
# Manual testing
# Automated testing

# 4. Commit changes
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name

# 5. Create pull request
# GitHub will show PR creation link
```

### Bug Fixing
```bash
# 1. Reproduce the bug
./jarvis.sh start --all
# Reproduce the issue

# 2. Debug the issue
tail -f logs/jarvis-*.log
# Check logs for errors

# 3. Fix the bug
# Make necessary changes

# 4. Test the fix
./jarvis.sh stop
./jarvis.sh start --all
# Verify fix works

# 5. Commit fix
git add .
git commit -m "Fix: description of the bug"
git push origin your-branch
```

## 🔍 Debugging

### Python Debugging
```bash
# Debug specific API
python -m pdb hockey-analytics/main_api.py

# Debug with logging
PYTHONPATH=. python hockey-analytics/main_api.py

# Check imports
python -c "import hockey-analytics.main_api"
```

### Node.js Debugging
```bash
# Debug dashboard
cd dashboard
npm run dev -- --inspect

# Debug with breakpoints
node --inspect-brk node_modules/.bin/next dev -p 8000
```

### Service Debugging
```bash
# Check service logs
tail -f logs/jarvis-*.log

# Check specific service
grep "ERROR" logs/jarvis-dashboard.log
grep "ERROR" logs/jarvis-core-api.log

# Check system resources
htop
df -h
free -h
```

## 🧪 Testing

### Running Tests
```bash
# Python tests
python -m pytest tests/
python -m pytest tests/ -v
python -m pytest tests/ --cov=hockey-analytics

# Node.js tests
cd dashboard
npm test
npm run test:watch
npm run test:coverage
```

### Writing Tests
```python
# tests/test_main_api.py
import pytest
from fastapi.testclient import TestClient
from hockey-analytics.main_api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_game_state():
    response = client.get("/api/game/state")
    assert response.status_code == 200
    assert "home_team" in response.json()
```

```javascript
// dashboard/tests/components/GameState.test.js
import { render, screen } from '@testing-library/react'
import GameState from '../components/GameState'

test('renders game state', () => {
  render(<GameState />)
  expect(screen.getByText('Toronto Maple Leafs')).toBeInTheDocument()
})
```

## 📊 Performance Monitoring

### Development Metrics
```bash
# Check response times
for port in 8000 8001 8007 8008 8009 8010 8011; do
  echo "Port $port:"
  time curl -s http://localhost:$port > /dev/null
done

# Check resource usage
htop
df -h
free -h

# Check API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8001/health
```

### Profiling
```bash
# Python profiling
python -m cProfile hockey-analytics/main_api.py

# Node.js profiling
cd dashboard
node --prof node_modules/.bin/next dev -p 8000
```

## 🔧 Common Development Tasks

### Adding New API Endpoint
```python
# hockey-analytics/main_api.py
@app.get("/api/new-endpoint")
async def new_endpoint():
    return {"message": "New endpoint", "status": "success"}
```

### Adding New Frontend Component
```javascript
// dashboard/src/components/NewComponent.js
import React from 'react'

export default function NewComponent() {
  return (
    <div>
      <h1>New Component</h1>
    </div>
  )
}
```

### Database Changes
```python
# hockey-analytics/main_api.py
import sqlite3

def init_database():
    conn = sqlite3.connect('analytics.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS new_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
```

## 🚨 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check for port conflicts
lsof -i :8000-8011

# Kill conflicting processes
for port in 8000 8001 8007 8008 8009 8010 8011; do
  lsof -ti:$port | xargs kill -9 2>/dev/null || true
done

# Restart services
./jarvis.sh start --all --force
```

#### Import Errors
```bash
# Check Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Reinstall dependencies
pip install -r requirements.txt
```

#### Node.js Issues
```bash
# Clear cache
cd dashboard
rm -rf node_modules package-lock.json
npm install
```

#### Database Issues
```bash
# Check database files
ls -la hockey-analytics/*.db

# Repair database
sqlite3 hockey-analytics/*.db "VACUUM;"
```

## 📚 Learning Resources

### Documentation
- [TSAI Jarvis CLI](../README-JARVIS-CLI.md)
- [API Documentation](../api/)
- [Architecture Documentation](../architecture/)
- [Testing Documentation](../testing/)

### External Resources
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Next.js Tutorial](https://nextjs.org/learn)
- [React Tutorial](https://react.dev/learn)
- [Python Tutorial](https://docs.python.org/3/tutorial/)

### Team Resources
- **Slack**: #tsai-jarvis-dev
- **Wiki**: Internal team wiki
- **Office Hours**: Tuesday/Thursday 2-4 PM
- **Mentor**: Assign a team mentor

## 🤝 Getting Help

### Internal Support
- **Slack**: #tsai-jarvis-dev for quick questions
- **Email**: dev-team@tsai-jarvis.com for detailed issues
- **Office Hours**: Tuesday/Thursday 2-4 PM
- **Mentor**: Your assigned mentor

### External Support
- **Stack Overflow**: Tag questions with `tsai-jarvis`
- **GitHub Issues**: For bug reports
- **GitHub Discussions**: For questions
- **Documentation**: This site

## 🎯 Next Steps

### Week 1
- [ ] Complete environment setup
- [ ] Run all services successfully
- [ ] Make first code change
- [ ] Write first test
- [ ] Submit first pull request

### Week 2
- [ ] Understand codebase structure
- [ ] Learn development workflow
- [ ] Contribute to documentation
- [ ] Fix first bug
- [ ] Attend team meetings

### Month 1
- [ ] Contribute to major feature
- [ ] Lead code review
- [ ] Mentor new team member
- [ ] Present at team meeting
- [ ] Complete onboarding checklist

## 📋 Onboarding Checklist

### Environment Setup
- [ ] Python 3.14+ installed
- [ ] Node.js 22+ installed
- [ ] Git configured
- [ ] IDE configured
- [ ] Virtual environment created
- [ ] Dependencies installed

### Development
- [ ] Repository cloned
- [ ] Services running
- [ ] Tests passing
- [ ] First code change made
- [ ] First test written
- [ ] First pull request submitted

### Team Integration
- [ ] Slack access granted
- [ ] Email access granted
- [ ] GitHub access granted
- [ ] Team meetings attended
- [ ] Mentor assigned
- [ ] Onboarding complete

## 🔄 Continuous Learning

### Technical Skills
- **Python**: Advanced Python features
- **FastAPI**: API development best practices
- **React/Next.js**: Modern frontend development
- **SQLite**: Database design and optimization
- **Testing**: Test-driven development
- **DevOps**: CI/CD and deployment

### Soft Skills
- **Communication**: Technical writing and presentation
- **Collaboration**: Code review and pair programming
- **Problem Solving**: Debugging and troubleshooting
- **Learning**: Continuous learning and adaptation
- **Leadership**: Mentoring and team leadership

## 📞 Contact Information

### Development Team
- **Team Lead**: team-lead@tsai-jarvis.com
- **Senior Engineers**: senior@tsai-jarvis.com
- **DevOps**: devops@tsai-jarvis.com
- **QA**: qa@tsai-jarvis.com

### Emergency Contacts
- **On-Call Engineer**: oncall@tsai-jarvis.com
- **Engineering Manager**: manager@tsai-jarvis.com
- **Director**: director@tsai-jarvis.com

Welcome to the TSAI Jarvis team! 🎉
