# Developer Guides

Comprehensive guides for developers working on the TSAI Jarvis platform.

## 📚 Developer Documentation

### 🚀 [Getting Started](./getting-started.md)
- Development environment setup
- First-time setup
- Quick start guide
- Common development tasks

### 🛠️ [Environment Setup](./environment-setup.md)
- Development environment configuration
- IDE setup and configuration
- Development tools installation
- Environment variables

### 📝 [Coding Standards](./coding-standards.md)
- Code style guidelines
- Naming conventions
- Documentation standards
- Review process

### 🏗️ [Architecture Guide](./architecture-guide.md)
- System architecture overview
- Microservices design
- API design principles
- Database design

### 🔌 [API Development](./api-development.md)
- API design guidelines
- Endpoint development
- Authentication implementation
- Error handling

### 🎨 [Frontend Development](./frontend-development.md)
- React/Next.js development
- Component development
- State management
- UI/UX guidelines

### 🤖 [Machine Learning](./machine-learning.md)
- ML model development
- Training pipelines
- Model deployment
- Performance monitoring

### 🧪 [Testing Guide](./testing-guide.md)
- Testing strategies
- Unit testing
- Integration testing
- End-to-end testing

### 🔧 [Development Tools](./development-tools.md)
- Development environment
- Debugging tools
- Performance profiling
- Code analysis

### 📦 [Deployment Guide](./deployment-guide.md)
- Local deployment
- Staging deployment
- Production deployment
- CI/CD pipelines

## 🎯 Quick Start for New Developers

### 1. Prerequisites
- Python 3.14+
- Node.js 22+
- Git
- Docker (optional)

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/baseonballs/tsai-jarvis.git
cd tsai-jarvis

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup Node.js environment
cd dashboard
npm install
cd ..

# Start development services
./jarvis.sh start --all
```

### 3. Development Workflow
```bash
# Check service status
./jarvis.sh status

# View logs
tail -f logs/jarvis-*.log

# Run tests
python -m pytest tests/
npm test

# Stop services
./jarvis.sh stop
```

## 🏗️ Development Architecture

### Frontend (Dashboard)
- **Technology**: Next.js 16
- **Port**: 8000
- **Location**: `dashboard/`
- **Key Features**: Real-time updates, 3D visualization, analytics dashboard

### Backend APIs
- **Core API**: Port 8001 - Main hockey analytics
- **Enterprise Integration**: Port 8007 - Professional features
- **Advanced Visualization**: Port 8008 - 3D visualization
- **Machine Learning**: Port 8009 - AI/ML models
- **Enterprise Security**: Port 8010 - Authentication
- **Advanced Analytics**: Port 8011 - Statistical analysis

### Database
- **Type**: SQLite
- **Location**: `hockey-analytics/*.db`
- **Schema**: Analytics, users, models, visualizations

## 🔧 Development Tools

### CLI Management
```bash
# Start all services
./jarvis.sh start --all

# Start specific service
./jarvis.sh start --dashboard

# Check status
./jarvis.sh status

# Stop services
./jarvis.sh stop
```

### Development Commands
```bash
# Python development
source venv/bin/activate
python -m pytest tests/
python -m black hockey-analytics/
python -m flake8 hockey-analytics/

# Node.js development
cd dashboard
npm run dev
npm test
npm run build
```

### Debugging
```bash
# View logs
tail -f logs/jarvis-*.log

# Check service health
curl http://localhost:8000
curl http://localhost:8001/health

# Debug specific service
python -m pdb hockey-analytics/main_api.py
```

## 📋 Development Checklist

### Before Starting Development
- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] Services running
- [ ] Tests passing
- [ ] Documentation reviewed

### During Development
- [ ] Follow coding standards
- [ ] Write tests
- [ ] Update documentation
- [ ] Test locally
- [ ] Code review

### Before Committing
- [ ] All tests passing
- [ ] Code formatted
- [ ] Documentation updated
- [ ] No console errors
- [ ] Performance acceptable

## 🚀 Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes
# Write tests
# Update documentation

# Test locally
./jarvis.sh start --all
# Run tests
# Manual testing

# Commit changes
git add .
git commit -m "Add new feature"
git push origin feature/new-feature
```

### 2. Code Review
- Create pull request
- Request review
- Address feedback
- Merge to main

### 3. Deployment
- Automated testing
- Staging deployment
- Production deployment
- Monitoring

## 🧪 Testing Strategy

### Unit Testing
- **Python**: pytest
- **Node.js**: Jest
- **Coverage**: > 80%
- **Location**: `tests/`

### Integration Testing
- **API Testing**: FastAPI TestClient
- **Database Testing**: SQLite in-memory
- **Service Testing**: Docker containers

### End-to-End Testing
- **Frontend**: Playwright
- **API**: Postman/Newman
- **Performance**: Load testing

## 📊 Performance Guidelines

### Response Times
- **API Endpoints**: < 200ms
- **Database Queries**: < 100ms
- **Frontend Rendering**: < 500ms
- **File Uploads**: < 5s

### Resource Usage
- **CPU**: < 80%
- **Memory**: < 85%
- **Disk**: < 90%
- **Network**: < 1Gbps

### Scalability
- **Concurrent Users**: 1000+
- **API Requests**: 10,000/min
- **Data Processing**: 1GB/hour
- **Storage**: 100GB+

## 🔒 Security Guidelines

### Authentication
- **JWT Tokens**: Secure token generation
- **Password Hashing**: bcrypt
- **Session Management**: Secure sessions
- **Rate Limiting**: API rate limits

### Data Protection
- **Encryption**: AES-256
- **Input Validation**: Strict validation
- **SQL Injection**: Parameterized queries
- **XSS Protection**: Content sanitization

### Access Control
- **RBAC**: Role-based access
- **API Keys**: Secure key management
- **CORS**: Proper CORS configuration
- **HTTPS**: SSL/TLS encryption

## 📚 Learning Resources

### Documentation
- [TSAI Jarvis CLI](../README-JARVIS-CLI.md)
- [API Documentation](../api/)
- [Architecture Documentation](../architecture/)
- [Testing Documentation](../testing/)

### External Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [React Documentation](https://react.dev/)
- [Python Documentation](https://docs.python.org/)

### Training
- **Internal**: Team knowledge sharing
- **External**: Online courses
- **Conferences**: Industry events
- **Books**: Technical books

## 🤝 Contributing

### Code Contributions
1. Fork repository
2. Create feature branch
3. Make changes
4. Write tests
5. Submit pull request

### Documentation Contributions
1. Identify gaps
2. Write documentation
3. Review with team
4. Submit pull request

### Bug Reports
1. Check existing issues
2. Create detailed report
3. Include reproduction steps
4. Provide system information

## 📞 Support

### Development Support
- **Slack**: #tsai-jarvis-dev
- **Email**: dev-team@tsai-jarvis.com
- **Office Hours**: Tuesday/Thursday 2-4 PM
- **Emergency**: On-call engineer

### Technical Support
- **Documentation**: This site
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Wiki**: Internal wiki

## 🔄 Continuous Improvement

### Process Improvement
- **Retrospectives**: Monthly team retrospectives
- **Feedback**: Regular feedback sessions
- **Tools**: Tool evaluation and updates
- **Training**: Continuous learning

### Code Quality
- **Reviews**: Code review process
- **Standards**: Coding standards updates
- **Tools**: Development tool improvements
- **Automation**: CI/CD improvements
