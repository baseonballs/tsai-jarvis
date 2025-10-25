# Testing Documentation

Comprehensive testing strategies and procedures for the TSAI Jarvis platform.

## 📚 Testing Documentation

### 🧪 [Test Strategy](./test-strategy.md)
- Testing approach and methodology
- Test types and coverage
- Quality assurance procedures
- Testing best practices

### 🤖 [Test Automation](./automation.md)
- Automated testing setup
- CI/CD integration
- Test execution pipelines
- Continuous testing

### 🔍 [Quality Assurance](./quality-assurance.md)
- QA processes and procedures
- Quality metrics and KPIs
- Defect management
- Release quality gates

### 📊 [Performance Testing](./performance-testing.md)
- Load testing strategies
- Performance benchmarks
- Stress testing procedures
- Performance monitoring

### 🔒 [Security Testing](./security-testing.md)
- Security test procedures
- Vulnerability assessment
- Penetration testing
- Security compliance

### 🧩 [Integration Testing](./integration-testing.md)
- API integration testing
- Service integration testing
- End-to-end testing
- System integration

## 🎯 Testing Overview

### Test Pyramid
```
    /\
   /  \
  / E2E \     End-to-End Tests (5%)
 /______\
/        \
/Integration\  Integration Tests (15%)
/__________\
/            \
/   Unit Tests   \  Unit Tests (80%)
/________________\
```

### Test Types
- **Unit Tests**: Individual components
- **Integration Tests**: Component interactions
- **End-to-End Tests**: Complete user workflows
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability assessment

## 🚀 Quick Start

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

# All tests
./run-tests.sh
```

### Test Coverage
```bash
# Python coverage
python -m pytest tests/ --cov=hockey-analytics --cov-report=html

# Node.js coverage
cd dashboard
npm run test:coverage

# Combined coverage
./coverage-report.sh
```

## 🏗️ Test Architecture

### Test Structure
```
tests/
├── unit/                    # Unit tests
│   ├── test_apis/          # API unit tests
│   ├── test_models/        # Model unit tests
│   └── test_utils/         # Utility unit tests
├── integration/            # Integration tests
│   ├── test_api_integration/
│   ├── test_database/
│   └── test_services/
├── e2e/                    # End-to-end tests
│   ├── test_user_flows/
│   ├── test_dashboard/
│   └── test_analytics/
├── performance/            # Performance tests
│   ├── test_load/
│   ├── test_stress/
│   └── test_benchmarks/
└── security/               # Security tests
    ├── test_auth/
    ├── test_permissions/
    └── test_vulnerabilities/
```

### Test Data
```
test-data/
├── fixtures/               # Test fixtures
│   ├── users.json
│   ├── games.json
│   └── analytics.json
├── mocks/                  # Mock data
│   ├── api_responses/
│   ├── database/
│   └── external_services/
└── samples/                # Sample data
    ├── images/
    ├── videos/
    └── datasets/
```

## 🔧 Testing Tools

### Python Testing
- **pytest**: Test framework
- **pytest-cov**: Coverage
- **pytest-mock**: Mocking
- **pytest-asyncio**: Async testing
- **factory_boy**: Test data generation

### Node.js Testing
- **Jest**: Test framework
- **React Testing Library**: Component testing
- **Playwright**: E2E testing
- **MSW**: API mocking
- **Storybook**: Component testing

### Performance Testing
- **Locust**: Load testing
- **Artillery**: Performance testing
- **k6**: Load testing
- **JMeter**: Performance testing

### Security Testing
- **Bandit**: Python security
- **ESLint**: JavaScript security
- **OWASP ZAP**: Security scanning
- **Snyk**: Vulnerability scanning

## 📊 Test Metrics

### Coverage Metrics
- **Line Coverage**: > 80%
- **Branch Coverage**: > 70%
- **Function Coverage**: > 90%
- **Statement Coverage**: > 85%

### Quality Metrics
- **Test Pass Rate**: > 95%
- **Test Execution Time**: < 10 minutes
- **Flaky Test Rate**: < 5%
- **Test Maintenance**: < 20% effort

### Performance Metrics
- **API Response Time**: < 200ms
- **Page Load Time**: < 2s
- **Database Query Time**: < 100ms
- **Memory Usage**: < 500MB

## 🧪 Test Categories

### Unit Tests
```python
# tests/unit/test_main_api.py
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

### Integration Tests
```python
# tests/integration/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from hockey-analytics.main_api import app

client = TestClient(app)

def test_api_integration():
    # Test complete API workflow
    response = client.get("/api/game/state")
    assert response.status_code == 200
    
    response = client.get("/api/players/stats")
    assert response.status_code == 200
    
    response = client.get("/api/events/live")
    assert response.status_code == 200
```

### End-to-End Tests
```javascript
// tests/e2e/test_dashboard.js
import { test, expect } from '@playwright/test'

test('dashboard loads correctly', async ({ page }) => {
  await page.goto('http://localhost:8000')
  await expect(page.locator('h1')).toContainText('TSAI Jarvis')
  await expect(page.locator('[data-testid="game-state"]')).toBeVisible()
})

test('user can view analytics', async ({ page }) => {
  await page.goto('http://localhost:8000')
  await page.click('[data-testid="analytics-tab"]')
  await expect(page.locator('[data-testid="analytics-chart"]')).toBeVisible()
})
```

### Performance Tests
```python
# tests/performance/test_load.py
import pytest
import requests
import time
from concurrent.futures import ThreadPoolExecutor

def test_api_load():
    def make_request():
        response = requests.get('http://localhost:8001/health')
        return response.status_code == 200
    
    # Test with 100 concurrent requests
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(make_request) for _ in range(100)]
        results = [future.result() for future in futures]
    
    assert all(results)
    assert len(results) == 100
```

## 🚀 CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.14'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          cd dashboard && npm install
      - name: Run tests
        run: |
          python -m pytest tests/
          cd dashboard && npm test
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Execution
```bash
# Local testing
./run-tests.sh

# CI testing
./ci-tests.sh

# Performance testing
./performance-tests.sh

# Security testing
./security-tests.sh
```

## 📋 Test Procedures

### Pre-Test Checklist
- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] Test data prepared
- [ ] Services running
- [ ] Database initialized

### Test Execution
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] E2E tests passing
- [ ] Performance tests passing
- [ ] Security tests passing

### Post-Test Procedures
- [ ] Test results reviewed
- [ ] Coverage report generated
- [ ] Defects logged
- [ ] Test data cleaned
- [ ] Reports published

## 🔍 Test Debugging

### Common Issues
```bash
# Test failures
python -m pytest tests/ -v --tb=short

# Coverage issues
python -m pytest tests/ --cov=hockey-analytics --cov-report=html

# Performance issues
python -m pytest tests/performance/ -v

# Security issues
bandit -r hockey-analytics/
```

### Debugging Tools
```bash
# Python debugging
python -m pdb -m pytest tests/test_specific.py

# Node.js debugging
cd dashboard
npm test -- --inspect

# Performance profiling
python -m cProfile -m pytest tests/performance/
```

## 📊 Test Reporting

### Coverage Reports
```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=hockey-analytics --cov-report=html

# Generate XML coverage report
python -m pytest tests/ --cov=hockey-analytics --cov-report=xml

# Generate terminal coverage report
python -m pytest tests/ --cov=hockey-analytics --cov-report=term
```

### Test Reports
```bash
# Generate test report
python -m pytest tests/ --html=reports/test-report.html

# Generate performance report
python -m pytest tests/performance/ --html=reports/performance-report.html

# Generate security report
bandit -r hockey-analytics/ -f json -o reports/security-report.json
```

## 🎯 Best Practices

### Test Design
- **Arrange-Act-Assert**: Clear test structure
- **Single Responsibility**: One test per scenario
- **Descriptive Names**: Clear test names
- **Independent Tests**: No test dependencies
- **Fast Execution**: Quick test runs

### Test Data
- **Fixtures**: Reusable test data
- **Factories**: Dynamic test data
- **Mocks**: External dependencies
- **Isolation**: Test data isolation
- **Cleanup**: Test data cleanup

### Test Maintenance
- **Regular Updates**: Keep tests current
- **Refactoring**: Improve test quality
- **Documentation**: Document test purpose
- **Review**: Regular test reviews
- **Metrics**: Track test metrics

## 🚨 Troubleshooting

### Common Issues
```bash
# Import errors
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Database issues
sqlite3 test.db "VACUUM;"

# Port conflicts
lsof -ti:8000-8011 | xargs kill -9

# Permission issues
chmod +x run-tests.sh
```

### Test Environment
```bash
# Reset test environment
./reset-test-env.sh

# Clean test data
rm -rf test-data/
mkdir test-data/

# Restart services
./jarvis.sh stop
./jarvis.sh start --all
```

## 📞 Support

### Testing Support
- **Slack**: #tsai-jarvis-testing
- **Email**: testing@tsai-jarvis.com
- **Office Hours**: Wednesday 2-4 PM
- **Documentation**: This site

### External Resources
- [pytest Documentation](https://docs.pytest.org/)
- [Jest Documentation](https://jestjs.io/docs/)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://testing-library.com/docs/)

## 🔄 Continuous Improvement

### Test Quality
- **Regular Reviews**: Monthly test reviews
- **Metrics Tracking**: Test metrics monitoring
- **Process Improvement**: Continuous improvement
- **Tool Updates**: Regular tool updates
- **Training**: Team training sessions

### Process Improvement
- **Retrospectives**: Test process retrospectives
- **Feedback**: Regular feedback collection
- **Automation**: Increased test automation
- **Efficiency**: Test execution efficiency
- **Quality**: Test quality improvement
