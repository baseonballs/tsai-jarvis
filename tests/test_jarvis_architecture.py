#!/usr/bin/env python3
"""
Architecture Test Suite for Jarvis Core Services
"""

import os
import sys
import json
import logging
from pathlib import Path

def test_jarvis_architecture():
    """Test Jarvis architecture and file structure"""
    print("🧪 Testing Jarvis Architecture...")
    
    try:
        # Test 1: Check if Jarvis services exist
        print("📁 Testing Jarvis services structure...")
        services_dir = Path("/Volumes/Thorage/wip/tsai-jarvis/services")
        if not services_dir.exists():
            print("❌ Services directory not found")
            return False
        
        expected_services = [
            "storage", "client-storage", "experiment", 
            "workflow", "analytics"
        ]
        
        for service in expected_services:
            service_path = services_dir / service
            if not service_path.exists():
                print(f"❌ Service {service} not found")
                return False
            print(f"✅ Service {service} found")
        
        # Test 2: Check if service files exist
        print("📄 Testing service files...")
        for service in expected_services:
            service_path = services_dir / service
            service_file = service_path / "service.py"
            config_file = service_path / "config.yaml"
            dockerfile = service_path / "Dockerfile"
            requirements = service_path / "requirements.txt"
            
            if not service_file.exists():
                print(f"❌ Service file for {service} not found")
                return False
            print(f"✅ Service file for {service} found")
            
            if not config_file.exists():
                print(f"❌ Config file for {service} not found")
                return False
            print(f"✅ Config file for {service} found")
            
            if not dockerfile.exists():
                print(f"❌ Dockerfile for {service} not found")
                return False
            print(f"✅ Dockerfile for {service} found")
            
            if not requirements.exists():
                print(f"❌ Requirements file for {service} not found")
                return False
            print(f"✅ Requirements file for {service} found")
        
        # Test 3: Check TSAI integration structure
        print("🔗 Testing TSAI integration structure...")
        tsai_dir = Path("/Volumes/Thorage/wip/tsai-jarvis/tsai-integration")
        if not tsai_dir.exists():
            print("❌ TSAI integration directory not found")
            return False
        
        expected_components = [
            "tsai_component_base.py",
            "toolchain_component.py", 
            "spotlight_component.py",
            "autopilot_component.py",
            "sherlock_component.py",
            "watson_component.py",
            "__init__.py",
            "README.md"
        ]
        
        for component in expected_components:
            component_path = tsai_dir / component
            if not component_path.exists():
                print(f"❌ Component {component} not found")
                return False
            print(f"✅ Component {component} found")
        
        # Test 4: Check Docker Compose
        print("🐳 Testing Docker Compose configuration...")
        docker_compose = Path("/Volumes/Thorage/wip/tsai-jarvis/docker-compose.yml")
        if not docker_compose.exists():
            print("❌ Docker Compose file not found")
            return False
        print("✅ Docker Compose file found")
        
        # Test 5: Check configuration files
        print("⚙️ Testing configuration files...")
        config_dir = Path("/Volumes/Thorage/wip/tsai-jarvis/config")
        if not config_dir.exists():
            print("❌ Config directory not found")
            return False
        
        config_file = config_dir / "jarvis-core.yaml"
        if not config_file.exists():
            print("❌ Jarvis core config file not found")
            return False
        print("✅ Jarvis core config file found")
        
        print("✅ Jarvis architecture test passed")
        return True
        
    except Exception as e:
        print(f"❌ Jarvis architecture test failed: {e}")
        return False

def test_service_implementations():
    """Test service implementations"""
    print("\n🧪 Testing Service Implementations...")
    
    try:
        services_dir = Path("/Volumes/Thorage/wip/tsai-jarvis/services")
        
        # Test each service implementation
        services = ["storage", "client-storage", "experiment", "workflow", "analytics"]
        
        for service in services:
            print(f"📦 Testing {service} service...")
            service_path = services_dir / service / "service.py"
            
            if not service_path.exists():
                print(f"❌ Service file for {service} not found")
                return False
            
            # Read and check service file content
            with open(service_path, 'r') as f:
                content = f.read()
                
            # Check for key components
            if "class" not in content:
                print(f"❌ No class definition found in {service} service")
                return False
            
            if "def __init__" not in content:
                print(f"❌ No __init__ method found in {service} service")
                return False
            
            print(f"✅ {service} service implementation looks good")
        
        print("✅ Service implementations test passed")
        return True
        
    except Exception as e:
        print(f"❌ Service implementations test failed: {e}")
        return False

def test_tsai_components():
    """Test TSAI component implementations"""
    print("\n🧪 Testing TSAI Component Implementations...")
    
    try:
        tsai_dir = Path("/Volumes/Thorage/wip/tsai-jarvis/tsai-integration")
        
        # Test each component
        components = [
            "tsai_component_base.py",
            "toolchain_component.py", 
            "spotlight_component.py",
            "autopilot_component.py",
            "sherlock_component.py",
            "watson_component.py"
        ]
        
        for component in components:
            print(f"📦 Testing {component}...")
            component_path = tsai_dir / component
            
            if not component_path.exists():
                print(f"❌ Component {component} not found")
                return False
            
            # Read and check component file content
            with open(component_path, 'r') as f:
                content = f.read()
                
            # Check for key components
            if "class" not in content:
                print(f"❌ No class definition found in {component}")
                return False
            
            if "def __init__" not in content:
                print(f"❌ No __init__ method found in {component}")
                return False
            
            print(f"✅ {component} implementation looks good")
        
        print("✅ TSAI component implementations test passed")
        return True
        
    except Exception as e:
        print(f"❌ TSAI component implementations test failed: {e}")
        return False

def test_docker_configuration():
    """Test Docker configuration"""
    print("\n🧪 Testing Docker Configuration...")
    
    try:
        # Test Docker Compose file
        docker_compose = Path("/Volumes/Thorage/wip/tsai-jarvis/docker-compose.yml")
        if not docker_compose.exists():
            print("❌ Docker Compose file not found")
            return False
        
        with open(docker_compose, 'r') as f:
            content = f.read()
        
        # Check for key services
        required_services = [
            "postgresql", "minio", "temporal", "mlflow", 
            "prometheus", "grafana", "jarvis-storage",
            "jarvis-client-storage", "jarvis-experiment",
            "jarvis-workflow", "jarvis-analytics"
        ]
        
        for service in required_services:
            if service not in content:
                print(f"❌ Service {service} not found in Docker Compose")
                return False
            print(f"✅ Service {service} found in Docker Compose")
        
        print("✅ Docker configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Docker configuration test failed: {e}")
        return False

def test_documentation():
    """Test documentation"""
    print("\n🧪 Testing Documentation...")
    
    try:
        # Test README files
        readme_files = [
            "/Volumes/Thorage/wip/tsai-jarvis/tsai-integration/README.md",
            "/Volumes/Thorage/wip/tsai-jarvis/README.md"
        ]
        
        for readme in readme_files:
            readme_path = Path(readme)
            if not readme_path.exists():
                print(f"❌ README file {readme} not found")
                return False
            
            with open(readme_path, 'r') as f:
                content = f.read()
            
            if len(content) < 100:  # Basic check for substantial content
                print(f"❌ README file {readme} seems too short")
                return False
            
            print(f"✅ README file {readme} looks good")
        
        print("✅ Documentation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Jarvis Architecture Tests...")
    print("="*60)
    
    tests = {
        "Jarvis Architecture": test_jarvis_architecture,
        "Service Implementations": test_service_implementations,
        "TSAI Components": test_tsai_components,
        "Docker Configuration": test_docker_configuration,
        "Documentation": test_documentation
    }
    
    results = {}
    for test_name, test_func in tests.items():
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print(f"\n{'='*60}")
    print("JARVIS ARCHITECTURE TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"{'='*60}")
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL ARCHITECTURE TESTS PASSED! Jarvis Core Services architecture is correct.")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the logs above.")
    
    print(f"{'='*60}")
    
    # Return exit code
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
