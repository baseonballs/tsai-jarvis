#!/usr/bin/env python3
"""
Direct Import Test for Jarvis Core Services
"""

import os
import sys
import logging

# Add the tsai-integration directory to the path
sys.path.insert(0, '/Volumes/Thorage/wip/tsai-jarvis/tsai-integration')

def test_direct_imports():
    """Test direct imports of TSAI components"""
    print("🧪 Testing Direct Imports...")
    
    try:
        # Test 1: Import TSAI Component Base
        print("📦 Testing TSAI Component Base import...")
        from tsai_component_base import TSAIComponent
        print("✅ TSAI Component Base imported successfully")
        
        # Test 2: Import Toolchain Component
        print("📦 Testing Toolchain Component import...")
        from toolchain_component import ToolchainComponent
        print("✅ Toolchain Component imported successfully")
        
        # Test 3: Import Spotlight Component
        print("📦 Testing Spotlight Component import...")
        from spotlight_component import SpotlightComponent
        print("✅ Spotlight Component imported successfully")
        
        # Test 4: Import Autopilot Component
        print("📦 Testing Autopilot Component import...")
        from autopilot_component import AutopilotComponent
        print("✅ Autopilot Component imported successfully")
        
        # Test 5: Import Sherlock Component
        print("📦 Testing Sherlock Component import...")
        from sherlock_component import SherlockComponent
        print("✅ Sherlock Component imported successfully")
        
        # Test 6: Import Watson Component
        print("📦 Testing Watson Component import...")
        from watson_component import WatsonComponent
        print("✅ Watson Component imported successfully")
        
        print("\n🎉 All direct imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_component_instantiation():
    """Test component instantiation"""
    print("\n🧪 Testing Component Instantiation...")
    
    try:
        # Test TSAI Component Base
        print("📦 Testing TSAI Component Base instantiation...")
        from tsai_component_base import TSAIComponent
        component = TSAIComponent("test-component")
        print(f"✅ TSAI Component Base created: {component.component_name}")
        
        # Test Toolchain Component
        print("📦 Testing Toolchain Component instantiation...")
        from toolchain_component import ToolchainComponent
        toolchain = ToolchainComponent()
        print(f"✅ Toolchain Component created: {toolchain.component_name}")
        
        # Test Spotlight Component
        print("📦 Testing Spotlight Component instantiation...")
        from spotlight_component import SpotlightComponent
        spotlight = SpotlightComponent()
        print(f"✅ Spotlight Component created: {spotlight.component_name}")
        
        # Test Autopilot Component
        print("📦 Testing Autopilot Component instantiation...")
        from autopilot_component import AutopilotComponent
        autopilot = AutopilotComponent()
        print(f"✅ Autopilot Component created: {autopilot.component_name}")
        
        # Test Sherlock Component
        print("📦 Testing Sherlock Component instantiation...")
        from sherlock_component import SherlockComponent
        sherlock = SherlockComponent()
        print(f"✅ Sherlock Component created: {sherlock.component_name}")
        
        # Test Watson Component
        print("📦 Testing Watson Component instantiation...")
        from watson_component import WatsonComponent
        watson = WatsonComponent()
        print(f"✅ Watson Component created: {watson.component_name}")
        
        print("\n🎉 All component instantiation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Component instantiation failed: {e}")
        return False

def test_component_lifecycle():
    """Test component lifecycle"""
    print("\n🧪 Testing Component Lifecycle...")
    
    try:
        from tsai_component_base import TSAIComponent
        
        # Test component lifecycle
        component = TSAIComponent("test-lifecycle")
        print(f"✅ Component created: {component.component_name}")
        
        # Test initialization
        component.initialize({"test": True})
        print("✅ Component initialized")
        
        # Test start
        component.start()
        print(f"✅ Component started: {component.metadata['status']}")
        
        # Test health check
        health = component.health_check()
        print(f"✅ Health check: {health['status']}")
        
        # Test stop
        component.stop()
        print(f"✅ Component stopped: {component.metadata['status']}")
        
        print("\n🎉 Component lifecycle test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Component lifecycle test failed: {e}")
        return False

def test_toolchain_functionality():
    """Test Toolchain component functionality"""
    print("\n🧪 Testing Toolchain Functionality...")
    
    try:
        from toolchain_component import ToolchainComponent
        
        # Create component
        toolchain = ToolchainComponent()
        toolchain.initialize({"model_type": "yolov8n"})
        toolchain.start()
        
        # Test hockey detection pipeline
        config = {
            'parameters': {'model_type': 'yolov8n', 'target_accuracy': 0.85},
            'data_config': {'dataset_path': '/data/hockey_players'},
            'training_config': {'epochs': 100, 'batch_size': 32}
        }
        
        workflow_id = toolchain.run_hockey_detection_pipeline(config)
        print(f"✅ Hockey detection pipeline started: {workflow_id}")
        
        # Test media import
        files = toolchain.import_hockey_media("google_drive")
        print(f"✅ Media import: {len(files)} files")
        
        # Test results export
        results = ["result1.jpg", "result2.jpg"]
        uploaded_files = toolchain.export_hockey_results(results, "google_drive")
        print(f"✅ Results export: {len(uploaded_files)} files")
        
        # Test component metrics
        metrics = toolchain.get_component_metrics()
        print(f"✅ Component metrics: {metrics['business_metrics']}")
        
        toolchain.stop()
        
        print("\n🎉 Toolchain functionality test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Toolchain functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Direct Import Tests for Jarvis Core Services...")
    print("="*60)
    
    tests = {
        "Direct Imports": test_direct_imports,
        "Component Instantiation": test_component_instantiation,
        "Component Lifecycle": test_component_lifecycle,
        "Toolchain Functionality": test_toolchain_functionality
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
    print("DIRECT IMPORT TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"{'='*60}")
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL DIRECT IMPORT TESTS PASSED! Jarvis Core Services are working correctly.")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the logs above.")
    
    print(f"{'='*60}")
    
    # Return exit code
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
