#!/usr/bin/env python3
"""
TSAI Jarvis Platform Structure Test

This script tests the platform structure and shows what services are available
without requiring external infrastructure services.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlatformStructureTest:
    """Test TSAI Jarvis platform structure"""
    
    def __init__(self):
        self.test_results = {}
        
    async def test_platform_structure(self):
        """Test platform structure and available services"""
        try:
            logger.info("🧪 Testing TSAI Jarvis Platform Structure")
            
            # Test 1: Check directory structure
            await self._test_directory_structure()
            
            # Test 2: Check core intelligence services
            await self._test_core_intelligence()
            
            # Test 3: Check foundation services
            await self._test_foundation_services()
            
            # Test 4: Check API gateway
            await self._test_api_gateway()
            
            # Test 5: Check infrastructure
            await self._test_infrastructure()
            
            # Print results
            self._print_results()
            
            logger.info("✅ Platform structure test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Platform structure test failed: {e}")
            return False
    
    async def _test_directory_structure(self):
        """Test directory structure"""
        try:
            logger.info("📁 Testing directory structure")
            
            required_dirs = [
                "core-intelligence",
                "foundation-services", 
                "api-gateway",
                "infrastructure",
                "docs"
            ]
            
            missing_dirs = []
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                self.test_results["directory_structure"] = {
                    "status": "failed",
                    "missing_directories": missing_dirs
                }
            else:
                self.test_results["directory_structure"] = {
                    "status": "passed",
                    "directories": required_dirs
                }
            
            logger.info("✅ Directory structure test completed")
            
        except Exception as e:
            logger.error(f"❌ Directory structure test failed: {e}")
            self.test_results["directory_structure"] = {"status": "failed", "error": str(e)}
    
    async def _test_core_intelligence(self):
        """Test core intelligence services"""
        try:
            logger.info("🧠 Testing core intelligence services")
            
            core_services = {
                "jarvis_brain": "core-intelligence/decision-engine/jarvis_brain.py",
                "jarvis_orchestrator": "core-intelligence/temporal-orchestration/jarvis_orchestrator.py",
                "jarvis_models": "core-intelligence/pydantic-ai-models/jarvis_models.py"
            }
            
            available_services = {}
            for service_name, file_path in core_services.items():
                if Path(file_path).exists():
                    available_services[service_name] = "available"
                else:
                    available_services[service_name] = "missing"
            
            self.test_results["core_intelligence"] = {
                "status": "passed" if all(status == "available" for status in available_services.values()) else "partial",
                "services": available_services
            }
            
            logger.info("✅ Core intelligence services test completed")
            
        except Exception as e:
            logger.error(f"❌ Core intelligence test failed: {e}")
            self.test_results["core_intelligence"] = {"status": "failed", "error": str(e)}
    
    async def _test_foundation_services(self):
        """Test foundation services"""
        try:
            logger.info("🔧 Testing foundation services")
            
            foundation_services = {
                "user_management": "foundation-services/user-management/user_service.py",
                "session_management": "foundation-services/session-management/session_service.py",
                "state_management": "foundation-services/state-management/state_service.py",
                "key_management": "foundation-services/key-management/key_service.py",
                "asset_management": "foundation-services/asset-management/asset_service.py"
            }
            
            available_services = {}
            for service_name, file_path in foundation_services.items():
                if Path(file_path).exists():
                    available_services[service_name] = "available"
                else:
                    available_services[service_name] = "missing"
            
            self.test_results["foundation_services"] = {
                "status": "passed" if all(status == "available" for status in available_services.values()) else "partial",
                "services": available_services
            }
            
            logger.info("✅ Foundation services test completed")
            
        except Exception as e:
            logger.error(f"❌ Foundation services test failed: {e}")
            self.test_results["foundation_services"] = {"status": "failed", "error": str(e)}
    
    async def _test_api_gateway(self):
        """Test API gateway"""
        try:
            logger.info("🌐 Testing API gateway")
            
            api_files = {
                "jarvis_api": "api-gateway/intelligence-api/jarvis_api.py",
                "main": "main.py"
            }
            
            available_files = {}
            for file_name, file_path in api_files.items():
                if Path(file_path).exists():
                    available_files[file_name] = "available"
                else:
                    available_files[file_name] = "missing"
            
            self.test_results["api_gateway"] = {
                "status": "passed" if all(status == "available" for status in available_files.values()) else "partial",
                "files": available_files
            }
            
            logger.info("✅ API gateway test completed")
            
        except Exception as e:
            logger.error(f"❌ API gateway test failed: {e}")
            self.test_results["api_gateway"] = {"status": "failed", "error": str(e)}
    
    async def _test_infrastructure(self):
        """Test infrastructure components"""
        try:
            logger.info("🏗️ Testing infrastructure components")
            
            infrastructure_files = {
                "database_schema": "infrastructure/database/schema.sql",
                "database_setup": "infrastructure/database/setup.py",
                "docker_compose": "docker-compose.yml",
                "dockerfile": "Dockerfile",
                "requirements": "requirements.txt"
            }
            
            available_files = {}
            for file_name, file_path in infrastructure_files.items():
                if Path(file_path).exists():
                    available_files[file_name] = "available"
                else:
                    available_files[file_name] = "missing"
            
            self.test_results["infrastructure"] = {
                "status": "passed" if all(status == "available" for status in available_files.values()) else "partial",
                "files": available_files
            }
            
            logger.info("✅ Infrastructure test completed")
            
        except Exception as e:
            logger.error(f"❌ Infrastructure test failed: {e}")
            self.test_results["infrastructure"] = {"status": "failed", "error": str(e)}
    
    def _print_results(self):
        """Print test results"""
        print("\n🎯 TSAI Jarvis Platform Structure Test Results")
        print("=" * 60)
        
        for component, details in self.test_results.items():
            print(f"\n📋 {component.upper().replace('_', ' ')}:")
            if details["status"] == "passed":
                print(f"  ✅ Status: {details['status']}")
            elif details["status"] == "partial":
                print(f"  ⚠️ Status: {details['status']}")
            else:
                print(f"  ❌ Status: {details['status']}")
            
            if "error" in details:
                print(f"  ❌ Error: {details['error']}")
            
            # Print detailed results
            for key, value in details.items():
                if key not in ["status", "error"]:
                    if isinstance(value, dict):
                        print(f"  📁 {key}:")
                        for sub_key, sub_value in value.items():
                            status_icon = "✅" if sub_value == "available" else "❌"
                            print(f"    {status_icon} {sub_key}: {sub_value}")
                    else:
                        print(f"  📁 {key}: {value}")
        
        # Summary
        total_components = len(self.test_results)
        passed_components = sum(1 for details in self.test_results.values() if details["status"] == "passed")
        partial_components = sum(1 for details in self.test_results.values() if details["status"] == "partial")
        failed_components = sum(1 for details in self.test_results.values() if details["status"] == "failed")
        
        print(f"\n📊 SUMMARY:")
        print(f"  ✅ Passed: {passed_components}/{total_components}")
        print(f"  ⚠️ Partial: {partial_components}/{total_components}")
        print(f"  ❌ Failed: {failed_components}/{total_components}")
        
        if passed_components == total_components:
            print(f"\n🚀 TSAI Jarvis Platform is ready for development!")
        elif passed_components + partial_components == total_components:
            print(f"\n⚠️ TSAI Jarvis Platform is mostly ready, some components need attention.")
        else:
            print(f"\n❌ TSAI Jarvis Platform needs significant work before it's ready.")

async def main():
    """Main test function"""
    test = PlatformStructureTest()
    success = await test.test_platform_structure()
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
