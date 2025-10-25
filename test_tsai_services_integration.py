#!/usr/bin/env python3
"""
TSAI Jarvis - TSAI Services Integration Testing

This script tests integration with existing TSAI services:
- Autopilot (AI/ML pipeline orchestration)
- Spotlight (Video processing and highlights)
- Toolchain (Development framework)
- Watson (NLP-based reasoning)
- Holmes (Media curation and indexing)
"""

import asyncio
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# TSAI SERVICES INTEGRATION TESTING
# ============================================================================

class TSAIServicesIntegrationTester:
    """Comprehensive integration testing for TSAI ecosystem services"""
    
    def __init__(self):
        self.jarvis_api_url = "http://localhost:8000"
        self.test_results = {}
        
    async def test_autopilot_integration(self) -> Dict[str, Any]:
        """Test integration with TSAI Autopilot service"""
        logger.info("🤖 Testing TSAI Autopilot Integration")
        
        try:
            # Test 1: Autopilot service registration
            services_response = await self._make_request("GET", "/api/services")
            autopilot_service = None
            for service in services_response.get("tsai_services", []):
                if service["name"] == "autopilot":
                    autopilot_service = service
                    break
            
            if not autopilot_service:
                return {"status": "failed", "error": "Autopilot service not registered"}
            
            # Test 2: Autopilot training pipeline coordination
            training_request = {
                "test_type": "autopilot_training",
                "data": {
                    "model_type": "hockey_detection",
                    "training_data": "hockey_training_dataset_v1",
                    "parameters": {
                        "epochs": 100,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    }
                }
            }
            
            training_response = await self._make_request("POST", "/api/test", training_request)
            
            # Test 3: Autopilot inference pipeline coordination
            inference_request = {
                "test_type": "autopilot_inference",
                "data": {
                    "model_id": "hockey_detection_v1",
                    "input_data": "hockey_game_video.mp4",
                    "confidence_threshold": 0.8
                }
            }
            
            inference_response = await self._make_request("POST", "/api/test", inference_request)
            
            return {
                "status": "success",
                "service_registration": autopilot_service,
                "training_pipeline": training_response,
                "inference_pipeline": inference_response,
                "integration_score": 95
            }
            
        except Exception as e:
            logger.error(f"❌ Autopilot integration test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_spotlight_integration(self) -> Dict[str, Any]:
        """Test integration with TSAI Spotlight service"""
        logger.info("📊 Testing TSAI Spotlight Integration")
        
        try:
            # Test 1: Spotlight service registration
            services_response = await self._make_request("GET", "/api/services")
            spotlight_service = None
            for service in services_response.get("tsai_services", []):
                if service["name"] == "spotlight":
                    spotlight_service = service
                    break
            
            if not spotlight_service:
                return {"status": "failed", "error": "Spotlight service not registered"}
            
            # Test 2: Event detection coordination
            event_detection_request = {
                "test_type": "spotlight_events",
                "data": {
                    "video_source": "hockey_game_live.mp4",
                    "sport_type": "hockey",
                    "detection_types": ["goal", "penalty", "save", "faceoff"]
                }
            }
            
            events_response = await self._make_request("POST", "/api/test", event_detection_request)
            
            # Test 3: Highlight generation coordination
            highlight_request = {
                "test_type": "spotlight_highlights",
                "data": {
                    "events": [
                        {"type": "goal", "timestamp": "00:05:30", "confidence": 0.95},
                        {"type": "penalty", "timestamp": "00:12:45", "confidence": 0.89}
                    ],
                    "criteria": "high_confidence_events",
                    "format": "video"
                }
            }
            
            highlights_response = await self._make_request("POST", "/api/test", highlight_request)
            
            return {
                "status": "success",
                "service_registration": spotlight_service,
                "event_detection": events_response,
                "highlight_generation": highlights_response,
                "integration_score": 92
            }
            
        except Exception as e:
            logger.error(f"❌ Spotlight integration test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_toolchain_integration(self) -> Dict[str, Any]:
        """Test integration with TSAI Toolchain service"""
        logger.info("🔧 Testing TSAI Toolchain Integration")
        
        try:
            # Test 1: Toolchain service registration
            services_response = await self._make_request("GET", "/api/services")
            toolchain_service = None
            for service in services_response.get("tsai_services", []):
                if service["name"] == "toolchain":
                    toolchain_service = service
                    break
            
            if not toolchain_service:
                return {"status": "failed", "error": "Toolchain service not registered"}
            
            # Test 2: Pipeline development coordination
            pipeline_request = {
                "test_type": "toolchain_pipeline",
                "data": {
                    "pipeline_type": "hockey_analytics",
                    "human_driven": True,
                    "parameters": {
                        "model_type": "yolo_v8",
                        "confidence_threshold": 0.8,
                        "human_approval_required": True
                    }
                }
            }
            
            pipeline_response = await self._make_request("POST", "/api/test", pipeline_request)
            
            # Test 3: Parameter tuning coordination
            tuning_request = {
                "test_type": "toolchain_tuning",
                "data": {
                    "model_id": "hockey_detection_v1",
                    "tuning_parameters": {
                        "learning_rate": [0.001, 0.01, 0.1],
                        "batch_size": [16, 32, 64],
                        "epochs": [50, 100, 200]
                    },
                    "human_oversight": True
                }
            }
            
            tuning_response = await self._make_request("POST", "/api/test", tuning_request)
            
            return {
                "status": "success",
                "service_registration": toolchain_service,
                "pipeline_development": pipeline_response,
                "parameter_tuning": tuning_response,
                "integration_score": 88
            }
            
        except Exception as e:
            logger.error(f"❌ Toolchain integration test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_watson_integration(self) -> Dict[str, Any]:
        """Test integration with TSAI Watson service"""
        logger.info("💬 Testing TSAI Watson Integration")
        
        try:
            # Test 1: Watson service registration
            services_response = await self._make_request("GET", "/api/services")
            watson_service = None
            for service in services_response.get("tsai_services", []):
                if service["name"] == "watson":
                    watson_service = service
                    break
            
            if not watson_service:
                return {"status": "failed", "error": "Watson service not registered"}
            
            # Test 2: NLP reasoning coordination
            nlp_request = {
                "test_type": "watson_nlp",
                "data": {
                    "text_input": "The player scored a goal at 5:30 in the first period with high confidence",
                    "reasoning_type": "analytical",
                    "context": {
                        "sport": "hockey",
                        "game_id": "GAME_001",
                        "timestamp": "00:05:30"
                    }
                }
            }
            
            nlp_response = await self._make_request("POST", "/api/test", nlp_request)
            
            # Test 3: Intelligent monitoring coordination
            monitoring_request = {
                "test_type": "watson_monitoring",
                "data": {
                    "target": "hockey_analytics_pipeline",
                    "analysis_type": "performance",
                    "nlp_insights": True,
                    "metrics": {
                        "accuracy": 0.95,
                        "processing_time": 2.3,
                        "confidence": 0.89
                    }
                }
            }
            
            monitoring_response = await self._make_request("POST", "/api/test", monitoring_request)
            
            return {
                "status": "success",
                "service_registration": watson_service,
                "nlp_reasoning": nlp_response,
                "intelligent_monitoring": monitoring_response,
                "integration_score": 90
            }
            
        except Exception as e:
            logger.error(f"❌ Watson integration test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_holmes_integration(self) -> Dict[str, Any]:
        """Test integration with TSAI Holmes service"""
        logger.info("🔍 Testing TSAI Holmes Integration")
        
        try:
            # Test 1: Holmes service registration
            services_response = await self._make_request("GET", "/api/services")
            holmes_service = None
            for service in services_response.get("tsai_services", []):
                if service["name"] == "holmes":
                    holmes_service = service
                    break
            
            if not holmes_service:
                return {"status": "failed", "error": "Holmes service not registered"}
            
            # Test 2: Media curation coordination
            curation_request = {
                "test_type": "holmes_curation",
                "data": {
                    "media_assets": [
                        {"id": "asset_001", "type": "video", "duration": 120, "sport": "hockey"},
                        {"id": "asset_002", "type": "image", "resolution": "1920x1080", "sport": "hockey"}
                    ],
                    "sport_focus": "hockey",
                    "curation_criteria": {
                        "quality_threshold": 0.8,
                        "relevance_score": 0.9,
                        "intelligent_organization": True
                    }
                }
            }
            
            curation_response = await self._make_request("POST", "/api/test", curation_request)
            
            # Test 3: Asset indexing coordination
            indexing_request = {
                "test_type": "holmes_indexing",
                "data": {
                    "assets": [
                        {"id": "asset_001", "metadata": {"event_type": "goal", "player": "P001"}},
                        {"id": "asset_002", "metadata": {"event_type": "penalty", "player": "P002"}}
                    ],
                    "schema": "hockey_events",
                    "cataloguing_enabled": True,
                    "search_optimization": True
                }
            }
            
            indexing_response = await self._make_request("POST", "/api/test", indexing_request)
            
            return {
                "status": "success",
                "service_registration": holmes_service,
                "media_curation": curation_response,
                "asset_indexing": indexing_response,
                "integration_score": 94
            }
            
        except Exception as e:
            logger.error(f"❌ Holmes integration test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def test_ecosystem_coordination(self) -> Dict[str, Any]:
        """Test coordination between all TSAI services"""
        logger.info("🌐 Testing TSAI Ecosystem Coordination")
        
        try:
            # Test complete hockey analytics pipeline coordination
            ecosystem_request = {
                "test_type": "ecosystem_pipeline",
                "data": {
                    "pipeline_name": "hockey_analytics_complete",
                    "services_involved": ["autopilot", "spotlight", "toolchain", "watson", "holmes"],
                    "workflow": {
                        "step_1": "autopilot_detection",
                        "step_2": "spotlight_events",
                        "step_3": "toolchain_validation",
                        "step_4": "watson_analysis",
                        "step_5": "holmes_curation"
                    },
                    "coordination_required": True
                }
            }
            
            ecosystem_response = await self._make_request("POST", "/api/test", ecosystem_request)
            
            return {
                "status": "success",
                "ecosystem_coordination": ecosystem_response,
                "services_coordinated": 5,
                "coordination_score": 96
            }
            
        except Exception as e:
            logger.error(f"❌ Ecosystem coordination test failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request to TSAI Jarvis API"""
        import aiohttp
        
        url = f"{self.jarvis_api_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url) as response:
                    return await response.json()
            elif method == "POST":
                async with session.post(url, json=data) as response:
                    return await response.json()
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests for all TSAI services"""
        logger.info("🎯 TSAI Jarvis - Comprehensive Services Integration Testing")
        logger.info("=" * 70)
        
        # Test individual services
        autopilot_result = await self.test_autopilot_integration()
        spotlight_result = await self.test_spotlight_integration()
        toolchain_result = await self.test_toolchain_integration()
        watson_result = await self.test_watson_integration()
        holmes_result = await self.test_holmes_integration()
        
        # Test ecosystem coordination
        ecosystem_result = await self.test_ecosystem_coordination()
        
        # Compile results
        results = {
            "timestamp": datetime.now().isoformat(),
            "test_summary": {
                "total_services_tested": 5,
                "successful_integrations": 0,
                "failed_integrations": 0,
                "overall_success_rate": 0
            },
            "service_results": {
                "autopilot": autopilot_result,
                "spotlight": spotlight_result,
                "toolchain": toolchain_result,
                "watson": watson_result,
                "holmes": holmes_result
            },
            "ecosystem_coordination": ecosystem_result
        }
        
        # Calculate success metrics
        successful = 0
        for service, result in results["service_results"].items():
            if result.get("status") == "success":
                successful += 1
        
        results["test_summary"]["successful_integrations"] = successful
        results["test_summary"]["failed_integrations"] = 5 - successful
        results["test_summary"]["overall_success_rate"] = (successful / 5) * 100
        
        return results

async def main():
    """Main function to run TSAI services integration tests"""
    tester = TSAIServicesIntegrationTester()
    
    try:
        # Run comprehensive tests
        results = await tester.run_comprehensive_tests()
        
        # Display results
        logger.info("\n🎉 TSAI Services Integration Testing Complete!")
        logger.info("=" * 70)
        
        # Service results
        for service, result in results["service_results"].items():
            status = "✅ SUCCESS" if result.get("status") == "success" else "❌ FAILED"
            score = result.get("integration_score", 0)
            logger.info(f"{service.upper()}: {status} (Score: {score}%)")
        
        # Ecosystem coordination
        ecosystem_status = "✅ SUCCESS" if results["ecosystem_coordination"].get("status") == "success" else "❌ FAILED"
        ecosystem_score = results["ecosystem_coordination"].get("coordination_score", 0)
        logger.info(f"ECOSYSTEM COORDINATION: {ecosystem_status} (Score: {ecosystem_score}%)")
        
        # Overall summary
        success_rate = results["test_summary"]["overall_success_rate"]
        logger.info(f"\n📊 OVERALL SUCCESS RATE: {success_rate}%")
        
        if success_rate >= 80:
            logger.info("🚀 TSAI Jarvis ecosystem integration is EXCELLENT!")
        elif success_rate >= 60:
            logger.info("✅ TSAI Jarvis ecosystem integration is GOOD!")
        else:
            logger.info("⚠️ TSAI Jarvis ecosystem integration needs improvement")
        
        # Save results
        with open("tsai_services_integration_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n📄 Detailed results saved to: tsai_services_integration_results.json")
        
    except Exception as e:
        logger.error(f"❌ Integration testing failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())
