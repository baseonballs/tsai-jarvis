#!/usr/bin/env python3
"""
Test Suite for Jarvis Core Services
"""

import os
import sys
import json
import time
import logging
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Add the services to the path
sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/services')

# Import Jarvis Core Services
from storage.service import JarvisStorageService
from client_storage.service import JarvisClientStorageService
from experiment.service import JarvisExperimentService
from workflow.service import JarvisWorkflowService
from analytics.service import JarvisAnalyticsService

# Import TSAI Integration
sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/tsai-integration')
from tsai_integration import (
    TSAIComponent, ToolchainComponent, SpotlightComponent, 
    AutopilotComponent, SherlockComponent, WatsonComponent
)

class JarvisCoreTester:
    """Test suite for Jarvis Core Services"""
    
    def __init__(self):
        self.test_results = {}
        self.logger = logging.getLogger("JarvisCoreTester")
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for tests"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_storage_service(self) -> bool:
        """Test Jarvis Storage Service"""
        try:
            self.logger.info("🧪 Testing Jarvis Storage Service...")
            
            # Initialize storage service
            storage = JarvisStorageService()
            
            # Test 1: Store artifact
            test_file = self._create_test_file("test_artifact.txt", "Test artifact content")
            artifact_id = storage.store_artifact(test_file, {"test": True, "component": "test"})
            
            if not artifact_id:
                self.logger.error("❌ Failed to store artifact")
                return False
            
            self.logger.info(f"✅ Stored artifact: {artifact_id}")
            
            # Test 2: Retrieve artifact
            temp_dir = tempfile.mkdtemp()
            retrieved_path = os.path.join(temp_dir, "retrieved_artifact.txt")
            success = storage.retrieve_artifact(artifact_id, retrieved_path)
            
            if not success:
                self.logger.error("❌ Failed to retrieve artifact")
                return False
            
            self.logger.info("✅ Retrieved artifact successfully")
            
            # Test 3: List artifacts
            artifacts = storage.list_artifacts(component="test")
            if not isinstance(artifacts, list):
                self.logger.error("❌ Failed to list artifacts")
                return False
            
            self.logger.info(f"✅ Listed {len(artifacts)} artifacts")
            
            # Test 4: Store model
            model_file = self._create_test_file("test_model.pt", "Mock model content")
            model_id = storage.store_model(model_file, {
                "model_type": "pytorch",
                "version": "1.0.0",
                "component": "test"
            })
            
            if not model_id:
                self.logger.error("❌ Failed to store model")
                return False
            
            self.logger.info(f"✅ Stored model: {model_id}")
            
            # Test 5: List models
            models = storage.list_models(component="test")
            if not isinstance(models, list):
                self.logger.error("❌ Failed to list models")
                return False
            
            self.logger.info(f"✅ Listed {len(models)} models")
            
            # Cleanup
            self._cleanup_test_files([test_file, model_file, retrieved_path])
            
            self.logger.info("✅ Jarvis Storage Service tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Storage service test failed: {e}")
            return False
    
    def test_client_storage_service(self) -> bool:
        """Test Jarvis Client Storage Service"""
        try:
            self.logger.info("🧪 Testing Jarvis Client Storage Service...")
            
            # Initialize client storage service
            client_storage = JarvisClientStorageService()
            
            # Test 1: Import user media (mock)
            files = client_storage.import_user_media("google_drive", component="test")
            if not isinstance(files, list):
                self.logger.error("❌ Failed to import user media")
                return False
            
            self.logger.info(f"✅ Imported {len(files)} files from Google Drive")
            
            # Test 2: Export results (mock)
            test_results = ["result1.jpg", "result2.jpg", "result3.jpg"]
            uploaded_files = client_storage.export_results(test_results, "google_drive", component="test")
            if not isinstance(uploaded_files, list):
                self.logger.error("❌ Failed to export results")
                return False
            
            self.logger.info(f"✅ Exported {len(uploaded_files)} results to Google Drive")
            
            # Test 3: Create shared folder (mock)
            folder_id = client_storage.create_shared_folder("Test Results", "google_drive")
            if not folder_id:
                self.logger.error("❌ Failed to create shared folder")
                return False
            
            self.logger.info(f"✅ Created shared folder: {folder_id}")
            
            self.logger.info("✅ Jarvis Client Storage Service tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Client storage service test failed: {e}")
            return False
    
    def test_experiment_service(self) -> bool:
        """Test Jarvis Experiment Service"""
        try:
            self.logger.info("🧪 Testing Jarvis Experiment Service...")
            
            # Initialize experiment service
            experiment = JarvisExperimentService(
                tracking_uri="http://localhost:5000",
                registry_uri="postgresql://user:pass@localhost:5432/mlflow",
                artifacts_uri="s3://test-bucket/mlflow"
            )
            
            # Test 1: Start experiment
            run_id = experiment.start_experiment("test-experiment", {"param1": "value1", "param2": 42})
            if not run_id:
                self.logger.error("❌ Failed to start experiment")
                return False
            
            self.logger.info(f"✅ Started experiment: {run_id}")
            
            # Test 2: Log parameters
            experiment.log_params({"test_param": "test_value"})
            self.logger.info("✅ Logged parameters")
            
            # Test 3: Log metrics
            experiment.log_metrics({"accuracy": 0.95, "loss": 0.05})
            self.logger.info("✅ Logged metrics")
            
            # Test 4: Log artifacts
            temp_dir = tempfile.mkdtemp()
            test_artifact = os.path.join(temp_dir, "test_artifact.txt")
            with open(test_artifact, 'w') as f:
                f.write("Test artifact content")
            
            experiment.log_artifacts(temp_dir)
            self.logger.info("✅ Logged artifacts")
            
            # Test 5: Log model
            model_path = os.path.join(temp_dir, "test_model.pt")
            with open(model_path, 'w') as f:
                f.write("Mock model content")
            
            experiment.log_model(model_path, "test_model", {"version": "1.0.0"})
            self.logger.info("✅ Logged model")
            
            # Test 6: Get experiment runs
            runs = experiment.get_experiment_runs(component="test")
            if not isinstance(runs, list):
                self.logger.error("❌ Failed to get experiment runs")
                return False
            
            self.logger.info(f"✅ Retrieved {len(runs)} experiment runs")
            
            # Cleanup
            self._cleanup_test_files([test_artifact, model_path])
            
            self.logger.info("✅ Jarvis Experiment Service tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Experiment service test failed: {e}")
            return False
    
    def test_workflow_service(self) -> bool:
        """Test Jarvis Workflow Service"""
        try:
            self.logger.info("🧪 Testing Jarvis Workflow Service...")
            
            # Initialize workflow service
            workflow = JarvisWorkflowService(
                temporal_endpoint="localhost:7233",
                namespace="test-namespace",
                task_queue="test-queue"
            )
            
            # Test 1: Start workflow (mock)
            workflow_id = asyncio.run(workflow.start_workflow("test-workflow", {"input": "test_data"}))
            if not workflow_id:
                self.logger.error("❌ Failed to start workflow")
                return False
            
            self.logger.info(f"✅ Started workflow: {workflow_id}")
            
            # Test 2: Get workflow status (mock)
            status = asyncio.run(workflow.get_workflow_status(workflow_id))
            if not isinstance(status, dict):
                self.logger.error("❌ Failed to get workflow status")
                return False
            
            self.logger.info(f"✅ Retrieved workflow status: {status.get('status', 'unknown')}")
            
            # Test 3: List workflows (mock)
            workflows = asyncio.run(workflow.list_workflows(component="test"))
            if not isinstance(workflows, list):
                self.logger.error("❌ Failed to list workflows")
                return False
            
            self.logger.info(f"✅ Listed {len(workflows)} workflows")
            
            self.logger.info("✅ Jarvis Workflow Service tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Workflow service test failed: {e}")
            return False
    
    def test_analytics_service(self) -> bool:
        """Test Jarvis Analytics Service"""
        try:
            self.logger.info("🧪 Testing Jarvis Analytics Service...")
            
            # Initialize analytics service
            analytics = JarvisAnalyticsService(
                prometheus_endpoint="http://localhost:9090",
                grafana_endpoint="http://localhost:3000"
            )
            
            # Test 1: Log metric
            analytics.log_metric("test_metric", 42.0, {"label": "test"})
            self.logger.info("✅ Logged metric")
            
            # Test 2: Log system metrics
            analytics.log_system_metrics("test-component", {
                "cpu_usage": 45.5,
                "memory_usage": 1024 * 1024 * 1024,
                "disk_usage": 50 * 1024 * 1024 * 1024
            })
            self.logger.info("✅ Logged system metrics")
            
            # Test 3: Log application metrics
            analytics.log_application_metrics("test-component", "/api/test", "GET", "200", 0.5)
            self.logger.info("✅ Logged application metrics")
            
            # Test 4: Log business metrics
            analytics.log_business_metrics("test-component", {
                "user_engagement": ["login", "upload"],
                "pipeline_success_rate": 0.95
            })
            self.logger.info("✅ Logged business metrics")
            
            # Test 5: Log workflow metrics
            analytics.log_workflow_metrics("test-component", "test-workflow", "completed", 30.5)
            self.logger.info("✅ Logged workflow metrics")
            
            # Test 6: Log activity metrics
            analytics.log_activity_metrics("test-component", "data_processing", "completed", 15.2)
            self.logger.info("✅ Logged activity metrics")
            
            # Test 7: Get Prometheus metrics
            metrics = analytics.get_prometheus_metrics()
            if not isinstance(metrics, bytes):
                self.logger.error("❌ Failed to get Prometheus metrics")
                return False
            
            self.logger.info("✅ Retrieved Prometheus metrics")
            
            self.logger.info("✅ Jarvis Analytics Service tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Analytics service test failed: {e}")
            return False
    
    def test_tsai_components(self) -> bool:
        """Test TSAI Components"""
        try:
            self.logger.info("🧪 Testing TSAI Components...")
            
            # Test 1: TSAIComponent base class
            base_component = TSAIComponent("test-base")
            base_component.initialize({"test": True})
            base_component.start()
            
            health = base_component.health_check()
            if health['status'] != 'healthy':
                self.logger.error("❌ Base component health check failed")
                return False
            
            self.logger.info("✅ TSAIComponent base class test passed")
            
            # Test 2: ToolchainComponent
            toolchain = ToolchainComponent()
            toolchain.initialize({"model_type": "yolov8n"})
            toolchain.start()
            
            config = {
                'parameters': {'model_type': 'yolov8n', 'target_accuracy': 0.85},
                'data_config': {'dataset_path': '/data/hockey_players'},
                'training_config': {'epochs': 100, 'batch_size': 32}
            }
            
            workflow_id = toolchain.run_hockey_detection_pipeline(config)
            if not workflow_id:
                self.logger.error("❌ Toolchain component test failed")
                return False
            
            self.logger.info(f"✅ ToolchainComponent test passed: {workflow_id}")
            
            # Test 3: SpotlightComponent
            spotlight = SpotlightComponent()
            spotlight.initialize({"detection_confidence": 0.5})
            spotlight.start()
            
            video_config = {
                'analysis_type': 'hockey_video',
                'target_fps': 30,
                'detection_confidence': 0.5
            }
            
            result = spotlight.analyze_hockey_video('/path/to/video.mp4', video_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Spotlight component test failed")
                return False
            
            self.logger.info(f"✅ SpotlightComponent test passed: {result['workflow_id']}")
            
            # Test 4: AutopilotComponent
            autopilot = AutopilotComponent()
            autopilot.initialize({"learning_rate": 0.001})
            autopilot.start()
            
            analysis_config = {
                'parameters': {'analysis_type': 'autonomous_hockey'},
                'analysis_config': {'target_accuracy': 0.85}
            }
            
            workflow_id = autopilot.start_autonomous_hockey_analysis(analysis_config)
            if not workflow_id:
                self.logger.error("❌ Autopilot component test failed")
                return False
            
            self.logger.info(f"✅ AutopilotComponent test passed: {workflow_id}")
            
            # Test 5: SherlockComponent
            sherlock = SherlockComponent()
            sherlock.initialize({"investigation_type": "security"})
            sherlock.start()
            
            incident_config = {
                'parameters': {'incident_type': 'data_breach', 'severity': 'high'},
                'incident_config': {'affected_systems': ['database', 'api']}
            }
            
            workflow_id = sherlock.start_security_investigation(incident_config)
            if not workflow_id:
                self.logger.error("❌ Sherlock component test failed")
                return False
            
            self.logger.info(f"✅ SherlockComponent test passed: {workflow_id}")
            
            # Test 6: WatsonComponent
            watson = WatsonComponent()
            watson.initialize({"nlp_model": "gpt-4"})
            watson.start()
            
            nlp_config = {
                'language': 'en',
                'type': 'general',
                'confidence_threshold': 0.7
            }
            
            result = watson.process_natural_language("Hello, how are you?", nlp_config)
            if not result.get('workflow_id'):
                self.logger.error("❌ Watson component test failed")
                return False
            
            self.logger.info(f"✅ WatsonComponent test passed: {result['workflow_id']}")
            
            # Stop all components
            base_component.stop()
            toolchain.stop()
            spotlight.stop()
            autopilot.stop()
            sherlock.stop()
            watson.stop()
            
            self.logger.info("✅ All TSAI Components tests passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ TSAI Components test failed: {e}")
            return False
    
    def _create_test_file(self, filename: str, content: str) -> str:
        """Create a test file"""
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def _cleanup_test_files(self, files: List[str]):
        """Clean up test files"""
        for file_path in files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                # Also remove parent directory if empty
                parent_dir = os.path.dirname(file_path)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {file_path}: {e}")
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests"""
        self.logger.info("🚀 Starting Jarvis Core Services Test Suite...")
        
        tests = {
            "Storage Service": self.test_storage_service,
            "Client Storage Service": self.test_client_storage_service,
            "Experiment Service": self.test_experiment_service,
            "Workflow Service": self.test_workflow_service,
            "Analytics Service": self.test_analytics_service,
            "TSAI Components": self.test_tsai_components
        }
        
        results = {}
        for test_name, test_func in tests.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running {test_name} Test")
            self.logger.info(f"{'='*50}")
            
            try:
                result = test_func()
                results[test_name] = result
                status = "✅ PASSED" if result else "❌ FAILED"
                self.logger.info(f"{test_name}: {status}")
            except Exception as e:
                self.logger.error(f"❌ {test_name} test crashed: {e}")
                results[test_name] = False
        
        return results
    
    def print_summary(self, results: Dict[str, bool]):
        """Print test summary"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("JARVIS CORE SERVICES TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.logger.info(f"{test_name:<25} {status}")
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"TOTAL: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 ALL TESTS PASSED! Jarvis Core Services are working correctly.")
        else:
            self.logger.error(f"⚠️  {total - passed} tests failed. Please check the logs above.")
        
        self.logger.info(f"{'='*60}")


def main():
    """Main test function"""
    tester = JarvisCoreTester()
    results = tester.run_all_tests()
    tester.print_summary(results)
    
    # Return exit code based on results
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
