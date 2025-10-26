#!/usr/bin/env python3
"""
TSAI Component Base Class - Integration with Jarvis Core Services
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import uuid

# Import Jarvis Core Services
from jarvis_core.storage import JarvisStorageService
from jarvis_core.client_storage import JarvisClientStorageService
from jarvis_core.experiment import JarvisExperimentService
from jarvis_core.workflow import JarvisWorkflowService
from jarvis_core.analytics import JarvisAnalyticsService

class TSAIComponent:
    """Base class for all TSAI components"""
    
    def __init__(self, component_name: str, jarvis_core: 'JarvisCoreServices' = None):
        self.component_name = component_name
        self.jarvis = jarvis_core or self._initialize_jarvis_core()
        self.logger = logging.getLogger(f"TSAI.{component_name}")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Component metadata
        self.metadata = {
            'component_name': component_name,
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'status': 'initialized'
        }
    
    def _initialize_jarvis_core(self) -> 'JarvisCoreServices':
        """Initialize Jarvis Core Services"""
        try:
            from jarvis_core import JarvisCoreServices
            return JarvisCoreServices()
        except ImportError:
            self.logger.warning("Jarvis Core Services not available, using mock services")
            return MockJarvisCoreServices()
    
    # Storage Operations
    def store_artifact(self, artifact_path: str, metadata: Dict[str, Any] = None) -> str:
        """Store artifact using Jarvis storage service"""
        try:
            artifact_id = self.jarvis.storage_service.store_artifact(
                artifact_path, 
                metadata=metadata,
                component=self.component_name
            )
            self.logger.info(f"Stored artifact: {artifact_id}")
            return artifact_id
        except Exception as e:
            self.logger.error(f"Failed to store artifact: {e}")
            raise
    
    def retrieve_artifact(self, artifact_id: str, local_path: str) -> bool:
        """Retrieve artifact from Jarvis storage service"""
        try:
            success = self.jarvis.storage_service.retrieve_artifact(artifact_id, local_path)
            if success:
                self.logger.info(f"Retrieved artifact: {artifact_id}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
            return False
    
    def store_model(self, model_path: str, model_metadata: Dict[str, Any]) -> str:
        """Store trained model using Jarvis storage service"""
        try:
            model_id = self.jarvis.storage_service.store_model(
                model_path, 
                model_metadata,
                component=self.component_name
            )
            self.logger.info(f"Stored model: {model_id}")
            return model_id
        except Exception as e:
            self.logger.error(f"Failed to store model: {e}")
            raise
    
    def retrieve_model(self, model_id: str, local_path: str) -> bool:
        """Retrieve model from Jarvis storage service"""
        try:
            success = self.jarvis.storage_service.retrieve_model(model_id, local_path)
            if success:
                self.logger.info(f"Retrieved model: {model_id}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to retrieve model {model_id}: {e}")
            return False
    
    def list_artifacts(self, experiment_id: str = None) -> List[Dict[str, Any]]:
        """List artifacts for this component"""
        try:
            artifacts = self.jarvis.storage_service.list_artifacts(
                component=self.component_name,
                experiment_id=experiment_id
            )
            return artifacts
        except Exception as e:
            self.logger.error(f"Failed to list artifacts: {e}")
            return []
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List models for this component"""
        try:
            models = self.jarvis.storage_service.list_models(component=self.component_name)
            return models
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    # Client Storage Operations
    def import_user_media(self, drive_name: str, folder_id: str = None, 
                         file_types: List[str] = None) -> List[str]:
        """Import user media using Jarvis client storage service"""
        try:
            files = self.jarvis.client_storage_service.import_user_media(
                drive_name, folder_id, file_types, component=self.component_name
            )
            self.logger.info(f"Imported {len(files)} files from {drive_name}")
            return files
        except Exception as e:
            self.logger.error(f"Failed to import media from {drive_name}: {e}")
            return []
    
    def export_results(self, results: List[str], drive_name: str, 
                      folder_id: str = None) -> List[str]:
        """Export results using Jarvis client storage service"""
        try:
            uploaded_files = self.jarvis.client_storage_service.export_results(
                results, drive_name, folder_id, component=self.component_name
            )
            self.logger.info(f"Exported {len(uploaded_files)} files to {drive_name}")
            return uploaded_files
        except Exception as e:
            self.logger.error(f"Failed to export results to {drive_name}: {e}")
            return []
    
    def create_shared_folder(self, folder_name: str, drive_name: str) -> str:
        """Create shared folder for results"""
        try:
            folder_id = self.jarvis.client_storage_service.create_shared_folder(
                folder_name, drive_name
            )
            self.logger.info(f"Created shared folder: {folder_name}")
            return folder_id
        except Exception as e:
            self.logger.error(f"Failed to create shared folder: {e}")
            return None
    
    # Experiment Operations
    def start_experiment(self, experiment_name: str, parameters: Dict[str, Any]) -> str:
        """Start experiment using Jarvis experiment service"""
        try:
            run_id = self.jarvis.experiment_service.start_experiment(
                experiment_name, parameters, component=self.component_name
            )
            self.logger.info(f"Started experiment: {experiment_name} (ID: {run_id})")
            return run_id
        except Exception as e:
            self.logger.error(f"Failed to start experiment {experiment_name}: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics using Jarvis experiment service"""
        try:
            self.jarvis.experiment_service.log_metrics(metrics, step)
            self.logger.info(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters using Jarvis experiment service"""
        try:
            self.jarvis.experiment_service.log_params(params)
            self.logger.info(f"Logged parameters: {list(params.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_artifacts(self, artifacts_dir: str, artifact_path: str = None):
        """Log artifacts using Jarvis experiment service"""
        try:
            self.jarvis.experiment_service.log_artifacts(artifacts_dir, artifact_path)
            self.logger.info(f"Logged artifacts from: {artifacts_dir}")
        except Exception as e:
            self.logger.error(f"Failed to log artifacts: {e}")
            raise
    
    def log_model(self, model_path: str, model_name: str, 
                  model_metadata: Dict[str, Any] = None, model_type: str = "pytorch"):
        """Log model using Jarvis experiment service"""
        try:
            self.jarvis.experiment_service.log_model(
                model_path, model_name, model_metadata, model_type
            )
            self.logger.info(f"Logged model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to log model: {e}")
            raise
    
    def get_experiment_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get experiment runs for this component"""
        try:
            runs = self.jarvis.experiment_service.get_experiment_runs(
                component=self.component_name, limit=limit
            )
            return runs
        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {e}")
            return []
    
    def get_best_run(self, metric: str) -> Optional[Dict[str, Any]]:
        """Get best run by metric for this component"""
        try:
            best_run = self.jarvis.experiment_service.get_best_run(
                self.component_name, metric, component=self.component_name
            )
            return best_run
        except Exception as e:
            self.logger.error(f"Failed to get best run: {e}")
            return None
    
    def compare_experiments(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare experiments for this component"""
        try:
            comparison = self.jarvis.experiment_service.compare_experiments(run_ids)
            return comparison
        except Exception as e:
            self.logger.error(f"Failed to compare experiments: {e}")
            return {'error': str(e)}
    
    # Workflow Operations
    async def start_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> str:
        """Start workflow using Jarvis workflow service"""
        try:
            workflow_id = await self.jarvis.workflow_service.start_workflow(
                workflow_name, input_data, component=self.component_name
            )
            self.logger.info(f"Started workflow: {workflow_name} (ID: {workflow_id})")
            return workflow_id
        except Exception as e:
            self.logger.error(f"Failed to start workflow {workflow_name}: {e}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status using Jarvis workflow service"""
        try:
            status = await self.jarvis.workflow_service.get_workflow_status(workflow_id)
            return status
        except Exception as e:
            self.logger.error(f"Failed to get workflow status {workflow_id}: {e}")
            return {'error': str(e)}
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow using Jarvis workflow service"""
        try:
            success = await self.jarvis.workflow_service.cancel_workflow(workflow_id)
            if success:
                self.logger.info(f"Cancelled workflow: {workflow_id}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow {workflow_id}: {e}")
            return False
    
    async def get_workflow_result(self, workflow_id: str) -> Any:
        """Get workflow result using Jarvis workflow service"""
        try:
            result = await self.jarvis.workflow_service.get_workflow_result(workflow_id)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get workflow result {workflow_id}: {e}")
            return None
    
    async def list_workflows(self, status: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List workflows for this component"""
        try:
            workflows = await self.jarvis.workflow_service.list_workflows(
                component=self.component_name, status=status, limit=limit
            )
            return workflows
        except Exception as e:
            self.logger.error(f"Failed to list workflows: {e}")
            return []
    
    # Analytics Operations
    def log_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Log metric using Jarvis analytics service"""
        try:
            labels = labels or {}
            labels['component'] = self.component_name
            self.jarvis.analytics_service.log_metric(metric_name, value, labels)
        except Exception as e:
            self.logger.error(f"Failed to log metric {metric_name}: {e}")
    
    def log_system_metrics(self, metrics: Dict[str, float]):
        """Log system metrics using Jarvis analytics service"""
        try:
            self.jarvis.analytics_service.log_system_metrics(self.component_name, metrics)
        except Exception as e:
            self.logger.error(f"Failed to log system metrics: {e}")
    
    def log_application_metrics(self, endpoint: str, method: str, status: str, duration: float):
        """Log application metrics using Jarvis analytics service"""
        try:
            self.jarvis.analytics_service.log_application_metrics(
                self.component_name, endpoint, method, status, duration
            )
        except Exception as e:
            self.logger.error(f"Failed to log application metrics: {e}")
    
    def log_business_metrics(self, metrics: Dict[str, Any]):
        """Log business metrics using Jarvis analytics service"""
        try:
            self.jarvis.analytics_service.log_business_metrics(self.component_name, metrics)
        except Exception as e:
            self.logger.error(f"Failed to log business metrics: {e}")
    
    def log_workflow_metrics(self, workflow_type: str, status: str, duration: float):
        """Log workflow metrics using Jarvis analytics service"""
        try:
            self.jarvis.analytics_service.log_workflow_metrics(
                self.component_name, workflow_type, status, duration
            )
        except Exception as e:
            self.logger.error(f"Failed to log workflow metrics: {e}")
    
    def log_activity_metrics(self, activity_type: str, status: str, duration: float):
        """Log activity metrics using Jarvis analytics service"""
        try:
            self.jarvis.analytics_service.log_activity_metrics(
                self.component_name, activity_type, status, duration
            )
        except Exception as e:
            self.logger.error(f"Failed to log activity metrics: {e}")
    
    # Component Lifecycle
    def initialize(self, config: Dict[str, Any] = None):
        """Initialize component with configuration"""
        try:
            self.config = config or {}
            self.metadata['status'] = 'initialized'
            self.logger.info(f"Initialized {self.component_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.component_name}: {e}")
            raise
    
    def start(self):
        """Start component"""
        try:
            self.metadata['status'] = 'running'
            self.metadata['started_at'] = datetime.now().isoformat()
            self.logger.info(f"Started {self.component_name}")
        except Exception as e:
            self.logger.error(f"Failed to start {self.component_name}: {e}")
            raise
    
    def stop(self):
        """Stop component"""
        try:
            self.metadata['status'] = 'stopped'
            self.metadata['stopped_at'] = datetime.now().isoformat()
            self.logger.info(f"Stopped {self.component_name}")
        except Exception as e:
            self.logger.error(f"Failed to stop {self.component_name}: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check component health"""
        try:
            # Check Jarvis services
            jarvis_health = self.jarvis.health_check()
            
            return {
                'component_name': self.component_name,
                'status': self.metadata['status'],
                'jarvis_services': jarvis_health,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'component_name': self.component_name,
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get component metadata"""
        return self.metadata.copy()
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update component metadata"""
        self.metadata.update(metadata)
        self.metadata['updated_at'] = datetime.now().isoformat()


class MockJarvisCoreServices:
    """Mock Jarvis Core Services for testing"""
    
    def __init__(self):
        self.storage_service = MockStorageService()
        self.client_storage_service = MockClientStorageService()
        self.experiment_service = MockExperimentService()
        self.workflow_service = MockWorkflowService()
        self.analytics_service = MockAnalyticsService()
    
    def health_check(self) -> Dict[str, Any]:
        return {'status': 'mock', 'services': 'available'}


class MockStorageService:
    """Mock storage service"""
    def store_artifact(self, path, metadata=None, component=None): return "mock-artifact-id"
    def retrieve_artifact(self, artifact_id, local_path): return True
    def store_model(self, path, metadata, component=None): return "mock-model-id"
    def retrieve_model(self, model_id, local_path): return True
    def list_artifacts(self, component=None, experiment_id=None): return []
    def list_models(self, component=None): return []


class MockClientStorageService:
    """Mock client storage service"""
    def import_user_media(self, drive_name, folder_id=None, file_types=None, component=None): return []
    def export_results(self, results, drive_name, folder_id=None, component=None): return []
    def create_shared_folder(self, folder_name, drive_name): return "mock-folder-id"


class MockExperimentService:
    """Mock experiment service"""
    def start_experiment(self, name, params, component=None): return "mock-run-id"
    def log_metrics(self, metrics, step=None): pass
    def log_params(self, params): pass
    def log_artifacts(self, artifacts_dir, artifact_path=None): pass
    def log_model(self, model_path, model_name, metadata=None, model_type="pytorch"): pass
    def get_experiment_runs(self, component=None, limit=100): return []
    def get_best_run(self, experiment_name, metric, component=None): return None
    def compare_experiments(self, run_ids): return {}


class MockWorkflowService:
    """Mock workflow service"""
    async def start_workflow(self, name, input_data, component=None): return "mock-workflow-id"
    async def get_workflow_status(self, workflow_id): return {'status': 'completed'}
    async def cancel_workflow(self, workflow_id): return True
    async def get_workflow_result(self, workflow_id): return {'result': 'mock'}
    async def list_workflows(self, component=None, status=None, limit=100): return []


class MockAnalyticsService:
    """Mock analytics service"""
    def log_metric(self, name, value, labels=None): pass
    def log_system_metrics(self, component, metrics): pass
    def log_application_metrics(self, component, endpoint, method, status, duration): pass
    def log_business_metrics(self, component, metrics): pass
    def log_workflow_metrics(self, component, workflow_type, status, duration): pass
    def log_activity_metrics(self, component, activity_type, status, duration): pass


def main():
    """Main function for TSAI component base"""
    
    # Test TSAI component
    component = TSAIComponent("test-component")
    
    # Health check
    health = component.health_check()
    print(f"TSAI Component Health: {health}")
    
    # Test operations
    component.initialize({"test": True})
    component.start()
    
    # Test storage
    artifact_id = component.store_artifact("test.txt", {"test": True})
    print(f"✅ Stored artifact: {artifact_id}")
    
    # Test experiment
    run_id = component.start_experiment("test-experiment", {"param": "value"})
    print(f"✅ Started experiment: {run_id}")
    
    # Test metrics
    component.log_metrics({"accuracy": 0.95})
    print("✅ Logged metrics")
    
    # Test workflow
    import asyncio
    async def test_workflow():
        workflow_id = await component.start_workflow("test-workflow", {"input": "data"})
        print(f"✅ Started workflow: {workflow_id}")
    
    asyncio.run(test_workflow())
    
    component.stop()
    print("✅ Component lifecycle completed")

if __name__ == "__main__":
    main()
