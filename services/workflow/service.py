#!/usr/bin/env python3
"""
Jarvis Workflow Service - Temporal Integration
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import uuid

from temporalio import workflow, activity
from temporalio.client import Client, WorkflowHandle
from temporalio.common import RetryPolicy
from temporalio.exceptions import WorkflowFailureError
import yaml

class JarvisWorkflowService:
    """Jarvis workflow service for Temporal integration"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('JARVIS_CONFIG_PATH', 'config/jarvis-core.yaml')
        self.config = self._load_config()
        self.temporal_client = None
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config['jarvis_core']['services']['workflow']
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                'type': 'temporal',
                'endpoint': 'temporal:7233',
                'namespace': 'jarvis-workflows',
                'task_queue': 'jarvis-pipeline-queue'
            }
    
    async def _create_temporal_client(self) -> Client:
        """Create Temporal client"""
        try:
            if not self.temporal_client:
                self.temporal_client = await Client.connect(
                    self.config['endpoint'],
                    namespace=self.config['namespace']
                )
            return self.temporal_client
        except Exception as e:
            self.logger.error(f"Failed to create Temporal client: {e}")
            raise
    
    async def start_workflow(self, workflow_name: str, input_data: Dict[str, Any], 
                           component: str = None) -> str:
        """Start workflow execution"""
        try:
            client = await self._create_temporal_client()
            
            # Create workflow options
            workflow_options = {
                'id': f"{component or 'jarvis'}-{workflow_name}-{int(time.time())}",
                'task_queue': self.config['task_queue'],
                'execution_timeout': timedelta(hours=24)
            }
            
            # Start workflow
            workflow_handle = await client.start_workflow(
                workflow_name,
                input_data,
                **workflow_options
            )
            
            self.logger.info(f"Started workflow: {workflow_name} (ID: {workflow_handle.id})")
            return workflow_handle.id
            
        except Exception as e:
            self.logger.error(f"Failed to start workflow {workflow_name}: {e}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status"""
        try:
            client = await self._create_temporal_client()
            workflow_handle = client.get_workflow_handle(workflow_id)
            
            # Get workflow info
            workflow_info = await workflow_handle.describe()
            
            return {
                'id': workflow_id,
                'status': workflow_info.status.name,
                'start_time': workflow_info.start_time.isoformat() if workflow_info.start_time else None,
                'execution_time': workflow_info.execution_time.isoformat() if workflow_info.execution_time else None,
                'close_time': workflow_info.close_time.isoformat() if workflow_info.close_time else None,
                'result': workflow_info.result if workflow_info.status.name == 'COMPLETED' else None,
                'error': str(workflow_info.failure) if workflow_info.failure else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow status {workflow_id}: {e}")
            return {
                'id': workflow_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel workflow execution"""
        try:
            client = await self._create_temporal_client()
            workflow_handle = client.get_workflow_handle(workflow_id)
            
            await workflow_handle.cancel()
            self.logger.info(f"Cancelled workflow: {workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow {workflow_id}: {e}")
            return False
    
    async def get_workflow_result(self, workflow_id: str) -> Any:
        """Get workflow result"""
        try:
            client = await self._create_temporal_client()
            workflow_handle = client.get_workflow_handle(workflow_id)
            
            result = await workflow_handle.result()
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow result {workflow_id}: {e}")
            return None
    
    async def list_workflows(self, component: str = None, status: str = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """List workflows with optional filtering"""
        try:
            client = await self._create_temporal_client()
            
            # Build filter
            filters = []
            if component:
                filters.append(f"WorkflowId LIKE '%{component}%'")
            if status:
                filters.append(f"Status = '{status}'")
            
            filter_string = " AND ".join(filters) if filters else None
            
            # List workflows
            workflows = []
            async for workflow in client.list_workflows(filter_string):
                workflow_info = {
                    'id': workflow.id,
                    'type': workflow.type,
                    'status': workflow.status.name,
                    'start_time': workflow.start_time.isoformat() if workflow.start_time else None,
                    'execution_time': workflow.execution_time.isoformat() if workflow.execution_time else None,
                    'close_time': workflow.close_time.isoformat() if workflow.close_time else None
                }
                workflows.append(workflow_info)
                
                if len(workflows) >= limit:
                    break
            
            return workflows
            
        except Exception as e:
            self.logger.error(f"Failed to list workflows: {e}")
            return []
    
    async def get_workflow_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        try:
            client = await self._create_temporal_client()
            workflow_handle = client.get_workflow_handle(workflow_id)
            
            history = []
            async for event in workflow_handle.fetch_history_events():
                event_info = {
                    'event_id': event.event_id,
                    'event_type': event.event_type.name,
                    'timestamp': event.event_time.isoformat(),
                    'data': str(event)
                }
                history.append(event_info)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow history {workflow_id}: {e}")
            return []
    
    async def signal_workflow(self, workflow_id: str, signal_name: str, 
                            signal_data: Any) -> bool:
        """Send signal to workflow"""
        try:
            client = await self._create_temporal_client()
            workflow_handle = client.get_workflow_handle(workflow_id)
            
            await workflow_handle.signal(signal_name, signal_data)
            self.logger.info(f"Sent signal {signal_name} to workflow {workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to signal workflow {workflow_id}: {e}")
            return False
    
    async def query_workflow(self, workflow_id: str, query_name: str, 
                           query_data: Any = None) -> Any:
        """Query workflow state"""
        try:
            client = await self._create_temporal_client()
            workflow_handle = client.get_workflow_handle(workflow_id)
            
            result = await workflow_handle.query(query_name, query_data)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to query workflow {workflow_id}: {e}")
            return None
    
    async def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow performance metrics"""
        try:
            client = await self._create_temporal_client()
            workflow_handle = client.get_workflow_handle(workflow_id)
            
            # Get workflow info
            workflow_info = await workflow_handle.describe()
            
            # Calculate metrics
            start_time = workflow_info.start_time
            end_time = workflow_info.close_time or datetime.now()
            duration = (end_time - start_time).total_seconds() if start_time else 0
            
            return {
                'workflow_id': workflow_id,
                'duration_seconds': duration,
                'status': workflow_info.status.name,
                'start_time': start_time.isoformat() if start_time else None,
                'end_time': end_time.isoformat() if end_time else None,
                'execution_time': workflow_info.execution_time.isoformat() if workflow_info.execution_time else None,
                'retry_count': getattr(workflow_info, 'retry_count', 0),
                'failure_reason': str(workflow_info.failure) if workflow_info.failure else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow metrics {workflow_id}: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check workflow service health"""
        try:
            client = await self._create_temporal_client()
            
            # Test connection by listing workflows
            workflows = []
            async for workflow in client.list_workflows(limit=1):
                workflows.append(workflow.id)
            
            return {
                'status': 'healthy',
                'endpoint': self.config['endpoint'],
                'namespace': self.config['namespace'],
                'task_queue': self.config['task_queue'],
                'connection_test': 'passed',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Workflow Definitions
@workflow.defn
class HockeyDetectionWorkflow:
    """Hockey detection workflow"""
    
    @workflow.run
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run hockey detection workflow"""
        try:
            # Step 1: Data Preparation
            data_result = await workflow.execute_activity(
                prepare_hockey_data,
                input_data.get('data_config', {}),
                start_to_close_timeout=timedelta(hours=2)
            )
            
            # Step 2: Model Training
            training_result = await workflow.execute_activity(
                train_hockey_model,
                input_data.get('training_config', {}),
                start_to_close_timeout=timedelta(hours=6)
            )
            
            # Step 3: Model Evaluation
            evaluation_result = await workflow.execute_activity(
                evaluate_hockey_model,
                input_data.get('evaluation_config', {}),
                start_to_close_timeout=timedelta(hours=1)
            )
            
            # Step 4: Model Deployment
            deployment_result = await workflow.execute_activity(
                deploy_hockey_model,
                input_data.get('deployment_config', {}),
                start_to_close_timeout=timedelta(hours=1)
            )
            
            return {
                'status': 'completed',
                'data_result': data_result,
                'training_result': training_result,
                'evaluation_result': evaluation_result,
                'deployment_result': deployment_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


@workflow.defn
class ComputerVisionWorkflow:
    """Computer vision workflow"""
    
    @workflow.run
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run computer vision workflow"""
        try:
            # Step 1: Image Processing
            image_result = await workflow.execute_activity(
                process_images,
                input_data.get('image_config', {}),
                start_to_close_timeout=timedelta(hours=1)
            )
            
            # Step 2: Feature Extraction
            feature_result = await workflow.execute_activity(
                extract_features,
                input_data.get('feature_config', {}),
                start_to_close_timeout=timedelta(hours=2)
            )
            
            # Step 3: Model Inference
            inference_result = await workflow.execute_activity(
                run_inference,
                input_data.get('inference_config', {}),
                start_to_close_timeout=timedelta(hours=1)
            )
            
            return {
                'status': 'completed',
                'image_result': image_result,
                'feature_result': feature_result,
                'inference_result': inference_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


@workflow.defn
class AutonomousAnalysisWorkflow:
    """Autonomous analysis workflow"""
    
    @workflow.run
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run autonomous analysis workflow"""
        try:
            # Step 1: Data Collection
            data_result = await workflow.execute_activity(
                collect_data,
                input_data.get('data_config', {}),
                start_to_close_timeout=timedelta(hours=1)
            )
            
            # Step 2: Analysis
            analysis_result = await workflow.execute_activity(
                run_analysis,
                input_data.get('analysis_config', {}),
                start_to_close_timeout=timedelta(hours=2)
            )
            
            # Step 3: Decision Making
            decision_result = await workflow.execute_activity(
                make_decision,
                input_data.get('decision_config', {}),
                start_to_close_timeout=timedelta(minutes=30)
            )
            
            # Step 4: Action Execution
            action_result = await workflow.execute_activity(
                execute_action,
                input_data.get('action_config', {}),
                start_to_close_timeout=timedelta(hours=1)
            )
            
            return {
                'status': 'completed',
                'data_result': data_result,
                'analysis_result': analysis_result,
                'decision_result': decision_result,
                'action_result': action_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Activity Definitions
@activity.defn
async def prepare_hockey_data(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare hockey data for training"""
    # Simulate data preparation
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'samples': data_config.get('samples', 1000),
        'features': data_config.get('features', 10)
    }

@activity.defn
async def train_hockey_model(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Train hockey detection model"""
    # Simulate model training
    await asyncio.sleep(2)
    return {
        'status': 'completed',
        'model_path': '/models/hockey_model.pt',
        'accuracy': 0.95,
        'epochs': training_config.get('epochs', 100)
    }

@activity.defn
async def evaluate_hockey_model(evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate hockey detection model"""
    # Simulate model evaluation
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'accuracy': 0.95,
        'precision': 0.92,
        'recall': 0.88,
        'f1_score': 0.90
    }

@activity.defn
async def deploy_hockey_model(deployment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deploy hockey detection model"""
    # Simulate model deployment
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'endpoint': 'http://api/hockey-detection',
        'version': '1.0.0'
    }

@activity.defn
async def process_images(image_config: Dict[str, Any]) -> Dict[str, Any]:
    """Process images for computer vision"""
    # Simulate image processing
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'processed_images': image_config.get('image_count', 100)
    }

@activity.defn
async def extract_features(feature_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features from images"""
    # Simulate feature extraction
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'features_extracted': feature_config.get('feature_count', 50)
    }

@activity.defn
async def run_inference(inference_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run model inference"""
    # Simulate inference
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'predictions': inference_config.get('prediction_count', 100)
    }

@activity.defn
async def collect_data(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect data for autonomous analysis"""
    # Simulate data collection
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'data_points': data_config.get('data_points', 1000)
    }

@activity.defn
async def run_analysis(analysis_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run autonomous analysis"""
    # Simulate analysis
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'analysis_type': analysis_config.get('analysis_type', 'statistical')
    }

@activity.defn
async def make_decision(decision_config: Dict[str, Any]) -> Dict[str, Any]:
    """Make autonomous decision"""
    # Simulate decision making
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'decision': decision_config.get('decision_type', 'optimize')
    }

@activity.defn
async def execute_action(action_config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute autonomous action"""
    # Simulate action execution
    await asyncio.sleep(1)
    return {
        'status': 'completed',
        'action': action_config.get('action_type', 'adjust_parameters')
    }


async def main():
    """Main function for workflow service"""
    
    # Initialize workflow service
    workflow_service = JarvisWorkflowService()
    
    # Health check
    health = await workflow_service.health_check()
    print(f"Workflow Service Health: {health}")
    
    if health['status'] == 'healthy':
        print("✅ Workflow service is healthy")
        
        # Test workflow creation
        workflow_id = await workflow_service.start_workflow(
            "hockey-detection",
            {"data_config": {"samples": 1000}, "training_config": {"epochs": 50}},
            component="test"
        )
        print(f"✅ Started workflow: {workflow_id}")
        
        # Test workflow status
        status = await workflow_service.get_workflow_status(workflow_id)
        print(f"✅ Workflow status: {status}")
        
        # Test workflow list
        workflows = await workflow_service.list_workflows(component="test")
        print(f"✅ Listed {len(workflows)} workflows")
        
    else:
        print(f"❌ Workflow service is unhealthy: {health['error']}")

if __name__ == "__main__":
    asyncio.run(main())
