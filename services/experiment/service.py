#!/usr/bin/env python3
"""
Jarvis Experiment Service - MLflow Integration
"""

import os
import json
import time
import logging
import sys
import platform
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import uuid

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus
import yaml

class JarvisExperimentService:
    """Jarvis experiment service for MLflow integration"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('JARVIS_CONFIG_PATH', 'config/jarvis-core.yaml')
        self.config = self._load_config()
        self.mlflow_client = self._create_mlflow_client()
        self.experiment_id = self._get_or_create_experiment()
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
            return config['jarvis_core']['services']['experiment']
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                'type': 'mlflow',
                'tracking_uri': 'http://mlflow:5000',
                'registry_uri': 'postgresql://postgres:password@postgres:5432/mlflow',
                'artifacts_uri': 's3://jarvis-artifacts/mlflow'
            }
    
    def _create_mlflow_client(self) -> MlflowClient:
        """Create MLflow client"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config['tracking_uri'])
            
            # Set registry URI
            if 'registry_uri' in self.config:
                mlflow.set_registry_uri(self.config['registry_uri'])
            
            client = MlflowClient()
            return client
        except Exception as e:
            self.logger.error(f"Failed to create MLflow client: {e}")
            raise
    
    def _get_or_create_experiment(self) -> str:
        """Get or create experiment"""
        try:
            experiment_name = self.config['experiment_settings']['name']
            
            # Try to get existing experiment
            try:
                experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
                if experiment:
                    return experiment.experiment_id
            except Exception:
                pass
            
            # Create new experiment
            experiment_id = self.mlflow_client.create_experiment(
                name=experiment_name,
                description=self.config['experiment_settings']['description'],
                tags=self.config['experiment_settings'].get('tags', {})
            )
            
            self.logger.info(f"Created experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Failed to get or create experiment: {e}")
            raise
    
    def start_experiment(self, experiment_name: str, parameters: Dict[str, Any], 
                        component: str = None) -> str:
        """Start new experiment"""
        try:
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=experiment_name,
                tags={"component": component}
            ):
                # Log parameters
                mlflow.log_params(parameters)
                
                # Log system info
                mlflow.log_params({
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "component": component,
                    "start_time": datetime.now().isoformat()
                })
                
                run_id = mlflow.active_run().info.run_id
                self.logger.info(f"Started experiment: {experiment_name} (ID: {run_id})")
                return run_id
                
        except Exception as e:
            self.logger.error(f"Failed to start experiment {experiment_name}: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to experiment"""
        try:
            mlflow.log_metrics(metrics, step=step)
            self.logger.info(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to experiment"""
        try:
            mlflow.log_params(params)
            self.logger.info(f"Logged parameters: {list(params.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_artifacts(self, artifacts_dir: str, artifact_path: str = None):
        """Log artifacts to experiment"""
        try:
            mlflow.log_artifacts(artifacts_dir, artifact_path)
            self.logger.info(f"Logged artifacts from: {artifacts_dir}")
        except Exception as e:
            self.logger.error(f"Failed to log artifacts: {e}")
            raise
    
    def log_model(self, model_path: str, model_name: str, 
                  model_metadata: Dict[str, Any] = None, model_type: str = "pytorch"):
        """Log model to experiment"""
        try:
            if model_type == "pytorch":
                mlflow.pytorch.log_model(
                    pytorch_model=model_path,
                    artifact_path="model",
                    registered_model_name=model_name,
                    metadata=model_metadata
                )
            elif model_type == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model_path,
                    artifact_path="model",
                    registered_model_name=model_name,
                    metadata=model_metadata
                )
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    tf_model=model_path,
                    artifact_path="model",
                    registered_model_name=model_name,
                    metadata=model_metadata
                )
            else:
                # Generic model logging
                mlflow.log_artifacts(model_path, "model")
            
            self.logger.info(f"Logged model: {model_name} ({model_type})")
            
        except Exception as e:
            self.logger.error(f"Failed to log model {model_name}: {e}")
            raise
    
    def get_experiment_runs(self, experiment_name: str = None, 
                           component: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get experiment runs"""
        try:
            # Build filter string
            filter_string = ""
            if component:
                filter_string += f"tags.component = '{component}'"
            
            # Search runs
            runs = self.mlflow_client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=limit
            )
            
            run_list = []
            for run in runs:
                run_info = {
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', ''),
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'component': run.data.tags.get('component', ''),
                    'parameters': run.data.params,
                    'metrics': run.data.metrics,
                    'artifacts': [artifact.path for artifact in run.data.artifacts]
                }
                run_list.append(run_info)
            
            return run_list
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {e}")
            return []
    
    def get_best_run(self, experiment_name: str, metric: str, 
                    component: str = None) -> Optional[Dict[str, Any]]:
        """Get best run by metric"""
        try:
            # Build filter string
            filter_string = ""
            if component:
                filter_string += f"tags.component = '{component}'"
            
            # Search runs ordered by metric
            runs = self.mlflow_client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                order_by=[f"metrics.{metric} DESC"]
            )
            
            if runs:
                best_run = runs[0]
                return {
                    'run_id': best_run.info.run_id,
                    'run_name': best_run.data.tags.get('mlflow.runName', ''),
                    'metric_value': best_run.data.metrics.get(metric, 0),
                    'component': best_run.data.tags.get('component', ''),
                    'parameters': best_run.data.params,
                    'metrics': best_run.data.metrics
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get best run: {e}")
            return None
    
    def compare_experiments(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments"""
        try:
            runs = []
            for run_id in run_ids:
                try:
                    run = self.mlflow_client.get_run(run_id)
                    runs.append(run)
                except Exception as e:
                    self.logger.warning(f"Failed to get run {run_id}: {e}")
                    continue
            
            if not runs:
                return {'error': 'No valid runs found'}
            
            # Compare parameters
            all_params = set()
            for run in runs:
                all_params.update(run.data.params.keys())
            
            param_comparison = {}
            for param in all_params:
                param_comparison[param] = {}
                for run in runs:
                    param_comparison[param][run.info.run_id] = run.data.params.get(param, 'N/A')
            
            # Compare metrics
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.data.metrics.keys())
            
            metric_comparison = {}
            for metric in all_metrics:
                metric_comparison[metric] = {}
                for run in runs:
                    metric_comparison[metric][run.info.run_id] = run.data.metrics.get(metric, 'N/A')
            
            # Compare artifacts
            artifact_comparison = {}
            for run in runs:
                artifact_comparison[run.info.run_id] = [artifact.path for artifact in run.data.artifacts]
            
            return {
                'runs': [run.info.run_id for run in runs],
                'parameters': param_comparison,
                'metrics': metric_comparison,
                'artifacts': artifact_comparison,
                'summary': {
                    'total_runs': len(runs),
                    'common_params': len([p for p in all_params if all(run.data.params.get(p) for run in runs)]),
                    'common_metrics': len([m for m in all_metrics if all(run.data.metrics.get(m) for run in runs)])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare experiments: {e}")
            return {'error': str(e)}
    
    def register_model(self, run_id: str, model_name: str, 
                      model_path: str, description: str = "") -> str:
        """Register model in MLflow model registry"""
        try:
            # Get run
            run = self.mlflow_client.get_run(run_id)
            
            # Register model
            model_version = self.mlflow_client.create_model_version(
                name=model_name,
                source=model_path,
                run_id=run_id,
                description=description
            )
            
            self.logger.info(f"Registered model: {model_name} (Version: {model_version.version})")
            return model_version.version
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to stage"""
        try:
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            self.logger.info(f"Promoted model {model_name} v{version} to {stage}")
            
        except Exception as e:
            self.logger.error(f"Failed to promote model {model_name} v{version}: {e}")
            raise
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get model versions"""
        try:
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            
            version_list = []
            for version in versions:
                version_info = {
                    'version': version.version,
                    'stage': version.current_stage,
                    'description': version.description,
                    'run_id': version.run_id,
                    'created_timestamp': version.creation_timestamp,
                    'last_updated_timestamp': version.last_updated_timestamp
                }
                version_list.append(version_info)
            
            return version_list
            
        except Exception as e:
            self.logger.error(f"Failed to get model versions for {model_name}: {e}")
            return []
    
    def get_model_stage(self, model_name: str, stage: str) -> Optional[Dict[str, Any]]:
        """Get model by stage"""
        try:
            versions = self.mlflow_client.get_latest_versions(
                model_name, 
                stages=[stage]
            )
            
            if versions:
                version = versions[0]
                return {
                    'version': version.version,
                    'stage': version.current_stage,
                    'description': version.description,
                    'run_id': version.run_id,
                    'source': version.source
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get model {model_name} in stage {stage}: {e}")
            return None
    
    def delete_run(self, run_id: str) -> bool:
        """Delete experiment run"""
        try:
            self.mlflow_client.delete_run(run_id)
            self.logger.info(f"Deleted run: {run_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete run {run_id}: {e}")
            return False
    
    def get_experiment_summary(self, component: str = None) -> Dict[str, Any]:
        """Get experiment summary"""
        try:
            # Get all runs
            runs = self.get_experiment_runs(component=component, limit=1000)
            
            if not runs:
                return {
                    'total_runs': 0,
                    'components': {},
                    'metrics_summary': {},
                    'status_summary': {}
                }
            
            # Calculate summary
            total_runs = len(runs)
            components = {}
            metrics_summary = {}
            status_summary = {}
            
            for run in runs:
                # Component summary
                component_name = run.get('component', 'unknown')
                if component_name not in components:
                    components[component_name] = 0
                components[component_name] += 1
                
                # Status summary
                status = run.get('status', 'unknown')
                if status not in status_summary:
                    status_summary[status] = 0
                status_summary[status] += 1
                
                # Metrics summary
                for metric, value in run.get('metrics', {}).items():
                    if metric not in metrics_summary:
                        metrics_summary[metric] = {'min': float('inf'), 'max': float('-inf'), 'count': 0}
                    
                    metrics_summary[metric]['min'] = min(metrics_summary[metric]['min'], value)
                    metrics_summary[metric]['max'] = max(metrics_summary[metric]['max'], value)
                    metrics_summary[metric]['count'] += 1
            
            return {
                'total_runs': total_runs,
                'components': components,
                'metrics_summary': metrics_summary,
                'status_summary': status_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get experiment summary: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check experiment service health"""
        try:
            # Test MLflow connection
            experiments = self.mlflow_client.search_experiments()
            
            return {
                'status': 'healthy',
                'tracking_uri': self.config['tracking_uri'],
                'experiment_id': self.experiment_id,
                'total_experiments': len(experiments),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Main function for experiment service"""
    
    # Initialize experiment service
    experiment_service = JarvisExperimentService()
    
    # Health check
    health = experiment_service.health_check()
    print(f"Experiment Service Health: {health}")
    
    if health['status'] == 'healthy':
        print("✅ Experiment service is healthy")
        
        # Test experiment creation
        run_id = experiment_service.start_experiment(
            "test-experiment",
            {"test_param": "test_value", "learning_rate": 0.001},
            component="test"
        )
        print(f"✅ Started experiment: {run_id}")
        
        # Test metrics logging
        experiment_service.log_metrics({
            "accuracy": 0.95,
            "loss": 0.05,
            "f1_score": 0.92
        })
        print("✅ Logged metrics")
        
        # Test experiment summary
        summary = experiment_service.get_experiment_summary(component="test")
        print(f"✅ Experiment summary: {summary}")
        
    else:
        print(f"❌ Experiment service is unhealthy: {health['error']}")

if __name__ == "__main__":
    main()
