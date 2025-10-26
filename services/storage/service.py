#!/usr/bin/env python3
"""
Jarvis Storage Service - MinIO Backend Implementation
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import hashlib
import uuid

from minio import Minio
from minio.error import S3Error
import yaml

class JarvisStorageService:
    """Jarvis storage service for durable backend storage"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('JARVIS_CONFIG_PATH', 'config/jarvis-core.yaml')
        self.config = self._load_config()
        self.minio_client = self._create_minio_client()
        self.buckets = self._initialize_buckets()
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
            return config['jarvis_core']['services']['storage']
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default config
            return {
                'type': 'minio',
                'endpoint': 'http://localhost:9000',
                'access_key': 'minioadmin',
                'secret_key': 'minioadmin',
                'buckets': ['jarvis-artifacts', 'jarvis-models', 'jarvis-datasets', 'jarvis-logs']
            }
    
    def _create_minio_client(self) -> Minio:
        """Create MinIO client"""
        try:
            client = Minio(
                self.config['endpoint'],
                access_key=self.config['access_key'],
                secret_key=self.config['secret_key'],
                secure=False  # Set to True for HTTPS
            )
            return client
        except Exception as e:
            self.logger.error(f"Failed to create MinIO client: {e}")
            raise
    
    def _initialize_buckets(self) -> Dict[str, str]:
        """Initialize MinIO buckets"""
        buckets = {}
        for bucket_name in self.config['buckets']:
            try:
                if not self.minio_client.bucket_exists(bucket_name):
                    self.minio_client.make_bucket(bucket_name)
                    self.logger.info(f"Created bucket: {bucket_name}")
                buckets[bucket_name.split('-')[1]] = bucket_name
            except S3Error as e:
                self.logger.error(f"Failed to create bucket {bucket_name}: {e}")
                raise
        return buckets
    
    def _generate_artifact_id(self) -> str:
        """Generate unique artifact ID"""
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        return f"artifact_{timestamp}_{random_id}"
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID"""
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        return f"model_{timestamp}_{random_id}"
    
    def _store_metadata(self, artifact_id: str, metadata: Dict[str, Any]) -> bool:
        """Store artifact metadata"""
        try:
            metadata_path = f"artifacts/{artifact_id}/metadata.json"
            metadata_json = json.dumps(metadata, indent=2)
            
            # Upload metadata
            self.minio_client.put_object(
                self.buckets['artifacts'],
                metadata_path,
                data=metadata_json.encode('utf-8'),
                length=len(metadata_json.encode('utf-8')),
                content_type='application/json'
            )
            
            self.logger.info(f"Stored metadata for artifact {artifact_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store metadata for artifact {artifact_id}: {e}")
            return False
    
    def _store_model_metadata(self, model_id: str, model_metadata: Dict[str, Any]) -> bool:
        """Store model metadata"""
        try:
            metadata_path = f"models/{model_id}/metadata.json"
            metadata_json = json.dumps(model_metadata, indent=2)
            
            # Upload metadata
            self.minio_client.put_object(
                self.buckets['models'],
                metadata_path,
                data=metadata_json.encode('utf-8'),
                length=len(metadata_json.encode('utf-8')),
                content_type='application/json'
            )
            
            self.logger.info(f"Stored metadata for model {model_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store metadata for model {model_id}: {e}")
            return False
    
    def store_artifact(self, artifact_path: str, metadata: Dict[str, Any] = None, 
                      component: str = None) -> str:
        """Store artifact with metadata"""
        try:
            artifact_id = self._generate_artifact_id()
            remote_path = f"artifacts/{artifact_id}/{Path(artifact_path).name}"
            
            # Upload artifact
            self.minio_client.fput_object(
                self.buckets['artifacts'],
                remote_path,
                artifact_path
            )
            
            # Prepare metadata
            artifact_metadata = {
                'id': artifact_id,
                'path': remote_path,
                'original_path': artifact_path,
                'component': component,
                'size': Path(artifact_path).stat().st_size,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # Store metadata
            self._store_metadata(artifact_id, artifact_metadata)
            
            self.logger.info(f"Stored artifact {artifact_id} from {artifact_path}")
            return artifact_id
            
        except Exception as e:
            self.logger.error(f"Failed to store artifact {artifact_path}: {e}")
            raise
    
    def retrieve_artifact(self, artifact_id: str, local_path: str) -> bool:
        """Retrieve artifact from storage"""
        try:
            # List objects with prefix to find the artifact
            objects = self.minio_client.list_objects(
                self.buckets['artifacts'],
                prefix=f"artifacts/{artifact_id}/",
                recursive=True
            )
            
            # Find the main artifact file (not metadata)
            artifact_path = None
            for obj in objects:
                if not obj.object_name.endswith('metadata.json'):
                    artifact_path = obj.object_name
                    break
            
            if not artifact_path:
                self.logger.error(f"Artifact {artifact_id} not found")
                return False
            
            # Download artifact
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.minio_client.fget_object(
                self.buckets['artifacts'],
                artifact_path,
                local_path
            )
            
            self.logger.info(f"Retrieved artifact {artifact_id} to {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve artifact {artifact_id}: {e}")
            return False
    
    def store_model(self, model_path: str, model_metadata: Dict[str, Any], 
                   component: str = None) -> str:
        """Store trained model with metadata"""
        try:
            model_id = self._generate_model_id()
            remote_path = f"models/{model_id}/{Path(model_path).name}"
            
            # Upload model
            self.minio_client.fput_object(
                self.buckets['models'],
                remote_path,
                model_path
            )
            
            # Prepare model metadata
            model_info = {
                'id': model_id,
                'path': remote_path,
                'original_path': model_path,
                'component': component,
                'size': Path(model_path).stat().st_size,
                'created_at': datetime.now().isoformat(),
                'metadata': model_metadata
            }
            
            # Store metadata
            self._store_model_metadata(model_id, model_info)
            
            self.logger.info(f"Stored model {model_id} from {model_path}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to store model {model_path}: {e}")
            raise
    
    def retrieve_model(self, model_id: str, local_path: str) -> bool:
        """Retrieve model from storage"""
        try:
            # List objects with prefix to find the model
            objects = self.minio_client.list_objects(
                self.buckets['models'],
                prefix=f"models/{model_id}/",
                recursive=True
            )
            
            # Find the main model file (not metadata)
            model_path = None
            for obj in objects:
                if not obj.object_name.endswith('metadata.json'):
                    model_path = obj.object_name
                    break
            
            if not model_path:
                self.logger.error(f"Model {model_id} not found")
                return False
            
            # Download model
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.minio_client.fget_object(
                self.buckets['models'],
                model_path,
                local_path
            )
            
            self.logger.info(f"Retrieved model {model_id} to {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve model {model_id}: {e}")
            return False
    
    def list_artifacts(self, component: str = None, experiment_id: str = None) -> List[Dict[str, Any]]:
        """List artifacts with optional filtering"""
        try:
            objects = self.minio_client.list_objects(
                self.buckets['artifacts'],
                prefix="artifacts/",
                recursive=True
            )
            
            artifacts = []
            for obj in objects:
                if obj.object_name.endswith('metadata.json'):
                    continue
                
                # Extract artifact ID from path
                artifact_id = obj.object_name.split('/')[1]
                
                artifact_info = {
                    'id': artifact_id,
                    'path': obj.object_name,
                    'size': obj.size,
                    'modified': obj.last_modified.isoformat(),
                    'component': component,
                    'experiment_id': experiment_id
                }
                
                artifacts.append(artifact_info)
            
            # Filter by component if specified
            if component:
                artifacts = [a for a in artifacts if a.get('component') == component]
            
            return artifacts
            
        except Exception as e:
            self.logger.error(f"Failed to list artifacts: {e}")
            return []
    
    def list_models(self, component: str = None) -> List[Dict[str, Any]]:
        """List models with optional filtering"""
        try:
            objects = self.minio_client.list_objects(
                self.buckets['models'],
                prefix="models/",
                recursive=True
            )
            
            models = []
            for obj in objects:
                if obj.object_name.endswith('metadata.json'):
                    continue
                
                # Extract model ID from path
                model_id = obj.object_name.split('/')[1]
                
                model_info = {
                    'id': model_id,
                    'path': obj.object_name,
                    'size': obj.size,
                    'modified': obj.last_modified.isoformat(),
                    'component': component
                }
                
                models.append(model_info)
            
            # Filter by component if specified
            if component:
                models = [m for m in models if m.get('component') == component]
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete artifact from storage"""
        try:
            # List objects with prefix to find all files for this artifact
            objects = self.minio_client.list_objects(
                self.buckets['artifacts'],
                prefix=f"artifacts/{artifact_id}/",
                recursive=True
            )
            
            # Delete all files for this artifact
            for obj in objects:
                self.minio_client.remove_object(
                    self.buckets['artifacts'],
                    obj.object_name
                )
            
            self.logger.info(f"Deleted artifact {artifact_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete artifact {artifact_id}: {e}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model from storage"""
        try:
            # List objects with prefix to find all files for this model
            objects = self.minio_client.list_objects(
                self.buckets['models'],
                prefix=f"models/{model_id}/",
                recursive=True
            )
            
            # Delete all files for this model
            for obj in objects:
                self.minio_client.remove_object(
                    self.buckets['models'],
                    obj.object_name
                )
            
            self.logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def get_artifact_metadata(self, artifact_id: str) -> Dict[str, Any]:
        """Get artifact metadata"""
        try:
            metadata_path = f"artifacts/{artifact_id}/metadata.json"
            
            # Get metadata object
            response = self.minio_client.get_object(
                self.buckets['artifacts'],
                metadata_path
            )
            
            # Read and parse metadata
            metadata_json = response.read().decode('utf-8')
            metadata = json.loads(metadata_json)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for artifact {artifact_id}: {e}")
            return {}
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get model metadata"""
        try:
            metadata_path = f"models/{model_id}/metadata.json"
            
            # Get metadata object
            response = self.minio_client.get_object(
                self.buckets['models'],
                metadata_path
            )
            
            # Read and parse metadata
            metadata_json = response.read().decode('utf-8')
            metadata = json.loads(metadata_json)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata for model {model_id}: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check storage service health"""
        try:
            # Test MinIO connection
            buckets = self.minio_client.list_buckets()
            
            return {
                'status': 'healthy',
                'endpoint': self.config['endpoint'],
                'buckets': len(buckets),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

def main():
    """Main function for storage service"""
    
    # Initialize storage service
    storage_service = JarvisStorageService()
    
    # Health check
    health = storage_service.health_check()
    print(f"Storage Service Health: {health}")
    
    # Example usage
    if health['status'] == 'healthy':
        print("✅ Storage service is healthy")
        
        # Test artifact storage
        test_file = "test_artifact.txt"
        with open(test_file, 'w') as f:
            f.write("Test artifact content")
        
        # Store artifact
        artifact_id = storage_service.store_artifact(
            test_file,
            metadata={'test': True, 'component': 'test'},
            component='test'
        )
        print(f"✅ Stored artifact: {artifact_id}")
        
        # List artifacts
        artifacts = storage_service.list_artifacts(component='test')
        print(f"✅ Listed {len(artifacts)} artifacts")
        
        # Clean up
        os.remove(test_file)
        storage_service.delete_artifact(artifact_id)
        print("✅ Cleaned up test artifacts")
        
    else:
        print(f"❌ Storage service is unhealthy: {health['error']}")

if __name__ == "__main__":
    main()
