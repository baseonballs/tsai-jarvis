"""
TSAI Jarvis - YOLO Model Manager for Hockey Analytics
Phase 1.1: YOLO Model Integration and Management

This module manages YOLO model deployment, versioning, and optimization
for real-time hockey player detection.
"""

import os
import logging
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import shutil

# YOLO and PyTorch imports
try:
    from ultralytics import YOLO
    import torch
    import torchvision
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available. Install with: pip install ultralytics torch torchvision")

# Model optimization imports
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX not available. Install with: pip install onnx onnxruntime")

logger = logging.getLogger(__name__)

class YOLOModelManager:
    """
    Manages YOLO model deployment, versioning, and optimization for hockey analytics
    """
    
    def __init__(self, models_dir: str = "models", cache_dir: str = "cache"):
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry = {}
        self.active_model = None
        self.model_versions = {}
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default YOLO models for hockey analytics"""
        default_models = {
            "yolov8n": {
                "name": "YOLOv8 Nano",
                "size": "nano",
                "speed": "fastest",
                "accuracy": "good",
                "description": "Lightweight model for real-time processing"
            },
            "yolov8s": {
                "name": "YOLOv8 Small",
                "size": "small", 
                "speed": "fast",
                "accuracy": "better",
                "description": "Balanced model for production use"
            },
            "yolov8m": {
                "name": "YOLOv8 Medium",
                "size": "medium",
                "speed": "medium",
                "accuracy": "high",
                "description": "High accuracy model for detailed analysis"
            },
            "yolov8l": {
                "name": "YOLOv8 Large",
                "size": "large",
                "speed": "slow",
                "accuracy": "highest",
                "description": "Maximum accuracy model for critical analysis"
            }
        }
        
        for model_id, config in default_models.items():
            self.model_registry[model_id] = config
            self.model_versions[model_id] = []
    
    async def download_model(self, model_id: str, force_download: bool = False) -> bool:
        """
        Download YOLO model if not already present
        
        Args:
            model_id: Model identifier (e.g., 'yolov8n', 'yolov8s')
            force_download: Force re-download even if model exists
            
        Returns:
            True if model is available, False otherwise
        """
        if model_id not in self.model_registry:
            logger.error(f"❌ Unknown model: {model_id}")
            return False
        
        model_path = self.models_dir / f"{model_id}.pt"
        
        if model_path.exists() and not force_download:
            logger.info(f"✅ Model already exists: {model_path}")
            return True
        
        try:
            logger.info(f"📥 Downloading model: {model_id}")
            
            # Download model using ultralytics
            model = YOLO(f"{model_id}.pt")
            
            # Save to our models directory
            shutil.copy(f"{model_id}.pt", str(model_path))
            
            # Record model version
            version_info = {
                "model_id": model_id,
                "version": "1.0.0",
                "download_date": datetime.now().isoformat(),
                "file_size": model_path.stat().st_size,
                "checksum": self._calculate_checksum(model_path)
            }
            
            self.model_versions[model_id].append(version_info)
            
            logger.info(f"✅ Model downloaded successfully: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download model {model_id}: {e}")
            return False
    
    async def load_model(self, model_id: str, device: str = "auto") -> bool:
        """
        Load YOLO model for inference
        
        Args:
            model_id: Model identifier
            device: Device to load model on ('auto', 'cpu', 'cuda')
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not YOLO_AVAILABLE:
            logger.error("❌ YOLO not available")
            return False
        
        try:
            model_path = self.models_dir / f"{model_id}.pt"
            
            if not model_path.exists():
                logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model
            self.active_model = YOLO(str(model_path))
            self.active_model.to(device)
            
            # Store model info
            self.active_model_id = model_id
            self.active_device = device
            
            logger.info(f"✅ Model loaded: {model_id} on {device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_id}: {e}")
            return False
    
    async def optimize_model(self, model_id: str, optimization_type: str = "onnx") -> bool:
        """
        Optimize model for production deployment
        
        Args:
            model_id: Model identifier
            optimization_type: Type of optimization ('onnx', 'tensorrt', 'torchscript')
            
        Returns:
            True if optimization successful, False otherwise
        """
        if not self.active_model:
            logger.error("❌ No active model to optimize")
            return False
        
        try:
            if optimization_type == "onnx":
                return await self._optimize_onnx(model_id)
            elif optimization_type == "torchscript":
                return await self._optimize_torchscript(model_id)
            else:
                logger.error(f"❌ Unsupported optimization type: {optimization_type}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model optimization failed: {e}")
            return False
    
    async def _optimize_onnx(self, model_id: str) -> bool:
        """Optimize model to ONNX format"""
        if not ONNX_AVAILABLE:
            logger.error("❌ ONNX not available for optimization")
            return False
        
        try:
            logger.info(f"🔄 Optimizing model to ONNX: {model_id}")
            
            # Export to ONNX
            onnx_path = self.models_dir / f"{model_id}.onnx"
            self.active_model.export(format="onnx", dynamic=True, simplify=True)
            
            # Move to models directory
            shutil.move(f"{model_id}.onnx", str(onnx_path))
            
            logger.info(f"✅ ONNX optimization completed: {onnx_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX optimization failed: {e}")
            return False
    
    async def _optimize_torchscript(self, model_id: str) -> bool:
        """Optimize model to TorchScript format"""
        try:
            logger.info(f"🔄 Optimizing model to TorchScript: {model_id}")
            
            # Export to TorchScript
            torchscript_path = self.models_dir / f"{model_id}.torchscript.pt"
            self.active_model.export(format="torchscript", dynamic=True)
            
            # Move to models directory
            shutil.move(f"{model_id}.torchscript.pt", str(torchscript_path))
            
            logger.info(f"✅ TorchScript optimization completed: {torchscript_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ TorchScript optimization failed: {e}")
            return False
    
    async def benchmark_model(self, model_id: str, test_images: int = 100) -> Dict[str, Any]:
        """
        Benchmark model performance
        
        Args:
            model_id: Model identifier
            test_images: Number of test images for benchmarking
            
        Returns:
            Performance metrics
        """
        if not self.active_model:
            logger.error("❌ No active model to benchmark")
            return {}
        
        try:
            logger.info(f"📊 Benchmarking model: {model_id}")
            
            # Create test images (random numpy arrays)
            import numpy as np
            test_data = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(test_images)]
            
            # Benchmark inference
            import time
            start_time = time.time()
            
            for img in test_data:
                _ = self.active_model(img, verbose=False)
            
            end_time = time.time()
            
            # Calculate metrics
            total_time = end_time - start_time
            fps = test_images / total_time
            avg_inference_time = total_time / test_images
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            else:
                memory_used = 0
            
            metrics = {
                "model_id": model_id,
                "test_images": test_images,
                "total_time": total_time,
                "fps": fps,
                "avg_inference_time": avg_inference_time,
                "memory_used_gb": memory_used,
                "device": self.active_device,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store metrics
            self.performance_metrics[model_id] = metrics
            
            logger.info(f"✅ Benchmark completed: {fps:.1f} FPS, {avg_inference_time*1000:.1f}ms per inference")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Benchmarking failed: {e}")
            return {}
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        if model_id not in self.model_registry:
            return {"error": f"Unknown model: {model_id}"}
        
        model_path = self.models_dir / f"{model_id}.pt"
        
        info = {
            "model_id": model_id,
            "registry_info": self.model_registry[model_id],
            "file_exists": model_path.exists(),
            "file_size": model_path.stat().st_size if model_path.exists() else 0,
            "versions": self.model_versions.get(model_id, []),
            "performance": self.performance_metrics.get(model_id, {}),
            "is_active": model_id == getattr(self, 'active_model_id', None)
        }
        
        return info
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        models = []
        
        for model_id in self.model_registry:
            info = await self.get_model_info(model_id)
            models.append(info)
        
        return models
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def cleanup_models(self, keep_versions: int = 3):
        """Clean up old model versions"""
        for model_id, versions in self.model_versions.items():
            if len(versions) > keep_versions:
                # Keep only the latest versions
                versions_to_keep = sorted(versions, key=lambda x: x['download_date'], reverse=True)[:keep_versions]
                self.model_versions[model_id] = versions_to_keep
                
                logger.info(f"🧹 Cleaned up old versions for {model_id}, keeping {len(versions_to_keep)} versions")

# Example usage and testing
async def test_yolo_model_manager():
    """Test YOLO model manager functionality"""
    logger.info("🧪 Testing YOLO Model Manager...")
    
    # Initialize manager
    manager = YOLOModelManager()
    
    # List available models
    models = await manager.list_models()
    logger.info(f"📋 Available models: {len(models)}")
    
    for model in models:
        logger.info(f"  - {model['model_id']}: {model['registry_info']['name']}")
    
    # Download and test a model
    model_id = "yolov8n"  # Start with nano for testing
    
    # Download model
    success = await manager.download_model(model_id)
    if not success:
        logger.error("❌ Failed to download model")
        return
    
    # Load model
    success = await manager.load_model(model_id)
    if not success:
        logger.error("❌ Failed to load model")
        return
    
    # Benchmark model
    metrics = await manager.benchmark_model(model_id, test_images=10)
    if metrics:
        logger.info(f"📊 Benchmark results: {metrics}")
    
    # Get model info
    info = await manager.get_model_info(model_id)
    logger.info(f"ℹ️  Model info: {info}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    asyncio.run(test_yolo_model_manager())
