#!/usr/bin/env python3
"""
Test Suite for Phase 1: Hockey Role Classification (Standalone)
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add phase1 to path
sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/phase1')

class Phase1StandaloneTester:
    """Test suite for Phase 1 standalone implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger("Phase1StandaloneTester")
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_imports(self) -> bool:
        """Test Phase 1 standalone imports"""
        try:
            self.logger.info("🧪 Testing Phase 1 standalone imports...")
            
            # Test core imports
            from hockey_role_classifier_standalone import (
                HockeyRoleDataset, HockeyRoleClassifier, 
                HockeyRoleTrainer, HockeyRoleClassificationPipeline,
                MockJarvisIntegration
            )
            self.logger.info("✅ Core classes imported successfully")
            
            # Test PyTorch imports
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import Dataset, DataLoader
            self.logger.info("✅ PyTorch imports successful")
            
            # Test other ML imports
            import numpy as np
            import cv2
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            self.logger.info("✅ ML library imports successful")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Import test failed: {e}")
            return False
    
    def test_dataset_creation(self) -> bool:
        """Test dataset creation"""
        try:
            self.logger.info("🧪 Testing dataset creation...")
            
            from hockey_role_classifier_standalone import HockeyRoleDataset
            
            # Create temporary data directory
            temp_dir = tempfile.mkdtemp()
            data_dir = Path(temp_dir) / "hockey_players"
            data_dir.mkdir(parents=True)
            
            # Create mock dataset
            dataset = HockeyRoleDataset(str(data_dir), split='train')
            
            if len(dataset) == 0:
                self.logger.error("❌ Dataset is empty")
                return False
            
            self.logger.info(f"✅ Dataset created with {len(dataset)} samples")
            
            # Test dataset iteration
            sample = dataset[0]
            required_keys = ['image', 'role', 'role_id', 'image_path']
            
            for key in required_keys:
                if key not in sample:
                    self.logger.error(f"❌ Missing key in sample: {key}")
                    return False
            
            self.logger.info("✅ Dataset sample structure correct")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dataset creation test failed: {e}")
            return False
    
    def test_model_creation(self) -> bool:
        """Test model creation"""
        try:
            self.logger.info("🧪 Testing model creation...")
            
            from hockey_role_classifier_standalone import HockeyRoleClassifier
            
            # Create model
            model = HockeyRoleClassifier(num_classes=3)
            
            # Test model structure
            if not hasattr(model, 'backbone'):
                self.logger.error("❌ Model missing backbone")
                return False
            
            if not hasattr(model, 'dropout'):
                self.logger.error("❌ Model missing dropout")
                return False
            
            self.logger.info("✅ Model structure correct")
            
            # Test forward pass
            import torch
            dummy_input = torch.randn(1, 3, 224, 224)
            output = model(dummy_input)
            
            if output.shape != (1, 3):
                self.logger.error(f"❌ Wrong output shape: {output.shape}")
                return False
            
            self.logger.info("✅ Model forward pass successful")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model creation test failed: {e}")
            return False
    
    def test_trainer_creation(self) -> bool:
        """Test trainer creation"""
        try:
            self.logger.info("🧪 Testing trainer creation...")
            
            from hockey_role_classifier_standalone import HockeyRoleClassifier, HockeyRoleTrainer
            
            # Create model and trainer
            model = HockeyRoleClassifier(num_classes=3)
            trainer = HockeyRoleTrainer(model, device='cpu')
            
            # Test trainer components
            if not hasattr(trainer, 'model'):
                self.logger.error("❌ Trainer missing model")
                return False
            
            if not hasattr(trainer, 'criterion'):
                self.logger.error("❌ Trainer missing criterion")
                return False
            
            if not hasattr(trainer, 'optimizer'):
                self.logger.error("❌ Trainer missing optimizer")
                return False
            
            if not hasattr(trainer, 'scheduler'):
                self.logger.error("❌ Trainer missing scheduler")
                return False
            
            self.logger.info("✅ Trainer components correct")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Trainer creation test failed: {e}")
            return False
    
    def test_mock_jarvis_integration(self) -> bool:
        """Test mock Jarvis integration"""
        try:
            self.logger.info("🧪 Testing mock Jarvis integration...")
            
            from hockey_role_classifier_standalone import MockJarvisIntegration
            
            # Create mock integration
            jarvis = MockJarvisIntegration()
            
            # Test experiment creation
            experiment_id = jarvis.start_experiment("test-experiment", {"param": "value"})
            if not experiment_id:
                self.logger.error("❌ Failed to create experiment")
                return False
            
            self.logger.info(f"✅ Experiment created: {experiment_id}")
            
            # Test metrics logging
            jarvis.log_metrics({"accuracy": 0.95, "loss": 0.05})
            self.logger.info("✅ Metrics logged successfully")
            
            # Test model storage
            model_id = jarvis.store_model("/tmp/test_model.pt", {"type": "test"})
            if not model_id:
                self.logger.error("❌ Failed to store model")
                return False
            
            self.logger.info(f"✅ Model stored: {model_id}")
            
            # Test artifact storage
            artifact_id = jarvis.store_artifact("/tmp/test_artifact.json", {"type": "test"})
            if not artifact_id:
                self.logger.error("❌ Failed to store artifact")
                return False
            
            self.logger.info(f"✅ Artifact stored: {artifact_id}")
            
            # Test business metrics
            jarvis.log_business_metrics({"success_rate": 1.0})
            self.logger.info("✅ Business metrics logged successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Mock Jarvis integration test failed: {e}")
            return False
    
    def test_pipeline_creation(self) -> bool:
        """Test pipeline creation"""
        try:
            self.logger.info("🧪 Testing pipeline creation...")
            
            from hockey_role_classifier_standalone import HockeyRoleClassificationPipeline
            
            # Create configuration
            config = {
                'target_accuracy': 0.85,
                'data_config': {
                    'dataset_path': '/tmp/mock_data'
                },
                'training_config': {
                    'epochs': 10,
                    'batch_size': 16
                },
                'device': 'cpu'
            }
            
            # Create pipeline
            pipeline = HockeyRoleClassificationPipeline(config)
            
            # Test pipeline components
            if not hasattr(pipeline, 'config'):
                self.logger.error("❌ Pipeline missing config")
                return False
            
            if not hasattr(pipeline, 'model'):
                self.logger.error("❌ Pipeline missing model")
                return False
            
            if not hasattr(pipeline, 'trainer'):
                self.logger.error("❌ Pipeline missing trainer")
                return False
            
            if not hasattr(pipeline, 'jarvis'):
                self.logger.error("❌ Pipeline missing Jarvis integration")
                return False
            
            self.logger.info("✅ Pipeline components correct")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline creation test failed: {e}")
            return False
    
    def test_mock_training(self) -> bool:
        """Test mock training process"""
        try:
            self.logger.info("🧪 Testing mock training...")
            
            from hockey_role_classifier_standalone import HockeyRoleClassificationPipeline
            
            # Create minimal configuration
            config = {
                'target_accuracy': 0.85,
                'data_config': {
                    'dataset_path': '/tmp/mock_data'
                },
                'training_config': {
                    'epochs': 2,  # Very short training
                    'batch_size': 4
                },
                'device': 'cpu'
            }
            
            # Create pipeline
            pipeline = HockeyRoleClassificationPipeline(config)
            
            # Test dataset preparation
            if not pipeline.prepare_datasets():
                self.logger.error("❌ Dataset preparation failed")
                return False
            
            self.logger.info("✅ Dataset preparation successful")
            
            # Test model training (short version)
            # Note: With mock data and only 2 epochs, we don't expect to reach 85% accuracy
            # So we'll just check that training completes without crashing
            try:
                pipeline.train_model()
                self.logger.info("✅ Model training completed (expected low accuracy with mock data)")
            except Exception as e:
                self.logger.error(f"❌ Model training crashed: {e}")
                return False
            
            # Test model evaluation
            test_metrics = pipeline.evaluate_model()
            if not test_metrics:
                self.logger.error("❌ Model evaluation failed")
                return False
            
            self.logger.info(f"✅ Model evaluation successful: {test_metrics}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Mock training test failed: {e}")
            return False
    
    def test_configuration_validation(self) -> bool:
        """Test configuration validation"""
        try:
            self.logger.info("🧪 Testing configuration validation...")
            
            from hockey_role_classifier_standalone import HockeyRoleClassificationPipeline
            
            # Test valid configuration
            valid_config = {
                'target_accuracy': 0.85,
                'data_config': {
                    'dataset_path': '/data/hockey_players',
                    'train_split': 0.8,
                    'val_split': 0.1,
                    'test_split': 0.1
                },
                'training_config': {
                    'epochs': 100,
                    'batch_size': 32,
                    'learning_rate': 0.001
                },
                'device': 'cpu'
            }
            
            pipeline = HockeyRoleClassificationPipeline(valid_config)
            
            # Check configuration values
            if pipeline.config['target_accuracy'] != 0.85:
                self.logger.error("❌ Target accuracy not set correctly")
                return False
            
            if pipeline.config['device'] != 'cpu':
                self.logger.error("❌ Device not set correctly")
                return False
            
            self.logger.info("✅ Configuration validation successful")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all Phase 1 standalone tests"""
        self.logger.info("🚀 Starting Phase 1 Standalone Test Suite...")
        
        tests = {
            "Import Test": self.test_imports,
            "Dataset Creation": self.test_dataset_creation,
            "Model Creation": self.test_model_creation,
            "Trainer Creation": self.test_trainer_creation,
            "Mock Jarvis Integration": self.test_mock_jarvis_integration,
            "Pipeline Creation": self.test_pipeline_creation,
            "Mock Training": self.test_mock_training,
            "Configuration Validation": self.test_configuration_validation
        }
        
        results = {}
        for test_name, test_func in tests.items():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running {test_name}")
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
        self.logger.info("PHASE 1 STANDALONE TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.logger.info(f"{test_name:<25} {status}")
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"TOTAL: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 ALL PHASE 1 STANDALONE TESTS PASSED! Implementation is ready.")
        else:
            self.logger.error(f"⚠️  {total - passed} tests failed. Please check the logs above.")
        
        self.logger.info(f"{'='*60}")

def main():
    """Main test function"""
    tester = Phase1StandaloneTester()
    results = tester.run_all_tests()
    tester.print_summary(results)
    
    # Return exit code based on results
    all_passed = all(results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
