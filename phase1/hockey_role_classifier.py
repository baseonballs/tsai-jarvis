#!/usr/bin/env python3
"""
Phase 1: Hockey Role Classification Implementation
Target: 85%+ accuracy for player/goalie/referee classification
"""

import os
import sys
import json
import time
import logging
import asyncio
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add TSAI integration to path
sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/tsai-integration')
from tsai_integration import ToolchainComponent

class HockeyRoleDataset(Dataset):
    """Dataset for hockey role classification"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples"""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            logging.warning(f"Split directory {split_dir} does not exist. Creating mock data.")
            return self._create_mock_samples()
        
        # Load real data if available
        for role_dir in split_dir.iterdir():
            if role_dir.is_dir():
                role = role_dir.name
                for img_path in role_dir.glob('*.jpg'):
                    samples.append({
                        'image_path': str(img_path),
                        'role': role,
                        'role_id': self._get_role_id(role)
                    })
        
        return samples
    
    def _create_mock_samples(self) -> List[Dict[str, Any]]:
        """Create mock samples for testing"""
        mock_samples = []
        roles = ['player', 'goalie', 'referee']
        
        for i in range(100):  # Create 100 mock samples
            role = roles[i % 3]
            mock_samples.append({
                'image_path': f'/mock/path/image_{i}.jpg',
                'role': role,
                'role_id': self._get_role_id(role)
            })
        
        return mock_samples
    
    def _get_role_id(self, role: str) -> int:
        """Get role ID for classification"""
        role_map = {'player': 0, 'goalie': 1, 'referee': 2}
        return role_map.get(role, 0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image (mock for now)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'role': sample['role'],
            'role_id': sample['role_id'],
            'image_path': sample['image_path']
        }

class HockeyRoleClassifier(nn.Module):
    """Hockey role classification model"""
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super(HockeyRoleClassifier, self).__init__()
        
        # Use a pre-trained backbone (ResNet18 for efficiency)
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Modify the final layer for our classification task
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return x

class HockeyRoleTrainer:
    """Trainer for hockey role classification"""
    
    def __init__(self, model: HockeyRoleClassifier, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in dataloader:
            images = batch['image'].to(self.device)
            labels = batch['role_id'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                labels = batch['role_id'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

class HockeyRoleClassificationPipeline:
    """Main pipeline for hockey role classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("HockeyRoleClassification")
        self.setup_logging()
        
        # Initialize TSAI Toolchain component
        self.toolchain = ToolchainComponent()
        self.toolchain.initialize(config.get('toolchain_config', {}))
        self.toolchain.start()
        
        # Initialize model and trainer
        self.model = HockeyRoleClassifier(num_classes=3)
        self.trainer = HockeyRoleTrainer(self.model, device=config.get('device', 'cpu'))
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def prepare_datasets(self) -> bool:
        """Prepare training, validation, and test datasets"""
        try:
            self.logger.info("📊 Preparing datasets...")
            
            data_config = self.config.get('data_config', {})
            data_dir = data_config.get('dataset_path', '/data/hockey_players')
            
            # Create datasets
            self.train_dataset = HockeyRoleDataset(data_dir, split='train')
            self.val_dataset = HockeyRoleDataset(data_dir, split='val')
            self.test_dataset = HockeyRoleDataset(data_dir, split='test')
            
            self.logger.info(f"✅ Datasets prepared:")
            self.logger.info(f"  - Train: {len(self.train_dataset)} samples")
            self.logger.info(f"  - Validation: {len(self.val_dataset)} samples")
            self.logger.info(f"  - Test: {len(self.test_dataset)} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare datasets: {e}")
            return False
    
    def train_model(self) -> bool:
        """Train the hockey role classification model"""
        try:
            self.logger.info("🏋️ Training hockey role classification model...")
            
            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.get('training_config', {}).get('batch_size', 32),
                shuffle=True
            )
            
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.get('training_config', {}).get('batch_size', 32),
                shuffle=False
            )
            
            # Training configuration
            epochs = self.config.get('training_config', {}).get('epochs', 100)
            target_accuracy = self.config.get('target_accuracy', 0.85)
            
            # Start experiment tracking
            experiment_id = self.toolchain.start_experiment(
                "hockey-role-classification",
                {
                    'model_type': 'resnet18',
                    'num_classes': 3,
                    'epochs': epochs,
                    'batch_size': train_loader.batch_size,
                    'target_accuracy': target_accuracy
                }
            )
            
            self.logger.info(f"📈 Started experiment: {experiment_id}")
            
            best_accuracy = 0.0
            best_model_state = None
            
            for epoch in range(epochs):
                # Train epoch
                train_metrics = self.trainer.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.trainer.validate(val_loader)
                
                # Log metrics
                self.toolchain.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall'],
                    'val_f1_score': val_metrics['f1_score']
                }, step=epoch)
                
                # Update learning rate
                self.trainer.scheduler.step()
                
                # Check if this is the best model
                if val_metrics['accuracy'] > best_accuracy:
                    best_accuracy = val_metrics['accuracy']
                    best_model_state = self.model.state_dict().copy()
                
                # Log progress
                self.logger.info(f"Epoch {epoch+1}/{epochs}: "
                               f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                               f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                               f"Val F1: {val_metrics['f1_score']:.3f}")
                
                # Early stopping if target accuracy reached
                if val_metrics['accuracy'] >= target_accuracy * 100:
                    self.logger.info(f"🎯 Target accuracy {target_accuracy*100}% reached!")
                    break
            
            # Load best model
            if best_model_state:
                self.model.load_state_dict(best_model_state)
            
            # Store trained model
            model_path = f"/tmp/hockey_role_classifier_epoch_{epochs}.pt"
            torch.save(self.model.state_dict(), model_path)
            
            model_id = self.toolchain.store_model(model_path, {
                'model_type': 'hockey_role_classifier',
                'num_classes': 3,
                'best_accuracy': best_accuracy,
                'epochs_trained': epoch + 1,
                'target_accuracy': target_accuracy
            })
            
            self.logger.info(f"💾 Model stored: {model_id}")
            self.logger.info(f"🏆 Best validation accuracy: {best_accuracy:.2f}%")
            
            return best_accuracy >= target_accuracy * 100
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return False
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate the trained model"""
        try:
            self.logger.info("📊 Evaluating model...")
            
            # Create test data loader
            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.get('training_config', {}).get('batch_size', 32),
                shuffle=False
            )
            
            # Evaluate on test set
            test_metrics = self.trainer.validate(test_loader)
            
            # Log evaluation metrics
            self.toolchain.log_metrics({
                'test_accuracy': test_metrics['accuracy'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_f1_score': test_metrics['f1_score']
            })
            
            self.logger.info(f"📈 Test Results:")
            self.logger.info(f"  - Accuracy: {test_metrics['accuracy']:.2f}%")
            self.logger.info(f"  - Precision: {test_metrics['precision']:.3f}")
            self.logger.info(f"  - Recall: {test_metrics['recall']:.3f}")
            self.logger.info(f"  - F1-Score: {test_metrics['f1_score']:.3f}")
            
            return test_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation failed: {e}")
            return {}
    
    def generate_report(self, test_metrics: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        try:
            self.logger.info("📄 Generating evaluation report...")
            
            # Create report
            report = {
                'model_info': {
                    'model_type': 'hockey_role_classifier',
                    'architecture': 'resnet18',
                    'num_classes': 3,
                    'classes': ['player', 'goalie', 'referee']
                },
                'performance': {
                    'accuracy': test_metrics.get('accuracy', 0),
                    'precision': test_metrics.get('precision', 0),
                    'recall': test_metrics.get('recall', 0),
                    'f1_score': test_metrics.get('f1_score', 0)
                },
                'target_achieved': test_metrics.get('accuracy', 0) >= self.config.get('target_accuracy', 0.85) * 100,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save report
            report_path = "/tmp/hockey_role_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Store report
            report_id = self.toolchain.store_artifact(report_path, {
                'type': 'evaluation_report',
                'model': 'hockey_role_classifier',
                'phase': 'phase1'
            })
            
            self.logger.info(f"📊 Report generated: {report_id}")
            
            return report_id
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return ""
    
    def run_pipeline(self) -> bool:
        """Run the complete hockey role classification pipeline"""
        try:
            self.logger.info("🚀 Starting Hockey Role Classification Pipeline...")
            
            # Step 1: Prepare datasets
            if not self.prepare_datasets():
                return False
            
            # Step 2: Train model
            if not self.train_model():
                return False
            
            # Step 3: Evaluate model
            test_metrics = self.evaluate_model()
            if not test_metrics:
                return False
            
            # Step 4: Generate report
            report_id = self.generate_report(test_metrics)
            if not report_id:
                return False
            
            # Step 5: Check if target accuracy achieved
            target_achieved = test_metrics.get('accuracy', 0) >= self.config.get('target_accuracy', 0.85) * 100
            
            if target_achieved:
                self.logger.info("🎯 TARGET ACCURACY ACHIEVED! Phase 1 successful!")
            else:
                self.logger.warning(f"⚠️ Target accuracy not achieved. Current: {test_metrics.get('accuracy', 0):.2f}%")
            
            # Log business metrics
            self.toolchain.log_business_metrics({
                'user_engagement': ['hockey_role_classification_completed'],
                'pipeline_success_rate': 1.0 if target_achieved else 0.0,
                'experiments': [{'status': 'completed', 'accuracy': test_metrics.get('accuracy', 0)}]
            })
            
            return target_achieved
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            return False
        finally:
            # Stop toolchain component
            self.toolchain.stop()

def main():
    """Main function for Phase 1 implementation"""
    
    # Configuration for Phase 1
    config = {
        'target_accuracy': 0.85,  # 85% target accuracy
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
        'evaluation_config': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'threshold': 0.5
        },
        'toolchain_config': {
            'model_type': 'resnet18',
            'target_accuracy': 0.85
        },
        'device': 'cpu'  # Use CPU for now, can be changed to 'cuda' if GPU available
    }
    
    # Create and run pipeline
    pipeline = HockeyRoleClassificationPipeline(config)
    success = pipeline.run_pipeline()
    
    if success:
        print("🎉 Phase 1: Hockey Role Classification - SUCCESS!")
        print("✅ Target accuracy of 85% achieved!")
        print("✅ Model trained and evaluated successfully!")
        print("✅ Report generated and stored!")
    else:
        print("❌ Phase 1: Hockey Role Classification - FAILED!")
        print("⚠️ Target accuracy not achieved or pipeline failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
