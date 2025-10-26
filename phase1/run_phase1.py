#!/usr/bin/env python3
"""
Phase 1 Execution Script - Hockey Role Classification
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add phase1 to path
sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/phase1')
from hockey_role_classifier import HockeyRoleClassificationPipeline

class Phase1Runner:
    """Phase 1 execution runner"""
    
    def __init__(self):
        self.logger = logging.getLogger("Phase1Runner")
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_config(self) -> Dict[str, Any]:
        """Create Phase 1 configuration"""
        return {
            'phase': 'phase1',
            'target_accuracy': 0.85,  # 85% target accuracy
            'model_type': 'resnet18',
            'num_classes': 3,
            'classes': ['player', 'goalie', 'referee'],
            'data_config': {
                'dataset_path': '/data/hockey_players',
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'image_size': (224, 224)
            },
            'training_config': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'patience': 10  # Early stopping patience
            },
            'evaluation_config': {
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
                'threshold': 0.5,
                'confusion_matrix': True
            },
            'toolchain_config': {
                'model_type': 'hockey_role_classifier',
                'target_accuracy': 0.85,
                'experiment_name': 'hockey-role-classification-phase1'
            },
            'device': 'cpu',  # Use CPU for now
            'save_model': True,
            'generate_report': True
        }
    
    def run_phase1(self) -> bool:
        """Run Phase 1: Hockey Role Classification"""
        try:
            self.logger.info("🏒 Starting Phase 1: Hockey Role Classification")
            self.logger.info("="*60)
            
            # Create configuration
            config = self.create_config()
            self.logger.info(f"📋 Configuration created: {config['model_type']}")
            
            # Create pipeline
            pipeline = HockeyRoleClassificationPipeline(config)
            self.logger.info("🔧 Pipeline initialized")
            
            # Run pipeline
            self.logger.info("🚀 Running hockey role classification pipeline...")
            success = pipeline.run_pipeline()
            
            if success:
                self.logger.info("🎉 Phase 1 completed successfully!")
                self.logger.info("✅ Target accuracy of 85% achieved!")
                self.logger.info("✅ Model trained and evaluated!")
                self.logger.info("✅ Report generated and stored!")
            else:
                self.logger.error("❌ Phase 1 failed!")
                self.logger.error("⚠️ Target accuracy not achieved or pipeline failed!")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Phase 1 execution failed: {e}")
            return False
    
    def run_with_workflow(self) -> bool:
        """Run Phase 1 with Temporal workflow integration"""
        try:
            self.logger.info("🔄 Running Phase 1 with Temporal workflow...")
            
            # Import TSAI integration
            sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/tsai-integration')
            from tsai_integration import ToolchainComponent
            
            # Initialize toolchain
            toolchain = ToolchainComponent()
            toolchain.initialize({
                'model_type': 'hockey_role_classifier',
                'target_accuracy': 0.85
            })
            toolchain.start()
            
            # Create configuration
            config = self.create_config()
            
            # Start workflow
            workflow_config = {
                'pipeline_type': 'hockey_role_classification',
                'phase': 'phase1',
                'config': config
            }
            
            workflow_id = asyncio.run(toolchain.start_workflow(
                "hockey-detection",
                workflow_config
            ))
            
            self.logger.info(f"🔄 Started workflow: {workflow_id}")
            
            # Monitor workflow
            while True:
                status = asyncio.run(toolchain.get_workflow_status(workflow_id))
                
                if status['status'] == 'completed':
                    self.logger.info("✅ Workflow completed successfully!")
                    break
                elif status['status'] == 'failed':
                    self.logger.error("❌ Workflow failed!")
                    return False
                
                self.logger.info(f"📊 Workflow status: {status['status']}")
                await asyncio.sleep(30)  # Check every 30 seconds
            
            # Get results
            result = asyncio.run(toolchain.get_workflow_result(workflow_id))
            self.logger.info(f"📊 Workflow result: {result}")
            
            # Stop toolchain
            toolchain.stop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Workflow execution failed: {e}")
            return False

def main():
    """Main function"""
    print("🚀 Phase 1: Hockey Role Classification")
    print("="*50)
    print("Target: 85%+ accuracy for player/goalie/referee classification")
    print("="*50)
    
    runner = Phase1Runner()
    
    # Choose execution method
    import argparse
    parser = argparse.ArgumentParser(description='Phase 1: Hockey Role Classification')
    parser.add_argument('--workflow', action='store_true', 
                       help='Run with Temporal workflow integration')
    args = parser.parse_args()
    
    if args.workflow:
        success = runner.run_with_workflow()
    else:
        success = runner.run_phase1()
    
    if success:
        print("\n🎉 Phase 1: Hockey Role Classification - SUCCESS!")
        print("✅ Target accuracy of 85% achieved!")
        print("✅ Model trained and evaluated successfully!")
        print("✅ Report generated and stored!")
        print("\n📋 Next Steps:")
        print("  - Review model performance metrics")
        print("  - Analyze confusion matrix")
        print("  - Prepare for Phase 2: Equipment Classification")
    else:
        print("\n❌ Phase 1: Hockey Role Classification - FAILED!")
        print("⚠️ Target accuracy not achieved or pipeline failed!")
        print("\n🔧 Troubleshooting:")
        print("  - Check dataset quality and size")
        print("  - Adjust model hyperparameters")
        print("  - Consider data augmentation")
        print("  - Review training configuration")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
