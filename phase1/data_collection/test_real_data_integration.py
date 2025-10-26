#!/usr/bin/env python3
"""
Test Suite for Real Hockey Data Integration
"""

import os
import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add data collection modules to path
sys.path.append('/Volumes/Thorage/wip/tsai-jarvis/phase1/data_collection')

class RealDataIntegrationTester:
    """Test suite for real hockey data integration"""
    
    def __init__(self):
        self.logger = logging.getLogger("RealDataIntegrationTester")
        self.setup_logging()
        
        # Create temporary data directory
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "hockey_players"
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def test_imports(self) -> bool:
        """Test real data integration imports"""
        try:
            self.logger.info("🧪 Testing real data integration imports...")
            
            # Test data collector imports
            from hockey_data_collector import HockeyDataCollector
            self.logger.info("✅ HockeyDataCollector imported successfully")
            
            # Test data preprocessor imports
            from data_preprocessor import HockeyDataPreprocessor
            self.logger.info("✅ HockeyDataPreprocessor imported successfully")
            
            # Test real data integration imports
            from real_data_integration import RealHockeyDataIntegration
            self.logger.info("✅ RealHockeyDataIntegration imported successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Import test failed: {e}")
            return False
    
    def test_data_collector_creation(self) -> bool:
        """Test data collector creation"""
        try:
            self.logger.info("🧪 Testing data collector creation...")
            
            from hockey_data_collector import HockeyDataCollector
            
            # Create data collector
            collector = HockeyDataCollector(str(self.data_dir))
            
            # Test collector components
            if not hasattr(collector, 'data_dir'):
                self.logger.error("❌ Data collector missing data_dir")
                return False
            
            if not hasattr(collector, 'sources'):
                self.logger.error("❌ Data collector missing sources")
                return False
            
            if not hasattr(collector, 'logger'):
                self.logger.error("❌ Data collector missing logger")
                return False
            
            self.logger.info("✅ Data collector creation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data collector creation test failed: {e}")
            return False
    
    def test_data_preprocessor_creation(self) -> bool:
        """Test data preprocessor creation"""
        try:
            self.logger.info("🧪 Testing data preprocessor creation...")
            
            from data_preprocessor import HockeyDataPreprocessor
            
            # Create data preprocessor
            preprocessor = HockeyDataPreprocessor(str(self.data_dir))
            
            # Test preprocessor components
            if not hasattr(preprocessor, 'data_dir'):
                self.logger.error("❌ Data preprocessor missing data_dir")
                return False
            
            if not hasattr(preprocessor, 'target_size'):
                self.logger.error("❌ Data preprocessor missing target_size")
                return False
            
            if not hasattr(preprocessor, 'train_augmentations'):
                self.logger.error("❌ Data preprocessor missing train_augmentations")
                return False
            
            if not hasattr(preprocessor, 'val_augmentations'):
                self.logger.error("❌ Data preprocessor missing val_augmentations")
                return False
            
            self.logger.info("✅ Data preprocessor creation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessor creation test failed: {e}")
            return False
    
    def test_real_data_integration_creation(self) -> bool:
        """Test real data integration creation"""
        try:
            self.logger.info("🧪 Testing real data integration creation...")
            
            from real_data_integration import RealHockeyDataIntegration
            
            # Create real data integration
            integration = RealHockeyDataIntegration(str(self.data_dir))
            
            # Test integration components
            if not hasattr(integration, 'data_dir'):
                self.logger.error("❌ Real data integration missing data_dir")
                return False
            
            if not hasattr(integration, 'data_collector'):
                self.logger.error("❌ Real data integration missing data_collector")
                return False
            
            if not hasattr(integration, 'data_preprocessor'):
                self.logger.error("❌ Real data integration missing data_preprocessor")
                return False
            
            if not hasattr(integration, 'config'):
                self.logger.error("❌ Real data integration missing config")
                return False
            
            self.logger.info("✅ Real data integration creation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Real data integration creation test failed: {e}")
            return False
    
    def test_sample_data_creation(self) -> bool:
        """Test sample data creation"""
        try:
            self.logger.info("🧪 Testing sample data creation...")
            
            from real_data_integration import RealHockeyDataIntegration
            
            # Create integration
            integration = RealHockeyDataIntegration(str(self.data_dir))
            
            # Create sample data
            success = integration.create_sample_data()
            
            if not success:
                self.logger.error("❌ Sample data creation failed")
                return False
            
            # Check if sample data was created
            roles = ['player', 'goalie', 'referee']
            splits = ['train', 'val', 'test']
            
            for role in roles:
                for split in splits:
                    split_dir = self.data_dir / 'raw' / split / role
                    if not split_dir.exists():
                        self.logger.error(f"❌ Missing directory: {split_dir}")
                        return False
                    
                    images = list(split_dir.glob('*.jpg'))
                    if len(images) == 0:
                        self.logger.error(f"❌ No images in {split_dir}")
                        return False
            
            self.logger.info("✅ Sample data creation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Sample data creation test failed: {e}")
            return False
    
    def test_data_preprocessing(self) -> bool:
        """Test data preprocessing"""
        try:
            self.logger.info("🧪 Testing data preprocessing...")
            
            from real_data_integration import RealHockeyDataIntegration
            
            # Create integration
            integration = RealHockeyDataIntegration(str(self.data_dir))
            
            # Create sample data first
            if not integration.create_sample_data():
                self.logger.error("❌ Sample data creation failed")
                return False
            
            # Test data preprocessing
            preprocessing_results = integration.preprocess_real_data()
            
            if not preprocessing_results:
                self.logger.error("❌ Data preprocessing failed")
                return False
            
            # Check if processed data exists
            roles = ['player', 'goalie', 'referee']
            splits = ['train', 'val', 'test']
            
            for role in roles:
                for split in splits:
                    split_dir = self.data_dir / split / role
                    if not split_dir.exists():
                        self.logger.error(f"❌ Missing processed directory: {split_dir}")
                        return False
            
            self.logger.info("✅ Data preprocessing successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing test failed: {e}")
            return False
    
    def test_dataset_validation(self) -> bool:
        """Test dataset validation"""
        try:
            self.logger.info("🧪 Testing dataset validation...")
            
            from real_data_integration import RealHockeyDataIntegration
            
            # Create integration
            integration = RealHockeyDataIntegration(str(self.data_dir))
            
            # Create sample data and preprocess
            if not integration.create_sample_data():
                self.logger.error("❌ Sample data creation failed")
                return False
            
            if not integration.preprocess_real_data():
                self.logger.error("❌ Data preprocessing failed")
                return False
            
            # Test dataset validation
            validation_results = integration.validate_dataset()
            
            if not validation_results:
                self.logger.error("❌ Dataset validation failed")
                return False
            
            # Check validation results
            if not validation_results.get('validation_passed', False):
                self.logger.error("❌ Dataset validation failed")
                return False
            
            if validation_results.get('total_samples', 0) == 0:
                self.logger.error("❌ No samples found in dataset")
                return False
            
            self.logger.info("✅ Dataset validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Dataset validation test failed: {e}")
            return False
    
    def test_complete_integration(self) -> bool:
        """Test complete integration pipeline"""
        try:
            self.logger.info("🧪 Testing complete integration pipeline...")
            
            from real_data_integration import RealHockeyDataIntegration
            
            # Create integration
            integration = RealHockeyDataIntegration(str(self.data_dir))
            
            # Run complete integration
            results = integration.run_complete_integration()
            
            if not results:
                self.logger.error("❌ Complete integration failed")
                return False
            
            # Check integration results
            if not results.get('overall_success', False):
                self.logger.error("❌ Integration pipeline failed")
                return False
            
            # Check steps completed
            required_steps = ['data_collection', 'data_preprocessing', 'dataset_validation', 'model_training']
            completed_steps = results.get('steps_completed', [])
            
            for step in required_steps:
                if step not in completed_steps:
                    self.logger.error(f"❌ Missing step: {step}")
                    return False
            
            self.logger.info("✅ Complete integration pipeline successful")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Complete integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all real data integration tests"""
        self.logger.info("🚀 Starting Real Data Integration Test Suite...")
        
        tests = {
            "Import Test": self.test_imports,
            "Data Collector Creation": self.test_data_collector_creation,
            "Data Preprocessor Creation": self.test_data_preprocessor_creation,
            "Real Data Integration Creation": self.test_real_data_integration_creation,
            "Sample Data Creation": self.test_sample_data_creation,
            "Data Preprocessing": self.test_data_preprocessing,
            "Dataset Validation": self.test_dataset_validation,
            "Complete Integration": self.test_complete_integration
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
        self.logger.info("REAL DATA INTEGRATION TEST SUMMARY")
        self.logger.info(f"{'='*60}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            self.logger.info(f"{test_name:<30} {status}")
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"TOTAL: {passed}/{total} tests passed")
        
        if passed == total:
            self.logger.info("🎉 ALL REAL DATA INTEGRATION TESTS PASSED! Ready for real hockey data!")
        else:
            self.logger.error(f"⚠️  {total - passed} tests failed. Please check the logs above.")
        
        self.logger.info(f"{'='*60}")
    
    def cleanup(self):
        """Cleanup temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            self.logger.info("🧹 Cleaned up temporary files")
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup failed: {e}")

def main():
    """Main test function"""
    tester = RealDataIntegrationTester()
    
    try:
        results = tester.run_all_tests()
        tester.print_summary(results)
        
        # Return exit code based on results
        all_passed = all(results.values())
        return 0 if all_passed else 1
        
    finally:
        tester.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
