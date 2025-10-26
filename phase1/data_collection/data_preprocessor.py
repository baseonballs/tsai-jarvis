#!/usr/bin/env python3
"""
Hockey Data Preprocessor - Process and prepare real hockey data for training
"""

import os
import sys
import json
import time
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import shutil
from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

class HockeyDataPreprocessor:
    """Preprocessor for hockey data to prepare it for training"""
    
    def __init__(self, data_dir: str = "/data/hockey_players"):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger("HockeyDataPreprocessor")
        self.setup_logging()
        
        # Image processing parameters
        self.target_size = (224, 224)
        self.image_quality = 95
        
        # Data augmentation configurations
        self.setup_augmentations()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def setup_augmentations(self):
        """Setup data augmentation pipelines"""
        # Simplified augmentations without albumentations
        self.train_augmentations = None  # Will use OpenCV-based augmentations
        self.val_augmentations = None    # Will use OpenCV-based augmentations
    
    def validate_image(self, image_path: Path) -> bool:
        """Validate if image is suitable for training"""
        try:
            # Check if file exists
            if not image_path.exists():
                return False
            
            # Check file size (minimum 1KB, maximum 50MB)
            file_size = image_path.stat().st_size
            if file_size < 1024 or file_size > 50 * 1024 * 1024:
                return False
            
            # Try to load image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Check image dimensions
            height, width = image.shape[:2]
            if height < 32 or width < 32:
                return False
            
            # Check if image is not corrupted
            if cv2.Laplacian(image, cv2.CV_64F).var() < 100:
                return False  # Too blurry
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Image validation failed for {image_path}: {e}")
            return False
    
    def preprocess_image(self, image_path: Path, output_path: Path, 
                        augmentation_type: str = 'val') -> bool:
        """Preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return False
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Simple preprocessing without albumentations
            # Resize to target size
            processed_image = cv2.resize(image, self.target_size)
            
            # Apply simple augmentations for training
            if augmentation_type == 'train':
                # Random horizontal flip
                if np.random.random() > 0.5:
                    processed_image = cv2.flip(processed_image, 1)
                
                # Random brightness/contrast
                if np.random.random() > 0.5:
                    alpha = np.random.uniform(0.8, 1.2)  # Contrast
                    beta = np.random.uniform(-20, 20)     # Brightness
                    processed_image = cv2.convertScaleAbs(processed_image, alpha=alpha, beta=beta)
            
            # Save processed image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Image preprocessing failed for {image_path}: {e}")
            return False
    
    def extract_frames_from_video(self, video_path: Path, output_dir: Path, 
                                 frame_interval: int = 30) -> List[Path]:
        """Extract frames from video file"""
        try:
            self.logger.info(f"🎬 Extracting frames from: {video_path}")
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"❌ Could not open video: {video_path}")
                return []
            
            frame_count = 0
            extracted_frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at specified intervals
                if frame_count % frame_interval == 0:
                    frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    extracted_frames.append(frame_path)
                
                frame_count += 1
            
            cap.release()
            self.logger.info(f"✅ Extracted {len(extracted_frames)} frames")
            return extracted_frames
            
        except Exception as e:
            self.logger.error(f"❌ Frame extraction failed: {e}")
            return []
    
    def process_dataset_split(self, split: str, role: str, 
                            augmentation_type: str = 'val') -> Dict[str, Any]:
        """Process a dataset split for a specific role"""
        try:
            self.logger.info(f"📊 Processing {split}/{role} dataset...")
            
            # Source and destination directories
            src_dir = self.data_dir / 'raw' / split / role
            dst_dir = self.data_dir / split / role
            
            if not src_dir.exists():
                self.logger.warning(f"⚠️ Source directory {src_dir} does not exist")
                return {'processed': 0, 'skipped': 0, 'errors': 0}
            
            # Create destination directory
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            processed_count = 0
            skipped_count = 0
            error_count = 0
            
            # Process all images in source directory
            for src_path in src_dir.glob('*.jpg'):
                dst_path = dst_dir / src_path.name
                
                # Validate image
                if not self.validate_image(src_path):
                    skipped_count += 1
                    continue
                
                # Preprocess image
                if self.preprocess_image(src_path, dst_path, augmentation_type):
                    processed_count += 1
                else:
                    error_count += 1
            
            result = {
                'processed': processed_count,
                'skipped': skipped_count,
                'errors': error_count,
                'total': processed_count + skipped_count + error_count
            }
            
            self.logger.info(f"✅ {split}/{role} processing complete: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Dataset split processing failed: {e}")
            return {'processed': 0, 'skipped': 0, 'errors': 0}
    
    def create_dataset_splits(self, train_split: float = 0.8, 
                            val_split: float = 0.1, 
                            test_split: float = 0.1) -> Dict[str, Any]:
        """Create train/validation/test splits from raw data"""
        try:
            self.logger.info("📊 Creating dataset splits...")
            
            raw_dir = self.data_dir / 'raw'
            if not raw_dir.exists():
                self.logger.error(f"❌ Raw data directory {raw_dir} does not exist")
                return {}
            
            roles = ['player', 'goalie', 'referee']
            splits = {}
            
            for role in roles:
                role_dir = raw_dir / role
                if not role_dir.exists():
                    self.logger.warning(f"⚠️ Role directory {role_dir} does not exist")
                    continue
                
                # Get all images for this role
                images = list(role_dir.glob('*.jpg'))
                if not images:
                    self.logger.warning(f"⚠️ No images found for role {role}")
                    continue
                
                # Shuffle images
                import random
                random.shuffle(images)
                
                # Calculate split indices
                total_images = len(images)
                train_end = int(total_images * train_split)
                val_end = train_end + int(total_images * val_split)
                
                # Create splits
                role_splits = {
                    'train': images[:train_end],
                    'val': images[train_end:val_end],
                    'test': images[val_end:]
                }
                
                splits[role] = role_splits
                
                self.logger.info(f"✅ {role} splits created:")
                self.logger.info(f"  - Train: {len(role_splits['train'])} images")
                self.logger.info(f"  - Val: {len(role_splits['val'])} images")
                self.logger.info(f"  - Test: {len(role_splits['test'])} images")
            
            return splits
            
        except Exception as e:
            self.logger.error(f"❌ Dataset split creation failed: {e}")
            return {}
    
    def process_all_data(self) -> Dict[str, Any]:
        """Process all hockey data for training"""
        try:
            self.logger.info("🚀 Starting hockey data preprocessing...")
            
            # Create dataset splits
            splits = self.create_dataset_splits()
            if not splits:
                self.logger.error("❌ No data splits created")
                return {}
            
            processing_results = {}
            
            # Process each role and split
            for role, role_splits in splits.items():
                processing_results[role] = {}
                
                for split_name, images in role_splits.items():
                    if not images:
                        continue
                    
                    # Determine augmentation type
                    augmentation_type = 'train' if split_name == 'train' else 'val'
                    
                    # Process images
                    result = self.process_dataset_split(split_name, role, augmentation_type)
                    processing_results[role][split_name] = result
            
            # Generate summary
            total_processed = sum(
                sum(role_results.values()) 
                for role_results in processing_results.values() 
                for role_results in role_results.values()
            )
            
            self.logger.info(f"🎉 Data preprocessing completed!")
            self.logger.info(f"📊 Total images processed: {total_processed}")
            
            return processing_results
            
        except Exception as e:
            self.logger.error(f"❌ Data preprocessing failed: {e}")
            return {}
    
    def generate_dataset_report(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive dataset report"""
        try:
            self.logger.info("📄 Generating dataset report...")
            
            report = {
                'dataset_info': {
                    'total_roles': len(processing_results),
                    'roles': list(processing_results.keys()),
                    'processing_date': datetime.now().isoformat()
                },
                'role_statistics': {},
                'split_statistics': {},
                'data_quality': {
                    'avg_processed_per_role': 0,
                    'total_processed': 0,
                    'total_skipped': 0,
                    'total_errors': 0
                }
            }
            
            # Calculate statistics
            total_processed = 0
            total_skipped = 0
            total_errors = 0
            
            for role, role_results in processing_results.items():
                role_processed = 0
                role_skipped = 0
                role_errors = 0
                
                for split, stats in role_results.items():
                    role_processed += stats.get('processed', 0)
                    role_skipped += stats.get('skipped', 0)
                    role_errors += stats.get('errors', 0)
                
                report['role_statistics'][role] = {
                    'processed': role_processed,
                    'skipped': role_skipped,
                    'errors': role_errors,
                    'total': role_processed + role_skipped + role_errors
                }
                
                total_processed += role_processed
                total_skipped += role_skipped
                total_errors += role_errors
            
            # Update data quality metrics
            report['data_quality'].update({
                'total_processed': total_processed,
                'total_skipped': total_skipped,
                'total_errors': total_errors,
                'avg_processed_per_role': total_processed / len(processing_results) if processing_results else 0
            })
            
            # Save report
            report_path = self.data_dir / 'metadata' / 'preprocessing_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"✅ Dataset report saved to {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return {}

def main():
    """Main function for data preprocessing"""
    preprocessor = HockeyDataPreprocessor()
    results = preprocessor.process_all_data()
    
    if results:
        report = preprocessor.generate_dataset_report(results)
        print("🎉 Hockey data preprocessing completed successfully!")
        print(f"📊 Total processed: {report['data_quality']['total_processed']}")
        print(f"📈 Role statistics: {report['role_statistics']}")
    else:
        print("❌ Hockey data preprocessing failed!")

if __name__ == "__main__":
    main()
