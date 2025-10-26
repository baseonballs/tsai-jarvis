# Real Hockey Data Integration

## Overview

This module provides comprehensive tools for collecting, processing, and integrating real hockey data for Phase 1: Hockey role classification. It includes data collection from multiple sources, preprocessing pipelines, and integration with the training system.

## Architecture

### Components

#### 1. HockeyDataCollector
- **YouTube Integration**: Collect hockey videos from YouTube
- **Web Scraping**: Extract images from hockey websites
- **Local Upload**: Process user-uploaded hockey media
- **Data Annotation**: Automatic role classification using heuristics

#### 2. HockeyDataPreprocessor
- **Image Validation**: Quality checks and filtering
- **Data Augmentation**: Training-specific augmentations
- **Frame Extraction**: Video to image conversion
- **Dataset Splits**: Train/validation/test organization

#### 3. RealHockeyDataIntegration
- **Complete Pipeline**: End-to-end data integration
- **Dataset Validation**: Quality assurance checks
- **Model Training**: Integration with training system
- **Report Generation**: Comprehensive analytics

## Features

### Data Collection
- **Multi-source Collection**: YouTube, web scraping, local uploads
- **Automatic Annotation**: Heuristic-based role classification
- **Quality Filtering**: Image validation and filtering
- **Metadata Tracking**: Comprehensive data provenance

### Data Processing
- **Image Preprocessing**: Resizing, normalization, augmentation
- **Video Processing**: Frame extraction and processing
- **Data Augmentation**: Training-specific augmentations
- **Split Management**: Automatic train/val/test splits

### Integration
- **Training Pipeline**: Seamless integration with model training
- **Quality Assurance**: Dataset validation and quality checks
- **Performance Monitoring**: Training metrics and analytics
- **Report Generation**: Comprehensive integration reports

## Usage

### Basic Data Collection
```python
from hockey_data_collector import HockeyDataCollector

# Initialize collector
collector = HockeyDataCollector("/data/hockey_players")

# Collect data from all sources
metadata = collector.run_data_collection()
```

### Data Preprocessing
```python
from data_preprocessor import HockeyDataPreprocessor

# Initialize preprocessor
preprocessor = HockeyDataPreprocessor("/data/hockey_players")

# Process all data
results = preprocessor.process_all_data()
```

### Complete Integration
```python
from real_data_integration import RealHockeyDataIntegration

# Initialize integration
integration = RealHockeyDataIntegration("/data/hockey_players")

# Run complete pipeline
results = integration.run_complete_integration()
```

## Configuration

### Data Collection Sources
```python
sources = {
    'youtube': {
        'enabled': True,
        'api_key': 'YOUR_YOUTUBE_API_KEY',
        'search_terms': [
            'hockey game highlights',
            'hockey players training',
            'hockey goalie saves'
        ]
    },
    'web_scraping': {
        'enabled': True,
        'sources': [
            'https://www.nhl.com',
            'https://www.espn.com/nhl'
        ]
    },
    'local_upload': {
        'enabled': True,
        'upload_dir': '/data/hockey_players/uploads'
    }
}
```

### Preprocessing Parameters
```python
preprocessing_config = {
    'target_size': (224, 224),
    'image_quality': 95,
    'augmentation': {
        'train': {
            'horizontal_flip': 0.5,
            'rotation': 30,
            'brightness_contrast': 0.2
        },
        'val': {
            'resize_only': True
        }
    }
}
```

## Data Structure

### Directory Layout
```
/data/hockey_players/
├── raw/                    # Raw collected data
│   ├── train/
│   │   ├── player/
│   │   ├── goalie/
│   │   └── referee/
│   ├── val/
│   └── test/
├── processed/              # Processed data
├── train/                  # Training data
│   ├── player/
│   ├── goalie/
│   └── referee/
├── val/                    # Validation data
├── test/                   # Test data
├── uploads/                # User uploads
└── metadata/               # Metadata and reports
    ├── dataset_metadata.json
    ├── preprocessing_report.json
    └── integration_report.json
```

### Data Formats
- **Images**: JPG format, 224x224 pixels
- **Videos**: MP4 format, frame extraction
- **Metadata**: JSON format with annotations
- **Reports**: JSON format with analytics

## Quality Assurance

### Image Validation
- **File Size**: 1KB - 50MB
- **Dimensions**: Minimum 32x32 pixels
- **Quality**: Blur detection and filtering
- **Format**: JPG, PNG support

### Dataset Validation
- **Minimum Samples**: 50 per role, 10 per split
- **Role Balance**: Balanced distribution
- **Split Integrity**: Proper train/val/test splits
- **Quality Metrics**: Processing success rates

## Performance Metrics

### Collection Metrics
- **Total Samples**: Number of collected items
- **Source Distribution**: Samples per source
- **Quality Rate**: Valid samples percentage
- **Collection Time**: Processing duration

### Processing Metrics
- **Processing Rate**: Images per second
- **Success Rate**: Successful processing percentage
- **Augmentation Quality**: Training augmentation effectiveness
- **Split Balance**: Dataset split distribution

### Training Metrics
- **Model Accuracy**: Classification performance
- **Training Time**: Model training duration
- **Convergence**: Training convergence metrics
- **Validation Performance**: Cross-validation results

## Troubleshooting

### Common Issues
1. **Data Collection Failures**: Check API keys and network connectivity
2. **Processing Errors**: Verify image formats and file permissions
3. **Validation Failures**: Check minimum data requirements
4. **Training Issues**: Verify dataset quality and balance

### Debugging
- **Logging**: Comprehensive logging for all operations
- **Error Handling**: Graceful error handling and recovery
- **Progress Tracking**: Real-time progress monitoring
- **Quality Reports**: Detailed quality analysis

## Dependencies

### Core Dependencies
- **OpenCV**: Image and video processing
- **NumPy**: Numerical operations
- **PIL**: Image manipulation
- **Pandas**: Data analysis

### Augmentation Dependencies
- **Albumentations**: Advanced data augmentation
- **Torchvision**: PyTorch vision utilities

### Collection Dependencies
- **Requests**: HTTP requests
- **BeautifulSoup**: Web scraping
- **Selenium**: Dynamic web content

### Development Dependencies
- **Pytest**: Testing framework
- **Pytest-asyncio**: Async testing support

## License

This module is part of the TSAI ecosystem and follows the same licensing terms.
