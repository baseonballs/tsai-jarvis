# Phase 1: Hockey Role Classification

## Overview

Phase 1 implements hockey role classification with a target accuracy of 85%+ for distinguishing between:
- **Player** - Regular hockey players
- **Goalie** - Goalkeepers with specialized equipment
- **Referee** - Game officials

## Architecture

### Model Architecture
- **Backbone**: ResNet18 (pre-trained on ImageNet)
- **Classification Head**: 3-class classifier (player/goalie/referee)
- **Input Size**: 224x224 RGB images
- **Output**: Softmax probabilities for each class

### Training Strategy
- **Pre-trained Backbone**: Transfer learning from ImageNet
- **Data Augmentation**: Random crops, flips, color jitter
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Cross-entropy loss
- **Early Stopping**: Patience-based stopping

## Implementation

### Core Components

#### 1. HockeyRoleDataset
```python
class HockeyRoleDataset(Dataset):
    """Dataset for hockey role classification"""
    - Loads images and labels
    - Supports train/val/test splits
    - Handles data augmentation
    - Mock data generation for testing
```

#### 2. HockeyRoleClassifier
```python
class HockeyRoleClassifier(nn.Module):
    """Hockey role classification model"""
    - ResNet18 backbone
    - Custom classification head
    - Dropout for regularization
    - Pre-trained weights initialization
```

#### 3. HockeyRoleTrainer
```python
class HockeyRoleTrainer:
    """Trainer for hockey role classification"""
    - Training loop implementation
    - Validation and metrics calculation
    - Learning rate scheduling
    - Model checkpointing
```

#### 4. HockeyRoleClassificationPipeline
```python
class HockeyRoleClassificationPipeline:
    """Main pipeline for hockey role classification"""
    - Dataset preparation
    - Model training
    - Model evaluation
    - Report generation
    - Jarvis integration
```

## Usage

### Basic Execution
```bash
# Run Phase 1 with basic pipeline
python run_phase1.py

# Run Phase 1 with Temporal workflow
python run_phase1.py --workflow
```

### Configuration
```python
config = {
    'target_accuracy': 0.85,  # 85% target
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
    }
}
```

## Integration with Jarvis Core

### TSAI Toolchain Integration
- **Experiment Tracking**: MLflow integration for metrics
- **Model Storage**: MinIO backend for model artifacts
- **Workflow Orchestration**: Temporal AI for pipeline execution
- **Analytics**: Prometheus metrics and monitoring

### Service Integration
```python
# Initialize TSAI Toolchain component
toolchain = ToolchainComponent()
toolchain.initialize(config)

# Start experiment
experiment_id = toolchain.start_experiment(
    "hockey-role-classification",
    parameters
)

# Log metrics
toolchain.log_metrics({
    'accuracy': 0.95,
    'precision': 0.92,
    'recall': 0.88,
    'f1_score': 0.90
})

# Store model
model_id = toolchain.store_model(
    model_path,
    model_metadata
)
```

## Performance Metrics

### Target Metrics
- **Accuracy**: ≥85%
- **Precision**: ≥80% (weighted average)
- **Recall**: ≥80% (weighted average)
- **F1-Score**: ≥80% (weighted average)

### Evaluation Metrics
- **Confusion Matrix**: Class-wise performance
- **Per-Class Metrics**: Precision, recall, F1 for each role
- **Training Curves**: Loss and accuracy over epochs
- **Validation Metrics**: Real-time validation performance

## Data Requirements

### Dataset Structure
```
/data/hockey_players/
├── train/
│   ├── player/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── goalie/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── referee/
│       ├── image1.jpg
│       └── image2.jpg
├── val/
│   ├── player/
│   ├── goalie/
│   └── referee/
└── test/
    ├── player/
    ├── goalie/
    └── referee/
```

### Data Quality Requirements
- **Image Resolution**: Minimum 224x224 pixels
- **Image Quality**: Clear, well-lit images
- **Label Accuracy**: Correct role annotations
- **Class Balance**: Balanced distribution across roles
- **Diversity**: Various poses, equipment, lighting conditions

## Training Process

### 1. Data Preparation
- Load dataset from specified path
- Create train/val/test splits
- Apply data augmentation
- Create data loaders

### 2. Model Training
- Initialize pre-trained ResNet18
- Freeze backbone initially
- Train classification head
- Fine-tune entire model
- Apply learning rate scheduling

### 3. Validation
- Evaluate on validation set
- Calculate metrics
- Track best model
- Apply early stopping

### 4. Testing
- Evaluate on test set
- Generate confusion matrix
- Calculate final metrics
- Create evaluation report

## Outputs

### Model Artifacts
- **Trained Model**: `hockey_role_classifier.pt`
- **Model Metadata**: Architecture, hyperparameters, performance
- **Training Logs**: Loss curves, accuracy curves
- **Evaluation Report**: Comprehensive performance analysis

### Reports
- **Performance Report**: JSON format with all metrics
- **Confusion Matrix**: Visual representation of predictions
- **Training Curves**: Loss and accuracy plots
- **Class-wise Analysis**: Per-role performance breakdown

## Monitoring and Analytics

### Real-time Metrics
- **Training Progress**: Loss and accuracy tracking
- **Validation Performance**: Real-time validation metrics
- **Resource Usage**: CPU/GPU utilization
- **Pipeline Status**: Execution progress

### Business Metrics
- **Success Rate**: Pipeline completion rate
- **Accuracy Achievement**: Target accuracy met
- **Model Quality**: Performance metrics
- **User Engagement**: Pipeline usage tracking

## Troubleshooting

### Common Issues
1. **Low Accuracy**: Check data quality, increase training epochs
2. **Overfitting**: Add regularization, reduce model complexity
3. **Class Imbalance**: Apply class weighting, data augmentation
4. **Memory Issues**: Reduce batch size, use gradient checkpointing

### Performance Optimization
1. **GPU Acceleration**: Use CUDA if available
2. **Data Loading**: Use multiple workers, prefetching
3. **Model Optimization**: Quantization, pruning
4. **Batch Processing**: Optimize batch size

## Next Steps

### Phase 2 Preparation
- **Equipment Classification**: Hockey equipment detection
- **Enhanced Features**: Additional role attributes
- **Model Ensemble**: Multiple model combination
- **Real-time Inference**: Production deployment

### Continuous Improvement
- **Data Collection**: Gather more training data
- **Model Updates**: Regular retraining
- **Performance Monitoring**: Ongoing evaluation
- **Feature Engineering**: Advanced preprocessing

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **Torchvision**: Computer vision utilities
- **Scikit-learn**: Machine learning metrics
- **OpenCV**: Image processing
- **Ultralytics**: YOLO integration

### TSAI Integration
- **TSAI Toolchain**: Hockey detection pipelines
- **Jarvis Core**: Storage, experiment, workflow services
- **Temporal AI**: Workflow orchestration
- **MLflow**: Experiment tracking

## License

This implementation is part of the TSAI ecosystem and follows the same licensing terms.
