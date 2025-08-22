# YOLO Training Project Documentation

This project implements a custom YOLOv8 object detection model training pipeline using Google Colab and Ultralytics. The project focuses on detecting and classifying objects in images using the state-of-the-art YOLOv8 architecture.

## Table of Contents

- [Overview](#overview)
- [Project Resources](#project-resources)
- [Dataset Structure](#dataset-structure)
- [Setup and Installation](#setup-and-installation)
- [Training Process](#training-process)
- [Results and Validation](#results-and-validation)
- [Understanding the Training Output](#understanding-the-training-output)
- [Model Performance Metrics](#model-performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Overview

This project trains a YOLOv8 model for object detection with three classes:
- `early` - Early stage objects
- `healthy` - Healthy objects  
- `not_early` - Non-early stage objects

The training process uses transfer learning from a pre-trained YOLOv8n model, fine-tuning it on custom dataset for 100 epochs.

## Project Resources

### Data and Code
- **Dataset**: [Google Drive - Training Data](https://drive.google.com/drive/folders/16bWvOFGsbTiWHP639rapec46xEMf-b6l?usp=sharing)
- **GitHub Repository**: [YOLO Training Notebook](https://github.com/tlb-unilag/yolo-training/blob/main/TLB_YOLO_Training.ipynb)
- **Project Website**: [TLB Project](https://tlbproject.com.ng/)
- **Data Repository**: [Mendeley Data](https://data.mendeley.com/datasets/3knm93dkc5/1)

### Setup Guides
- **Label Studio Setup**: [GitHub - Label Studio](https://github.com/HumanSignal/label-studio)
- **Video Tutorial**: [YouTube - Label Studio Guide](https://www.youtube.com/watch?v=i7pHt0701zo)

## Dataset Structure

The dataset follows the standard YOLO format with the following directory structure:

```
taro_test/full/
├── classes.txt          # Class names definition
├── config.yml           # Training configuration file
├── images/              # All images directory
├── labels/              # All labels directory
├── notes.json           # Dataset metadata
├── test/                # Test split
│   ├── images/
│   └── labels/
├── test.txt             # Test set file paths
├── train/               # Training split
│   ├── images/
│   └── labels/
├── train.txt            # Training set file paths
├── val/                 # Validation split
│   ├── images/
│   └── labels/
└── val.txt              # Validation set file paths
```

### Dataset Statistics
- **Training Images**: 3,679 images (3 corrupt images ignored)
- **Validation Images**: 1,050 images
- **Total Instances**: 1,065 labeled objects
- **Classes**: 3 (early: 326, healthy: 374, not_early: 365)

## Setup and Installation

### 1. Google Colab Environment Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
```

### 2. Install Required Packages

```bash
# Install Ultralytics YOLOv8
!pip install -U ultralytics
```

### 3. Verify Dataset Structure

```bash
# Check dataset contents
!ls '/gdrive/My Drive/taro_test/full'
```

## Training Process

### Step 1: Initialize Model and Configuration

```python
import torch
from ultralytics import YOLO

# Define path to training configuration
training_config = '/gdrive/My Drive/taro_test/full/config.yml'

# Load pre-trained YOLOv8 nano model
model = YOLO('yolov8n.pt')
```

### Step 2: Start Training

```python
# Train the model for 100 epochs
results = model.train(data=training_config, epochs=100)
```

### Key Training Parameters

The model uses the following default parameters:
- **Base Model**: YOLOv8n (nano - fastest, smallest)
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640x640
- **Optimizer**: AdamW (automatically selected)
- **Learning Rate**: 0.001429 (automatically determined)
- **Device**: CUDA GPU (Tesla T4)
- **Mixed Precision**: Enabled (AMP)

### Step 3: Save Results

```python
import os
import shutil

def copy_files(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            source_file_path = os.path.join(root, file)
            destination_file_path = os.path.join(destination_dir, file)
            shutil.copy(source_file_path, destination_file_path)

# Copy training results to Google Drive
source_directory = "runs"
destination_directory = "/gdrive/My Drive/taro_test/full/results"
copy_files(source_directory, destination_directory)
```

## Results and Validation

### Final Model Performance

After 100 epochs of training, the model achieved:

- **Precision**: 0.768
- **Recall**: 0.857
- **mAP@0.5**: 0.857
- **mAP@0.5-0.95**: 0.673

### Per-Class Performance

| Class | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|-------|-----------|--------|---------|--------------|
| early | 0.709 | 0.843 | 0.822 | 0.674 |
| healthy | 0.794 | 0.912 | 0.918 | 0.727 |
| not_early | 0.801 | 0.816 | 0.832 | 0.619 |

### Inference Speed
- **Preprocessing**: 0.3ms per image
- **Inference**: 2.5ms per image
- **Postprocessing**: 2.9ms per image
- **Total**: ~5.7ms per image

## Understanding the Training Output

### Training Progress Explanation

The training output shows key metrics for each epoch:

1. **GPU Memory**: Memory usage on the GPU (stabilized around 2.24-2.28G)
2. **Loss Functions**:
   - `box_loss`: Bounding box regression loss
   - `cls_loss`: Classification loss
   - `dfl_loss`: Distribution focal loss
3. **Instances**: Number of objects detected per batch
4. **Validation Metrics**:
   - **Box(P)**: Precision for bounding box detection
   - **R**: Recall
   - **mAP50**: Mean Average Precision at IoU threshold 0.5
   - **mAP50-95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95

### Training Phases

The training showed two distinct phases:

1. **Epochs 1-90**: Normal training with mosaic augmentation
   - Loss decreased steadily from ~2.0 to ~0.7
   - mAP@0.5 improved from 0.57 to 0.85

2. **Epochs 91-100**: Mosaic augmentation disabled (fine-tuning phase)
   - Further loss reduction to ~0.4-0.5
   - Final performance stabilization

### Data Quality Issues

The training process identified and handled:
- **3 corrupt images** with invalid bounding box coordinates
- **Automatic data validation** and filtering
- **Robust training** despite data quality issues

## Model Performance Metrics

### Visualization Results

The training generates several important visualization files:

1. **Training Curves** (`results.png`): Shows loss and metric progression
2. **F1 Curve** (`F1_curve.png`): F1 score vs confidence threshold
3. **Precision Curve** (`P_curve.png`): Precision vs confidence threshold  
4. **Recall Curve** (`R_curve.png`): Recall vs confidence threshold
5. **Confusion Matrix** (`confusion_matrix.png`): Classification accuracy breakdown
6. **Sample Predictions** (`val_batch2_pred.jpg`): Visual validation results

### Model Files

Training produces two key model files:
- `best.pt`: Best performing model weights during training
- `last.pt`: Final model weights after all epochs

## Troubleshooting

### Common Issues and Solutions

**1. CUDA Out of Memory**
```python
# Reduce batch size in config.yml
batch: 8  # Instead of 16
```

**2. Corrupt Image Warnings**
- The model automatically handles corrupt images
- Review and fix bounding box annotations if needed
- Ensure coordinates are normalized (0-1 range)

**3. Low Performance**
- Increase training epochs
- Add more diverse training data
- Verify label quality and consistency
- Consider using a larger model (yolov8s.pt, yolov8m.pt)

**4. Google Drive Mount Issues**
```python
# Force remount if needed
drive.mount('/gdrive', force_remount=True)
```

## Next Steps

### Model Deployment
1. **Export Model**: Convert to deployment format (ONNX, TensorRT, etc.)
2. **Inference Pipeline**: Implement real-time detection pipeline
3. **Performance Optimization**: Model pruning, quantization

### Model Improvement
1. **Data Augmentation**: Experiment with different augmentation strategies
2. **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.
3. **Ensemble Methods**: Combine multiple models for better performance
4. **Active Learning**: Identify and label challenging cases

### Validation Commands

```python
# Run custom validation
validation_results = model.val(data=training_config)

# Display results
def display_image(image_path):
    image = cv2.imread(image_path)
    cv2_imshow(image)

# View training metrics
display_image("/path/to/results/results.png")
```

### Inference Example

```python
# Load trained model
model = YOLO('/path/to/best.pt')

# Run inference on new images
results = model.predict('path/to/new/image.jpg')

# Process results
for result in results:
    # Extract bounding boxes, classes, confidence scores
    boxes = result.boxes
    print(f"Detected {len(boxes)} objects")
```

## Additional Resources

- **Ultralytics Documentation**: [YOLOv8 Docs](https://docs.ultralytics.com/)
- **Model Architecture**: YOLOv8n with 3,011,433 parameters
- **Training Hardware**: Tesla T4 GPU with 15GB VRAM
- **Framework**: PyTorch 2.1.0 with CUDA 12.1

---

**Note**: This documentation is based on the training session that achieved 85.7% mAP@0.5 and completed successfully in 3.345 hours. The model shows strong performance across all three classes with the 'healthy' class performing best (91.8% mAP@0.5).