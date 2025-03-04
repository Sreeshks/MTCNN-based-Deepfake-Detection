# Deepfake Detection System

A robust, deep learning-based system for detecting AI-generated fake videos (deepfakes) using computer vision and neural networks.

## Overview

This project implements a deepfake detection system that analyzes videos to determine whether they contain AI-manipulated facial content. The system uses a combination of face detection and feature extraction techniques, powered by a modified ResNet-50 architecture.

## Features

- **Face Detection**: Utilizes MTCNN (Multi-task Cascaded Convolutional Networks) to accurately detect faces in video frames
- **Feature Extraction**: Employs a customized ResNet-50 model to extract facial features and discriminate between real and fake content
- **Video Processing**: Frame-by-frame analysis with customizable sample rates for efficient processing
- **Visual Output**: Option to generate annotated output videos with bounding boxes and confidence scores
- **Training Pipeline**: Complete dataset handling and model training functionality
- **Evaluation Tools**: Performance metrics calculation on test datasets

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- facenet-pytorch
- NumPy
- PIL

## Installation

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

## Dataset

This project uses the "1000 Videos Split" dataset available on Kaggle:
[https://www.kaggle.com/datasets/nanduncs/1000-videos-split](https://www.kaggle.com/datasets/nanduncs/1000-videos-split)

The dataset contains a collection of real and fake videos split into training, testing, and validation sets, making it ideal for training and evaluating deepfake detection models.

## Usage

### Training the Model

```python
from deepfake_detector import DeepfakeDetector, DeepfakeDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Initialize detector
detector = DeepfakeDetector()

# Create datasets and loaders
train_dataset = DeepfakeDataset("path/to/train_data", transform=detector.transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(detector.resnet.parameters(), lr=0.001)
detector.train(train_loader, criterion, optimizer, num_epochs=10)
detector.save_model("deepfake_model.pth")
```

### Analyzing a Video

```python
from deepfake_detector import DeepfakeDetector

# Load trained model
detector = DeepfakeDetector()
detector.load_model("deepfake_model.pth")

# Process video
result = detector.process_video(
    "path/to/video.mp4", 
    output_path="output_video.mp4",
    frame_sample_rate=5
)

# Check results
if result:
    print(f"Video prediction: {result['prediction']}")
    print(f"Confidence score: {result['avg_probability_fake']:.2f}")
```

## Dataset Structure

The system expects a dataset organized as follows:

```
dataset_root/
├── train/
│   ├── real/
│   │   └── [real video files]
│   └── fake/
│       └── [fake video files]
├── test/
│   ├── real/
│   └── fake/
└── validation/
    ├── real/
    └── fake/
```

## How It Works

1. **Face Detection**: The system uses MTCNN to detect faces in each sampled video frame
2. **Feature Extraction**: Each detected face is passed through a customized ResNet-50 model
3. **Classification**: The model outputs a probability score indicating whether the face is real or fake
4. **Video-Level Decision**: Scores across all frames are aggregated to make a final prediction

## Model Architecture

The system uses a modified ResNet-50 architecture with:
- Pre-trained ImageNet weights
- Custom fully connected layers
- Dropout for regularization
- Data augmentation during training (rotation, color jitter, etc.)

## Performance

The system's performance depends on:
- Quality and diversity of the training dataset
- Resolution of input videos
- Number of frames sampled per video
- Face detection confidence thresholds

## Future Improvements

- Integration with additional feature extraction techniques
- Support for multi-modal analysis (audio + video)
- Temporal analysis across frame sequences
- Attention mechanisms for focusing on manipulation artifacts

## Acknowledgements

- The facenet-pytorch library for MTCNN implementation
- PyTorch for the pretrained ResNet-50 model
- Dataset provided by [1000 Videos Split](https://www.kaggle.com/datasets/nanduncs/1000-videos-split) on Kaggle
