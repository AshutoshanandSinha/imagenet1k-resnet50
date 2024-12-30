# ResNet50 ImageNet-1K Classifier

This repository contains a PyTorch implementation of ResNet50 for training on the ImageNet-1K dataset (ILSVRC2012), targeting 70% top-1 accuracy. It includes support for both full dataset training and subset training for faster experimentation.

## Project Structure 

- `models/resnet50.py`: Defines the ResNet50 architecture.
- `train.py`: Main training script for full ImageNet-1K dataset.
- `train_subset.py`: Training script for subset of ImageNet-1K dataset.
- `download_dataset.py`: Script to download the ImageNet-1K dataset from Kaggle.
- `requirements.txt`: List of dependencies.
- `config/config.yaml`: Configuration file for training.

## Training Goals

- Target: 70% top-1 accuracy on ImageNet-1K validation set
- Training from scratch (no pre-trained weights)
- Expected training time: ~100 epochs
- Hardware requirements: Multi-GPU setup recommended (8x GPUs ideal)
- Estimated training time: 3-4 days on 8x V100 GPUs

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Kaggle credentials:
   - Get your Kaggle API credentials from your account settings
   - The script will download the ImageNet-1K (ILSVRC2012) dataset from the "imagenet-object-localization-challenge"
   - Run `download_dataset.py` and enter your credentials when prompted

3. Download the dataset:
```bash
python download_dataset.py
```

## Configuration

The `config/config.yaml` file contains all training parameters. Key settings for achieving 70% accuracy:

```yaml
training:
  epochs: 100          # Required for target accuracy
  batch_size: 256      # Adjust based on available GPU memory
  learning_rate: 0.1   # Initial learning rate
  weight_decay: 1e-4   # L2 regularization
  
data:
  data_path: "data/imagenet"
  num_workers: 8
  num_classes: 1000    # ImageNet-1K classes
  
model:
  architecture: "resnet50"
  pretrained: false    # Training from scratch
```

## Training Strategy

To achieve 70% top-1 accuracy:

1. **Learning Rate Schedule**:
   - Initial LR: 0.1
   - Cosine decay over 100 epochs
   - Warm-up period: 5 epochs

2. **Data Augmentation Pipeline**:
   - Random resized crop (224x224)
   - Random horizontal flip
   - Color jitter (brightness=0.4, contrast=0.4, saturation=0.4)
   - Normalization with ImageNet statistics

3. **Optimization**:
   - SGD optimizer with momentum (0.9)
   - Weight decay: 1e-4
   - Batch size: 256 per GPU
   - Mixed precision training (FP16)

## Usage

### Full Dataset Training

To train on the complete ImageNet-1K dataset (1000 classes, ~1.2M training images):

```bash
python train.py --config config/config.yaml
```

Features:
- Training on all 1000 classes of ImageNet-1K
- ~1.2M training images, ~50K validation images
- Cosine learning rate scheduling
- Weights & Biases logging
- Automatic checkpointing
- Multi-GPU support

### Subset Training

To train on a subset of ImageNet-1K (useful for prototyping):

```bash
python train_subset.py
```

Features:
- Training on 10 classes from ImageNet-1K
- 100 samples per class for training (1000 total)
- 50 samples per class for validation (500 total)
- Faster experimentation and debugging

## Model Architecture

The ResNet50 implementation includes:
- Bottleneck blocks with expansion factor of 4
- Batch normalization layers
- Residual connections
- Kaiming initialization
- Final layer configured for 1000 classes (ImageNet-1K)

## Data Augmentation

Both training scripts use standard ImageNet augmentations:
- Random resized crop (224x224)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation)
- Normalization with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Monitoring

Training progress is monitored through:
1. Command-line progress bars (tqdm)
2. Weights & Biases dashboard, tracking:
   - Training/validation loss
   - Training/validation accuracy
   - Learning rate
   - Model parameters

## Expected Training Progress

Typical accuracy progression:
- 20 epochs: ~40% top-1 accuracy
- 50 epochs: ~60% top-1 accuracy
- 100 epochs: ~70% top-1 accuracy

Monitor training through Weights & Biases dashboard for:
- Training/validation loss curves
- Top-1/Top-5 accuracy
- Learning rate schedule
- GPU utilization and throughput

## Requirements

```
torch>=1.8.0
torchvision>=0.9.0
wandb
tqdm
numpy
Pillow
pyyaml
kaggle
```





