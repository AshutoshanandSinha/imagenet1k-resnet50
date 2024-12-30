import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet
from models.resnet50 import ResNet50
import wandb
from tqdm import tqdm
import yaml
import random
import numpy as np
from pathlib import Path
from datasets.imagenet import CustomImageNet
from torchsummary import summary
from math import floor

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_subset_indices(dataset, num_classes=10, samples_per_class=100):
    """Create indices for a balanced subset of ImageNet-1K data
    
    Args:
        dataset: ImageNet-1K dataset
        num_classes: Number of classes to include in subset (default: 10)
        samples_per_class: Number of samples per class (default: 100)
    
    Returns:
        list: Indices of samples to include in subset
    """
    if num_classes > 1000:
        raise ValueError("ImageNet-1K only contains 1000 classes")
        
    class_counts = {i: 0 for i in range(num_classes)}
    indices = []
    
    for idx, (_, label) in enumerate(dataset):
        if label < num_classes and class_counts[label] < samples_per_class:
            indices.append(idx)
            class_counts[label] += 1
        
        # Early exit condition
        if all(count >= samples_per_class for count in class_counts.values()):
            break
            
    if any(count < samples_per_class for count in class_counts.values()):
        print("Warning: Some classes have fewer than requested samples")
    
    return indices

def calculate_rf(model):
    """Calculate receptive field for ResNet50"""
    # Initial values
    rf = 1
    stride = 1
    padding = 0
    
    # Layer parameters for ResNet50
    layers = [
        # Initial conv
        {"kernel": 7, "stride": 2, "padding": 3},
        # MaxPool
        {"kernel": 3, "stride": 2, "padding": 1},
        # Conv layers in blocks (only counting 3x3 convs)
        *[{"kernel": 3, "stride": 1, "padding": 1}] * 3,  # Layer1
        {"kernel": 3, "stride": 2, "padding": 1},         # First block of Layer2
        *[{"kernel": 3, "stride": 1, "padding": 1}] * 3,  # Rest of Layer2
        {"kernel": 3, "stride": 2, "padding": 1},         # First block of Layer3
        *[{"kernel": 3, "stride": 1, "padding": 1}] * 5,  # Rest of Layer3
        {"kernel": 3, "stride": 2, "padding": 1},         # First block of Layer4
        *[{"kernel": 3, "stride": 1, "padding": 1}] * 2   # Rest of Layer4
    ]
    
    for layer in layers:
        rf = calculate_layer_rf(
            rf, 
            layer["kernel"], 
            layer["stride"], 
            layer["padding"]
        )
    
    return rf

def calculate_layer_rf(rf_in, kernel_size, stride, padding):
    """Calculate receptive field after a layer"""
    return (rf_in - 1) * stride + kernel_size

def print_model_summary(model, input_size=(3, 224, 224)):
    """Print model summary and calculations"""
    print("\nModel Summary:")
    print("=" * 50)
    summary(model, input_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nParameter Counts:")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate receptive field
    rf = calculate_rf(model)
    print("\nReceptive Field Analysis:")
    print("=" * 50)
    print(f"Theoretical receptive field: {rf}x{rf} pixels")
    
def train_model_subset(config):
    # Validate dataset path
    data_path = Path(config['data']['data_path'])
    if not (data_path / 'train').exists() or not (data_path / 'val').exists():
        raise ValueError(f"Dataset not found at {data_path}. Please run download_dataset.py first.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create transforms (same as original)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['data']['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=config['augmentation']['color_jitter']['brightness'],
            contrast=config['augmentation']['color_jitter']['contrast'],
            saturation=config['augmentation']['color_jitter']['saturation']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['normalize']['mean'],
            std=config['augmentation']['normalize']['std']
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = CustomImageNet(root=config['data']['data_path'], 
                                 split='train', 
                                 transform=train_transform)
    val_dataset = CustomImageNet(root=config['data']['data_path'], 
                               split='val', 
                               transform=val_transform)
    
    # Create subsets
    train_indices = create_subset_indices(train_dataset, 
                                        num_classes=10, 
                                        samples_per_class=100)
    val_indices = create_subset_indices(val_dataset, 
                                      num_classes=10, 
                                      samples_per_class=50)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, 
                            batch_size=32,  # Smaller batch size for subset
                            shuffle=True, 
                            num_workers=4)
    val_loader = DataLoader(val_subset, 
                          batch_size=32, 
                          shuffle=False, 
                          num_workers=4)

    # Initialize model
    model = ResNet50(num_classes=10).to(device)
    
    # Print model summary before training
    print_model_summary(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                               lr=0.01,
                               momentum=0.9,
                               weight_decay=1e-4)
    
    # Increase epochs from 10 to 50
    num_epochs = 50  # Changed from 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize wandb for subset training
    if config['logging']['wandb_enabled']:
        wandb.init(
            project=f"{config['logging']['project_name']}-subset",
            config={**config, 'subset_training': {
                'num_classes': 10,
                'samples_per_class': 100,
                'val_samples_per_class': 50
            }}
        )

    # Training loop
    for epoch in range(num_epochs):  # Changed from range(10)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:  # Updated epoch display
            for images, targets in pbar:
                images, targets = images.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'Loss': train_loss/(pbar.n+1), 
                                'Acc': 100.*correct/total})
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss/len(train_loader),
            'train_acc': 100.*correct/total,
            'val_loss': val_loss/len(val_loader),
            'val_acc': 100.*val_correct/val_total,
            'learning_rate': scheduler.get_last_lr()[0]
        }
        
        if config['logging']['wandb_enabled']:
            wandb.log(metrics)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {metrics["train_loss"]:.4f}, Train Acc: {metrics["train_acc"]:.2f}%')
        print(f'Val Loss: {metrics["val_loss"]:.4f}, Val Acc: {metrics["val_acc"]:.2f}%')
        scheduler.step()

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_model_subset(config) 