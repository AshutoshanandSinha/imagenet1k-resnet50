import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import yaml
from pathlib import Path
from datasets.imagenet import CustomImageNet
from models.resnet50 import ResNet50
from torchsummary import summary
from math import floor

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

def train_model(config):
    # Validate dataset path
    data_path = Path(config['data']['data_path'])
    if not (data_path / 'train').exists() or not (data_path / 'val').exists():
        raise ValueError(f"Dataset not found at {data_path}. Please run download_dataset.py first.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create transforms
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

    # Create datasets and dataloaders
    train_dataset = CustomImageNet(root=config['data']['data_path'], 
                                 split='train', 
                                 transform=train_transform)
    val_dataset = CustomImageNet(root=config['data']['data_path'], 
                               split='val', 
                               transform=val_transform)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['training']['batch_size'],
                            shuffle=True, 
                            num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, 
                          batch_size=config['training']['batch_size'], 
                          shuffle=False, 
                          num_workers=config['data']['num_workers'])

    # Initialize model
    model = ResNet50(num_classes=1000).to(device)
    
    # Print model summary before training
    print_model_summary(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        momentum=float(config['training']['momentum']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                          T_max=config['training']['epochs'])

    # Initialize wandb
    if config['logging']['wandb_enabled']:
        wandb.init(project=config['logging']['project_name'])
        wandb.config.update(config)


    # Initialize best accuracy tracking
    best_val_acc = 0.0
    best_model_path = Path(config['training']['model_save_path'])
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize gradient scaler for AMP
    scaler = torch.amp.GradScaler()

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}') as pbar:
            for images, targets in pbar:
                images, targets = images.to(device), targets.to(device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
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
        
        # Calculate metrics
        val_acc = 100. * val_correct / val_total
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss/len(train_loader),
            'train_acc': 100.*correct/total,
            'val_loss': val_loss/len(val_loader),
            'val_acc': val_acc,
            'learning_rate': scheduler.get_last_lr()[0]
        }

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss/len(val_loader),
            }, best_model_path)
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

        if config['logging']['wandb_enabled']:
            wandb.log(metrics)
        
        print(f'Epoch {epoch+1}/{config["training"]["epochs"]}:')
        print(f'Train Loss: {metrics["train_loss"]:.4f}, Train Acc: {metrics["train_acc"]:.2f}%')
        print(f'Val Loss: {metrics["val_loss"]:.4f}, Val Acc: {metrics["val_acc"]:.2f}%')
        
        scheduler.step()

if __name__ == "__main__":
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config) 