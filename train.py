import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from models.resnet50 import ResNet50
import wandb
from tqdm import tqdm
import yaml
from pathlib import Path
from datasets.imagenet import CustomImageNet

def train_model(config):
    # Validate dataset path
    data_path = Path(config['data']['data_path'])
    if not (data_path / 'train').exists() or not (data_path / 'val').exists():
        raise ValueError(f"Dataset not found at {data_path}. Please run download_dataset.py first.")

    # Set device and handle multi-GPU
    if torch.cuda.is_available() and config['hardware']['device'] == 'cuda':
        device = torch.device('cuda')
        if config['hardware']['multi_gpu'] and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            use_multi_gpu = True
        else:
            use_multi_gpu = False
    else:
        device = torch.device('cpu')
        use_multi_gpu = False
        print("Warning: Using CPU. Training will be slow!")

    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['data']['input_size']),
        transforms.RandomHorizontalFlip() if config['augmentation']['random_horizontal_flip'] else None,
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
        transforms.CenterCrop(config['data']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['augmentation']['normalize']['mean'],
            std=config['augmentation']['normalize']['std']
        )
    ])

    # Create datasets
    train_dataset = CustomImageNet(root=config['data']['data_path'], 
                                 split='train', 
                                 transform=train_transform)
    val_dataset = CustomImageNet(root=config['data']['data_path'], 
                               split='val', 
                               transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize model
    model = ResNet50(num_classes=config['data']['num_classes'])
    if use_multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Create output directories
    output_dir = Path('outputs')
    checkpoint_dir = output_dir / 'checkpoints'
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    # Initialize wandb
    if config['logging']['wandb_enabled']:
        try:
            wandb.init(project=config['logging']['project_name'], config=config)
            wandb.watch(model)
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {str(e)}")
            config['logging']['wandb_enabled'] = False

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['learning_rate'],
                               momentum=config['training']['momentum'], weight_decay=config['training']['weight_decay'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config['training']['epochs']}') as pbar:
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
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss/len(train_loader),
            'train_acc': 100.*correct/total,
            'val_loss': val_loss/len(val_loader),
            'val_acc': 100.*val_correct/val_total
        })
        
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'checkpoints/resnet50_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_model(config) 