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

def train_model(config):
    # Validate dataset size
    if config['data']['num_classes'] != 1000:
        raise ValueError("ImageNet-1K requires num_classes=1000")
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Initialize datasets and dataloaders
    train_dataset = ImageNet(root=config.data_path, split='train', transform=train_transform)
    val_dataset = ImageNet(root=config.data_path, split='val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers)

    # Initialize model
    model = ResNet50(num_classes=1000).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                               momentum=0.9, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}') as pbar:
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
        if (epoch + 1) % config.save_freq == 0:
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
    
    # Initialize wandb
    if config['logging']['wandb_enabled']:
        wandb.init(project=config['logging']['project_name'], config=config)
    
    train_model(config) 