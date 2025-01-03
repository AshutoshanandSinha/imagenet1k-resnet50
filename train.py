import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pathlib import Path
from datasets.imagenet import CustomImageNet
import yaml
import time
import logging
from torch.amp import GradScaler, autocast
from torchvision.models import ResNet50_Weights
from torchsummary import torchsummary
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_lr(optimizer):
    """Helper function to get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def create_transforms():
    # Training transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform

def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Log device information
    device = torch.device(config['hardware']['device'])
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    # Create transforms
    train_transform, val_transform = create_transforms()

    # Create datasets
    train_dataset = CustomImageNet(
        root=config['data']['data_path'],
        split='train',
        transform=train_transform
    )

    val_dataset = CustomImageNet(
        root=config['data']['data_path'],
        split='val',
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        prefetch_factor=config['training']['prefetch_factor']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        prefetch_factor=config['training']['prefetch_factor']
    )

    # Create model (explicitly not pretrained)
    model = models.resnet50(weights=None)  # Ensure model is not pretrained
    model = model.to(device)

    # Print model summary
    logger.info("\nModel Summary:")
    torchsummary.summary(model, (3, 224, 224))

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("\nParameter Counts:")
    logger.info("=" * 50)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info("=" * 50 + "\n")

    if config['hardware']['multi_gpu']:
        model = nn.DataParallel(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )

    # Create checkpoint directory
    Path(config['training']['model_save_path']).parent.mkdir(parents=True, exist_ok=True)

    # Create gradient scaler
    scaler = GradScaler()

    # Training loop
    best_acc = 0.0
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()

        # Add tqdm progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config["training"]["epochs"]}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Warmup learning rate
            if epoch < config['training']['warmup_epochs']:
                lr_scale = min(1., (epoch * len(train_loader) + batch_idx + 1) / 
                             (config['training']['warmup_epochs'] * len(train_loader)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['training']['learning_rate'] * lr_scale
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            optimizer.step()
            scheduler.step()
            
            # Calculate metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # Update progress bar
            current_lr = get_lr(optimizer)
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%',
                'LR': f'{current_lr:.6f}'
            })

            if (batch_idx + 1) % config['logging']['log_interval'] == 0:
                logger.info(f'Epoch: {epoch + 1}/{config["training"]["epochs"]} | '
                          f'Batch: {batch_idx + 1}/{len(train_loader)} | '
                          f'Loss: {train_loss/(batch_idx+1):.3f} | '
                          f'Acc: {100.*train_correct/train_total:.2f}% | '
                          f'LR: {current_lr:.6f}')

        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch + 1} training completed in {epoch_time:.2f}s')

        # Validation phase with progress bar
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc='Validation')
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # Update validation progress bar
                val_pbar.set_postfix({
                    'Loss': f'{val_loss/val_total:.3f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })

        # Calculate accuracy
        val_acc = 100. * val_correct / val_total

        logger.info(f'Validation Loss: {val_loss/len(val_loader):.3f} | '
                   f'Validation Acc: {val_acc:.2f}%')

        # Save checkpoint if validation accuracy improves
        if val_acc > best_acc:
            logger.info(f'Validation accuracy improved from {best_acc:.2f} to {val_acc:.2f}')
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, config['training']['model_save_path'])

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with exception")
        raise